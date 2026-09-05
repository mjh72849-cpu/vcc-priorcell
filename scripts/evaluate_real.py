#!/usr/bin/env python3
"""Evaluate a real checkpoint on held-out perturbation targets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from vcc_prior_model import GeneVocabularyArtifacts, ModelConfig, build_model
from vcc_prior_model.losses import sliced_wasserstein_distance
from vcc_prior_model.real_data import BackedH5adDataset

WORKSPACE = Path(__file__).resolve().parents[2]


def move(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)


def normalize(counts: torch.Tensor) -> torch.Tensor:
    library = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.log1p(counts * (10_000.0 / library))


def effect_metrics(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    control: torch.Tensor,
    *,
    top_k: int,
    projections: int,
    projection_seed: int,
) -> dict[str, float]:
    predicted_normalized = normalize(predicted)
    observed_normalized = normalize(observed)
    control_normalized = normalize(control)
    predicted_delta = predicted_normalized.mean(dim=1) - control_normalized.mean(dim=1)
    observed_delta = observed_normalized.mean(dim=1) - control_normalized.mean(dim=1)
    delta_mse = F.mse_loss(predicted_delta, observed_delta)
    cosine = F.cosine_similarity(predicted_delta, observed_delta, dim=-1).mean()
    k = min(top_k, predicted_delta.shape[-1])
    predicted_top = set(
        predicted_delta.abs().topk(k, dim=-1).indices[0].detach().cpu().tolist()
    )
    observed_top = set(
        observed_delta.abs().topk(k, dim=-1).indices[0].detach().cpu().tolist()
    )
    generator = torch.Generator(device=predicted.device).manual_seed(projection_seed)
    directions = torch.randn(
        predicted.shape[-1],
        projections,
        device=predicted.device,
        dtype=predicted.dtype,
        generator=generator,
    )
    directions = F.normalize(directions, dim=0)
    distribution = sliced_wasserstein_distance(
        predicted,
        observed,
        num_projections=projections,
        projection_directions=directions,
    )
    return {
        "delta_mse": float(delta_mse),
        "delta_cosine": float(cosine),
        "top_gene_jaccard": len(predicted_top & observed_top)
        / max(1, len(predicted_top | observed_top)),
        "pseudobulk_mae": float(
            (predicted_normalized.mean(dim=1) - observed_normalized.mean(dim=1))
            .abs()
            .mean()
        ),
        "sliced_wasserstein": float(distribution),
    }


def expand_context(context: Any, batch: int) -> Any:
    """Expand batch-indexed context fields without copying global gene embeddings."""

    return type(context)(
        context.context_embedding.expand(batch, -1),
        context.state_prototypes.expand(batch, -1, -1),
        context.gene_embeddings,
        context.gene_context_stats.expand(batch, -1, -1),
        context.gene_context_observed.expand(batch, -1),
    )


def average(records: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    keys = records[0][prefix]
    return {
        key: float(np.mean([record[prefix][key] for record in records])) for key in keys
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if any(not 0.0 <= weight <= 1.0 for weight in args.decoder_baseline_weights):
        raise ValueError("decoder baseline weights must be in [0, 1]")
    if any(scale < 0.0 for scale in args.effect_scales):
        raise ValueError("effect scales must be non-negative")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    if config.num_genes != vocabulary.num_genes:
        raise ValueError("checkpoint and current global vocabulary have different sizes")
    if tuple(checkpoint["global_gene_symbols"]) != vocabulary.global_gene_symbols:
        raise ValueError("checkpoint and current global vocabulary have different symbols")
    model = build_model(
        checkpoint["level"], config, vocabulary.output_gene_index
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    use_amp = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    records: list[dict[str, Any]] = []
    with BackedH5adDataset(args.data, vocabulary) as dataset, torch.inference_mode():
        dataset.validate_raw_counts()
        context_rows = dataset.sample_indices(
            dataset.control_indices,
            min(args.context_cells, len(dataset.control_indices)),
            rng,
        )
        gene_embeddings = model.encode_genes()

        def context_chunks() -> Any:
            for counts in dataset.input_chunks(context_rows, args.context_chunk_size):
                yield move(counts, device)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            context = model.encode_context_chunks(
                context_chunks(),
                gene_embeddings,
                input_gene_index=dataset.alignment.input_gene_index.to(device),
            )
        targets = list(dataset.perturbation_targets)
        if args.target_list is not None:
            with args.target_list.open(newline="", encoding="utf-8") as handle:
                requested = [row["target_gene"] for row in csv.DictReader(handle)]
            available = set(targets)
            targets = [target for target in requested if target in available]
        if args.max_targets is not None:
            targets = targets[: args.max_targets]
        for target_number, target in enumerate(targets):
            query, control_observed = dataset.sample_control(args.cells, rng)
            perturbed_observed = dataset.sample_perturbed(target, args.cells, rng)
            query_device = move(query, device)
            control_device = move(control_observed, device)
            observed_device = move(perturbed_observed, device)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                output = model.predict_from_context(
                    query_device,
                    vocabulary.target_index([target]).to(device),
                    context,
                    query_gene_index=dataset.alignment.input_gene_index.to(device),
                    decode_gene_index=dataset.alignment.decode_gene_index.to(device),
                )
            raw_log_effect = output.log_effect.float()
            decoder = output.mean.float()
            baseline = effect_metrics(
                control_device,
                observed_device,
                control_device,
                top_k=args.top_k,
                projections=args.distribution_projections,
                projection_seed=args.seed + target_number,
            )
            record = {
                "target": target,
                "cells": args.cells,
                "baseline": baseline,
                "decoder": effect_metrics(
                    decoder,
                    observed_device,
                    control_device,
                    top_k=args.top_k,
                    projections=args.distribution_projections,
                    projection_seed=args.seed + target_number,
                ),
            }
            for decoder_weight in args.decoder_baseline_weights:
                prediction_baseline = (
                    (1.0 - decoder_weight) * control_device
                    + decoder_weight * output.control_mean.float()
                )
                for effect_scale in args.effect_scales:
                    variant = f"anchor_w{decoder_weight:g}_effect_s{effect_scale:g}"
                    predicted = prediction_baseline * (
                        effect_scale * raw_log_effect
                    ).exp()
                    record[variant] = effect_metrics(
                        predicted,
                        observed_device,
                        control_device,
                        top_k=args.top_k,
                        projections=args.distribution_projections,
                        projection_seed=args.seed + target_number,
                    )
            records.append(record)
            print(json.dumps(record), flush=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_level": checkpoint["level"],
        "checkpoint_step": checkpoint["step"],
        "model_config": asdict(config),
        "dataset": str(args.data),
        "targets": len(records),
        "summary": {
            name: average(records, name)
            for name in records[0]
            if name not in {"target", "cells"}
        },
        "per_target": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--context-cells", type=int, default=18_400)
    parser.add_argument("--context-chunk-size", type=int, default=256)
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--distribution-projections", type=int, default=16)
    parser.add_argument(
        "--decoder-baseline-weights", nargs="+", type=float, default=[0.0, 0.1]
    )
    parser.add_argument(
        "--effect-scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--target-list", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
