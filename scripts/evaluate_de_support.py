#!/usr/bin/env python3
"""Evaluate Level 3.1 DE support/sign heads on a held-out dataset cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vcc_prior_model import (
    DifferentialExpressionCache,
    GeneVocabularyArtifacts,
    ModelConfig,
    build_model,
)
from vcc_prior_model.prior_data import PriorArtifacts
from vcc_prior_model.real_data import BackedH5adDataset

WORKSPACE = Path(__file__).resolve().parents[2]


def move(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ranked = label[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def target_metrics(
    probability: np.ndarray,
    sign_logits: np.ndarray,
    observed_effect: np.ndarray,
    label: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    true_count = int(label.sum())
    top_index = np.argpartition(probability, -true_count)[-true_count:]
    predicted_top = np.zeros_like(label)
    predicted_top[top_index] = True
    intersection = int((predicted_top & label).sum())
    union = int((predicted_top | label).sum())
    called = probability >= threshold
    called_true = int((called & label).sum())
    twice_k = min(len(label), 2 * true_count)
    twice_index = np.argpartition(probability, -twice_k)[-twice_k:]
    top_twice = np.zeros_like(label)
    top_twice[twice_index] = True
    return {
        "average_precision": average_precision(probability, label),
        "top_true_count_jaccard": intersection / max(1, union),
        "recall_at_true_count": intersection / true_count,
        "recall_at_twice_true_count": int((top_twice & label).sum()) / true_count,
        "threshold_precision": called_true / max(1, int(called.sum())),
        "threshold_recall": called_true / true_count,
        "predicted_de_count": float(called.sum()),
        "true_de_count": float(true_count),
        "de_count_ratio": float(called.sum()) / true_count,
        "support_brier": float(np.mean((probability - label.astype(float)) ** 2)),
        "de_sign_accuracy": float(
            np.mean(np.sign(sign_logits[label]) == np.sign(observed_effect[label]))
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    if not config.factorized_effect:
        raise ValueError("checkpoint does not use the Level 3.1 factorized effect head")
    model = build_model(checkpoint["level"], config, vocabulary.output_gene_index).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    priors = PriorArtifacts(args.priors, vocabulary).load(device)
    cache = DifferentialExpressionCache(args.de_cache)
    use_amp = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    records: list[dict[str, Any]] = []
    with BackedH5adDataset(args.data, vocabulary) as dataset, torch.inference_mode():
        context_rows = dataset.sample_indices(
            dataset.control_indices,
            min(args.context_cells, len(dataset.control_indices)),
            rng,
        )
        gene_embeddings = model.encode_genes(priors)

        def context_chunks() -> Any:
            for counts in dataset.input_chunks(context_rows, args.context_chunk_size):
                yield move(counts, device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            context = model.encode_context_chunks(
                context_chunks(),
                gene_embeddings,
                input_gene_index=dataset.alignment.input_gene_index.to(device),
            )
        targets = cache.targets(dataset.path, informative_only=True)
        for target in targets:
            rows = dataset.sample_indices(dataset.control_indices, args.query_cells, rng)
            query = dataset.read_inputs(rows)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model.predict_from_context(
                    move(query, device),
                    vocabulary.target_index([target]).to(device),
                    context,
                    query_gene_index=dataset.alignment.input_gene_index.to(device),
                    decode_gene_index=dataset.alignment.decode_gene_index.to(device),
                )
            if output.effect_support_probability is None or output.effect_sign_logits is None:
                raise RuntimeError("factorized effect diagnostics are missing")
            cached = cache.get(dataset.path, target, "cpu")
            probability = output.effect_support_probability[0].float().cpu().numpy()
            sign_logits = output.effect_sign_logits[0].float().mean(dim=0).cpu().numpy()
            label = cached.de_mask.numpy()
            observed_effect = cached.log_fold_change.numpy()
            record = {
                "target": target,
                **target_metrics(
                    probability,
                    sign_logits,
                    observed_effect,
                    label,
                    args.threshold,
                ),
                "effect_strength": float(output.effect_strength[0, 0]),
            }
            records.append(record)
    metrics = [key for key in records[0] if key != "target"]
    summary = {key: float(np.nanmean([record[key] for record in records])) for key in metrics}
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "model_config": asdict(config),
        "dataset": str(args.data),
        "targets": len(records),
        "threshold": args.threshold,
        "summary": summary,
        "per_target": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--de-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument("--priors", type=Path, default=WORKSPACE / "priorcell/artifacts/priors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--context-cells", type=int, default=18_400)
    parser.add_argument("--context-chunk-size", type=int, default=256)
    parser.add_argument("--query-cells", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2036)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
