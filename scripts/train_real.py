#!/usr/bin/env python3
"""Train a learned PriorCell architecture level on real raw-count H5AD data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vcc_prior_model import (
    CompositePerturbationLoss,
    GeneVocabularyArtifacts,
    LossConfig,
    ModelConfig,
    build_model,
    control_reconstruction_loss,
)
from vcc_prior_model.real_data import BackedH5adDataset
from vcc_prior_model.prior_data import PriorArtifacts
from vcc_prior_model.training import training_schedule

WORKSPACE = Path(__file__).resolve().parents[2]


def read_targets(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["target_gene"] for row in csv.DictReader(handle)]


def move(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)


def choose_target(
    dataset: BackedH5adDataset,
    vcc_targets: set[str],
    rng: np.random.Generator,
    minimum_cells: int,
    vcc_target_probability: float,
    preference_minimum_targets: int,
) -> str:
    eligible = [
        target
        for target in dataset.perturbation_targets
        if len(dataset.groups[target]) >= minimum_cells
    ]
    preferred = (
        [target for target in eligible if target in vcc_targets]
        if len(eligible) >= preference_minimum_targets
        else []
    )
    pool = (
        preferred
        if preferred and rng.random() < vcc_target_probability
        else eligible
    )
    if not pool:
        raise ValueError(f"{dataset.name}: no perturbations pass minimum_cells")
    return pool[int(rng.integers(len(pool)))]


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    *,
    level: str,
    step: int,
    args: argparse.Namespace,
    vocabulary: GeneVocabularyArtifacts,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "level": level,
            "step": step,
            "model_config": asdict(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "output_gene_index": vocabulary.output_gene_index,
            "global_gene_symbols": vocabulary.global_gene_symbols,
            "output_gene_symbols": vocabulary.output_gene_symbols,
            "arguments": vars(args),
            "rng_state": rng.bit_generator.state,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def load_shared_weights(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint["model_state"].items()
        if key in current and current[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        json.dumps(
            {
                "initialized_from": str(path),
                "loaded_tensors": len(compatible),
                "new_or_missing_tensors": len(missing),
                "unexpected_tensors": len(unexpected),
            }
        )
    )
    return checkpoint


def resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    level: str,
    config: ModelConfig,
    vocabulary: GeneVocabularyArtifacts,
    rng: np.random.Generator,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("level") != level:
        raise ValueError("resume checkpoint architecture level does not match")
    if checkpoint.get("model_config") != asdict(config):
        raise ValueError("resume checkpoint model configuration does not match")
    if tuple(checkpoint.get("global_gene_symbols", ())) != vocabulary.global_gene_symbols:
        raise ValueError("resume checkpoint gene vocabulary does not match")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["rng_state"]
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    step = int(checkpoint["step"])
    print(json.dumps({"resumed_from": str(path), "completed_step": step}))
    return step


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    prior_artifacts = (
        PriorArtifacts(args.priors, vocabulary)
        if args.level in {"functional_prior", "full_prior"}
        else None
    )
    config = ModelConfig(
        num_genes=vocabulary.num_genes,
        num_output_genes=vocabulary.num_output_genes,
        gene_dim=args.model_dim,
        cell_dim=args.model_dim,
        context_dim=args.model_dim,
        perturbation_dim=args.model_dim,
        decoder_dim=args.decoder_dim,
        num_go_terms=(prior_artifacts.num_go_terms if prior_artifacts else 0),
        num_pathway_terms=(
            prior_artifacts.num_pathway_terms if prior_artifacts else 0
        ),
        graph_layers=args.graph_layers,
        context_prototypes=args.context_prototypes,
        dropout=args.dropout,
        prior_dropout=(args.prior_dropout if prior_artifacts else 0.0),
        prior_gate_init_bias=args.prior_gate_init_bias,
    )
    model = build_model(args.level, config, vocabulary.output_gene_index).to(device)
    priors = prior_artifacts.load(device) if prior_artifacts else None
    if args.initialize_from is not None and args.resume is not None:
        raise ValueError("--initialize-from and --resume are mutually exclusive")
    if args.initialize_from is not None:
        load_shared_weights(model, args.initialize_from)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    if args.resume is not None:
        start_step = resume_checkpoint(
            model,
            optimizer,
            args.resume,
            level=args.level,
            config=config,
            vocabulary=vocabulary,
            rng=rng,
        )
    perturbation_loss = CompositePerturbationLoss(
        LossConfig(
            expression_weight=args.expression_weight,
            distribution_weight=args.distribution_weight,
            delta_weight=args.delta_weight,
            direction_weight=args.direction_weight,
            de_weight=args.de_weight,
            magnitude_weight=args.magnitude_weight,
            target_weight=args.target_weight,
            de_direction_weight=args.de_direction_weight,
            unsupervised_effect_weight=0.0,
            distribution_projections=args.distribution_projections,
        )
    )
    vcc_targets = set(read_targets(args.perturbations))
    control_datasets = [
        BackedH5adDataset(path, vocabulary, name=f"vcc_{name}")
        for name, path in zip(("A", "B", "C"), args.vcc_controls, strict=True)
    ]
    perturbation_datasets = [
        BackedH5adDataset(path, vocabulary) for path in args.perturbation_data
    ]
    if args.dataset_weights is None:
        dataset_weights = np.full(
            len(perturbation_datasets), 1.0 / len(perturbation_datasets)
        )
    else:
        if len(args.dataset_weights) != len(perturbation_datasets):
            raise ValueError("--dataset-weights must match --perturbation-data length")
        dataset_weights = np.asarray(args.dataset_weights, dtype=np.float64)
        if not np.isfinite(dataset_weights).all() or (dataset_weights <= 0).any():
            raise ValueError("dataset weights must be finite and positive")
        dataset_weights /= dataset_weights.sum()
    for dataset_number, dataset in enumerate(
        [*control_datasets, *perturbation_datasets]
    ):
        perturbation_number = dataset_number - len(control_datasets)
        dataset.validate_raw_counts()
        print(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "cells": dataset.num_cells,
                    "input_genes": int(dataset.alignment.input_gene_index.numel()),
                    "output_overlap": int(dataset.alignment.decode_gene_index.numel()),
                    "controls": len(dataset.control_indices),
                    "perturbations": len(dataset.perturbation_targets),
                    "sampling_weight": (
                        float(dataset_weights[perturbation_number])
                        if perturbation_number >= 0
                        else None
                    ),
                }
            )
        )

    use_amp = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.amp == "fp16")
    log_path = args.output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phases = training_schedule(
        args.control_steps, args.perturbation_steps, args.control_every
    )
    total_steps = len(phases)
    if start_step > total_steps:
        raise ValueError(
            f"checkpoint step {start_step} exceeds requested schedule length {total_steps}"
        )
    model.train()
    completed_step = start_step
    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            for step, phase in enumerate(phases[start_step:], start=start_step + 1):
                completed_step = step
                optimizer.zero_grad(set_to_none=True)
                is_control = phase == "control"
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    if is_control:
                        dataset = control_datasets[int(rng.integers(len(control_datasets)))]
                        inputs, observed = dataset.sample_control(args.query_cells, rng)
                        output = model.reconstruct_control(
                            move(inputs, device),
                            priors=priors,
                            query_gene_index=dataset.alignment.input_gene_index.to(device),
                            decode_gene_index=dataset.alignment.decode_gene_index.to(device),
                        )
                        loss = control_reconstruction_loss(output, move(observed, device))
                        parts = {"control": loss.detach()}
                        target = "control"
                    else:
                        dataset = perturbation_datasets[
                            int(
                                rng.choice(
                                    len(perturbation_datasets), p=dataset_weights
                                )
                            )
                        ]
                        target = choose_target(
                            dataset,
                            vcc_targets,
                            rng,
                            args.minimum_perturbed_cells,
                            args.vcc_target_probability,
                            args.preference_minimum_targets,
                        )
                        context, _ = dataset.sample_control(args.context_cells, rng)
                        query, control_observed = dataset.sample_control(
                            args.query_cells, rng
                        )
                        perturbed_observed = dataset.sample_perturbed(
                            target, args.query_cells, rng
                        )
                        perturbed_library = perturbed_observed.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1.0)
                        control_library = control_observed.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1.0)
                        observed_delta = torch.log1p(
                            perturbed_observed * (10_000.0 / perturbed_library)
                        ).mean(dim=0) - torch.log1p(
                            control_observed * (10_000.0 / control_library)
                        ).mean(dim=0)
                        de_count = min(args.de_genes, observed_delta.numel())
                        de_mask = torch.zeros_like(observed_delta, dtype=torch.bool)
                        if de_count:
                            de_mask[observed_delta.abs().topk(de_count).indices] = True
                        output = model(
                            move(context, device),
                            move(query, device),
                            vocabulary.target_index([target]).to(device),
                            priors=priors,
                            context_gene_index=dataset.alignment.input_gene_index.to(device),
                            query_gene_index=dataset.alignment.input_gene_index.to(device),
                            decode_gene_index=dataset.alignment.decode_gene_index.to(device),
                        )
                        parts = perturbation_loss(
                            output,
                            move(perturbed_observed, device),
                            move(control_observed, device),
                            de_gene_mask=de_mask.unsqueeze(0).to(device),
                            target_gene_index=vocabulary.target_index([target]).to(device),
                        )
                        loss = parts["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()

                if step == 1 or step % args.log_every == 0:
                    record = {
                        "step": step,
                        "phase": "control" if is_control else "perturbation",
                        "dataset": dataset.name,
                        "target": target,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        **{
                            name: float(value)
                            for name, value in parts.items()
                            if name != "loss"
                        },
                    }
                    print(json.dumps(record), flush=True)
                    log_handle.write(json.dumps(record) + "\n")
                    log_handle.flush()
                if step % args.save_every == 0:
                    save_checkpoint(
                        args.output,
                        model,
                        optimizer,
                        config,
                        level=args.level,
                        step=step,
                        args=args,
                        vocabulary=vocabulary,
                        rng=rng,
                    )
        save_checkpoint(
            args.output,
            model,
            optimizer,
            config,
            level=args.level,
            step=completed_step,
            args=args,
            vocabulary=vocabulary,
            rng=rng,
        )
    finally:
        for dataset in [*control_datasets, *perturbation_datasets]:
            dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        choices=("global", "cell_state", "functional_prior", "full_prior"),
        required=True,
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument(
        "--priors",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/priors",
    )
    parser.add_argument(
        "--perturbations",
        type=Path,
        default=WORKSPACE / "data/vcc_2026_controls/pert_counts.csv",
    )
    parser.add_argument(
        "--vcc-controls",
        nargs=3,
        type=Path,
        default=[
            WORKSPACE / f"data/vcc_2026_controls/context_{name}.h5ad"
            for name in "ABC"
        ],
    )
    parser.add_argument("--perturbation-data", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset-weights", nargs="+", type=float)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--decoder-dim", type=int, default=64)
    parser.add_argument("--context-prototypes", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prior-dropout", type=float, default=0.2)
    parser.add_argument("--prior-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--control-steps", type=int, default=1_000)
    parser.add_argument("--perturbation-steps", type=int, default=5_000)
    parser.add_argument("--control-every", type=int, default=4)
    parser.add_argument("--context-cells", type=int, default=128)
    parser.add_argument("--query-cells", type=int, default=64)
    parser.add_argument("--minimum-perturbed-cells", type=int, default=16)
    parser.add_argument("--vcc-target-probability", type=float, default=0.7)
    parser.add_argument("--preference-minimum-targets", type=int, default=1_000)
    parser.add_argument("--distribution-projections", type=int, default=16)
    parser.add_argument("--de-genes", type=int, default=200)
    parser.add_argument("--expression-weight", type=float, default=0.0)
    parser.add_argument("--distribution-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--direction-weight", type=float, default=0.1)
    parser.add_argument("--de-weight", type=float, default=0.2)
    parser.add_argument("--magnitude-weight", type=float, default=0.0)
    parser.add_argument("--target-weight", type=float, default=0.0)
    parser.add_argument("--de-direction-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
