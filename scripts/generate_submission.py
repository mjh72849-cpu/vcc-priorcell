#!/usr/bin/env python3
"""Generate a complete three-context VCC raw-count H5AD from a checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from vcc_prior_model import GeneVocabularyArtifacts, ModelConfig, build_model
from vcc_prior_model.real_data import BackedH5adDataset
from vcc_prior_model.prior_data import PriorArtifacts

WORKSPACE = Path(__file__).resolve().parents[2]


def move(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)


def read_targets(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        targets = [row["target_gene"] for row in csv.DictReader(handle)]
    if len(targets) != len(set(targets)):
        raise ValueError("perturbation target list contains duplicates")
    return targets


def stochastic_round(mean: torch.Tensor) -> torch.Tensor:
    floor = mean.floor()
    return (floor + (torch.rand_like(mean) < mean - floor)).to(torch.int32)


def draw_counts(
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    method: str,
    eps: float,
) -> torch.Tensor:
    if method == "round":
        return stochastic_round(mean)
    if method == "poisson":
        return torch.poisson(mean).to(torch.int32)
    if method == "negative_binomial":
        theta = dispersion.view(1, 1, -1).float()
        probability = mean / (theta + mean)
        return torch.distributions.NegativeBinomial(
            total_count=theta,
            probs=probability.clamp(max=1.0 - eps),
        ).sample().to(torch.int32)
    raise ValueError(f"unknown count sampling method: {method}")


def encode_context(
    model: torch.nn.Module,
    dataset: BackedH5adDataset,
    device: torch.device,
    chunk_size: int,
    gene_embeddings: torch.Tensor,
) -> Any:
    rows = np.arange(dataset.num_cells, dtype=np.int64)
    def chunks() -> Any:
        for counts in dataset.input_chunks(rows, chunk_size):
            yield move(counts, device)

    return model.encode_context_chunks(
        chunks(),
        gene_embeddings,
        input_gene_index=dataset.alignment.input_gene_index.to(device),
    )


def generate_context(
    args: argparse.Namespace,
    context_name: str,
    control_path: Path,
    targets: list[str],
    model: torch.nn.Module,
    vocabulary: GeneVocabularyArtifacts,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    shard_path: Path,
    gene_embeddings: torch.Tensor,
) -> dict[str, Any]:
    matrices: list[sparse.csr_matrix] = []
    target_column: list[str] = []
    context_column: list[str] = []
    obs_names: list[str] = []
    maximum_library = 0
    with BackedH5adDataset(
        control_path, vocabulary, name=f"vcc_{context_name}"
    ) as dataset, torch.inference_mode():
        if len(dataset.alignment.output_local_index) != vocabulary.num_output_genes:
            raise ValueError(f"context {context_name} does not contain the full VCC panel")
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            context = encode_context(
                model, dataset, device, args.context_chunk_size, gene_embeddings
            )
        for target_number, target in enumerate(targets):
            target_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, ord(context_name), target_number])
            )
            rows = dataset.sample_indices(
                dataset.control_indices, args.cells_per_perturbation, target_rng
            )
            query = dataset.read_inputs(rows)
            empirical_control = dataset.read_outputs(rows)
            query_device = move(query, device)
            empirical_device = move(empirical_control, device)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                output = model.predict_from_context(
                    query_device,
                    vocabulary.target_index([target]).to(device),
                    context,
                    query_gene_index=dataset.alignment.input_gene_index.to(device),
                )
            decoder_weight = args.decoder_baseline_weight
            baseline = (
                (1.0 - decoder_weight) * empirical_device
                + decoder_weight * output.control_mean.float()
            )
            prediction_mean = baseline * (
                args.effect_scale * output.log_effect.float()
            ).exp()
            counts = draw_counts(
                prediction_mean,
                output.dispersion,
                args.sampling,
                model.config.eps,
            )[0].cpu().numpy()
            if (counts < 0).any():
                raise RuntimeError("count sampler produced a negative value")
            libraries = counts.sum(axis=1, dtype=np.int64)
            maximum_library = max(maximum_library, int(libraries.max(initial=0)))
            matrices.append(sparse.csr_matrix(counts, dtype=np.int32))
            target_column.extend([target] * args.cells_per_perturbation)
            context_column.extend([context_name] * args.cells_per_perturbation)
            obs_names.extend(
                f"{context_name}_{target}_{cell}"
                for cell in range(args.cells_per_perturbation)
            )
            if (target_number + 1) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "context": context_name,
                            "completed_targets": target_number + 1,
                            "total_targets": len(targets),
                            "maximum_library": maximum_library,
                        }
                    ),
                    flush=True,
                )
    matrix = sparse.vstack(matrices, format="csr", dtype=np.int32)
    obs = pd.DataFrame(
        {
            "target_gene": pd.Categorical(target_column, categories=targets),
            "context": pd.Categorical(context_column, categories=list("ABC")),
        },
        index=pd.Index(obs_names, name="cell_id"),
    )
    var = pd.DataFrame(
        index=pd.Index(vocabulary.output_gene_symbols, name="gene_symbol")
    )
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.uns["prediction_manifest"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_level": args.checkpoint_level,
        "checkpoint_step": args.checkpoint_step,
        "seed": args.seed,
        "sampling": args.sampling,
        "decoder_baseline_weight": args.decoder_baseline_weight,
        "effect_scale": args.effect_scale,
        "context": context_name,
    }
    result.write_h5ad(shard_path, compression="gzip")
    return {
        "context": context_name,
        "path": str(shard_path),
        "shape": [int(value) for value in result.shape],
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / np.prod(matrix.shape)),
        "maximum_library": maximum_library,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.decoder_baseline_weight <= 1.0:
        raise ValueError("decoder baseline weight must be in [0, 1]")
    if args.effect_scale < 0.0:
        raise ValueError("effect scale must be non-negative")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists; pass --force to replace it")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["global_gene_symbols"]) != vocabulary.global_gene_symbols:
        raise ValueError("checkpoint and current global vocabulary differ")
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(
        checkpoint["level"], config, vocabulary.output_gene_index
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    priors = (
        PriorArtifacts(args.priors, vocabulary).load(device)
        if checkpoint["level"] in {"functional_prior", "full_prior"}
        else None
    )
    with torch.inference_mode():
        gene_embeddings = model.encode_genes(priors)
    args.checkpoint_level = checkpoint["level"]
    args.checkpoint_step = int(checkpoint["step"])
    targets = read_targets(args.perturbations)
    unknown = [target for target in targets if target not in vocabulary.global_index_by_symbol]
    if unknown:
        raise ValueError(f"submission targets absent from vocabulary: {unknown[:5]}")
    use_amp = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    shard_dir = args.output.with_suffix(".shards")
    if shard_dir.exists():
        if not args.force:
            raise FileExistsError(f"{shard_dir} already exists; pass --force")
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    summaries = []
    shards = []
    for context_name, control_path in zip("ABC", args.vcc_controls, strict=True):
        shard = shard_dir / f"context_{context_name}.h5ad"
        summaries.append(
            generate_context(
                args,
                context_name,
                control_path,
                targets,
                model,
                vocabulary,
                device,
                amp_dtype,
                use_amp,
                shard,
                gene_embeddings,
            )
        )
        shards.append(shard)
    if args.output.exists():
        args.output.unlink()
    ad.experimental.concat_on_disk(
        shards,
        args.output,
        axis="obs",
        join="inner",
        merge="same",
        max_loaded_elems=args.concat_max_loaded_elements,
    )
    result = {
        "output": str(args.output),
        "shape": [
            len(targets) * args.cells_per_perturbation * len(args.vcc_controls),
            vocabulary.num_output_genes,
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_level": checkpoint["level"],
        "checkpoint_step": int(checkpoint["step"]),
        "sampling": args.sampling,
        "decoder_baseline_weight": args.decoder_baseline_weight,
        "effect_scale": args.effect_scale,
        "seed": args.seed,
        "contexts": summaries,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--context-chunk-size", type=int, default=256)
    parser.add_argument("--cells-per-perturbation", type=int, default=400)
    parser.add_argument(
        "--sampling",
        choices=("round", "poisson", "negative_binomial"),
        default="round",
    )
    parser.add_argument("--decoder-baseline-weight", type=float, default=0.1)
    parser.add_argument("--effect-scale", type=float, default=1.0)
    parser.add_argument("--concat-max-loaded-elements", type=int, default=100_000_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
