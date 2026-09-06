#!/usr/bin/env python3
"""Precompute deterministic dataset-level DE targets for Level 3.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from vcc_prior_model import GeneVocabularyArtifacts
from vcc_prior_model.real_data import BackedH5adDataset

WORKSPACE = Path(__file__).resolve().parents[2]


def read_target_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["target_gene"] for row in csv.DictReader(handle)}


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, preserving the original gene order."""

    pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
    order = np.argsort(pvalues, kind="stable")
    ranks = np.arange(1, len(pvalues) + 1, dtype=np.float64)
    adjusted_sorted = pvalues[order] * len(pvalues) / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def log_normalize(counts: np.ndarray) -> np.ndarray:
    library = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(counts * (10_000.0 / library)).astype(np.float32, copy=False)


def select_rows(rows: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(rows) <= maximum:
        return rows
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def prepare_dataset(
    path: Path,
    vocabulary: GeneVocabularyArtifacts,
    requested_targets: set[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    seed = int.from_bytes(hashlib.sha256(str(path.resolve()).encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, seed]))
    with BackedH5adDataset(path, vocabulary) as dataset:
        available = [
            target
            for target in dataset.perturbation_targets
            if len(dataset.groups[target]) >= args.minimum_perturbed_cells
        ]
        if len(available) > args.max_unlisted_targets:
            available = [target for target in available if target in requested_targets]
        if not available:
            raise ValueError(f"{path}: no targets selected for DE caching")
        control_rows = select_rows(dataset.control_indices, args.control_cells, rng)
        control = log_normalize(dataset.read_outputs(control_rows).numpy())
        gene_symbols = np.asarray(
            [
                vocabulary.output_gene_symbols[position]
                for position in dataset.alignment.output_position.tolist()
            ],
            dtype=str,
        )
        effects = np.empty((len(available), len(gene_symbols)), dtype=np.float32)
        qvalues = np.empty_like(effects)
        de_masks = np.empty_like(effects, dtype=np.uint8)
        confidence = np.empty_like(effects)
        cell_counts = np.empty(len(available), dtype=np.int32)
        for target_index, target in enumerate(available):
            rows = select_rows(dataset.groups[target], args.perturbed_cells, rng)
            perturbed = log_normalize(dataset.read_outputs(rows).numpy())
            effect = perturbed.mean(axis=0) - control.mean(axis=0)
            result = mannwhitneyu(
                perturbed,
                control,
                axis=0,
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            )
            qvalue = bh_adjust(np.asarray(result.pvalue, dtype=np.float64))
            mask = (qvalue < args.fdr) & (np.abs(effect) >= args.minimum_logfc)
            effects[target_index] = effect
            qvalues[target_index] = qvalue.astype(np.float32)
            de_masks[target_index] = mask
            confidence[target_index] = np.clip(-np.log10(qvalue + 1e-12) / 10.0, 0.0, 1.0)
            cell_counts[target_index] = len(rows)
            if target_index == 0 or (target_index + 1) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "dataset": dataset.name,
                            "target": target,
                            "completed": target_index + 1,
                            "total": len(available),
                            "perturbed_cells": len(rows),
                            "de_genes": int(mask.sum()),
                        }
                    ),
                    flush=True,
                )
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        filename = f"{path.stem}_{digest}.npz"
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output / filename,
            targets=np.asarray(available, dtype=str),
            gene_symbols=gene_symbols,
            log_fold_change=effects,
            qvalue=qvalues,
            de_mask=de_masks,
            confidence=confidence,
            perturbed_cell_count=cell_counts,
            control_cell_count=np.asarray([len(control_rows)], dtype=np.int32),
        )
        return {
            "dataset_path": str(path.resolve()),
            "file": filename,
            "targets": len(available),
            "genes": len(gene_symbols),
            "control_cells": len(control_rows),
            "median_de_genes": float(np.median(de_masks.sum(axis=1))),
            "mean_de_genes": float(de_masks.sum(axis=1).mean()),
        }


def main(args: argparse.Namespace) -> None:
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    requested = read_target_list(args.target_list)
    entries = [prepare_dataset(path, vocabulary, requested, args) for path in args.data]
    manifest = {
        "schema_version": 1,
        "method": "Mann-Whitney U/Wilcoxon rank-sum, BH correction",
        "fdr": args.fdr,
        "minimum_logfc": args.minimum_logfc,
        "maximum_control_cells": args.control_cells,
        "maximum_perturbed_cells": args.perturbed_cells,
        "seed": args.seed,
        "datasets": entries,
    }
    temporary = args.output / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temporary.replace(args.output / "manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument("--target-list", type=Path)
    parser.add_argument("--max-unlisted-targets", type=int, default=500)
    parser.add_argument("--control-cells", type=int, default=512)
    parser.add_argument("--perturbed-cells", type=int, default=512)
    parser.add_argument("--minimum-perturbed-cells", type=int, default=16)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--minimum-logfc", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2031)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
