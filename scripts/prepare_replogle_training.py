#!/usr/bin/env python3
"""Create a row-efficient raw-count Replogle training subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from vcc_prior_model.real_data import BackedH5adDataset, read_h5ad_strings, read_var_names
from vcc_prior_model.vocabulary import GeneVocabularyArtifacts

WORKSPACE = Path(__file__).resolve().parents[2]


def targets(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["target_gene"] for row in csv.DictReader(handle)}


def choose_rows(args: argparse.Namespace, dataset: BackedH5adDataset) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    vcc_targets = targets(args.perturbations)
    selected = [
        dataset.sample_indices(dataset.control_indices, args.control_cells, rng)
    ]
    retained_targets = 0
    retained_vcc_targets = 0
    for target in dataset.perturbation_targets:
        pool = dataset.groups[target]
        if len(pool) < args.minimum_cells:
            continue
        cap = args.vcc_cells_per_target if target in vcc_targets else args.cells_per_target
        selected.append(dataset.sample_indices(pool, min(cap, len(pool)), rng))
        retained_targets += 1
        retained_vcc_targets += int(target in vcc_targets)
    rows = np.unique(np.concatenate(selected)).astype(np.int64, copy=False)
    print(
        json.dumps(
            {
                "selected_cells": len(rows),
                "retained_targets": retained_targets,
                "retained_vcc_targets": retained_vcc_targets,
            }
        ),
        flush=True,
    )
    return rows


def prepare(args: argparse.Namespace) -> None:
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    with BackedH5adDataset(args.input, vocabulary) as dataset:
        dataset.validate_raw_counts()
        rows = choose_rows(args, dataset)
        handle = dataset.handle
        x = handle["X"]
        if not isinstance(x, h5py.Dataset):
            raise ValueError("this preparation is intended for the dense source H5AD")
        chunks: list[sparse.csr_matrix] = []
        cursor = 0
        for start in range(0, dataset.num_cells, args.source_row_chunk):
            stop = min(start + args.source_row_chunk, dataset.num_cells)
            right = np.searchsorted(rows, stop)
            if right == cursor:
                continue
            local_rows = rows[cursor:right] - start
            dense = np.asarray(x[start:stop], dtype=np.float32)[local_rows]
            chunks.append(sparse.csr_matrix(dense))
            cursor = right
            if len(chunks) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "source_rows_scanned": int(stop),
                            "selected_rows_loaded": int(cursor),
                        }
                    ),
                    flush=True,
                )
        if cursor != len(rows):
            raise RuntimeError("not all selected rows were loaded")
        matrix = sparse.vstack(chunks, format="csr")
        target_values = np.asarray(
            read_h5ad_strings(handle["obs"][dataset.target_key]), dtype=object
        )[rows]
        obs = pd.DataFrame(
            {
                "target_gene": pd.Categorical(target_values),
                "batch": np.asarray(handle["obs"]["batch"][:])[rows],
            },
            index=[f"replogle_{row}" for row in rows],
        )
        var_names = read_var_names(handle)
        var = pd.DataFrame(index=pd.Index(var_names, name="gene_symbol"))
        if "ensembl_id" in handle["var"]:
            var["ensembl_id"] = read_h5ad_strings(handle["var"]["ensembl_id"])
        result = ad.AnnData(X=matrix, obs=obs, var=var)
        result.uns["subset_manifest"] = {
            "source": str(args.input.resolve()),
            "seed": args.seed,
            "control_cells": args.control_cells,
            "cells_per_target": args.cells_per_target,
            "vcc_cells_per_target": args.vcc_cells_per_target,
            "minimum_cells": args.minimum_cells,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(args.output, compression="gzip")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "shape": [int(value) for value in result.shape],
                    "nnz": int(result.X.nnz),
                    "density": float(result.X.nnz / np.prod(result.shape)),
                }
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE / "data/references/ReplogleWeissman2022_K562_gwps.h5ad",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "data/training/replogle_k562_vcc_balanced_raw.h5ad",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument(
        "--perturbations",
        type=Path,
        default=WORKSPACE / "data/vcc_2026_controls/pert_counts.csv",
    )
    parser.add_argument("--control-cells", type=int, default=30_000)
    parser.add_argument("--cells-per-target", type=int, default=32)
    parser.add_argument("--vcc-cells-per-target", type=int, default=512)
    parser.add_argument("--minimum-cells", type=int, default=16)
    parser.add_argument("--source-row-chunk", type=int, default=3_886)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
