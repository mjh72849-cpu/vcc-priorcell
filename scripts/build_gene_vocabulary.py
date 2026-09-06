#!/usr/bin/env python3
"""Build the real VCC/Replogle global vocabulary from local metadata only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vcc_prior_model.vocabulary import ordered_union, symbols_sha256

WORKSPACE = Path(__file__).resolve().parents[2]


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _read_encoded_strings(node: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return _decode(node[:])
    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding == "nullable-string-array":
        values = _decode(node["values"][:])
        mask = node["mask"][:]
        if bool(mask.any()):
            raise ValueError("gene index contains missing values")
        return values
    if encoding == "categorical":
        categories = _decode(node["categories"][:])
        return [categories[int(code)] for code in node["codes"][:]]
    raise ValueError(f"unsupported H5AD string encoding: {encoding!r}")


def read_h5ad_var(path: Path) -> tuple[list[str], list[str | None]]:
    """Read gene symbols and optional Ensembl IDs without touching ``X``."""

    with h5py.File(path, "r") as handle:
        var = handle["var"]
        index_name = var.attrs["_index"]
        if isinstance(index_name, bytes):
            index_name = index_name.decode("utf-8")
        symbols = _read_encoded_strings(var[str(index_name)])
        ensembl = (
            _read_encoded_strings(var["ensembl_id"])
            if "ensembl_id" in var
            else [None] * len(symbols)
        )
    symbols = [symbol.strip() for symbol in symbols]
    if any(not symbol for symbol in symbols):
        raise ValueError(f"{path} contains an empty gene symbol")
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{path} contains duplicate gene symbols")
    return symbols, ensembl


def read_h5ad_obs_category_counts(path: Path, column: str) -> dict[str, int]:
    """Count one categorical obs column without reading the expression matrix."""

    with h5py.File(path, "r") as handle:
        node = handle["obs"][column]
        if not isinstance(node, h5py.Group):
            raise ValueError(f"{path}: obs/{column} is not categorical")
        categories = _decode(node["categories"][:])
        codes = node["codes"][:]
    valid_codes = codes[codes >= 0].astype(np.int64, copy=False)
    counts = np.bincount(valid_codes, minlength=len(categories))
    return dict(zip(categories, counts.tolist(), strict=True))


def read_single_column_csv(path: Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise ValueError(f"{path} does not contain column {column!r}")
    return [row[column].strip() for row in rows]


def read_headerless_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or any(len(row) != 1 or not row[0].strip() for row in rows):
        raise ValueError(f"{path} is not a non-empty single-column CSV")
    return [row[0].strip() for row in rows]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_record(path: Path, symbols: list[str]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "gene_count": len(symbols),
        "symbols_sha256": symbols_sha256(symbols),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    vcc_csv_symbols = read_single_column_csv(args.vcc_gene_names, "gene_name")
    vcc2025_symbols = read_headerless_csv(args.vcc2025_gene_names)
    vcc_sources: dict[str, tuple[Path, list[str]]] = {}
    for name, path in args.vcc_contexts.items():
        symbols, _ = read_h5ad_var(path)
        if symbols != vcc_csv_symbols:
            raise ValueError(f"{name} gene order differs from gene_names.csv")
        vcc_sources[name] = (path, symbols)

    replogle_symbols, replogle_ensembl = read_h5ad_var(args.replogle)
    replogle_perturbation_counts = read_h5ad_obs_category_counts(args.replogle, "gene")
    replogle_target_symbols = [
        symbol
        for symbol in replogle_perturbation_counts
        if symbol not in {"non-targeting", "control", "NTC"}
    ]
    additional_sources: dict[str, tuple[Path, list[str]]] = {}
    for path in args.additional_h5ad:
        symbols, _ = read_h5ad_var(path)
        name = path.stem.lower().replace("-", "_")
        if name in additional_sources:
            raise ValueError(f"duplicate additional dataset name: {name}")
        additional_sources[name] = (path, symbols)
    global_symbols = ordered_union(
        vcc_csv_symbols,
        vcc2025_symbols,
        replogle_symbols,
        replogle_target_symbols,
        *(symbols for _, symbols in additional_sources.values()),
    )
    global_by_symbol = {symbol: index for index, symbol in enumerate(global_symbols)}
    output_by_symbol = {symbol: index for index, symbol in enumerate(vcc_csv_symbols)}
    replogle_ensembl_by_symbol = dict(zip(replogle_symbols, replogle_ensembl, strict=True))
    replogle_local_by_symbol = {
        symbol: index for index, symbol in enumerate(replogle_symbols)
    }

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    global_rows = []
    vcc_set = set(vcc_csv_symbols)
    replogle_set = set(replogle_symbols)
    vcc2025_set = set(vcc2025_symbols)
    vcc2025_local_by_symbol = {
        symbol: index for index, symbol in enumerate(vcc2025_symbols)
    }
    for index, symbol in enumerate(global_symbols):
        global_rows.append(
            {
                "global_index": index,
                "gene_symbol": symbol,
                "ensembl_id": replogle_ensembl_by_symbol.get(symbol) or "",
                "in_vcc_output": int(symbol in vcc_set),
                "vcc_output_index": output_by_symbol.get(symbol, -1),
                "in_vcc2025": int(symbol in vcc2025_set),
                "vcc2025_local_index": vcc2025_local_by_symbol.get(symbol, -1),
                "in_replogle": int(symbol in replogle_set),
                "replogle_local_index": replogle_local_by_symbol.get(symbol, -1),
                "is_replogle_perturbation_target": int(
                    symbol in replogle_perturbation_counts
                ),
                "replogle_perturbed_cell_count": replogle_perturbation_counts.get(
                    symbol, 0
                ),
            }
        )
    write_csv(
        output / "global_gene_vocabulary.csv",
        list(global_rows[0]),
        global_rows,
    )

    output_rows = [
        {
            "output_index": output_index,
            "gene_symbol": symbol,
            "global_index": global_by_symbol[symbol],
        }
        for output_index, symbol in enumerate(vcc_csv_symbols)
    ]
    write_csv(output / "vcc_output_mapping.csv", list(output_rows[0]), output_rows)

    datasets = {
        **vcc_sources,
        "replogle_k562_gwps": (args.replogle, replogle_symbols),
        **additional_sources,
    }
    for name, path in args.vcc2025_splits.items():
        if not path.exists() or path.with_suffix(path.suffix + ".aria2").exists():
            continue
        symbols, _ = read_h5ad_var(path)
        if symbols != vcc2025_symbols:
            raise ValueError(f"{name} gene order differs from VCC 2025 gene_names.csv")
        datasets[name] = (path, symbols)
    dataset_summaries: dict[str, Any] = {}
    for name, (path, symbols) in datasets.items():
        dataset_set = set(symbols)
        ensembl_by_symbol = (
            replogle_ensembl_by_symbol if name == "replogle_k562_gwps" else {}
        )
        rows = [
            {
                "local_index": local_index,
                "gene_symbol": symbol,
                "ensembl_id": ensembl_by_symbol.get(symbol) or "",
                "global_index": global_by_symbol[symbol],
                "output_index": output_by_symbol.get(symbol, -1),
            }
            for local_index, symbol in enumerate(symbols)
        ]
        write_csv(output / "datasets" / f"{name}.csv", list(rows[0]), rows)
        gene_index = np.asarray([global_by_symbol[symbol] for symbol in symbols], dtype=np.int64)
        global_mask = np.zeros(len(global_symbols), dtype=np.bool_)
        global_mask[gene_index] = True
        output_mask = np.asarray([symbol in dataset_set for symbol in vcc_csv_symbols])
        local_to_output = np.asarray(
            [output_by_symbol.get(symbol, -1) for symbol in symbols], dtype=np.int64
        )
        output_to_local = np.full(len(vcc_csv_symbols), -1, dtype=np.int64)
        observed_local = np.flatnonzero(local_to_output >= 0)
        output_to_local[local_to_output[observed_local]] = observed_local
        array_dir = output / "arrays"
        array_dir.mkdir(exist_ok=True)
        np.save(array_dir / f"{name}_gene_index.npy", gene_index, allow_pickle=False)
        np.save(array_dir / f"{name}_global_observed_mask.npy", global_mask, allow_pickle=False)
        np.save(array_dir / f"{name}_output_observed_mask.npy", output_mask, allow_pickle=False)
        np.save(
            array_dir / f"{name}_local_to_output_index.npy",
            local_to_output,
            allow_pickle=False,
        )
        np.save(
            array_dir / f"{name}_output_to_local_index.npy",
            output_to_local,
            allow_pickle=False,
        )
        dataset_summaries[name] = {
            **source_record(path, symbols),
            "vcc_output_overlap": int(output_mask.sum()),
            "dataset_only_vs_vcc": len(dataset_set - vcc_set),
            "missing_from_vcc_output": len(vcc_set - dataset_set),
        }

    np.save(
        output / "arrays" / "vcc_output_gene_index.npy",
        np.asarray([global_by_symbol[symbol] for symbol in vcc_csv_symbols], dtype=np.int64),
        allow_pickle=False,
    )

    targets = read_single_column_csv(args.perturbations, "target_gene")
    if len(set(targets)) != len(targets):
        raise ValueError("perturbation target list contains duplicates")
    unknown_targets = sorted(set(targets) - set(global_symbols))
    if unknown_targets:
        raise ValueError(f"perturbation targets absent from global vocabulary: {unknown_targets}")
    target_rows = [
        {
            "perturbation_index": index,
            "target_gene": symbol,
            "global_index": global_by_symbol[symbol],
            "vcc_output_index": output_by_symbol[symbol],
            "has_replogle_expression_gene": int(symbol in replogle_set),
            "has_replogle_perturbation": int(symbol in replogle_perturbation_counts),
            "replogle_perturbed_cell_count": replogle_perturbation_counts.get(symbol, 0),
        }
        for index, symbol in enumerate(targets)
    ]
    write_csv(output / "perturbation_target_mapping.csv", list(target_rows[0]), target_rows)

    manifest = {
        "schema_version": 1,
        "ordering_policy": (
            "all VCC 2026 genes in submission order, then VCC 2025-only genes, "
            "then Replogle expression-only genes, then target-only genes; each "
            "suffix preserves its source order"
        ),
        "identity_policy": "exact case-sensitive gene_symbol match after stripping whitespace",
        "global_vocabulary": {
            "gene_count": len(global_symbols),
            "symbols_sha256": symbols_sha256(global_symbols),
            "vcc_gene_count": len(vcc_csv_symbols),
            "vcc2025_gene_count": len(vcc2025_symbols),
            "vcc2025_overlap": len(vcc_set & vcc2025_set),
            "vcc2025_only": len(vcc2025_set - vcc_set),
            "vcc2026_only_vs_vcc2025": len(vcc_set - vcc2025_set),
            "replogle_gene_count": len(replogle_symbols),
            "overlap": len(vcc_set & replogle_set),
            "vcc_only": len(vcc_set - replogle_set),
            "replogle_only": len(replogle_set - vcc_set),
            "replogle_target_gene_count": len(replogle_target_symbols),
            "replogle_target_only_gene_count": len(
                set(replogle_target_symbols)
                - set(vcc_csv_symbols)
                - vcc2025_set
                - replogle_set
            ),
        },
        "perturbations": {
            "target_count": len(targets),
            "targets_with_replogle_expression_gene": len(set(targets) & replogle_set),
            "targets_without_replogle_expression_gene": len(set(targets) - replogle_set),
            "targets_with_replogle_perturbation": len(
                set(targets) & set(replogle_perturbation_counts)
            ),
            "targets_without_replogle_perturbation": len(
                set(targets) - set(replogle_perturbation_counts)
            ),
        },
        "sources": {
            "vcc_gene_names": source_record(args.vcc_gene_names, vcc_csv_symbols),
            "vcc_2025_gene_names": source_record(
                args.vcc2025_gene_names, vcc2025_symbols
            ),
            **dataset_summaries,
        },
        "prior_only_genes_included": 0,
        "notes": [
            "All Replogle perturbation target symbols are included even when absent "
            "from the Replogle expression feature matrix.",
            "Prior-only genes are intentionally not appended; priors are aligned "
            "onto the fixed expression/perturbation vocabulary.",
            "Ensembl IDs are available only from the local Replogle var metadata.",
        ],
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vcc-gene-names",
        type=Path,
        default=WORKSPACE / "data/vcc_2026_controls/gene_names.csv",
    )
    parser.add_argument(
        "--perturbations",
        type=Path,
        default=WORKSPACE / "data/vcc_2026_controls/pert_counts.csv",
    )
    parser.add_argument(
        "--vcc2025-gene-names",
        type=Path,
        default=WORKSPACE / "data/vcc_2025/gene_names.csv",
    )
    parser.add_argument(
        "--replogle",
        type=Path,
        default=WORKSPACE / "data/references/ReplogleWeissman2022_K562_gwps.h5ad",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument(
        "--additional-h5ad",
        nargs="*",
        type=Path,
        default=[],
        help="Additional expression panels appended to the global vocabulary",
    )
    args = parser.parse_args()
    args.vcc_contexts = {
        name: WORKSPACE / f"data/vcc_2026_controls/context_{name[-1]}.h5ad"
        for name in ("context_A", "context_B", "context_C")
    }
    args.vcc2025_splits = {
        "vcc_2025_train": WORKSPACE / "data/vcc_2025/train/adata_Training.h5ad",
        "vcc_2025_validation": WORKSPACE
        / "data/vcc_2025/validation/adata_Validation.h5ad",
        "vcc_2025_test": WORKSPACE / "data/vcc_2025/test/adata_Test.h5ad",
    }
    return args


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result["global_vocabulary"], ensure_ascii=False, indent=2))
