"""Build and load a global gene vocabulary without reading expression matrices."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _require_unique(values: Sequence[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"{label} contains duplicate gene symbols: {examples}")


def ordered_union(primary: Sequence[str], *additional: Sequence[str]) -> list[str]:
    """Return an order-stable union, keeping the complete primary panel first."""

    _require_unique(primary, "primary vocabulary")
    result = list(primary)
    present = set(primary)
    for source_number, source in enumerate(additional, start=1):
        _require_unique(source, f"additional source {source_number}")
        for symbol in source:
            if symbol not in present:
                present.add(symbol)
                result.append(symbol)
    return result


def symbols_sha256(symbols: Iterable[str]) -> str:
    """Hash an ordered symbol sequence with unambiguous newline delimiters."""

    digest = hashlib.sha256()
    for symbol in symbols:
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetGeneMapping:
    """Indices needed to move between one dataset panel and the global/VCC panels."""

    gene_symbols: tuple[str, ...]
    gene_index: Tensor
    global_observed_mask: Tensor
    output_observed_mask: Tensor
    local_to_output_index: Tensor
    output_to_local_index: Tensor


class GeneVocabularyArtifacts:
    """Dependency-light loader for generated CSV/JSON mapping artifacts."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        with (self.directory / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        with (self.directory / "global_gene_vocabulary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: int(row["global_index"]))
        self.global_gene_symbols = tuple(row["gene_symbol"] for row in rows)
        self.global_index_by_symbol = {
            symbol: index for index, symbol in enumerate(self.global_gene_symbols)
        }
        expected_hash = self.manifest["global_vocabulary"]["symbols_sha256"]
        if symbols_sha256(self.global_gene_symbols) != expected_hash:
            raise ValueError("global vocabulary hash does not match manifest")

        with (self.directory / "vcc_output_mapping.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            output_rows = list(csv.DictReader(handle))
        output_rows.sort(key=lambda row: int(row["output_index"]))
        self.output_gene_symbols = tuple(row["gene_symbol"] for row in output_rows)
        self.output_gene_index = torch.tensor(
            [int(row["global_index"]) for row in output_rows], dtype=torch.long
        )

    @property
    def num_genes(self) -> int:
        return len(self.global_gene_symbols)

    @property
    def num_output_genes(self) -> int:
        return len(self.output_gene_symbols)

    def target_index(self, symbols: Sequence[str]) -> Tensor:
        """Map target symbols to global indices, failing loudly on unknown genes."""

        unknown = [symbol for symbol in symbols if symbol not in self.global_index_by_symbol]
        if unknown:
            raise KeyError(f"unknown target gene symbols: {unknown[:5]}")
        return torch.tensor(
            [self.global_index_by_symbol[symbol] for symbol in symbols], dtype=torch.long
        )

    def load_dataset(self, dataset_name: str) -> DatasetGeneMapping:
        path = self.directory / "datasets" / f"{dataset_name}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: int(row["local_index"]))
        symbols = tuple(row["gene_symbol"] for row in rows)
        source_manifest = self.manifest["sources"].get(dataset_name)
        if source_manifest is None:
            raise ValueError(f"dataset {dataset_name!r} is absent from manifest")
        if symbols_sha256(symbols) != source_manifest["symbols_sha256"]:
            raise ValueError(f"{dataset_name} mapping hash does not match manifest")
        gene_index = torch.tensor(
            [int(row["global_index"]) for row in rows], dtype=torch.long
        )
        global_mask = torch.zeros(self.num_genes, dtype=torch.bool)
        global_mask[gene_index] = True
        local_to_output = torch.tensor(
            [int(row["output_index"]) for row in rows], dtype=torch.long
        )
        output_to_local = torch.full(
            (self.num_output_genes,), -1, dtype=torch.long
        )
        observed_local = torch.nonzero(local_to_output >= 0, as_tuple=False).flatten()
        output_to_local[local_to_output[observed_local]] = observed_local
        return DatasetGeneMapping(
            gene_symbols=symbols,
            gene_index=gene_index,
            global_observed_mask=global_mask,
            output_observed_mask=output_to_local >= 0,
            local_to_output_index=local_to_output,
            output_to_local_index=output_to_local,
        )
