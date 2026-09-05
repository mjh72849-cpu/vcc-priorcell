"""Backed H5AD access for real raw-count perturbation training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .vocabulary import GeneVocabularyArtifacts


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def read_h5ad_strings(node: h5py.Dataset | h5py.Group) -> list[str]:
    """Decode the string encodings used by AnnData 0.8+ dataframes."""

    if isinstance(node, h5py.Dataset):
        return _decode(node[:])
    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding == "nullable-string-array":
        if bool(node["mask"][:].any()):
            raise ValueError("gene identifiers contain missing values")
        return _decode(node["values"][:])
    if encoding == "categorical":
        categories = _decode(node["categories"][:])
        return [categories[int(code)] if code >= 0 else "" for code in node["codes"][:]]
    raise ValueError(f"unsupported AnnData string encoding {encoding!r}")


def read_var_names(handle: h5py.File) -> list[str]:
    var = handle["var"]
    index_name = var.attrs["_index"]
    if isinstance(index_name, bytes):
        index_name = index_name.decode()
    return read_h5ad_strings(var[str(index_name)])


def detect_target_key(handle: h5py.File) -> str | None:
    for candidate in ("target_gene", "gene", "perturbation", "condition"):
        if candidate in handle["obs"]:
            return candidate
    return None


def _categorical_groups(node: h5py.Dataset | h5py.Group) -> dict[str, np.ndarray]:
    if not isinstance(node, h5py.Group) or "categories" not in node or "codes" not in node:
        values = np.asarray(read_h5ad_strings(node), dtype=object)
        return {str(value): np.flatnonzero(values == value) for value in np.unique(values)}
    categories = _decode(node["categories"][:])
    codes = node["codes"][:]
    valid_rows = np.flatnonzero(codes >= 0)
    order = valid_rows[np.argsort(codes[valid_rows], kind="stable")]
    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(len(categories) + 1))
    return {
        category: np.sort(order[boundaries[index] : boundaries[index + 1]])
        for index, category in enumerate(categories)
    }


@dataclass(frozen=True)
class PanelAlignment:
    """One H5AD panel aligned to global input and fixed VCC output spaces."""

    input_local_index: np.ndarray
    input_gene_index: Tensor
    output_local_index: np.ndarray
    decode_gene_index: Tensor
    output_position: Tensor


class BackedH5adDataset:
    """Read small row batches from dense or CSR H5AD ``X`` datasets."""

    def __init__(
        self,
        path: str | Path,
        vocabulary: GeneVocabularyArtifacts,
        *,
        name: str | None = None,
        target_key: str | None = None,
        control_labels: Sequence[str] = ("non-targeting", "control", "NTC"),
    ) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.handle = h5py.File(self.path, "r")
        self.var_names = tuple(read_var_names(self.handle))
        self.num_cells = int(self._shape()[0])
        global_by_symbol = vocabulary.global_index_by_symbol
        output_by_symbol = {
            symbol: index for index, symbol in enumerate(vocabulary.output_gene_symbols)
        }
        input_local = [
            index for index, symbol in enumerate(self.var_names) if symbol in global_by_symbol
        ]
        input_symbols = [self.var_names[index] for index in input_local]
        local_symbol_set = set(self.var_names)
        output_symbols = [
            symbol for symbol in vocabulary.output_gene_symbols if symbol in local_symbol_set
        ]
        local_by_symbol = {symbol: index for index, symbol in enumerate(self.var_names)}
        self.alignment = PanelAlignment(
            input_local_index=np.asarray(input_local, dtype=np.int64),
            input_gene_index=torch.tensor(
                [global_by_symbol[symbol] for symbol in input_symbols], dtype=torch.long
            ),
            output_local_index=np.asarray(
                [local_by_symbol[symbol] for symbol in output_symbols], dtype=np.int64
            ),
            decode_gene_index=torch.tensor(
                [global_by_symbol[symbol] for symbol in output_symbols], dtype=torch.long
            ),
            output_position=torch.tensor(
                [output_by_symbol[symbol] for symbol in output_symbols], dtype=torch.long
            ),
        )
        self.target_key = target_key or detect_target_key(self.handle)
        self.groups: dict[str, np.ndarray] = {}
        if self.target_key is not None:
            self.groups = _categorical_groups(self.handle["obs"][self.target_key])
        self.control_label = next((x for x in control_labels if x in self.groups), None)
        self.control_indices = (
            self.groups[self.control_label]
            if self.control_label is not None
            else np.arange(self.num_cells, dtype=np.int64)
        )

    def _shape(self) -> tuple[int, int]:
        x = self.handle["X"]
        if isinstance(x, h5py.Dataset):
            return int(x.shape[0]), int(x.shape[1])
        shape = x.attrs.get("shape")
        if shape is None:
            raise ValueError(f"{self.path}: sparse X has no shape attribute")
        return int(shape[0]), int(shape[1])

    @property
    def perturbation_targets(self) -> tuple[str, ...]:
        return tuple(sorted(target for target in self.groups if target != self.control_label))

    def _read_unique_rows(self, rows: np.ndarray) -> np.ndarray:
        x = self.handle["X"]
        if isinstance(x, h5py.Dataset):
            return np.asarray(x[rows], dtype=np.float32)
        encoding = x.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding != "csr_matrix":
            raise ValueError(f"{self.path}: only dense and CSR X are supported, got {encoding}")
        width = self._shape()[1]
        result = np.zeros((len(rows), width), dtype=np.float32)
        indptr = x["indptr"]
        indices = x["indices"]
        data = x["data"]
        if len(rows) and (len(rows) == 1 or np.all(np.diff(rows) == 1)):
            source_indptr = np.asarray(
                indptr[int(rows[0]) : int(rows[-1]) + 2], dtype=np.int64
            )
            data_start = int(source_indptr[0])
            data_stop = int(source_indptr[-1])
            block_indices = np.asarray(indices[data_start:data_stop])
            block_data = np.asarray(data[data_start:data_stop])
            local_indptr = source_indptr - data_start
            for output_row, (start, stop) in enumerate(
                zip(local_indptr[:-1], local_indptr[1:], strict=True)
            ):
                result[output_row, block_indices[start:stop]] = block_data[start:stop]
            return result
        for output_row, source_row in enumerate(rows):
            start = int(indptr[source_row])
            stop = int(indptr[source_row + 1])
            result[output_row, indices[start:stop]] = data[start:stop]
        return result

    def _read_rows_all_columns(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        return self._read_unique_rows(unique_rows)[inverse]

    def read_rows(self, rows: np.ndarray, local_columns: np.ndarray) -> np.ndarray:
        """Read arbitrary rows while satisfying h5py's sorted-unique constraint."""

        return self._read_rows_all_columns(rows)[:, local_columns]

    def read_inputs(self, rows: np.ndarray) -> Tensor:
        """Read selected rows aligned to this dataset's global input indices."""

        return torch.from_numpy(self.read_rows(rows, self.alignment.input_local_index))

    def read_outputs(self, rows: np.ndarray) -> Tensor:
        """Read selected rows aligned to the observed VCC-output intersection."""

        return torch.from_numpy(self.read_rows(rows, self.alignment.output_local_index))

    def input_chunks(self, rows: np.ndarray, chunk_size: int) -> Iterable[Tensor]:
        """Return a lazy iterable of globally aligned input-panel row chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        def chunks() -> Iterable[Tensor]:
            for start in range(0, len(rows), chunk_size):
                yield self.read_inputs(rows[start : start + chunk_size])

        return chunks()

    def sample_indices(
        self, pool: np.ndarray, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        if len(pool) == 0:
            raise ValueError(f"{self.name}: cannot sample from an empty cell group")
        return rng.choice(pool, size=count, replace=len(pool) < count)

    def sample_control(self, count: int, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
        rows = self.sample_indices(self.control_indices, count, rng)
        values = self._read_rows_all_columns(rows)
        inputs = values[:, self.alignment.input_local_index]
        outputs = values[:, self.alignment.output_local_index]
        return torch.from_numpy(inputs), torch.from_numpy(outputs)

    def sample_perturbed(
        self, target: str, count: int, rng: np.random.Generator
    ) -> Tensor:
        rows = self.sample_indices(self.groups[target], count, rng)
        outputs = self.read_rows(rows, self.alignment.output_local_index)
        return torch.from_numpy(outputs)

    def validate_raw_counts(self, sample_size: int = 32) -> None:
        rows = np.linspace(0, self.num_cells - 1, min(sample_size, self.num_cells)).astype(
            np.int64
        )
        sample = self.read_rows(rows, self.alignment.input_local_index)
        if not np.isfinite(sample).all() or (sample < 0).any():
            raise ValueError(f"{self.path}: X contains invalid count values")
        if not np.equal(sample, np.floor(sample)).all():
            raise ValueError(
                f"{self.path}: X is not raw integer counts; do not use it with the NB loss"
            )

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> BackedH5adDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
