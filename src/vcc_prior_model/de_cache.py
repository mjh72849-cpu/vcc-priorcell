"""Stable, dataset-level differential-expression supervision artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class DifferentialExpressionTarget:
    log_fold_change: Tensor
    de_mask: Tensor
    confidence: Tensor


class DifferentialExpressionCache:
    """Load precomputed full-cell DE statistics by dataset path and target."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        with (self.root / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported DE cache schema")
        self.entries = {
            str(Path(entry["dataset_path"]).resolve()): entry for entry in manifest["datasets"]
        }
        self._loaded: dict[str, dict[str, object]] = {}

    def _entry(self, dataset_path: str | Path) -> tuple[str, dict[str, object]]:
        key = str(Path(dataset_path).resolve())
        if key not in self.entries:
            raise KeyError(f"dataset is absent from DE cache: {key}")
        return key, self.entries[key]

    def contains_dataset(self, dataset_path: str | Path) -> bool:
        """Whether this cache can supervise a dataset (for mixed cached/online runs)."""

        return str(Path(dataset_path).resolve()) in self.entries

    def targets(
        self, dataset_path: str | Path, *, informative_only: bool = False
    ) -> tuple[str, ...]:
        key, entry = self._entry(dataset_path)
        loaded = self._load(key, entry)
        targets = tuple(loaded["targets"])  # type: ignore[arg-type]
        if not informative_only:
            return targets
        informative = np.asarray(loaded["de_mask"]).any(axis=1)
        return tuple(target for target, keep in zip(targets, informative, strict=True) if keep)

    def gene_symbols(self, dataset_path: str | Path) -> tuple[str, ...]:
        key, entry = self._entry(dataset_path)
        loaded = self._load(key, entry)
        return tuple(loaded["gene_symbols"])  # type: ignore[arg-type]

    def get(
        self, dataset_path: str | Path, target: str, device: torch.device | str
    ) -> DifferentialExpressionTarget:
        key, entry = self._entry(dataset_path)
        loaded = self._load(key, entry)
        target_to_index = loaded["target_to_index"]
        if target not in target_to_index:  # type: ignore[operator]
            raise KeyError(f"{target!r} is absent from DE cache for {key}")
        index = target_to_index[target]  # type: ignore[index]
        return DifferentialExpressionTarget(
            log_fold_change=torch.from_numpy(loaded["log_fold_change"][index]).to(  # type: ignore[index]
                device=device, dtype=torch.float32
            ),
            de_mask=torch.from_numpy(loaded["de_mask"][index]).to(  # type: ignore[index]
                device=device, dtype=torch.bool
            ),
            confidence=torch.from_numpy(loaded["confidence"][index]).to(  # type: ignore[index]
                device=device, dtype=torch.float32
            ),
        )

    def _load(self, key: str, entry: dict[str, object]) -> dict[str, object]:
        if key in self._loaded:
            return self._loaded[key]
        path = self.root / str(entry["file"])
        with np.load(path, allow_pickle=False) as arrays:
            targets = tuple(str(value) for value in arrays["targets"].tolist())
            loaded: dict[str, object] = {
                "targets": targets,
                "target_to_index": {target: index for index, target in enumerate(targets)},
                "gene_symbols": tuple(str(value) for value in arrays["gene_symbols"].tolist()),
                "log_fold_change": arrays["log_fold_change"].copy(),
                "de_mask": arrays["de_mask"].copy(),
                "confidence": arrays["confidence"].copy(),
            }
        self._loaded[key] = loaded
        return loaded
