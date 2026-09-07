#!/usr/bin/env python3
"""Validate every STATE cache required by the Level 4 training entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXPECTED = {
    "vcc2025_train.npy": 221_273,
    "vcc2025_test.npy": 170_846,
    "replogle_k562.npy": 383_061,
    "replogle_rpe1.npy": 247_914,
    "jiang24.npy": 1_628_476,
    "vcc2026_A.npy": 18_400,
    "vcc2026_B.npy": 18_400,
    "vcc2026_C.npy": 18_400,
}


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "state_embeddings"
    for name, rows in EXPECTED.items():
        path = root / name
        array = np.load(path, mmap_mode="r")
        expected_shape = (rows, 2048)
        if array.shape != expected_shape:
            raise ValueError(f"{path}: expected {expected_shape}, found {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{path}: expected float32, found {array.dtype}")
        print(f"verified {name}: {array.shape} {array.dtype}")


if __name__ == "__main__":
    main()
