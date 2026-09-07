#!/usr/bin/env python3
"""Keep STATE's biological cell embedding and drop its dataset-ID suffix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(args: argparse.Namespace) -> None:
    source = np.load(args.input, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(f"expected a rank-2 embedding matrix, got {source.shape}")
    if source.shape[1] < args.cell_dim:
        raise ValueError(
            f"STATE embedding width {source.shape[1]} is smaller than {args.cell_dim}"
        )
    if args.expected_rows is not None and source.shape[0] != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} rows, found {source.shape[0]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float32,
        shape=(source.shape[0], args.cell_dim),
    )
    for start in range(0, source.shape[0], args.chunk_rows):
        stop = min(source.shape[0], start + args.chunk_rows)
        output[start:stop] = source[start:stop, : args.cell_dim]
    output.flush()
    print(
        f"wrote {args.output}: {tuple(output.shape)}; "
        f"dropped {source.shape[1] - args.cell_dim} dataset dimensions"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-dim", type=int, default=2048)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
