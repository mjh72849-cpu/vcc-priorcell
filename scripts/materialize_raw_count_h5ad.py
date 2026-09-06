#!/usr/bin/env python3
"""Copy an H5AD and promote layers/counts to X without loading cells into RAM."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py


def main(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with h5py.File(temporary, "r+") as handle:
        if "layers/counts" not in handle:
            raise KeyError(f"{source}: layers/counts is absent")
        del handle["X"]
        handle.move("layers/counts", "X")
        handle["X"].attrs["priorcell_source_layer"] = "layers/counts"
    temporary.replace(output)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.source, arguments.output)
