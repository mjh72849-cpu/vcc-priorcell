#!/usr/bin/env python3
"""Resumable parallel-range downloader for large Hugging Face LFS files."""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def resolved_url(url: str) -> str:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "priorcell/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.geturl()


def fetch_part(
    url: str, part: Path, start: int, stop: int, retries: int
) -> tuple[int, int]:
    expected = stop - start + 1
    if part.exists() and part.stat().st_size == expected:
        return start, expected
    temporary = part.with_suffix(part.suffix + ".tmp")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{stop}", "User-Agent": "priorcell/1"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise OSError(f"range request returned HTTP {response.status}")
                with temporary.open("wb") as handle:
                    while block := response.read(4 << 20):
                        handle.write(block)
            if temporary.stat().st_size != expected:
                raise OSError(f"short part: {temporary.stat().st_size} != {expected}")
            temporary.replace(part)
            return start, expected
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("unreachable")


def main(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        output.exists()
        and output.stat().st_size == args.size
        and (args.sha256 is None or digest(output) == args.sha256)
    ):
        print(f"already verified: {output}")
        return
    url = resolved_url(args.url)
    part_dir = output.with_suffix(output.suffix + ".parts")
    part_dir.mkdir(exist_ok=True)
    ranges = []
    for start in range(0, args.size, args.chunk_size):
        stop = min(args.size - 1, start + args.chunk_size - 1)
        ranges.append((start, stop, part_dir / f"{start:012d}.part"))
    complete = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_part, url, part, start, stop, args.retries)
            for start, stop, part in ranges
        ]
        for future in as_completed(futures):
            _, size = future.result()
            complete += size
            print(f"downloaded {complete / args.size:.1%}", flush=True)
    temporary = output.with_suffix(output.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for _, _, part in ranges:
            with part.open("rb") as source:
                while block := source.read(16 << 20):
                    destination.write(block)
    if temporary.stat().st_size != args.size:
        raise OSError("assembled file has the wrong size")
    if args.sha256 is not None:
        actual = digest(temporary)
        if actual != args.sha256:
            raise OSError(f"SHA256 mismatch: {actual} != {args.sha256}")
    temporary.replace(output)
    for _, _, part in ranges:
        part.unlink()
    part_dir.rmdir()
    print(f"verified: {output}")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            checksum.update(block)
    return checksum.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--workers", type=int, default=24)
    # The execution environment's HTTP proxy caps Range responses at 4 MiB.
    # Matching that cap avoids repeated short-part retries on large LFS files.
    parser.add_argument("--chunk-size", type=int, default=4 << 20)
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
