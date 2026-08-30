#!/usr/bin/env python3
"""Pool a 2.4 µm surface volume to the ~9.6 µm isotropic 21-slice input Zarr.

Reads one XY pyramid level of a surface-volume OME-Zarr (level 2 of a
2.399 µm render is 9.596 µm per pixel), takes the 84 centered Z planes, and
mean-pools them 4× to 21 slices so all three axes sit at the same ~9.6 µm
spacing. Native ~9 µm surface volumes do not need this step.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import time

import numpy as np
from numcodecs import Blosc
import zarr

OUTPUT_Z = 21
POOL_Z = 4
INPUT_Z = OUTPUT_Z * POOL_Z
TILE = 512
CHUNK_XY = 128


def centered_slice(length: int, requested: int) -> tuple[int, int]:
    if requested > length:
        raise ValueError(f"Cannot take {requested} centered planes from {length}")
    start = math.ceil((length - requested) / 2)
    return start, start + requested


def open_source_array(path: str, level: str) -> zarr.Array:
    node = zarr.open(path, mode="r")
    if isinstance(node, zarr.Array):
        return node
    if level not in node:
        available = sorted(node.array_keys())
        raise KeyError(f"Pyramid level '{level}' not found in {path}; available: {available}")
    return node[level]



def publish_partial(staged: Path, output: Path, *, attempts: int = 6, delay: float = 0.5) -> None:
    """Rename a finished staging directory onto the output path.

    POSIX renames a directory whose files are open. Windows refuses with
    WinError 5 while any file inside is still open, and one handle is enough --
    measured on this failure: renaming succeeds with nothing held, and fails with
    exactly WinError 5 with a single file inside opened for read. After writing a
    multi-gigabyte zarr, something holding a handle for a moment (an indexer, a
    scanner) is ordinary, so retry briefly before giving up.

    If it still fails, say what is true: every tile was written, so nothing needs
    recomputing, and a manual rename finishes the job. The bare traceback this
    replaces looked like a lost conversion.
    """
    for attempt in range(attempts):
        try:
            staged.replace(output)
            return
        except PermissionError as exc:
            if attempt + 1 == attempts:
                raise SystemExit(
                    f"The conversion finished -- every tile was written -- but the result "
                    f"could not be published:\n"
                    f"  {staged}\n"
                    f"  -> {output}\n"
                    f"  {exc}\n"
                    f"On Windows a directory cannot be renamed while any file inside it is "
                    f"open. Nothing needs recomputing: rename the staging directory by hand."
                ) from None
            time.sleep(delay * (2 ** attempt))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zarr", help="Surface-volume OME-Zarr group path or URL (or a bare 3D array).")
    parser.add_argument("output_zarr", type=Path, help="Output Zarr path; refuses to overwrite.")
    parser.add_argument(
        "--level",
        default="2",
        help="Input pyramid level to read (default 2: one level-2 pixel of a 2.399 µm render is ~9.6 µm).",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = open_source_array(args.input_zarr, args.level)
    shape = tuple(int(value) for value in source.shape)
    if source.dtype != np.uint8 or len(shape) != 3:
        raise ValueError(f"Expected 3D uint8 source, got {shape} {source.dtype}")
    z0, z1 = centered_slice(shape[0], INPUT_Z)

    partial = args.output_zarr.with_name(args.output_zarr.name + ".partial")
    if args.output_zarr.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace {args.output_zarr} or {partial}")

    group = zarr.open_group(str(partial), mode="w")
    group.attrs.update(
        {
            "format": "level2-zmean4-21slice-v1",
            "source": str(args.input_zarr),
            "source_level": str(args.level),
            "source_shape_zyx": list(shape),
            "source_z_slice": [z0, z1],
            "z_pool": "rounded mean of 4 centered source planes",
        }
    )
    target = group.create_dataset(
        "0",
        shape=(OUTPUT_Z, shape[1], shape[2]),
        chunks=(OUTPUT_Z, min(CHUNK_XY, shape[1]), min(CHUNK_XY, shape[2])),
        dtype=np.uint8,
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        fill_value=0,
    )
    tiles = [
        (y0, min(shape[1], y0 + TILE), x0, min(shape[2], x0 + TILE))
        for y0 in range(0, shape[1], TILE)
        for x0 in range(0, shape[2], TILE)
    ]

    def process(tile: tuple[int, int, int, int]) -> int:
        y0, y1, x0, x1 = tile
        block = np.asarray(source[z0:z1, y0:y1, x0:x1], dtype=np.float32)
        pooled = np.rint(
            block.reshape(OUTPUT_Z, POOL_Z, y1 - y0, x1 - x0).mean(axis=1)
        ).astype(np.uint8)
        target[:, y0:y1, x0:x1] = pooled
        return 1

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count in pool.map(process, tiles):
            completed += count
            if completed % 50 == 0 or completed == len(tiles):
                print(f"tiles={completed}/{len(tiles)}", flush=True)
    output_shape = tuple(target.shape)
    del target, group  # release the store's handles before renaming its directory
    publish_partial(partial, args.output_zarr)
    print(f"wrote {args.output_zarr} shape={output_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
