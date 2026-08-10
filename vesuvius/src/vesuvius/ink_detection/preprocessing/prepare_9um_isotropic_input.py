"""Pool a 2.4 µm surface volume into a 21-slice isotropic input Zarr."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numcodecs import Blosc
import zarr

from vesuvius.utils.cli import HyphenUnderscoreParser


OUTPUT_Z = 21
POOL_Z = 4
INPUT_Z = OUTPUT_Z * POOL_Z
TILE = 512
CHUNK_XY = 128
FORMAT_TAG = "level2-zmean4-21slice-v1"


def centered_slice(length: int, requested: int) -> tuple[int, int]:
    """Return a centered half-open slice with an odd margin before the slice."""
    if requested > length:
        raise ValueError(f"Cannot take {requested} centered planes from {length}")
    start = math.ceil((length - requested) / 2)
    return start, start + requested


def open_source_array(path: str, level: str) -> zarr.Array:
    """Open a bare ZYX array or one named level from a local or remote group."""
    node = zarr.open(path, mode="r")
    if isinstance(node, zarr.Array):
        return node
    if level not in node:
        available = sorted(node.array_keys())
        raise KeyError(
            f"Pyramid level '{level}' not found in {path}; available: {available}"
        )
    return node[level]


def _create_target_array(
    partial_path: Path,
    *,
    shape_zyx: tuple[int, int, int],
) -> tuple[zarr.Group, zarr.Array]:
    group_kwargs: dict[str, object] = {"mode": "w"}
    zarr_v3 = int(zarr.__version__.split(".", 1)[0]) >= 3
    if zarr_v3:
        group_kwargs["zarr_format"] = 2
    group = zarr.open_group(str(partial_path), **group_kwargs)
    chunks = (
        OUTPUT_Z,
        min(CHUNK_XY, shape_zyx[1]),
        min(CHUNK_XY, shape_zyx[2]),
    )
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    create_kwargs = {
        "shape": shape_zyx,
        "chunks": chunks,
        "dtype": np.uint8,
        "compressor": compressor,
        "fill_value": 0,
    }
    if zarr_v3:
        target = group.create_array("0", **create_kwargs)
    else:
        target = group.create_dataset("0", **create_kwargs)
    return group, target


def prepare_isotropic_input(
    input_zarr: str,
    output_zarr: Path,
    *,
    level: str = "2",
    workers: int = 8,
) -> Path:
    """Write the centered four-plane rounded-mean ZYX volume transactionally."""
    source = open_source_array(input_zarr, level)
    shape = tuple(int(value) for value in source.shape)
    if source.dtype != np.uint8 or len(shape) != 3:
        raise ValueError(f"Expected 3D uint8 source, got {shape} {source.dtype}")
    z0, z1 = centered_slice(shape[0], INPUT_Z)

    output_zarr = Path(output_zarr)
    partial = output_zarr.with_name(output_zarr.name + ".partial")
    if output_zarr.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace {output_zarr} or {partial}")

    group, target = _create_target_array(
        partial,
        shape_zyx=(OUTPUT_Z, shape[1], shape[2]),
    )
    group.attrs.update(
        {
            "format": FORMAT_TAG,
            "source": str(input_zarr),
            "source_level": str(level),
            "source_shape_zyx": list(shape),
            "source_z_slice": [z0, z1],
            "z_pool": "rounded mean of 4 centered source planes",
        }
    )
    tiles = [
        (y0, min(shape[1], y0 + TILE), x0, min(shape[2], x0 + TILE))
        for y0 in range(0, shape[1], TILE)
        for x0 in range(0, shape[2], TILE)
    ]

    def process(tile: tuple[int, int, int, int]) -> int:
        y0, y1, x0, x1 = tile
        block_ZYX = np.asarray(
            source[z0:z1, y0:y1, x0:x1], dtype=np.float32
        )
        pooled_ZYX = np.rint(
            block_ZYX.reshape(
                OUTPUT_Z, POOL_Z, y1 - y0, x1 - x0
            ).mean(axis=1)
        ).astype(np.uint8)
        target[:, y0:y1, x0:x1] = pooled_ZYX
        return 1

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count in pool.map(process, tiles):
            completed += count
            if completed % 50 == 0 or completed == len(tiles):
                print(f"tiles={completed}/{len(tiles)}", flush=True)
    partial.replace(output_zarr)
    print(f"wrote {output_zarr} shape={tuple(target.shape)}")
    return output_zarr


def parse_args(argv: Sequence[str] | None = None):
    """Parse the surface-volume preparation command line."""
    parser = HyphenUnderscoreParser(description=__doc__)
    parser.add_argument(
        "input_zarr",
        help="Surface-volume OME-Zarr group path or URL (or a bare 3D array).",
    )
    parser.add_argument(
        "output_zarr", type=Path, help="Output Zarr path; refuses to overwrite."
    )
    parser.add_argument(
        "--level",
        default="2",
        help=(
            "Input pyramid level to read (default 2: one level-2 pixel of a "
            "2.399 µm render is ~9.6 µm)."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 9 µm preparation command."""
    args = parse_args(argv)
    prepare_isotropic_input(
        args.input_zarr,
        args.output_zarr,
        level=args.level,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
