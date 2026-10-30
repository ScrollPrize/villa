"""Render a flat surface volume from a tifxyz mesh and a scan volume.

The output is what flat ink inference reads: a ZYX uint8 group whose level "0"
has one plane per normal offset. Plane j samples the scan trilinearly at
xyz + ((depth - 1) / 2 - j) * normal, with xyz and normals taken from the
vc_render_tifxyz-convention geometry in self_render_tifxyz. On PHerc0139 w035
this reproduces the published 9.362 µm surface volume (per-plane median r
0.993-0.9999 over the whole segment).

Only the scan chunks the surface touches are read, so a public mesh can be
rendered straight from the open-data bucket without VC3D or a full download.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
from typing import Sequence
import urllib.request

import numpy as np
from numcodecs import Blosc
from scipy.ndimage import map_coordinates

from vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input import (
    open_source_array,
)
from vesuvius.ink_detection.preprocessing.staged_write import publish_staged_output
from vesuvius.ink_detection.volume_io import ZARR_V3
from vesuvius.label_zarr import open_v2_group
from vesuvius.tifxyz_label_transfer.core import Surface
from vesuvius.tifxyz_label_transfer.io import load_surface
from vesuvius.tifxyz_label_transfer.self_render_tifxyz import surface_tile_geometry
from vesuvius.utils.cli import HyphenUnderscoreParser


DEFAULT_DEPTH = 28
TILE = 256
CHUNK_XY = 128
FORMAT_TAG = "tifxyz-surface-volume-v1"
READ_ATTEMPTS = 4
TIFXYZ_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")


def plane_offsets(depth: int) -> np.ndarray:
    """Normal offset of each output plane, matching published surface volumes."""
    return ((depth - 1) / 2.0 - np.arange(depth)).astype(np.float32)


def load_tifxyz(path: str):
    """Load a tifxyz directory from a local path or an http(s) URL."""
    if not path.startswith(("http://", "https://")):
        return load_surface(Path(path))
    local = Path(tempfile.mkdtemp(prefix="tifxyz_"))
    for name in TIFXYZ_FILES:
        urllib.request.urlretrieve(path.rstrip("/") + "/" + name, local / name)
    return load_surface(local)


class ChunkReader:
    """Decoded scan chunks with an LRU cache and parallel fetching."""

    def __init__(self, array, *, cache_chunks: int, workers: int) -> None:
        self.array = array
        self.chunks = np.asarray(array.chunks, dtype=np.int64)
        self.cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.cache_chunks = cache_chunks
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def _read(self, key: tuple[int, int, int]) -> np.ndarray:
        start = np.asarray(key) * self.chunks
        stop = np.minimum(start + self.chunks, self.array.shape)
        region = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        for attempt in range(READ_ATTEMPTS):
            try:
                return np.asarray(self.array[region], dtype=np.uint8)
            except (OSError, TimeoutError):  # transient remote-read failures
                if attempt == READ_ATTEMPTS - 1:
                    raise
                time.sleep(2**attempt)

    def keys_for(self, points_zyx: np.ndarray) -> list[tuple[int, int, int]]:
        """Chunks holding every trilinear corner of the given points."""
        base = np.floor(points_zyx).astype(np.int64)
        corners = np.concatenate([base, base + 1])
        corners = corners[np.all((corners >= 0) & (corners < self.array.shape), axis=1)]
        k = corners // self.chunks
        codes = np.unique((k[:, 0] << 40) | (k[:, 1] << 20) | k[:, 2])
        return [(int(c >> 40), int((c >> 20) & 0xFFFFF), int(c & 0xFFFFF)) for c in codes]

    def block(self, keys, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Assemble [lo, hi) in ZYX from the given chunks; other voxels stay 0."""
        missing = [key for key in keys if key not in self.cache]
        for key, data in zip(missing, self.pool.map(self._read, missing)):
            self.cache[key] = data
        out = np.zeros(tuple(hi - lo), dtype=np.uint8)
        for key in keys:
            self.cache.move_to_end(key)
            data = self.cache[key]
            start = np.asarray(key) * self.chunks
            a = np.maximum(lo, start)
            b = np.minimum(hi, start + data.shape)
            if np.any(b <= a):
                continue
            out[tuple(slice(i, j) for i, j in zip(a - lo, b - lo))] = data[
                tuple(slice(i, j) for i, j in zip(a - start, b - start))
            ]
        while len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return out


def level_surface(surface: Surface, level: int) -> Surface:
    """Express level-0 tifxyz coordinates and canvas at pyramid level ``level``.

    The canvas shrinks by 2**level, so one output pixel and one plane step are
    one level-``level`` voxel. Pyramid levels are 2x2x2 means, so a level-L voxel
    i is centred on level-0 coordinate (i + 0.5) * 2**level - 0.5.
    """
    if level == 0:
        return surface
    factor = float(2**level)
    to_level = lambda values: (values + 0.5) / factor - 0.5  # noqa: E731
    return Surface(
        x=to_level(surface.x),
        y=to_level(surface.y),
        z=to_level(surface.z),
        scale_yx=(surface.scale_yx[0] * factor, surface.scale_yx[1] * factor),
        valid=surface.valid,
        name=surface.name,
    )


def render_tile(surface, canvas_hw, bounds, offsets, reader: ChunkReader) -> np.ndarray:
    """Render one (depth, h, w) canvas tile; pixels off the mesh stay 0."""
    y0, y1, x0, x1 = bounds
    out = np.zeros((len(offsets), y1 - y0, x1 - x0), dtype=np.uint8)
    xyz, normals, valid = surface_tile_geometry(surface, canvas_hw, bounds)
    if not valid.any():
        return out
    points = xyz[valid][None] + offsets[:, None, None] * normals[valid][None]
    zyx = points[..., ::-1].reshape(-1, 3)
    upper = np.asarray(reader.array.shape) - 1
    lo = np.clip(np.floor(zyx.min(axis=0)).astype(np.int64), 0, upper)
    hi = np.clip(np.floor(zyx.max(axis=0)).astype(np.int64) + 2, 1, upper + 1)
    block = reader.block(reader.keys_for(zyx), lo, hi)
    values = map_coordinates(
        block, (zyx - lo).T, order=1, mode="constant", cval=0.0, prefilter=False
    )
    out[:, valid] = np.clip(np.rint(values), 0, 255).reshape(len(offsets), -1)
    return out


def render_surface_volume(
    tifxyz: str,
    volume_zarr: str,
    output_zarr: Path,
    *,
    level: str = "0",
    depth: int = DEFAULT_DEPTH,
    workers: int = 16,
    cache_chunks: int = 1024,
) -> Path:
    """Write the flat surface volume of one tifxyz mesh transactionally."""
    surface = level_surface(load_tifxyz(tifxyz), int(level))
    source = open_source_array(volume_zarr, level)
    if source.dtype != np.uint8 or source.ndim != 3:
        raise ValueError(f"Expected a 3D uint8 scan, got {source.shape} {source.dtype}")
    stored_h, stored_w = surface.shape
    canvas_hw = (
        int(round(stored_h / surface.scale_yx[0])),
        int(round(stored_w / surface.scale_yx[1])),
    )
    offsets = plane_offsets(depth)

    output_zarr = Path(output_zarr)
    partial = output_zarr.with_name(output_zarr.name + ".partial")
    if output_zarr.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace {output_zarr} or {partial}")
    group = open_v2_group(partial)
    create_kwargs = {
        "shape": (depth, *canvas_hw),
        "chunks": (depth, min(CHUNK_XY, canvas_hw[0]), min(CHUNK_XY, canvas_hw[1])),
        "dtype": np.uint8,
        "compressor": Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        "fill_value": 0,
    }
    target = (
        group.create_array("0", **create_kwargs)
        if ZARR_V3
        else group.create_dataset("0", **create_kwargs)
    )
    group.attrs.update(
        {
            "format": FORMAT_TAG,
            "tifxyz": str(tifxyz),
            "volume": str(volume_zarr),
            "volume_level": str(level),
            "plane_offsets": [float(value) for value in offsets],
            "plane_rule": "plane j samples xyz + ((depth - 1) / 2 - j) * normal",
        }
    )
    reader = ChunkReader(source, cache_chunks=cache_chunks, workers=workers)
    tiles = [
        (y0, min(canvas_hw[0], y0 + TILE), x0, min(canvas_hw[1], x0 + TILE))
        for y0 in range(0, canvas_hw[0], TILE)
        for x0 in range(0, canvas_hw[1], TILE)
    ]
    for completed, bounds in enumerate(tiles, start=1):
        y0, y1, x0, x1 = bounds
        target[:, y0:y1, x0:x1] = render_tile(surface, canvas_hw, bounds, offsets, reader)
        if completed % 50 == 0 or completed == len(tiles):
            print(f"tiles={completed}/{len(tiles)}", flush=True)
    publish_staged_output(partial, output_zarr)
    print(f"wrote {output_zarr} shape={tuple(target.shape)}")
    return output_zarr


def parse_args(argv: Sequence[str] | None = None):
    """Parse the surface-volume rendering command line."""
    parser = HyphenUnderscoreParser(description=__doc__)
    parser.add_argument("tifxyz", help="tifxyz directory path or URL.")
    parser.add_argument(
        "volume_zarr", help="Scan OME-Zarr group path or URL (or a bare 3D array)."
    )
    parser.add_argument(
        "output_zarr", type=Path, help="Output Zarr path; refuses to overwrite."
    )
    parser.add_argument(
        "--level",
        default="0",
        help=(
            "Scan pyramid level to render from (default 0). tifxyz coordinates are "
            "level-0 voxels; at level L the canvas and plane step are 2**L voxels."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="Number of planes (default 28, as the native 9.362 µm volumes).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--cache-chunks",
        type=int,
        default=1024,
        help="Decoded scan chunks kept in memory (default 1024, 2 GiB for 128³ uint8).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the surface-volume rendering command."""
    args = parse_args(argv)
    render_surface_volume(
        args.tifxyz,
        args.volume_zarr,
        args.output_zarr,
        level=args.level,
        depth=args.depth,
        workers=args.workers,
        cache_chunks=args.cache_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
