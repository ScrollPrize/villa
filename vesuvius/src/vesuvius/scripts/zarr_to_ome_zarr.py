#!/usr/bin/env python
"""Convert a plain Zarr array (v2 or v3) into a multi-level OME-Zarr pyramid.

Output layout (as written by ``tifs_to_ome_zarr.py``): a Zarr v2 group with
arrays ``0 .. levels-1``, ZYX, blosc zstd + bitshuffle, chunks entirely equal
to the fill value not stored, and OME-NGFF 0.4 ``multiscales`` metadata.

``--clear-plateau V`` removes flat blocks of value ``V`` (e.g. 127: sigmoid 0.5 where inference saw
nothing, which a blend leaves as a gray shell around the scroll): voxels whose whole cube of edge
``--plateau-size`` is ``V`` are plateau cores; every voxel <= ``V`` within core edge/2 + ``--plateau-ramp``
of a core is set to 0, which also takes the blend ramp at the plateau's edge.

Leading non-spatial axes (e.g. a ``(1, Z, Y, X)`` prediction) are dropped:
size-1 axes automatically, larger ones with ``--channel``.

Level 0 is copied in blocks aligned to both the input and output chunk grids,
one block per task in a process pool. Levels 1+ are reduced by vc_reader's
``write_pyramid_level``, each worker process writing a disjoint range of
output chunks. Every stage shows a tqdm bar (done/total, elapsed, ETA).

Example::

    python zarr_to_ome_zarr.py in.zarr out.ome.zarr --levels 6 --workers 16
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm
from vc_reader.reader import (
    ZarrWriter,
    create_group,
    create_pyramid_levels,
    ome_multiscales,
    open_zarr,
    write_pyramid_level,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_zyx(value: str) -> tuple[int, int, int]:
    parts = [int(p) for p in value.split(",")]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError(f"expected N or Z,Y,X positive integers, got {value!r}")
    return tuple(parts)


def leading_index(shape: tuple[int, ...], channel: int | None) -> tuple[int, ...]:
    """Integer indices for the axes in front of the last three (ZYX)."""
    lead = shape[:-3]
    if not lead:
        return ()
    if channel is None:
        if any(n != 1 for n in lead):
            raise SystemExit(f"input has leading axes {lead}; pick one with --channel")
        return (0,) * len(lead)
    if len(lead) != 1 or not 0 <= channel < lead[0]:
        raise SystemExit(f"--channel {channel} does not fit leading axes {lead}")
    return (channel,)


def block_shape(in_chunks: tuple[int, ...], out_chunks: tuple[int, ...], target: int) -> tuple[int, ...]:
    """Per axis: a multiple of the output chunk, aligned to the input chunk where that stays small,
    grown to about ``target`` voxels."""
    block = []
    for ic, oc in zip(in_chunks, out_chunks):
        b = math.lcm(ic, oc)
        if b > max(target, oc) * 2:
            b = oc
        b *= max(1, target // b)
        block.append(b)
    return tuple(block)


# ---------------------------------------------------------------- level 0 workers

_src = None
_dst = None
_lead: tuple[int, ...] = ()
_plateau: tuple[int, int, int] | None = None  # (value, cube edge, ramp)


def _init_copy(src_path: str, dst_path: str, lead: tuple[int, ...], plateau) -> None:
    global _src, _dst, _lead, _plateau
    _src = open_zarr(src_path, threads=1)
    _dst = ZarrWriter.open(dst_path, skip_empty=True, auth=None)
    _lead = lead
    _plateau = plateau


def plateau_halo(size: int, ramp: int) -> int:
    return 2 * (size // 2) + ramp


def clear_plateau(data: np.ndarray, value: int, size: int, ramp: int) -> int:
    """In place: zero voxels <= value within size//2 + ramp (Chebyshev) of a voxel whose size^3 cube is all
    ``value``. Exact for the interior of ``data`` that lies plateau_halo(size, ramp) from its faces.
    Returns the number of voxels cleared."""
    from scipy import ndimage as ndi

    core = ndi.minimum_filter((data == value).view(np.uint8), size=size, mode="constant", cval=0)
    if not core.any():
        return 0
    zone = ndi.maximum_filter(core, size=2 * (size // 2 + ramp) + 1, mode="constant", cval=0).view(bool)
    zone &= data <= value
    zone &= data != 0
    data[zone] = 0
    return int(zone.sum())


def _copy_block(start: tuple[int, int, int], stop: tuple[int, int, int]) -> tuple[int, int]:
    halo = 0 if _plateau is None else plateau_halo(_plateau[1], _plateau[2])
    shape = _src.shape[-3:]
    lo = tuple(max(0, a - halo) for a in start)
    hi = tuple(min(n, b + halo) for b, n in zip(stop, shape))
    data = _src.read(_lead + tuple(slice(a, b) for a, b in zip(lo, hi)))
    if not data.any():
        return 0, 0  # nothing to store (skip_empty would drop every chunk anyway)
    cleared = 0
    if _plateau is not None:
        inner = tuple(slice(a - l, b - l) for a, b, l in zip(start, stop, lo))
        before = np.count_nonzero(data[inner])
        if clear_plateau(data, *_plateau):
            cleared = before - np.count_nonzero(data[inner])
        data = data[inner]
    if not data.any():
        return 0, cleared
    _dst[tuple(slice(a, b) for a, b in zip(start, stop))] = data
    return 1, cleared


def _reduce_range(root: str, level: int, factors, reduction: str, begin, end) -> None:
    write_pyramid_level(root, level, factors=factors, reduction=reduction, threads=1,
                        chunk_range=(begin, end))


# ---------------------------------------------------------------- driver

def convert(args: argparse.Namespace) -> None:
    t_all = time.perf_counter()
    src = open_zarr(str(args.input))
    lead = leading_index(tuple(src.shape), args.channel)
    shape = tuple(src.shape[-3:])
    in_chunks = tuple(src.chunks[-3:])
    dtype = np.dtype(src.dtype)
    fill = src.fill_value if getattr(src, "fill_value", None) is not None else 0
    log(f"input {args.input}: shape {tuple(src.shape)} chunks {tuple(src.chunks)} {dtype}, "
        f"using ZYX {shape} (leading index {lead})")

    out = str(args.output)
    if Path(out).exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        log(f"removing existing {out}")

    root = create_group(out, overwrite=args.overwrite, auth=None)
    root.create_array("0", shape=shape, chunks=args.chunks, dtype=dtype, compressor="blosc",
                      cname="zstd", shuffle=2, level=args.compression_level, fill_value=fill)
    root.attrs["multiscales"] = [ome_multiscales(
        name=args.name, levels=args.levels - 1, scale=args.voxel_size, unit=args.unit,
        reduction=args.reduction)]

    # Level 0 ------------------------------------------------------------
    block = block_shape(in_chunks, args.chunks, args.block)
    starts = [range(0, n, b) for n, b in zip(shape, block)]
    tasks = [(s, tuple(min(a + b, n) for a, b, n in zip(s, block, shape))) for s in product(*starts)]
    log(f"level 0: {len(tasks)} blocks of {block} with {args.workers} processes")
    t0 = time.perf_counter()
    plateau = None
    if args.clear_plateau is not None:
        plateau = (args.clear_plateau, args.plateau_size, args.plateau_ramp)
        log(f"clearing plateaus of value {plateau[0]} (core cube {plateau[1]}, ramp {plateau[2]}, "
            f"halo {plateau_halo(plateau[1], plateau[2])})")
    nonempty = cleared = 0
    with ProcessPoolExecutor(args.workers, initializer=_init_copy,
                             initargs=(str(args.input), out + "/0", lead, plateau)) as pool:
        futures = [pool.submit(_copy_block, a, b) for a, b in tasks]
        bar = tqdm(as_completed(futures), total=len(futures), desc="level 0", unit="block", smoothing=0.05)
        for f in bar:
            n, c = f.result()
            nonempty += n
            cleared += c
            if plateau is not None:
                bar.set_postfix(cleared=f"{cleared:.3g}", refresh=False)
    log(f"level 0 done in {time.perf_counter() - t0:.1f} s ({nonempty}/{len(tasks)} blocks non-empty"
        + (f", {cleared} plateau voxels cleared)" if plateau is not None else ")"))

    # Levels 1+ ----------------------------------------------------------
    if args.levels > 1:
        create_pyramid_levels(out, args.levels - 1, factors=args.factors, overwrite=True)
    for level in range(1, args.levels):
        arr = open_zarr(out, level=str(level))
        grid = [math.ceil(n / c) for n, c in zip(arr.shape, arr.chunks)]
        # one task per row of chunks along x; whole level in one task once it is small
        ranges = [((z, y, 0), (z + 1, y + 1, grid[2])) for z in range(grid[0]) for y in range(grid[1])]
        if len(ranges) < args.workers:
            ranges = [((z, 0, 0), (z + 1, grid[1], grid[2])) for z in range(grid[0])]
        log(f"level {level}: shape {tuple(arr.shape)}, chunk grid {grid}, {len(ranges)} tasks")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(_reduce_range, out, level, args.factors, args.reduction, b, e)
                       for b, e in ranges]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"level {level}",
                          unit="task", smoothing=0.05):
                f.result()
        log(f"level {level} done in {time.perf_counter() - t0:.1f} s")

    log(f"wrote {out} ({args.levels} levels) in {time.perf_counter() - t_all:.1f} s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path, help="input Zarr array")
    p.add_argument("output", type=Path, help="output OME-Zarr group")
    p.add_argument("--levels", type=int, default=6, help="pyramid levels including level 0 (default 6)")
    p.add_argument("--chunks", type=parse_zyx, default=(128, 128, 128), help="output chunks, N or Z,Y,X")
    p.add_argument("--factors", type=parse_zyx, default=(2, 2, 2), help="downsampling per level")
    p.add_argument("--reduction", choices=("mean", "nearest", "mode"), default="mean",
                   help="mean for intensities/probabilities, mode or nearest for labels")
    p.add_argument("--channel", type=int, default=None, help="index into a leading channel axis")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes")
    p.add_argument("--block", type=int, default=384,
                   help="approximate level-0 copy block edge in voxels (per task)")
    p.add_argument("--compression-level", type=int, default=3, help="blosc zstd level")
    p.add_argument("--voxel-size", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"))
    p.add_argument("--unit", default="", help='OME axis unit, e.g. "micrometer"')
    p.add_argument("--name", default="volume", help="multiscales name")
    p.add_argument("--clear-plateau", type=int, default=None, metavar="V",
                   help="zero flat blocks of value V and their blend ramp (e.g. 127 for an inference border)")
    p.add_argument("--plateau-size", type=int, default=9, help="cube edge a plateau core must fill (default 9)")
    p.add_argument("--plateau-ramp", type=int, default=24,
                   help="voxels beyond a core's cube also cleared when <= V (default 24)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing output")
    args = p.parse_args()
    if args.levels < 1:
        p.error("--levels must be >= 1")
    convert(args)


if __name__ == "__main__":
    main()
