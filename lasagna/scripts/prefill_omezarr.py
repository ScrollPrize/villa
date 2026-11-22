#!/usr/bin/env python3
"""Fill the vc_reader disk cache with the levels of a remote OME-Zarr volume.

Every chunk is fetched through :class:`vc_reader.reader.ZarrReader` into
``cache_dir``, the directory tiled inference (``InputLevel``) and the manager
read their input through, so a prefilled level is read without any further
network access. A cancelled or failed run resumes: chunks already in the cache
are not fetched again. No separate local copy of the volume is written.

Examples::

    lasagna-prefill s3://bucket/vol.zarr /cache --scales 0,1 --anon
    lasagna-prefill s3://bucket/vol.zarr /cache --scales 0 --bbox 100,200,300,900,1000,1100
    lasagna-prefill s3://bucket/vol.zarr /cache --scales 0 --center 500,600,700 --radius 256
    lasagna-prefill s3://bucket/vol.zarr /cache --scales 0 --zrange 1000,1200

Regions are given in level-0 voxels and are mapped to each level through the
group's level placement (:meth:`~vc_reader.reader.ZarrGroup.placement`), so
physical multiscale metadata is handled the same as unit level-0 spacing.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any, Callable, Iterable, Sequence

from vc_reader._reader import PrefillStatus
from vc_reader.reader import ZarrGroup, ZarrReader

BoundsZYX = tuple[int, int, int, int, int, int]
DEFAULT_CONNECTIONS = 64


def level_location(location: str, scale: int) -> str:
    """``<group>/<scale>``, the array location :class:`ZarrReader` opens."""
    if int(scale) < 0:
        raise ValueError("scale must be a non-negative OME-Zarr group index")
    return f"{location.rstrip('/')}/{int(scale)}"


def store_auth(anon: bool) -> Any:
    """ObjectStore/ZarrReader ``auth`` for an anonymous or credentialed read."""
    return None if anon else "auto"


def level_placement(location: str, scale: int, *, anon: bool) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where level ``scale`` sits in level-0 voxels: ``level0 = factor * index + offset`` (ZYX)."""
    group = ZarrGroup.open(location.rstrip("/"), auth=store_auth(anon))
    try:
        return group.placement(str(int(scale)))
    finally:
        group.close()


def level_chunks_in_region(
    shape_zyx: Sequence[int],
    chunks_zyx: Sequence[int],
    region_level0_zyx: BoundsZYX,
    placement: tuple[Sequence[float], Sequence[float]],
) -> list[tuple[int, int, int]]:
    """Chunk-grid indices of a level intersecting a level-0 voxel region (z0,y0,x0,z1,y1,x1).

    ``placement`` is the level's ``(factor, offset)`` in level-0 voxels per level index.
    """
    factor_zyx, offset_zyx = placement
    z0, y0, x0, z1, y1, x1 = (int(v) for v in region_level0_zyx)
    lows = (z0, y0, x0)
    highs = (z1, y1, x1)
    ranges = []
    for axis in range(3):
        factor, offset = float(factor_zyx[axis]), float(offset_zyx[axis])
        if not factor > 0:
            raise ValueError("level placement factor must be positive")
        n, c = int(shape_zyx[axis]), int(chunks_zyx[axis])
        lo = max(0, int(math.floor((lows[axis] - offset) / factor)))
        hi = min(n, int(math.ceil((highs[axis] - offset) / factor)))
        if hi <= lo:
            return []
        ranges.append(range(lo // c, (hi + c - 1) // c))
    return [(cz, cy, cx) for cz in ranges[0] for cy in ranges[1] for cx in ranges[2]]


def prefill_level(
    location: str,
    cache_dir: str,
    *,
    anon: bool = True,
    connections: int = DEFAULT_CONNECTIONS,
    chunks: Iterable[Sequence[int]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Any:
    """Fetch one array level (or the listed chunks of it) into ``cache_dir``.

    Returns the native ``PrefillResult``; raises ``RuntimeError`` unless its
    status is ``Completed``.
    """
    if int(connections) <= 0:
        raise ValueError("connections must be a positive integer")
    reader = ZarrReader.open(location, cache_dir=cache_dir, connections=int(connections), auth=store_auth(anon))
    try:
        if chunks is None:
            result = reader.prefill(progress=progress)
        else:
            result = reader.prefill_chunks([list(int(v) for v in chunk) for chunk in chunks], progress=progress)
    finally:
        reader.close()
    if result.status != PrefillStatus.Completed:
        raise RuntimeError(
            f"prefill of {location} {result.status.name.lower()}: {result.message or 'no message'} "
            f"({result.fetched_chunks}/{result.total_chunks} chunks, {result.error_chunks} errors)"
        )
    return result


def _console_progress(label: str, stream) -> Callable[[int, int], None]:
    started = time.monotonic()
    last = [-1]

    def progress(done: int, total: int) -> None:
        step = max(1, total // 100)
        if total and (done == total or done // step != last[0]):
            last[0] = done // step
            print(f"{label} {done}/{total} chunks {time.monotonic() - started:.0f}s", file=stream, flush=True)

    return progress


def _parse_ints(value: str, count: int, flag: str) -> list[int]:
    parts = [p for p in value.replace(" ", "").split(",") if p]
    if len(parts) != count:
        raise SystemExit(f"{flag} expects {count} comma-separated integers")
    try:
        return [int(p) for p in parts]
    except ValueError as error:
        raise SystemExit(f"{flag} expects integers: {error}") from error


def parse_region(args: argparse.Namespace) -> BoundsZYX | None:
    """The base-voxel ZYX region of ``--bbox``, ``--center``/``--radius`` or ``--zrange``."""
    given = [flag for flag in ("bbox", "center", "zrange") if getattr(args, flag)]
    if len(given) > 1:
        raise SystemExit("--bbox, --center and --zrange are mutually exclusive")
    if args.center and args.radius is None:
        raise SystemExit("--center requires --radius")
    if args.radius is not None and not args.center:
        raise SystemExit("--radius requires --center")
    if args.bbox:
        x0, y0, z0, x1, y1, z1 = _parse_ints(args.bbox, 6, "--bbox")
        region = (z0, y0, x0, z1, y1, x1)
    elif args.center:
        cx, cy, cz = _parse_ints(args.center, 3, "--center")
        r = int(args.radius)
        region = (cz - r, cy - r, cx - r, cz + r, cy + r, cx + r)
    elif args.zrange:
        z0, z1 = _parse_ints(args.zrange, 2, "--zrange")
        region = (z0, 0, 0, z1, sys.maxsize, sys.maxsize)
    else:
        return None
    if any(region[i] >= region[i + 3] for i in range(3)):
        raise SystemExit("region must have positive extent on every axis")
    return region


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill the vc_reader disk cache with OME-Zarr levels of a remote volume.")
    parser.add_argument("source", help="OME-Zarr group location (s3://bucket/vol.zarr)")
    parser.add_argument("cache_dir", help="vc_reader disk cache directory to fill")
    parser.add_argument("--scales", default="0", help="Comma-separated OME-Zarr levels (default: 0)")
    parser.add_argument("--anon", action="store_true", help="Read anonymously (public data)")
    parser.add_argument("--connections", type=int, default=DEFAULT_CONNECTIONS,
                        help=f"Parallel remote reads (default: {DEFAULT_CONNECTIONS})")
    parser.add_argument("--bbox", help="x0,y0,z0,x1,y1,z1 in level-0 voxels (half-open)")
    parser.add_argument("--center", help="x,y,z in level-0 voxels, with --radius")
    parser.add_argument("--radius", type=int, help="Half extent in level-0 voxels around --center")
    parser.add_argument("--zrange", help="z0,z1 in level-0 voxels")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.connections <= 0:
        raise SystemExit("--connections must be at least 1")
    scales = sorted({int(v) for v in _parse_ints(args.scales, len([p for p in args.scales.split(",") if p]), "--scales")})
    if any(s < 0 for s in scales):
        raise SystemExit("--scales must be non-negative")
    region = parse_region(args)
    source = args.source.rstrip("/")
    print(f"Source: {source}\nCache:  {args.cache_dir}\nLevels: {scales}", file=sys.stderr, flush=True)
    for scale in scales:
        location = level_location(source, scale)
        chunks = None
        if region is not None:
            placement = level_placement(source, scale, anon=args.anon)
            reader = ZarrReader.open(location, auth=store_auth(args.anon))
            try:
                chunks = level_chunks_in_region(reader.shape, reader.chunks, region, placement)
            finally:
                reader.close()
            print(f"Level {scale}: {len(chunks)} chunks in region", file=sys.stderr, flush=True)
        try:
            result = prefill_level(location, args.cache_dir, anon=args.anon, connections=args.connections,
                                   chunks=chunks, progress=_console_progress(f"Level {scale}:", sys.stderr))
        except RuntimeError as error:
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
            return 1
        print(f"Level {scale}: done, {result.fetched_chunks} fetched ({result.data_chunks} with data, "
              f"{result.empty_chunks} absent) of {result.total_chunks} chunks", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
