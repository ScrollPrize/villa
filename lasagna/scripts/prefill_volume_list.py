#!/usr/bin/env python3
"""Prefill scale 0 of a list of public Vesuvius OME-Zarr volumes into one vc_reader cache.

Each non-empty, non-comment line in the input file must be a full URI like::

    s3://vesuvius-challenge-open-data/PHerc0125/volumes/example.zarr

Volumes are processed one at a time; within a volume, chunks are fetched
concurrently. The chunks land in ``<cache_dir>/s3/<bucket>/<key>/0``, the
layout the manager and tiled inference read from, so a prefilled volume is
later read without network access.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lasagna.scripts.prefill_omezarr import DEFAULT_CONNECTIONS, level_location, prefill_level

BUCKET_URI = "s3://vesuvius-challenge-open-data/"


@dataclass(frozen=True)
class Volume:
    source: str
    scroll: str
    name: str

    def cached_level(self, cache_dir: Path, scale: int = 0) -> Path:
        """Where vc_reader keeps this volume's level: ``<cache>/s3/<bucket>/<key>/<scale>``."""
        bucket, _, key = self.source.removeprefix("s3://").partition("/")
        return cache_dir / "s3" / bucket / key / str(int(scale))


def parse_volume_uri(raw: str, *, line_number: int) -> Volume:
    source = raw.strip().rstrip("/")
    if not source.startswith(BUCKET_URI):
        raise ValueError(
            f"line {line_number}: expected URI below {BUCKET_URI}, got {raw!r}"
        )

    relative = PurePosixPath(source.removeprefix(BUCKET_URI))
    parts = relative.parts
    if len(parts) != 3 or parts[1] != "volumes":
        raise ValueError(
            f"line {line_number}: expected <scroll>/volumes/<name>.zarr, got {raw!r}"
        )
    scroll, _volumes, name = parts
    if not scroll.startswith("PHerc") or not scroll.replace("_", "").isalnum():
        raise ValueError(f"line {line_number}: invalid scroll directory {scroll!r}")
    if not name.endswith(".zarr") or name in {".zarr", "..zarr"}:
        raise ValueError(f"line {line_number}: invalid volume directory {name!r}")
    return Volume(source=source, scroll=scroll, name=name)


def read_volume_list(path: Path) -> list[Volume]:
    volumes: list[Volume] = []
    seen_sources: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        volume = parse_volume_uri(stripped, line_number=line_number)
        if volume.source in seen_sources:
            raise ValueError(f"line {line_number}: duplicate volume {volume.source}")
        seen_sources.add(volume.source)
        volumes.append(volume)
    if not volumes:
        raise ValueError(f"{path}: no volume URIs found")
    return volumes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prefill scale 0 of each S3 OME-Zarr URI in a text file into a vc_reader cache."
    )
    parser.add_argument("volume_list", type=Path, help="One full S3 volume URI per line")
    parser.add_argument(
        "cache_dir", type=Path,
        help="vc_reader disk cache directory (the manager's <cache_dir>/volumes)",
    )
    parser.add_argument(
        "--connections", type=int, default=DEFAULT_CONNECTIONS,
        help=f"Parallel remote reads within each volume (default: {DEFAULT_CONNECTIONS})",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="Continue with later volumes after a failure",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sources and cache destinations without fetching",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.connections < 1:
        raise SystemExit("--connections must be at least 1")
    try:
        volumes = read_volume_list(args.volume_list)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    cache_dir = args.cache_dir.resolve()
    failures = 0
    for index, volume in enumerate(volumes, 1):
        print(f"[{index}/{len(volumes)}] {volume.source}", flush=True)
        print(f"  -> {volume.cached_level(cache_dir)}", flush=True)
        if args.dry_run:
            continue
        try:
            result = prefill_level(level_location(volume.source, 0), str(cache_dir),
                                   anon=True, connections=args.connections)
        except Exception as error:
            failures += 1
            print(f"prefill failed for {volume.source}: {error}", file=sys.stderr, flush=True)
            if not args.keep_going:
                return 1
            continue
        print(f"  {result.fetched_chunks} fetched of {result.total_chunks} chunks", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
