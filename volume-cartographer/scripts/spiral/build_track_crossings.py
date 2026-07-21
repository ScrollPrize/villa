#!/usr/bin/env python3
"""Build an exact-crossing CSR sidecar beside a Spiral tracks DBM.

The default output is ``TRACKS_DBM.crossings.npz``.  ``fit_spiral.py`` loads
that file automatically when its DBM fingerprint matches, avoiding the large
packed-coordinate sort at every session load.

Example:

    python scripts/spiral/build_track_crossings.py /path/to/tracks.dbm

To build only the tracks wholly contained in a fitting ROI:

    python scripts/spiral/build_track_crossings.py /path/to/tracks.dbm \
        --z-min 4000 --z-max 17000 --temp-dir /fast/disk/tmp

The z range is half-open: ``[z_min, z_max)``. A range-limited build replaces
the adjacent sidecar, so the fitter using it must select the same or a narrower
track range.

The builder stages 20 bytes per selected point in temporary disk files beside
the tracks DBM by default (12 bytes of coordinates plus an 8-byte packed voxel
key). The native VC kernel uses a parallel radix sort and compact crossing
records, and releases its sort arrays before CSR consolidation. Those files are
removed after the sidecar is written.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from tracks import (
    _tracks_db_signature,
    build_crossing_partner_csr_disk_backed,
    load_track_crossing_cache,
    normalize_tracks_dbm_path,
    track_crossing_cache_path,
    write_track_crossing_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'tracks_dbm',
        help='logical DBM path (a .db backing-file path is also accepted)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='replace an already valid sidecar',
    )
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='crossing-analysis worker processes (default: min(8, CPU count))',
    )
    parser.add_argument(
        '--z-min', type=int,
        help='include only tracks entirely at or above this z voxel',
    )
    parser.add_argument(
        '--z-max', type=int,
        help='include only tracks entirely below this z voxel (exclusive)',
    )
    parser.add_argument(
        '--temp-dir',
        help='directory for memory-mapped coordinates and packed voxel keys '
             '(default: beside the tracks DBM)',
    )
    return parser.parse_args()


def build_cache(
        path, force=False, workers=1, show_progress=True,
        z_lo=None, z_hi=None, temporary_directory=None):
    path = normalize_tracks_dbm_path(path)
    if temporary_directory is None:
        temporary_directory = os.fspath(Path(path).parent)
    if workers < 1:
        raise ValueError('workers must be positive')
    if z_lo is not None and z_hi is not None and z_lo >= z_hi:
        raise ValueError('z_min must be less than z_max')
    destination = track_crossing_cache_path(path)
    requested_z_range = (z_lo, z_hi)
    if (not force
            and load_track_crossing_cache(
                path, warn=False,
                expected_z_range=requested_z_range) is not None):
        print(f'track crossing cache is already current: {destination}')
        return destination

    signature_before = _tracks_db_signature(path)
    started = time.perf_counter()
    csr = build_crossing_partner_csr_disk_backed(
        path, z_lo=z_lo, z_hi=z_hi,
        workers=workers, show_progress=show_progress,
        temporary_directory=temporary_directory)

    signature_after = _tracks_db_signature(path)
    if signature_after != signature_before:
        raise RuntimeError(
            'tracks DBM changed while crossings were being indexed; not writing cache')
    write_track_crossing_cache(
        path, csr, destination=destination, source_signature=signature_before,
        z_lo=z_lo, z_hi=z_hi)
    elapsed = time.perf_counter() - started
    size_mib = destination.stat().st_size / (1 << 20)
    print(
        f'wrote {destination}: {len(csr["source_ids"])} tracks, '
        f'{len(csr["partners"])} directed partner records, {size_mib:.1f} MiB '
        f'in {elapsed:.1f}s'
    )
    return destination


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit('--workers must be positive')
    if (args.z_min is not None and args.z_max is not None
            and args.z_min >= args.z_max):
        raise SystemExit('--z-min must be less than --z-max')
    build_cache(
        args.tracks_dbm, force=args.force, workers=args.workers,
        z_lo=args.z_min, z_hi=args.z_max,
        temporary_directory=args.temp_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
