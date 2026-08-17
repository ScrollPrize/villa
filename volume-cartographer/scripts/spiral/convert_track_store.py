#!/usr/bin/env python3
"""Convert a legacy pickle/DBM track database to VC's packed track store."""

import argparse
import time

from tracks import track_store_path, write_packed_track_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracks_dbm', help='logical DBM path or its .db backing file')
    parser.add_argument(
        '-o', '--output', help='output directory (default: DBM + .vctracks)')
    parser.add_argument(
        '--force', action='store_true', help='replace an existing packed store')
    parser.add_argument(
        '--quiet', action='store_true', help='disable conversion progress')
    args = parser.parse_args()

    destination = args.output or track_store_path(args.tracks_dbm)
    started = time.perf_counter()
    result = write_packed_track_store(
        args.tracks_dbm, destination, force=args.force,
        show_progress=not args.quiet)
    print(f'packed store ready at {result} in {time.perf_counter() - started:.1f}s')


if __name__ == '__main__':
    main()
