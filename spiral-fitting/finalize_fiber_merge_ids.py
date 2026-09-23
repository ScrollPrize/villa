"""Repair a completed continuous-strip batch's segment UUIDs; optionally wait.

The export lock prevents changing metadata until every export attempt finishes.
Original metadata is backed up, and only its uuid field is changed.
"""
import argparse
import fcntl
from pathlib import Path

from batch_fiber_merge import finalize_segment_ids_locked, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--wait', action='store_true', help='wait for the active export batch to release its lock')
    args = parser.parse_args()
    root = args.output.resolve()
    if not (root / 'manifest.json').is_file():
        parser.error('output is not an existing fiber merge batch')
    with (root / '.batch.lock').open('a') as lock:
        print('Waiting for exports to finish' if args.wait else 'Checking completed exports', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
        try:
            result = finalize_segment_ids_locked(root)
        except Exception as error:
            write_json(root / 'metadata_ids_status.json', {'complete': False, 'error': str(error)})
            raise
        print(f"Verified {result['unique_ids']} unique segment IDs; original metadata backed up", flush=True)


if __name__ == '__main__':
    main()
