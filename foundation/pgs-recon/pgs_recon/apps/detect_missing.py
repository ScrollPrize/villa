import argparse
from pathlib import Path
import sys
import json

from pgs_recon.utils.apps import ANSICode


def process_scan(input_dir):
    # Log the scan directory
    print(f'{ANSICode.BOLD}{str(input_dir)}{ANSICode.ENDC}')

    # Check for metadata file
    meta_path = input_dir / 'metadata.json'
    if not meta_path.exists():
        print('  metadata.json')
        print(f'{ANSICode.FAIL}Missing 1 file{ANSICode.ENDC}\n')
        return

    # Load the metadata
    with meta_path.open() as f:
        meta = json.loads(f.read())

    # Get the number of capture positions
    num_positions = len(meta['scan']['capture_positions'])

    # Get the number of captures at each position
    num_caps = len(meta['scan']['capture_settings'])

    # Get the camera list for each capture
    cams_per_cap = []
    for cap in meta['scan']['capture_settings']:
        cams_per_cap.append(sorted(cap['cameras']))

    # Get the file pattern info
    prefix = meta['scan']['file_prefix']
    ext = meta['scan']['format'].lower()

    # Check each file
    missing = 0
    for pos in range(num_positions):
        for cap in range(num_caps):
            for cam in cams_per_cap[cap]:
                f = f'{prefix}{cam:03}_{pos:05}_{cap:02}.{ext}'
                if not (input_dir / f).exists():
                    missing += 1
                    print(f'  {f}')
    s = ANSICode.FAIL if missing > 0 else ANSICode.OKGREEN
    e = ANSICode.ENDC
    print(f'{s}Missing {missing} files{e}\n')


def main():
    parser = argparse.ArgumentParser('pgs-detect-missing')
    parser.add_argument('input', metavar='DIR', nargs='+',
                        help='Input directories containing scans')
    args = parser.parse_args()

    # Load scan directory
    for input_dir in sorted(args.input):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            print('Error: Input is not a directory')
            continue

        if (input_dir / 'metadata.json').exists():
            process_scan(input_dir)
        else:
            for d in sorted(list(input_dir.iterdir())):
                if not d.is_dir():
                    continue
                process_scan(d)


if __name__ == '__main__':
    main()
