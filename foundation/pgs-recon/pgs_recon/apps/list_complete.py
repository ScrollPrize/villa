import argparse
import json
import logging
import sys
from pathlib import Path


def scan_is_complete(scan_dir: Path, meta=None) -> bool:
    # Load metadata dict if we haven't been given one
    if meta is None:
        # Check if we have a metadata file
        meta_path = scan_dir / 'metadata.json'
        if not meta_path.exists():
            logging.info(f'[{str(scan_dir)}] missing metadata.json')
            return False

        # Load the metadata
        with meta_path.open() as f:
            meta = json.loads(f.read())

    # Check for primary keys
    if 'scan' not in meta.keys():
        logging.info(f'[{str(scan_dir)}] missing meta.scan')
        return False
    # check for complete or success tag
    if 'complete' in meta['scan'].keys():
        if not meta['scan']['complete']:
            logging.info(f'[{str(scan_dir)}] meta.scan.complete == False')
            return False
    elif 'success' in meta['scan'].keys():
        if not meta['scan']['success']:
            logging.info(f'[{str(scan_dir)}] meta.scan.success == False')
            return False
    else:
        logging.info(f'[{str(scan_dir)}] missing scan.[complete|success]')
        return False

    # return if we don't have all the keys we need
    if 'capture_settings' not in meta['scan'].keys():
        logging.info(f'[{str(scan_dir)}] missing scan.capture_settings')
        return False
    if 'capture_positions' not in meta['scan'].keys():
        logging.info(f'[{str(scan_dir)}] missing scan.capture_positions')
        return False

    # Get the number of expected images per position
    num_images = 0
    for cap in meta['scan']['capture_settings']:
        num_images += len(cap['cameras'])

    # Get the number of capture positions
    num_positions = len(meta['scan']['capture_positions'])
    if num_positions == 0:
        logging.info(f'[{str(scan_dir)}] scan.capture_positions == 0')
        return False

    # Get the total number of expected images
    num_images *= num_positions

    # Collect a list of images
    prefix = meta['scan']['file_prefix']
    ext = meta['scan']['format'].lower()
    images = list(scan_dir.glob(f'{prefix}*.{ext}'))

    # Check if we have all images
    if len(images) == num_images:
        return True
    else:
        missing = num_images - len(images)
        logging.info(f'[{str(scan_dir)}] missing {missing} images')
        return False


def main():
    parser = argparse.ArgumentParser('pgs-list-complete')
    parser.add_argument('input', metavar='DIR', nargs='+',
                        help='One or more directories containing scans')
    parser.add_argument('--report-complete',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Print complete datasets to stdout')
    parser.add_argument('--report-incomplete',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Print incomplete datasets to stderr')
    parser.add_argument('--complete-file', '-o', metavar='FILE',
                        help='If specified, save the list of complete datasets '
                             'to the given file')
    parser.add_argument('--incomplete-file', metavar='FILE',
                        help='Save the list of incomplete datasets to the '
                             'given file')
    parser.add_argument('--log-level', type=str.upper,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='WARNING', help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Iterate over all input directories
    complete = []
    incomplete = []
    for input_dir in args.input:
        input_dir = Path(input_dir)
        for d in sorted(list(input_dir.rglob('*'))):
            # Skip things that aren't directories
            if not d.is_dir():
                continue

            if scan_is_complete(d):
                complete.append(d)
            else:
                incomplete.append(d)

    # print to console
    if args.report_complete:
        for d in complete:
            print(str(d))

    if args.report_incomplete:
        for d in incomplete:
            print(str(d), file=sys.stderr)

    # save to file
    if args.complete_file is not None:
        with Path(args.complete_file).open('w') as of:
            of.writelines(f'{str(d)}\n' for d in complete)

    if args.incomplete_file is not None:
        with Path(args.incomplete_file).open('w') as of:
            of.writelines(f'{str(d)}\n' for d in incomplete)


if __name__ == '__main__':
    main()
