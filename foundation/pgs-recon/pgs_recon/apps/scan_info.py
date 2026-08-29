import argparse
import json
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

from pgs_recon.apps.list_complete import scan_is_complete
from pgs_recon.utils.apps import ANSICode

DATETIME_FMT = '%m/%d/%Y, %H:%M:%S (%Z)'


def get_by_path(data, key_list):
    """Get a nested dictionary entry from its key path"""
    for key in key_list:
        data = data.get(key, None)
        if data is None:
            break
    return data


def parse_filter(filter_str: str):
    key, _, value = filter_str.partition('=')
    key = key.split('/')
    if not isinstance(key, list):
        key = list(key)
    return key, value


def get_notes(scan_dir: Path) -> Tuple[Dict, Dict]:
    info = {
        'software': '',
        'scanner': 'Unknown',
        'notes': '',
        'complete': False,
        'num_captures': '?',
        'num_positions': '?',
        'duration': '?'
    }

    # Check if we have a metadata file
    meta_path = scan_dir / 'metadata.json'
    if not meta_path.exists():
        return info, {}

    # Load the metadata
    with meta_path.open() as f:
        meta = json.loads(f.read())

    # Get software and scanner info
    info['software'] = meta['software']
    if 'scanner' in meta.keys():
        hw = meta["scanner"]
        info['scanner'] = f'{hw["make"]} {hw["model"]} ({hw.get("sn", "")})'

    # Get scan info
    if 'scan' in meta.keys():
        scan = meta['scan']
        info['complete'] = scan_is_complete(scan_dir, meta=meta)
        if 'capture_settings' in scan.keys():
            info['num_captures'] = len(scan['capture_settings'])
        if 'capture_positions' in scan.keys():
            info['num_positions'] = len(scan['capture_positions'])
        if 'datetime_start' in scan.keys() and 'datetime_end' in scan.keys():
            start_time = scan['datetime_start']
            end_time = scan['datetime_end']
            if len(start_time) > 0 and len(end_time) > 0:
                start_time = dt.strptime(start_time, DATETIME_FMT)
                end_time = dt.strptime(end_time, DATETIME_FMT)
                info['duration'] = str(end_time - start_time)

    # Get sample info
    if 'sample' in meta.keys() and 'Notes' in meta['sample'].keys():
        info['notes'] = meta['sample']['Notes']

    return info, meta


def print_dir(dir_path, detail_level, status_filter, meta_filter=None) -> Tuple[
    bool, bool]:
    """Returns (is complete, passed meta filter)"""
    # Get info
    info, meta = get_notes(dir_path)

    # Filter by metadata filter first
    if meta_filter is not None:
        filter_key, filter_val = meta_filter
        val = get_by_path(meta, filter_key)
        if val is None or str(val) != filter_val:
            return info['complete'], False

    # Skip printing if not enabled
    if info['complete'] and status_filter == 'incomplete':
        return info['complete'], True
    elif not info['complete'] and status_filter == 'complete':
        return info['complete'], True

    # Start info msg
    s = ANSICode.OKGREEN if info['complete'] else ANSICode.FAIL
    e = ANSICode.ENDC
    scan = str(dir_path)
    cap = f'captures: {info["num_captures"]}'
    pos = f'positions: {info["num_positions"]}'
    dur = f'time: {info["duration"]}'
    notes = info['notes']

    if detail_level == 'minimal':
        notes = f', notes: {notes}' if len(notes) > 0 else ''
        print(f'{s}[{scan}] {cap}, {pos}, {dur}{notes}{e}')
    elif detail_level == 'full':
        notes = f'  notes: {notes}\n' if len(notes) > 0 else ''
        print(f'{scan}:\n'
              f'  scanner: {info["scanner"]}\n'
              f'  software: {info["software"]}\n'
              f'  complete: {s}{info["complete"]}{e}\n'
              f'  {cap}, {pos}, {dur}\n'
              f'{notes}')
    return info['complete'], True


def main():
    parser = argparse.ArgumentParser('pgs-info',
                                     description='Print metadata for every scan'
                                                 'in the provided directories')
    parser.add_argument('input', metavar='DIR', nargs='+',
                        help='Input directories containing scans')
    parser.add_argument('--detail', default='minimal', type=str.lower,
                        choices=['minimal', 'full'])
    parser.add_argument('--status', default='all', type=str.lower,
                        choices=['complete', 'incomplete', 'all'])
    parser.add_argument('--filter', '-f',
                        help='Filter scans by a metadata value. For example, '
                             '\'--filter scanner/sn=G01-001\' will return all '
                             'scans captured on the scanner with serial no. '
                             'G01-001. Supported operators: =')
    args = parser.parse_args()

    # Parse the metadata filter
    meta_filter = None
    if args.filter is not None:
        meta_filter = parse_filter(args.filter)

    # Iterate over inputs
    complete = 0
    incomplete = 0
    skipped = 0
    for input_dir in sorted(args.input):
        input_dir = Path(input_dir)
        # Handle input which is scan directory
        if (input_dir / 'metadata.json').exists():
            c, p = print_dir(input_dir, args.detail, args.print, meta_filter)
            if not p:
                skipped += 1
                continue
            if c:
                complete += 1
            else:
                incomplete += 1
        # Handle directory of scans
        else:
            for d in sorted(list(input_dir.iterdir())):
                if not d.is_dir():
                    continue
                c, p = print_dir(d, args.detail, args.status, meta_filter)
                if not p:
                    skipped += 1
                    continue
                if c:
                    complete += 1
                else:
                    incomplete += 1
    total = complete + incomplete
    print(
        f'{ANSICode.BOLD}Processed {total} scans ({skipped} skipped):{ANSICode.ENDC} '
        f'{ANSICode.OKGREEN}{complete} complete{ANSICode.ENDC}, '
        f'{ANSICode.FAIL}{incomplete} incomplete{ANSICode.ENDC}')


if __name__ == '__main__':
    main()
