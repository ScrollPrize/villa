#!/usr/bin/env python3
"""Read a real uint8 Zarr crop with vc_reader, then benchmark native thinning."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np
from vc_reader.reader import ZarrReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array', required=True)
    parser.add_argument('--start', nargs=3, type=int, default=(0, 0, 0), metavar=('Z', 'Y', 'X'))
    parser.add_argument('--shape', nargs=3, type=int, default=(4, 2048, 2048), metavar=('Z', 'Y', 'X'))
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(s < 0 for s in args.start) or any(n <= 0 for n in args.shape):
        parser.error('start must be nonnegative and shape positive')
    reader = ZarrReader.open(args.array, cache_writes=False)
    crop = reader.read(tuple(slice(s, s+n) for s, n in zip(args.start, args.shape)))
    if crop.dtype != np.uint8 or tuple(crop.shape) != tuple(args.shape):
        raise ValueError('expected a uint8 prediction crop with the requested shape')
    if not np.any(crop):
        raise ValueError('empty prediction crop; choose a populated cached region')
    if not np.all((crop == 0) | (crop == 255)):
        raise ValueError('OpenCV comparison requires predictions encoded as 0/255')
    with tempfile.TemporaryDirectory(prefix='vcr-thinning-') as directory:
        raw = Path(directory) / 'crop.raw'
        np.ascontiguousarray(crop).tofile(raw)
        z, y, x = crop.shape
        command = [str(args.executable.resolve()), str(raw), str(y), str(x), str(z), str(args.repetitions)]
        if args.cuda:
            command.append('cuda')
        report = json.loads(subprocess.check_output(command, text=True))
    cache = args.executable.resolve().parent / 'CMakeCache.txt'
    if cache.is_file():
        report['build'] = {line.split(':', 1)[0]: line.split('=', 1)[1]
                           for line in cache.read_text().splitlines()
                           if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CUDA_ARCHITECTURES:'))}
    cpu_info = Path('/proc/cpuinfo')
    report['cpu'] = platform.processor() or platform.machine()
    if cpu_info.is_file():
        report['cpu'] = next((line.split(':', 1)[1].strip() for line in cpu_info.read_text().splitlines()
                              if line.startswith('model name')), report['cpu'])
    report['command'] = [sys.executable, *sys.argv]
    report.update(array=args.array, start_zyx=args.start, foreground_voxels=int(np.count_nonzero(crop)),
                  platform=platform.platform(), executable=str(args.executable.resolve()))
    if args.cuda:
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
