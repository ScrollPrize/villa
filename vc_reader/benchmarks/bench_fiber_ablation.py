"""Run the four CUDA fiber configurations sequentially in counterbalanced order."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vc-site-packages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configurations = [('sample', 'separate'), ('batch', 'separate'),
                      ('batch', 'shared'), ('sample', 'shared')]
    runs = []
    for repetition, order in enumerate((configurations, configurations[::-1]), 1):
        for augmentation, cache in order:
            name = f'{augmentation}_{cache}_r{repetition}'
            output = args.output / name
            command = [sys.executable, str(Path(__file__).with_name('bench_fiber_trainer.py')),
                       '--checkpoint', str(args.checkpoint.resolve()), '--output', str(output.resolve()),
                       '--backend', 'vc-reader', '--heading', 'tensor', '--steps', '500', '--warmup', '20',
                       '--grad-steps', '1', '--lookahead', '0', '--cuda-augmentation', augmentation,
                       '--cuda-cache', cache, '--vc-site-packages', str(args.vc_site_packages.resolve())]
            print(f'START {name}', flush=True)
            with (args.output / f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((output / 'summary.json').read_text())
            runs.append(dict(name=name, repetition=repetition, augmentation=augmentation,
                             cache=cache, wall_seconds=report['measured_wall_seconds'],
                             loader_seconds=report['loader_seconds']['total'],
                             preparation_seconds=report['input_seconds']['total'] - report['loader_seconds']['total'],
                             update_seconds=report['update_seconds']['total']))
            (args.output / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
            print(f'DONE {name}: {runs[-1]}', flush=True)


if __name__ == '__main__':
    main()
