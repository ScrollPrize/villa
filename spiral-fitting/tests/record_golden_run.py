"""Calibrate the golden-run tolerance bands from repeated baseline runs.

Run this against *unmodified* code on the development machine (needs the
PHercParis4 dataset and a GPU):

    uv run python tests/record_golden_run.py --runs 4

Each run executes tests/golden_run_driver.py in a fresh subprocess. Leaves
identical across runs become exact assertions; numeric leaves that vary
get bands from the observed spread plus margin (see golden_run_compare.py).
The result is written to tests/golden/golden_bands.json, which
tests/test_golden_run.py then checks fresh runs against.

Re-run this recorder only to re-baseline after an *intentional* numerics
change; refactoring commits must pass against the existing golden file.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SPIRAL_DIR, 'tests'))

from golden_run_compare import calibrate  # noqa: E402


def run_driver_once(spec_path, work_dir, index):
    result_path = os.path.join(work_dir, f'result_{index}.json')
    out_dir = os.path.join(work_dir, f'out_{index}')
    subprocess.run(
        [sys.executable, os.path.join(SPIRAL_DIR, 'tests', 'golden_run_driver.py'),
         spec_path, result_path, out_dir],
        cwd=SPIRAL_DIR, check=True,
    )
    with open(result_path) as result_file:
        return json.load(result_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--spec', default=os.path.join(
        SPIRAL_DIR, 'tests', 'golden', 'run_spec.json'))
    parser.add_argument('--out', default=os.path.join(
        SPIRAL_DIR, 'tests', 'golden', 'golden_bands.json'))
    parser.add_argument('--work-dir', default=None,
                        help='keep per-run outputs here instead of a temp dir')
    arguments = parser.parse_args()

    with open(arguments.spec) as spec_file:
        spec = json.load(spec_file)

    work_dir = arguments.work_dir or tempfile.mkdtemp(prefix='golden-run-')
    os.makedirs(work_dir, exist_ok=True)
    print(f'calibrating from {arguments.runs} runs; per-run outputs in {work_dir}')

    results = []
    for index in range(arguments.runs):
        print(f'--- baseline run {index + 1}/{arguments.runs} ---')
        results.append(run_driver_once(arguments.spec, work_dir, index))

    golden = calibrate(results, spec)
    with open(arguments.out, 'w') as out_file:
        json.dump(golden, out_file, indent=2, sort_keys=True)

    print(f'wrote {arguments.out}:')
    print(f'  exact assertions: {len(golden["exact"])}')
    print(f'  banded assertions: {len(golden["bands"])}')
    print(f'  ignored leaves: {len(golden["ignored"])}')
    for entry in golden['ignored']:
        print(f'    ignored {entry["path"]}: {entry["reason"]}')


if __name__ == '__main__':
    main()
