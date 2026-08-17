"""Golden-run characterization test for the headless fit path
(REFACTOR_PLAN.md, PR 1 commit 0).

Runs one short headless fit (tests/golden_run_driver.py, subprocess) on the
local PHercParis4 dataset and asserts it against tests/golden/golden_bands.json:
structural invariants exactly (checkpoint key sets and tensor shapes,
iteration counts, config round-trip, host RNG-state hashes where stable) and
loss traces within calibrated tolerance bands.

This is a local end-to-end regression test, not a CI job: it needs the
dataset, a GPU, and ~10+ minutes, so it only runs when opted in:

    RUN_GOLDEN=1 uv run python -m pytest tests/test_golden_run.py -s

Calibrate/re-baseline with tests/record_golden_run.py (only when a numerics
change is intentional).
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.environ.get(
    'GOLDEN_RUN_SPEC', os.path.join(SPIRAL_DIR, 'tests', 'golden', 'run_spec.json'))
GOLDEN_PATH = os.environ.get(
    'GOLDEN_RUN_BANDS', os.path.join(SPIRAL_DIR, 'tests', 'golden', 'golden_bands.json'))

sys.path.insert(0, os.path.join(SPIRAL_DIR, 'tests'))
from golden_run_compare import compare  # noqa: E402


def _skip_reason():
    if os.environ.get('RUN_GOLDEN') != '1':
        return 'set RUN_GOLDEN=1 to run the golden characterization fit (GPU, ~10+ min)'
    if not os.path.exists(GOLDEN_PATH):
        return f'no golden file at {GOLDEN_PATH}; run tests/record_golden_run.py first'
    with open(SPEC_PATH) as spec_file:
        spec = json.load(spec_file)
    if not os.path.isdir(spec['dataset_path']):
        return f'dataset not available at {spec["dataset_path"]}'
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or '')
def test_golden_run_matches_baseline():
    with open(GOLDEN_PATH) as golden_file:
        golden = json.load(golden_file)

    with tempfile.TemporaryDirectory(prefix='golden-run-check-') as work_dir:
        result_path = os.path.join(work_dir, 'result.json')
        subprocess.run(
            [sys.executable,
             os.path.join(SPIRAL_DIR, 'tests', 'golden_run_driver.py'),
             SPEC_PATH, result_path, os.path.join(work_dir, 'out')],
            cwd=SPIRAL_DIR, check=True,
        )
        with open(result_path) as result_file:
            result = json.load(result_file)

    failures = compare(golden, result)
    if failures:
        pytest.fail(
            f'{len(failures)} golden-run mismatches:\n' + '\n'.join(failures))
