"""bench_metal_kernels.py runs, and reports, the path each variant names.

Each worker decides its own path: an inherited FIT_SPIRAL_DIRECT_LR or
FIT_SPIRAL_METAL must neither relabel a run (an eager flow timed as
flow-metal) nor stop one (flow-eager refusing an explicit direct request),
and the result names the path that actually ran.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SPIRAL_DIR = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason='the benchmark times the Metal kernels on an Apple GPU (MPS)')


def _worker(variant, inherited):
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith('FIT_SPIRAL_')}
    environ.update(inherited)
    done = subprocess.run(
        [sys.executable, str(SPIRAL_DIR / 'bench_metal_kernels.py'), '--worker',
         '--variant', variant, '--resolution', '12', '24', '24',
         '--points', '2000', '--windings', '12', '--warmups', '0',
         '--iterations', '1'],
        cwd=SPIRAL_DIR, env=environ, capture_output=True, text=True,
        timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('inherited', [
    {}, {'FIT_SPIRAL_DIRECT_LR': '0'}, {'FIT_SPIRAL_DIRECT_LR': '1'},
    {'FIT_SPIRAL_METAL': '0'}], ids=['clean', 'direct-0', 'direct-1', 'metal-0'])
@pytest.mark.parametrize('variant,path', [
    ('flow-metal', 'metal'), ('flow-eager', 'eager')])
def test_flow_variants_run_the_path_their_label_names(variant, path, inherited):
    assert _worker(variant, inherited)['path'] == path


@pytest.mark.parametrize('variant,path', [
    ('gap-metal', 'metal'), ('gap-eager', 'eager')])
def test_gap_variants_run_the_path_their_label_names(variant, path):
    assert _worker(variant, {'FIT_SPIRAL_METAL': '1' if path == 'eager' else '0'})['path'] == path
