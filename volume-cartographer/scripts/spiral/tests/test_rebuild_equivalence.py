"""A model-stage rebuild produces the session a full build would have.

The defence behind config.MODEL_STAGE_KEYS is that retaining the host inputs
and the brick pools across an allowlisted change is unobservable. The
source-scan test in tests/test_config.py checks that no allowlisted key is
even named during host preparation; this checks the other half end to end —
that the rebuilt session's checkpoint is structurally identical to one built
from scratch with the same value.

Like the golden run it needs the dataset and a GPU, so it only runs when
opted in:

    RUN_GOLDEN=1 uv run python -m pytest tests/test_rebuild_equivalence.py -s
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


def _skip_reason():
    if os.environ.get('RUN_GOLDEN') != '1':
        return 'set RUN_GOLDEN=1 to run the rebuild-equivalence build (GPU)'
    with open(SPEC_PATH) as spec_file:
        spec = json.load(spec_file)
    if not os.path.isdir(spec['dataset_path']):
        return f'dataset not available at {spec["dataset_path"]}'
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or '')
def test_a_model_stage_rebuild_matches_a_session_built_from_scratch():
    with tempfile.TemporaryDirectory(prefix='rebuild-equivalence-') as work_dir:
        result_path = os.path.join(work_dir, 'result.json')
        subprocess.run(
            [sys.executable,
             os.path.join(SPIRAL_DIR, 'tests', 'rebuild_equivalence_driver.py'),
             SPEC_PATH, result_path, os.path.join(work_dir, 'out')],
            cwd=SPIRAL_DIR, check=True)
        with open(result_path) as result_file:
            result = json.load(result_file)

    key, value = result['rebuild_key'], result['rebuild_value']
    # The change has teeth: it alters the model the session carries.
    assert result['before']['model_state'] != result['rebuilt']['model_state']
    assert result['before']['cfg'][key] != value
    assert result['rebuilt']['cfg'][key] == value
    # And the rebuilt session is the one a full rebuild would have produced.
    assert result['rebuilt'] == result['fresh']
