import json
import math
from pathlib import Path

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _reviewed_trace_unlocked, _save_snapshot, _load_snapshot,
)


@pytest.mark.parametrize('angle,expected', [(11, False), (10, False), (9.999, True),
                                         (math.nan, False), (math.inf, False)])
def test_strict_raw_angle_threshold(angle, expected):
    losses = {'angle_mean_deg': angle, 'angle_count': 12,
              'ema_0p99/angle_mean_deg': 1.0}
    assert _reviewed_trace_unlocked(False, losses) is expected


def test_no_labels_and_permanent_latch():
    assert not _reviewed_trace_unlocked(False, {})
    assert not _reviewed_trace_unlocked(False, {'angle_mean_deg': 0, 'angle_count': 0})
    assert _reviewed_trace_unlocked(True, {'angle_mean_deg': 20, 'angle_count': 12})
    assert _reviewed_trace_unlocked(True, {})


@pytest.mark.parametrize('unlocked', [False, True])
def test_gate_survives_checkpoint_resume(tmp_path, unlocked):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / 'snapshot.pt'
    _save_snapshot(path, model=model, optimizer=optimizer, step=7, config={},
                   metric=1.0, reviewed_trace_unlocked=unlocked)
    progress = {}
    assert _load_snapshot(path, model=model, progress_metadata=progress) == 7
    assert progress['reviewed_trace_unlocked'] is unlocked


def test_host_baked_test_copy_preserves_fixed_selection_and_parent_labels():
    import hashlib
    from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import baked_normals
    from vesuvius.neural_tracing.fiber_trace_3d.reviewed_failure_test import validate_failure_union
    import os
    source = os.environ.get('FIBER_REVIEWED_TEST_SOURCE')
    baked = os.environ.get('FIBER_REVIEWED_TEST_BAKED')
    if not source or not baked:
        pytest.skip('set FIBER_REVIEWED_TEST_SOURCE and FIBER_REVIEWED_TEST_BAKED')
    old, new = Path(source), Path(baked)
    previous = validate_failure_union(old)
    current = validate_failure_union(new)
    assert [(r['source'], r['path']) for r in previous['files']] == [
        (r['source'], r['path']) for r in current['files']]
    for row in current['files']:
        raw = (old/'test'/row['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['original_piece_sha256']
        before = json.loads(raw)
        after = json.loads((new/'test'/row['path']).read_text())
        allowed = {'baked_annotations', 'hv_classification', 'vc_open_data_coordinate_space'}
        assert {k:v for k,v in before.items() if k not in allowed} == {
            k:v for k,v in after.items() if k not in allowed}
        parent_bytes = Path(row['parent']).read_bytes()
        assert hashlib.sha256(parent_bytes).hexdigest() == row['parent_sha256']
        hv = json.loads(parent_bytes)['hv_classification']
        assert after['hv_classification']['manual_tag'] == (hv.get('manual_tag') or hv['automatic_tag'])
        assert row['parent_split'] == 'test'
        for fiber in load_fiber_file(new/'test'/row['path']):
            axes = baked_normals(fiber)
            assert axes is not None and len(axes) == len(fiber.line_points_xyz)
