"""A checkpoint keeps the umbilicus end rule it was fitted with."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.interpolate
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_spiral
import flatten_spiral_checkpoint
from checkpoint_io import model_state_sha256
from config import Config
from test_checkpoint_load import _checkpoint, _live_context
from transforms import SpiralAndTransform
from umbilicus import (EXTRAPOLATE_ENDS, HOLD_ENDS, checkpoint_end_rule,
                       json_umbilicus_z_to_yx)

POINTS = [(100, 200, 20), (110, 220, 30), (160, 220, 60)]  # x, y, z
CROSSING, INSIDE = (0, 50), (25, 55)


@pytest.fixture
def path(tmp_path):
    out = tmp_path / 'umbilicus.json'
    out.write_text(json.dumps(
        {'control_points': [dict(x=x, y=y, z=z) for x, y, z in POINTS]}))
    return str(out)


def _main_axis(window):
    """The reader before ends were held."""
    zs = np.asarray([z for _, _, z in POINTS], dtype=np.float32)
    yxs = np.asarray([(y, x) for x, y, _ in POINTS], dtype=np.float32)
    return scipy.interpolate.interp1d(
        zs, yxs, axis=0, fill_value='extrapolate')(np.arange(*window))


def _preflight_context(path, window, rule, model_window=None):
    context = _live_context()
    context.z_begin, context.z_end = window
    context.model_z_begin, context.model_z_end = model_window or window
    context.umbilicus_end_rule = rule
    context.umbilicus_z_to_yx = lambda end_rule=None: json_umbilicus_z_to_yx(
        path, end_rule=end_rule or context.umbilicus_end_rule)
    context.umbilicus_end_rule_moves_axis = (
        fit_spiral.FitContext._end_rule_moves_model_axis(context))
    Path(path).unlink()
    return context


def test_a_checkpoint_without_the_rule_reads_as_extrapolated(path):
    rule = checkpoint_end_rule({})
    assert rule == EXTRAPOLATE_ENDS
    read = json_umbilicus_z_to_yx(path, end_rule=rule)
    np.testing.assert_array_equal(read(np.arange(*CROSSING)), _main_axis(CROSSING))
    with pytest.raises(ValueError, match='unknown umbilicus end rule'):
        checkpoint_end_rule({'umbilicus_end_rule': 'other'})


@pytest.mark.parametrize('saved, expected', [
    (None, EXTRAPOLATE_ENDS), (EXTRAPOLATE_ENDS, EXTRAPOLATE_ENDS),
    (HOLD_ENDS, HOLD_ENDS)])
def test_resume_adopts_the_saved_rule(path, tmp_path, saved, expected):
    checkpoint = tmp_path / 'a.ckpt'
    torch.save({} if saved is None else {'umbilicus_end_rule': saved}, checkpoint)
    context = SimpleNamespace(resume_path=str(checkpoint), umbilicus_end_rule=HOLD_ENDS)
    fit_spiral.FitContext._adopt_resume_end_rule(context)
    assert context.umbilicus_end_rule == expected


@pytest.mark.parametrize('saved', [None, EXTRAPOLATE_ENDS, HOLD_ENDS])
def test_flatten_rebuilds_the_saved_axis(path, saved):
    config = Config().as_dict()
    config.update(model_flow_bounds_radius=128, model_flow_bounds_z_margin=32)
    rule = saved or EXTRAPOLATE_ENDS
    zs = np.arange(*CROSSING)
    yx = json_umbilicus_z_to_yx(path, end_rule=rule)(zs)
    live = SpiralAndTransform(
        flow_integration_steps=config['model_num_flow_integration_steps'],
        flow_integration_solver=config['model_flow_integration_solver'],
        flow_min_corner_zyx=torch.tensor([-32, -128, -128]),
        flow_max_corner_zyx=torch.tensor([CROSSING[1] + 32, 128, 128]),
        umbilicus_zyx=torch.from_numpy(
            np.concatenate([zs[:, None], yx], axis=-1).astype(np.float32)),
        config=config)
    checkpoint = dict(z_begin=CROSSING[0], z_end=CROSSING[1],
                      spiral_and_transform=live.state_dict())
    if saved is not None:
        checkpoint['umbilicus_end_rule'] = saved
    rebuilt = flatten_spiral_checkpoint._build_model(
        checkpoint, config, Path(path), torch.device('cpu'))
    assert torch.equal(rebuilt.umbilicus_transform._yx, live.umbilicus_transform._yx)


@pytest.mark.parametrize('window, model_window, accepted', [
    (CROSSING, None, False), (INSIDE, None, True), (INSIDE, CROSSING, True)])
def test_load_refuses_only_a_rule_that_moves_the_axis(
        path, window, model_window, accepted):
    context = _preflight_context(path, window, HOLD_ENDS, model_window)
    domain = model_window or window
    checkpoint = _checkpoint(z_begin=domain[0], z_end=domain[1])
    before = copy.deepcopy(vars(context)['config'])
    verdict = fit_spiral.FitContext.inspect_checkpoint(context, checkpoint)
    assert verdict.accepted is accepted
    assert accepted or any('umbilicus end rule' in r for r in verdict.reasons)
    assert context.config == before and context.umbilicus_end_rule == HOLD_ENDS
    checkpoint['umbilicus_end_rule'] = HOLD_ENDS
    assert fit_spiral.FitContext.inspect_checkpoint(context, checkpoint).accepted


@pytest.mark.parametrize('window, rule, same', [
    (CROSSING, EXTRAPOLATE_ENDS, True), (CROSSING, HOLD_ENDS, False),
    (INSIDE, HOLD_ENDS, True)])
def test_digest_changes_only_when_the_axis_does(path, window, rule, same):
    context = _preflight_context(path, window, rule)
    digest = fit_spiral.FitContext.model_state_digest(context)
    unmarked = model_state_sha256(
        context.spiral_and_transform.state_dict(), (), *window)
    assert (digest == unmarked) is same
