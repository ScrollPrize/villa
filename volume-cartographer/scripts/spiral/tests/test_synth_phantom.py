"""Tests for the synth_phantom / winding_error pair.

A shared tiny phantom is generated once through the real CLI entry point (so
the file contract -- volume/winding/mask/checkpoint/meta -- is what's tested),
then the metric harness is exercised against candidates with known defects:
identical (zero error), constant offset (pure gauge, must align away), partial
sheet shift (must show up as switch rate, not gauge), and a perturbed
deformation (must score worse than identical).
"""

import json
import os

import numpy as np
import pytest
import tifffile
import torch
from click.testing import CliRunner

import synth_phantom
from winding_error import evaluate, load_phantom

SEED = 3
PHANTOM_ARGS = ['--seed', str(SEED), '--z-size', '16', '--yx-size', '192',
                '--dr-per-winding', '12', '--num-tears', '2']


def generate_phantom(out_dir):
    result = CliRunner().invoke(synth_phantom.main, ['--out', out_dir, *PHANTOM_ARGS])
    assert result.exit_code == 0, result.output
    return out_dir


@pytest.fixture(scope='module')
def phantom_dir(tmp_path_factory):
    return generate_phantom(str(tmp_path_factory.mktemp('phantom')))


@pytest.fixture(scope='module')
def phantom(phantom_dir):
    checkpoint, model, mask = load_phantom(phantom_dir, torch.device('cpu'))
    return {'dir': phantom_dir, 'checkpoint': checkpoint, 'model': model, 'mask': mask}


def run_eval(phantom, candidate, num_points=20_000):
    return evaluate(phantom['model'], phantom['mask'], candidate,
                    num_points=num_points, cut_margin=0.1, seed=0,
                    device=torch.device('cpu'))


def test_outputs_exist_and_consistent(phantom_dir):
    volume = tifffile.imread(os.path.join(phantom_dir, 'volume.tif'))
    winding = tifffile.imread(os.path.join(phantom_dir, 'winding.tif'))
    mask = tifffile.imread(os.path.join(phantom_dir, 'mask.tif'))
    assert volume.shape == winding.shape == mask.shape == (16, 192, 192)
    assert volume.dtype == np.uint8 and winding.dtype == np.float32
    meta = json.load(open(os.path.join(phantom_dir, 'meta.json')))
    assert meta['valid_voxels'] == int((mask > 0).sum())
    # Winding truth must respect the annulus the mask claims.
    w_valid = winding[mask > 0]
    assert w_valid.min() >= meta['first_winding']
    assert w_valid.max() <= meta['last_winding']


def test_tears_land_on_papyrus(phantom_dir):
    # Regression: volume-uniform tear boxes usually missed the thin annulus,
    # silently producing phantoms with zero torn voxels.
    meta = json.load(open(os.path.join(phantom_dir, 'meta.json')))
    assert meta['torn_voxels'] > 0
    mask = tifffile.imread(os.path.join(phantom_dir, 'mask.tif'))
    assert (mask == 2).sum() == meta['torn_voxels']


def test_roundtrip_invertibility(phantom_dir):
    meta = json.load(open(os.path.join(phantom_dir, 'meta.json')))
    assert meta['roundtrip_p95_vox'] < 0.01


def test_determinism(phantom_dir, tmp_path):
    other = generate_phantom(str(tmp_path / 'phantom_again'))
    for name in ('volume.tif', 'winding.tif', 'mask.tif'):
        a = tifffile.imread(os.path.join(phantom_dir, name))
        b = tifffile.imread(os.path.join(other, name))
        np.testing.assert_array_equal(a, b)


def test_identical_candidate_scores_zero(phantom):
    result = run_eval(phantom, ('model', phantom['model']))
    assert result['overall']['mae'] < 1e-6
    assert result['overall']['switch_rate'] == 0.
    assert abs(result['gauge_offset']) < 1e-6


def test_winding_volume_candidate_scores_zero(phantom):
    winding = tifffile.imread(os.path.join(phantom['dir'], 'winding.tif'))
    result = run_eval(phantom, ('volume', winding))
    assert result['overall']['mae'] < 1e-6


def test_constant_offset_is_gauge_aligned_away(phantom):
    # A constant winding offset is pure gauge (theta-origin choice) and must
    # be absorbed by alignment, not reported as error.
    winding = tifffile.imread(os.path.join(phantom['dir'], 'winding.tif')) + 0.37
    result = run_eval(phantom, ('volume', winding))
    assert abs(result['gauge_offset'] - 0.37) < 0.02
    assert result['overall']['mae'] < 0.02
    assert result['raw_mae'] > 0.3  # ...but the unaligned figure still shows it


def test_partial_sheet_shift_shows_as_switches(phantom):
    # Shifting a clear minority of the field by one winding is a genuine
    # sheet-switch defect: median alignment must pin the gauge to the
    # unshifted majority, and the shifted minority must land in the
    # switch-rate figure -- roughly its volume fraction.
    winding = tifffile.imread(os.path.join(phantom['dir'], 'winding.tif')).copy()
    shifted_region = np.zeros_like(winding, dtype=bool)
    shifted_region[3 * winding.shape[0] // 4 :] = True  # top z-quarter
    winding[shifted_region] += 1.
    result = run_eval(phantom, ('volume', winding))
    assert abs(result['gauge_offset']) < 0.05
    assert 0.1 < result['overall']['switch_rate'] < 0.45
    mask = phantom['mask']
    shifted_frac = (shifted_region & (mask > 0)).sum() / (mask > 0).sum()
    assert result['overall']['switch_rate'] == pytest.approx(shifted_frac, abs=0.05)


def test_perturbed_deformation_scores_worse(phantom):
    from synth_phantom import build_model_from_phantom_checkpoint
    perturbed = build_model_from_phantom_checkpoint(
        phantom['checkpoint'], torch.device('cpu'))
    with torch.no_grad():
        hr = perturbed.flow_field.flows[1]
        hr.add_(torch.from_numpy(
            np.random.default_rng(1).standard_normal(tuple(hr.shape))
            .astype(np.float32)) * 0.005)
    identical = run_eval(phantom, ('model', phantom['model']))
    worse = run_eval(phantom, ('model', perturbed))
    assert worse['overall']['mae'] > max(identical['overall']['mae'] * 10, 1e-4)


def test_preset_texture_preserves_truth(tmp_path):
    # The pherc0009b preset changes RENDERING (fiber texture, speckles, murky
    # gaps) and deformation magnitudes -- never the truth convention. The
    # identical candidate must still score exactly zero, and the volume must
    # actually carry the textured intensity statistics.
    out = str(tmp_path / 'preset')
    result = CliRunner().invoke(synth_phantom.main, [
        '--out', out, '--preset', 'pherc0009b', '--seed', str(SEED),
        '--z-size', '16', '--yx-size', '448', '--num-tears', '0'])
    assert result.exit_code == 0, result.output
    checkpoint, model, mask = load_phantom(out, torch.device('cpu'))
    scored = evaluate(model, mask, ('model', model), num_points=10_000,
                      cut_margin=0.1, seed=0, device=torch.device('cpu'))
    assert scored['overall']['mae'] < 1e-6
    meta = json.load(open(os.path.join(out, 'meta.json')))
    assert meta['preset'] == 'pherc0009b'
    assert meta['fiber_amp'] > 0 and meta['gap_level'] == pytest.approx(0.29)
    volume = tifffile.imread(os.path.join(out, 'volume.tif'))
    sheets = volume[mask > 0]
    # Textured sheets must vary well beyond the additive noise floor.
    assert sheets.std() > 20  # uint8 units; noise_std alone would be ~9


def test_shape_mismatch_rejected(phantom):
    with pytest.raises(Exception):
        bad = np.zeros((4, 4, 4), dtype=np.float32)
        # evaluate indexes the volume at phantom voxel coordinates; a wrong
        # grid must fail loudly, not silently score garbage.
        run_eval(phantom, ('volume', bad))
