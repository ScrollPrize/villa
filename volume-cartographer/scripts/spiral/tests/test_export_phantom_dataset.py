"""Tests for export_phantom_dataset + fit_phantom_reference.

Reuses test_synth_phantom's tiny CLI-generated phantom. The load-bearing
invariants: exported patch vertices must lie exactly on true integer windings
(in the fit's shifted_radius/dr convention -- the annotation-convention bug
this would have caught injected a mean +0.5-winding screw dislocation), and
the reference fitter must measurably improve on the unfitted model when
scored against phantom truth.
"""

import json
import os

import numpy as np
import pytest
import torch
from click.testing import CliRunner

import export_phantom_dataset
import fit_phantom_reference
from checkpoint_io import load_checkpoint_cpu
from synth_phantom import build_model_from_phantom_checkpoint
from tifxyz import load_tifxyz
from winding_error import evaluate, load_phantom, winding_and_theta
from test_synth_phantom import generate_phantom


@pytest.fixture(scope='module')
def phantom_dir(tmp_path_factory):
    return generate_phantom(str(tmp_path_factory.mktemp('phantom')))


@pytest.fixture(scope='module')
def dataset_dir(phantom_dir, tmp_path_factory):
    out = str(tmp_path_factory.mktemp('dataset'))
    result = CliRunner().invoke(export_phantom_dataset.main, [
        '--phantom', phantom_dir, '--out', out, '--step', '2', '--coverage', '0.7'])
    assert result.exit_code == 0, result.output
    return out


def test_umbilicus_json_schema(dataset_dir):
    data = json.load(open(os.path.join(dataset_dir, 'umbilicus.json')))
    points = data['control_points']
    assert len(points) > 0 and all(set(p) == {'z', 'y', 'x'} for p in points)


def test_patch_vertices_lie_on_true_windings(phantom_dir, dataset_dir):
    # The whole point of truth-derived patches: every vertex sits exactly on
    # its annotated winding, in the SAME convention the fit anchors
    # (shifted_radius == winding * dr). A convention mismatch (e.g. annotating
    # the radius parameter k + theta/2pi) shows up here as a 0..1 sawtooth.
    _, model, _ = load_phantom(phantom_dir, torch.device('cpu'))
    patches_dir = os.path.join(dataset_dir, 'verified_patches')
    for entry in sorted(os.listdir(patches_dir)):
        patch = load_tifxyz(os.path.join(patches_dir, entry))
        w_true, _ = winding_and_theta(model, patch.zyxs.reshape(-1, 3))
        annotation = patch.winding.reshape(-1).numpy()
        assert np.abs(w_true - annotation).max() < 0.02, entry
        # Annotations are constant per patch: the winding index.
        assert np.allclose(annotation, np.round(annotation[0]))


def test_position_noise_corrupts_vertices(phantom_dir, dataset_dir, tmp_path):
    noisy = str(tmp_path / 'noisy')
    result = CliRunner().invoke(export_phantom_dataset.main, [
        '--phantom', phantom_dir, '--out', noisy, '--step', '2',
        '--coverage', '0.7', '--position-noise', '2.0'])
    assert result.exit_code == 0, result.output
    entry = sorted(os.listdir(os.path.join(dataset_dir, 'verified_patches')))[0]
    clean_patch = load_tifxyz(os.path.join(dataset_dir, 'verified_patches', entry))
    noisy_patch = load_tifxyz(os.path.join(noisy, 'verified_patches', entry))
    displacement = (clean_patch.zyxs - noisy_patch.zyxs).norm(dim=-1)
    assert 1.0 < displacement.mean() < 6.0  # ~ chi(3) mean at sigma 2


def test_tifxyz_candidate_mode(phantom_dir, dataset_dir, tmp_path):
    # Exported patches ARE tifxyz surfaces on true windings, so scoring them
    # through the mesh mode must report near-zero on-surface error, zero
    # grid discontinuities, and near-zero annotation disagreement -- while a
    # noisy export must score measurably off-surface.
    from winding_error import evaluate_tifxyz, load_tifxyz_candidates
    _, model, _ = load_phantom(phantom_dir, torch.device('cpu'))
    clean = evaluate_tifxyz(
        model, load_tifxyz_candidates(os.path.join(dataset_dir, 'verified_patches')),
        cut_margin=0.1)
    assert clean['overall']['on_surface_mae'] < 0.01
    assert clean['overall']['grid_discontinuity_frac'] == 0.
    for p in clean['per_patch'].values():
        assert p['winding_agreement']['mae'] < 0.01

    noisy_dir = str(tmp_path / 'noisy_for_mesh')
    result = CliRunner().invoke(export_phantom_dataset.main, [
        '--phantom', phantom_dir, '--out', noisy_dir, '--step', '2',
        '--coverage', '0.7', '--position-noise', '3.0'])
    assert result.exit_code == 0, result.output
    noisy = evaluate_tifxyz(
        model, load_tifxyz_candidates(os.path.join(noisy_dir, 'verified_patches')),
        cut_margin=0.1)
    assert noisy['overall']['on_surface_mae'] > clean['overall']['on_surface_mae'] * 5


def test_reference_fit_improves_over_init(phantom_dir, dataset_dir, tmp_path):
    fitted_path = str(tmp_path / 'fitted.pt')
    result = CliRunner().invoke(fit_phantom_reference.main, [
        '--dataset', dataset_dir, '--out', fitted_path, '--steps', '200',
        '--reg-gap', '0.01', '--reg-flow', '0.1'])
    assert result.exit_code == 0, result.output

    _, phantom_model, mask = load_phantom(phantom_dir, torch.device('cpu'))
    checkpoint = load_checkpoint_cpu(fitted_path)
    fitted = build_model_from_phantom_checkpoint(checkpoint, torch.device('cpu'))
    # Init baseline: the same architecture, freshly constructed (identity
    # deformation, dr at its configured init) -- NOT zeroed parameters, which
    # would also zero the dr logit and produce a nonsense spiral.
    from synth_phantom import build_model
    init = build_model(checkpoint['cfg'], int(checkpoint['z_begin']),
                       int(checkpoint['z_end']), checkpoint['umbilicus_zyx'],
                       checkpoint['spiral_outward_sense'], torch.device('cpu'))

    def mae(model):
        return evaluate(phantom_model, mask, ('model', model), num_points=20_000,
                        cut_margin=0.1, seed=0, device=torch.device('cpu'))['overall']['mae']

    init_mae, fitted_mae = mae(init), mae(fitted)
    assert fitted_mae < init_mae * 0.7, (init_mae, fitted_mae)
