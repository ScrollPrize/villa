"""Acceptance exam for the finite_paint consume path in winding_error.

winding_error can score a candidate winding raster against an external
finite_paint-format .npz (turn_id ground truth), making the ruler
source-agnostic. finite_paint itself lives in Diego-dcv/vesuvius-topological-grid
and isn't a dependency here, so the fixture below MIMICS its documented on-disk
contract -- turn_id int16 [Z,Y,X], 0=air, winding t stored as t+1 -- rather
than being real finite_paint output. This exam pins the adapter against that
contract without needing the external repo in CI.

Verified against REAL finite_paint output (2026-08-10, repo main):

  python scripts/finite_paint.py volume --columns 12 --voxel-um 60 \
      --z-window 3 --fuse 3,5,80,140 --out small_fused.npz

emits exactly the assumed contract (turn_id int16 [50,206,380], 0=air, 1-based,
max 23); winding_error --truth-npz scored a perfect candidate at MAE 0.0000 /
switch 0% and localized a single mislabeled turn to 100% on that turn, 0%
elsewhere -- so the fixture's assumptions match the live painter.
"""

import os

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

import winding_error as we


def make_finite_paint_npz(path, z=8, size=128, pitch=12.0, thickness=2.0, num_turns=6):
    """Concentric integer-winding sheets in the documented finite_paint format.
    Returns the continuous true winding field (radius/pitch) for building
    candidates."""
    cy = cx = size / 2.0
    zz, yy, xx = np.mgrid[0:z, 0:size, 0:size]
    r = np.hypot(yy - cy, xx - cx).astype(np.float32)
    winding_cont = r / pitch
    nearest = np.round(winding_cont)
    on_sheet = ((np.abs(r - nearest * pitch) <= thickness / 2)
                & (nearest >= 1) & (nearest <= num_turns))
    turn_id = np.where(on_sheet, (nearest + 1), 0).astype(np.int16)  # t stored as t+1
    volume = np.where(on_sheet, 180, 30).astype(np.uint8)
    np.savez(path, volume=volume, turn_id=turn_id,
             gt_surface=(turn_id > 0).astype(np.uint8))
    return winding_cont, turn_id


@pytest.fixture(scope='module')
def phantom_npz(tmp_path_factory):
    d = tmp_path_factory.mktemp('fp')
    npz = str(d / 'twin_cell.npz')
    winding_cont, turn_id = make_finite_paint_npz(npz)
    return {'npz': npz, 'winding_cont': winding_cont, 'turn_id': turn_id, 'dir': str(d)}


def score(npz, candidate):
    truth_w, mask = we.load_finite_paint_truth(npz)
    return we.score_turn_id(truth_w, mask, candidate.astype(np.float32),
                            num_points=1_000_000, rng=np.random.default_rng(0))


def test_loads_turn_id_as_winding(phantom_npz):
    truth_w, mask = we.load_finite_paint_truth(phantom_npz['npz'])
    # winding truth == turn_id - 1 on material voxels; masked-out are NaN.
    assert mask.sum() > 0
    tid = phantom_npz['turn_id']
    np.testing.assert_array_equal(truth_w[mask], (tid[mask].astype(np.float32) - 1))
    assert np.isnan(truth_w[~mask]).all()


def test_exact_instance_candidate_scores_zero(phantom_npz):
    # Candidate = the true integer winding -> exact.
    result = score(phantom_npz['npz'], phantom_npz['turn_id'].astype(np.float32) - 1)
    assert result['overall']['mae'] < 1e-6
    assert result['overall']['switch_rate'] == 0.0


def test_continuous_candidate_recovers_instances(phantom_npz):
    # A continuous winding field (radius/pitch) rounds to the right turn on every
    # sheet voxel: switch 0, with a small sub-winding continuous residual.
    result = score(phantom_npz['npz'], phantom_npz['winding_cont'])
    assert result['overall']['switch_rate'] == 0.0
    assert result['overall']['mae'] < 0.15


def test_global_offset_is_gauge_aligned(phantom_npz):
    # A constant winding-number offset is gauge (numbering choice) -> absorbed.
    result = score(phantom_npz['npz'], phantom_npz['turn_id'].astype(np.float32) - 1 + 3.0)
    assert abs(result['gauge_offset'] - 3.0) < 1e-6
    assert result['overall']['mae'] < 1e-6
    assert result['raw_mae'] > 2.9  # ...but the unaligned figure still shows it


def test_wrong_turn_region_shows_as_switches(phantom_npz):
    # Corrupt a spatial block by +1 winding: those voxels are assigned the wrong
    # turn -> switch rate ~ corrupted fraction, gauge pinned by the majority.
    cand = phantom_npz['turn_id'].astype(np.float32) - 1
    corrupt = np.zeros(cand.shape, bool)
    corrupt[:, :, 3 * cand.shape[2] // 4:] += True  # minority (~a quarter)
    mask = phantom_npz['turn_id'] > 0
    cand[corrupt] += 1.0
    result = score(phantom_npz['npz'], cand)
    frac = (corrupt & mask).sum() / mask.sum()
    assert abs(result['gauge_offset']) < 0.5
    assert result['overall']['switch_rate'] == pytest.approx(frac, abs=0.05)


def test_cli_end_to_end_and_json(phantom_npz, tmp_path):
    cand_tif = str(tmp_path / 'cand.tif')
    tifffile.imwrite(cand_tif, (phantom_npz['turn_id'].astype(np.float32) - 1))
    out = str(tmp_path / 'score.json')
    r = CliRunner().invoke(we.main, ['--truth-npz', phantom_npz['npz'],
                                     '--candidate-winding-volume', cand_tif,
                                     '--output', out])
    assert r.exit_code == 0, r.output
    assert os.path.exists(out)


def test_non_finite_paint_npz_rejected(tmp_path):
    bad = str(tmp_path / 'bad.npz')
    np.savez(bad, volume=np.zeros((4, 4, 4), np.uint8))  # no turn_id
    with pytest.raises(Exception):
        we.load_finite_paint_truth(bad)


def test_shape_mismatch_rejected(phantom_npz, tmp_path):
    cand_tif = str(tmp_path / 'cand.tif')
    tifffile.imwrite(cand_tif, np.zeros((3, 3, 3), np.float32))
    r = CliRunner().invoke(we.main, ['--truth-npz', phantom_npz['npz'],
                                     '--candidate-winding-volume', cand_tif])
    assert r.exit_code != 0 and 'shape' in r.output.lower()
