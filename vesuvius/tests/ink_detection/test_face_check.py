"""Face check: one-sided agreement scores two-model ink maps as writing-like, two-sided as texture-like."""

from __future__ import annotations

import json

import numpy as np
import tifffile

from vesuvius.ink_detection.face_check import face_scores, main


def _maps(rng, one_sided: bool, size: int = 1536):
    shared = rng.random((size, size)) > 0.97          # where both models fire
    noise = lambda: rng.random((size, size)).astype(np.float32) * 0.5
    a_f, b_f = noise() + shared, noise() + shared
    if one_sided:
        a_r, b_r = noise(), noise()                   # other face: independent noise only
    else:
        a_r, b_r = noise() + shared, noise() + shared # same agreement on both faces
    return {'a': a_f, 'b': b_f}, {'a': a_r, 'b': b_r}


def test_one_sided_agreement_is_asymmetric_and_consistent():
    fwd, rev = _maps(np.random.default_rng(0), one_sided=True)
    r = face_scores(fwd, rev, np.ones(fwd['a'].shape, bool))['a/b']
    assert r['best_face'] == 'forward'
    assert r['face_asymmetry'] > 10
    assert r['consistent_face'] == 'forward'
    assert r['face_consistency'] == 1.0


def test_two_sided_agreement_is_flat_and_a_tie():
    fwd, rev = _maps(np.random.default_rng(1), one_sided=False)
    r = face_scores(fwd, rev, np.ones(fwd['a'].shape, bool))['a/b']
    assert 0.8 < r['face_asymmetry'] < 1.25
    assert r['consistent_face'] == 'tie'
    assert r['face_consistency'] == 0.0


def test_cli_finds_reverse_siblings(tmp_path):
    fwd, rev = _maps(np.random.default_rng(2), one_sided=True)
    for k in fwd:
        tifffile.imwrite(tmp_path / f'{k}.tif', fwd[k])
        tifffile.imwrite(tmp_path / f'{k}_reverse.tif', rev[k])
    out = tmp_path / 'fc.json'
    assert main([str(tmp_path / 'a.tif'), str(tmp_path / 'b.tif'), '--out', str(out)]) == 0
    res = json.loads(out.read_text())
    assert res['pairs']['a/b']['best_face'] == 'forward'
