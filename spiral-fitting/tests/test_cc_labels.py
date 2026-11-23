"""Parity tests for vc_reader components and CPU slab workers."""
from __future__ import annotations

import dbm
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cc3d = pytest.importorskip('cc3d')

import extract_surface_tracks as est  # noqa: E402


def _blobs(shape, density, seed):
    # Random foreground grown into sheets and blobs of many sizes, so dust removes components.
    rng = np.random.default_rng(seed)
    x = rng.random(shape) < density
    for ax in range(3):
        x = x | np.roll(x, 1, axis=ax)
    return x.astype(np.uint8)


@pytest.mark.parametrize('shape,density,seed', [
    ((4, 256, 256), 0.08, 0),     # horizontal ribbon
    ((96, 4, 300), 0.10, 1),      # vertical zx slab
    ((96, 300, 4), 0.12, 2),      # vertical zy slab
    ((33, 47, 61), 0.05, 3),      # shape not divisible by the downsample factor
    ((8, 64, 64), 0.0, 4),        # empty
])
def test_vc_reader_labels_equal_cc3d(shape, density, seed):
    pred = _blobs(shape, density, seed)
    want = _reference_labels(pred.copy())
    got = est.prepare_cc_labels(pred.copy())
    np.testing.assert_array_equal(got, want)


def _reference_labels(pred):
    labels = cc3d.connected_components(pred, connectivity=6)
    labels = est.downsample_maxpool(labels, est.downsample_factor)
    scale = est.downsample_factor ** 3
    cc3d.dust(labels, threshold=[max(1, est.dust_threshold // scale),
                               max(1, est.max_area_threshold // scale)],
              in_place=True, precomputed_ccl=True)
    return labels


def test_max_area_equal_cc3d(monkeypatch):
    monkeypatch.setattr(est, 'max_area_threshold', 20_000)
    pred = _blobs((4, 256, 256), 0.08, 0)
    want = _reference_labels(pred.copy())
    np.testing.assert_array_equal(est.prepare_cc_labels(pred.copy()), want)
    monkeypatch.setattr(est, 'max_area_threshold', 640_000)
    assert np.count_nonzero(want) < np.count_nonzero(est.prepare_cc_labels(pred.copy()))


def _run(out_dir, name, volume, **opts):
    pytest.importorskip('kimimaro')
    out = str(Path(out_dir) / f'{name}.dbm')
    saved = {k: getattr(est, k) for k in ('z_min', 'z_max', *opts)}
    try:
        est.z_min, est.z_max = 0, volume.shape[0]
        for k, v in opts.items():
            setattr(est, k, v)
        with dbm.open(out, 'c') as db:
            est.extract_horizontal(volume, db)
            lo, hi = est.find_yx_range(volume)
            est.extract_vertical(volume, db, 'y', lo, hi)
            est.extract_vertical(volume, db, 'x', lo, hi)
        with dbm.open(out, 'r') as db:
            return {k: [t.tolist() for t in pickle.loads(db[k])] for k in db.keys()}
    finally:
        for k, v in saved.items():
            setattr(est, k, v)


def _canonical(tracks):
    # Kimimaro's own parallelism can return skeletons in another order; the set of tracks per key is what counts.
    return {k: sorted(v) for k, v in tracks.items()}


@pytest.mark.parametrize('opts', [
    dict(kimimaro_parallel=1),
    dict(slab_workers=3, kimimaro_parallel=1),
])
def test_dbm_equal_to_default(tmp_path, opts, monkeypatch):
    volume = _blobs((64, 128, 128), 0.03, 7)
    with monkeypatch.context() as reference:
        reference.setattr(est, 'prepare_cc_labels', _reference_labels)
        want = _run(tmp_path, 'ref', volume)
    got = _run(tmp_path, 'new', volume, **opts)
    assert want.keys() == got.keys()
    assert _canonical(got) == _canonical(want)


def test_slab_workers_on_zarr(tmp_path):
    # Workers forked after the parent has read the zarr array (find_yx_range) must still read it.
    zarr = pytest.importorskip('zarr')
    volume = _blobs((64, 128, 128), 0.03, 7)
    arr = zarr.open_array(str(tmp_path / 'pred.zarr'), mode='w', shape=volume.shape, chunks=(32, 64, 64), dtype='u1')
    arr[:] = volume
    want = _run(tmp_path, 'ref', volume)
    got = _run(tmp_path, 'new', zarr.open_array(str(tmp_path / 'pred.zarr'), mode='r'),
               slab_workers=3, kimimaro_parallel=1)
    assert _canonical(got) == _canonical(want)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_real_scroll_labels_equal_cc3d(axis):
    fixture = Path(__file__).resolve().parents[2] / 'vesuvius/tests/ink_detection/fixtures/ink_roi_case.npz'
    if not fixture.exists():
        pytest.skip('monorepo real-scroll fixture required')
    with np.load(fixture) as data:
        # Exercise slab labeling on foreground derived from the real CT crop.
        volume = (data['ct'] > 128).astype(np.uint8)
    index = [slice(None)] * 3
    index[axis] = slice(4, 8)
    slab = np.ascontiguousarray(volume[tuple(index)])
    want = _reference_labels(slab)
    assert np.count_nonzero(want)
    np.testing.assert_array_equal(est.prepare_cc_labels(slab), want)
