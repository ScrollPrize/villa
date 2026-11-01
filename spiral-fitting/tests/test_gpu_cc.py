"""Tests for the CPU-backend speed options of extract_surface_tracks.py.

``--cc-device gpu`` must give the same connected-component labels as cc3d, and
``--cc-device gpu``, ``--slab-workers`` and ``--kimimaro-parallel`` must write
the same DBM as the defaults. The GPU tests run only when PyTorch sees a CUDA
or ROCm GPU.
"""

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

try:
    import torch
except ImportError:
    torch = None

needs_gpu = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason='needs a CUDA or ROCm GPU')


def _blobs(shape, density, seed):
    # Random foreground grown into sheets and blobs of many sizes, so dust removes components.
    rng = np.random.default_rng(seed)
    x = rng.random(shape) < density
    for ax in range(3):
        x = x | np.roll(x, 1, axis=ax)
    return x.astype(np.uint8)


@needs_gpu
@pytest.mark.parametrize('shape,density,seed', [
    ((4, 256, 256), 0.08, 0),     # horizontal ribbon
    ((96, 4, 300), 0.10, 1),      # vertical zx slab
    ((96, 300, 4), 0.12, 2),      # vertical zy slab
    ((33, 47, 61), 0.05, 3),      # shape not divisible by the downsample factor
    ((8, 64, 64), 0.0, 4),        # empty
])
def test_gpu_labels_equal_cc3d(shape, density, seed, monkeypatch):
    pred = _blobs(shape, density, seed)
    monkeypatch.setattr(est, 'cc_device', 'cpu')
    want = est.prepare_cc_labels(pred.copy())
    monkeypatch.setattr(est, 'cc_device', 'gpu')
    got = est.prepare_cc_labels(pred.copy())
    # cc3d picks uint16 or uint32 from the volume size; the GPU path always returns uint32.
    np.testing.assert_array_equal(got.astype(np.uint32), want.astype(np.uint32))


@needs_gpu
def test_gpu_max_area_equal_cc3d(monkeypatch):
    # Low enough that the max-area filter removes components of this volume.
    monkeypatch.setattr(est, 'max_area_threshold', 20_000)
    pred = _blobs((4, 256, 256), 0.08, 0)
    monkeypatch.setattr(est, 'cc_device', 'cpu')
    want = est.prepare_cc_labels(pred.copy())
    monkeypatch.setattr(est, 'cc_device', 'gpu')
    got = est.prepare_cc_labels(pred.copy())
    monkeypatch.setattr(est, 'max_area_threshold', 640_000)
    assert np.count_nonzero(want) < np.count_nonzero(est.prepare_cc_labels(pred.copy()))
    np.testing.assert_array_equal(got.astype(np.uint32), want.astype(np.uint32))


@pytest.mark.skipif(torch is None, reason='needs PyTorch')
def test_gpu_oom_falls_back_to_cc3d(monkeypatch):
    import gpu_cc

    def oom(*args):
        raise torch.cuda.OutOfMemoryError('test')

    pred = _blobs((8, 64, 64), 0.05, 5)
    monkeypatch.setattr(est, 'cc_device', 'cpu')
    want = est.prepare_cc_labels(pred.copy())
    monkeypatch.setattr(est, 'cc_device', 'gpu')
    monkeypatch.setattr(gpu_cc, 'prepare_cc_labels', oom)
    np.testing.assert_array_equal(est.prepare_cc_labels(pred.copy()), want)


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
    pytest.param(dict(cc_device='gpu'), marks=needs_gpu),
    pytest.param(dict(cc_device='gpu', slab_workers=3, kimimaro_parallel=1), marks=needs_gpu),
    # A pass with a single slab (one horizontal ribbon here).
    pytest.param(dict(cc_device='gpu', slab_workers=3, kimimaro_parallel=1, z_chunk_stride_h=64), marks=needs_gpu),
])
def test_dbm_equal_to_default(tmp_path, opts):
    volume = _blobs((64, 128, 128), 0.03, 7)
    layout = {k: v for k, v in opts.items() if k == 'z_chunk_stride_h'}
    want = _run(tmp_path, 'ref', volume, **layout)
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
