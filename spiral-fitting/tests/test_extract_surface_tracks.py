"""Tests for extract_surface_tracks.py, fast_tracks.py and the GPU backend.

The fast_tracks tests compare every engine with the original networkx walk,
``extract_surface_tracks.extract_inter_branch_paths``, and with the original
per-slab track loop. The GPU end-to-end test runs only when Brook, CuPy and
cuCIM are installed and the test GPU (the first CUDA_VISIBLE_DEVICES entry when
that is set, else the first GPU) has compute capability 8.0 or newer; it
compares the GPU backend with the per-slab loop run with ``brook.skeletonize``.
"""

from __future__ import annotations

import contextlib
import dbm
import functools
import importlib.util
import os
import pickle
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extract_surface_tracks as est  # noqa: E402


ENGINES = [
    pytest.param('numba', marks=pytest.mark.skipif(
        importlib.util.find_spec('numba') is None, reason='numba is not installed')),
    'numpy',
    'python',
]


# Edge lists covering the graph shapes the walk must handle like networkx.
FIXED = {
    'empty': [],
    'self-loop only': [(0, 0)],
    'duplicate self-loop': [(0, 0), (0, 0)],
    'isolated edge': [(0, 1)],
    'isolated edge, reversed duplicate': [(0, 1), (1, 0)],
    'isolated edge, reversed first': [(1, 0)],
    'triangle (pure cycle)': [(0, 1), (1, 2), (2, 0)],
    'triangle + self-loop': [(0, 1), (1, 2), (2, 0), (0, 0)],
    'self-loop first, triangle': [(0, 0), (0, 1), (1, 2), (2, 0)],
    'chain ending in a self-loop': [(0, 1), (1, 2), (2, 2)],
    'chain with self-loop inside': [(0, 1), (1, 2), (2, 3), (1, 1)],
    'star': [(0, i) for i in range(1, 6)],
    'star, leaves first': [(i, 0) for i in range(1, 6)],
    'figure eight': [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
    'theta': [(0, 1), (1, 2), (2, 9), (0, 3), (3, 9), (0, 4), (4, 5), (5, 6), (6, 9)],
    'lollipop': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (5, 0)],
    'two cycles and a path': [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8)],
    'far end first in node order': [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0), (1, 6), (1, 7)],
    'duplicate middle edge': [(0, 1), (1, 2), (2, 1), (2, 3), (1, 2)],
    'long path': [(i, i + 1) for i in range(30)],
    'long cycle through hub': (
        [(0, i) for i in (1, 50, 60)] + [(i, i + 1) for i in range(1, 25)] + [(25, 0)]),
}


def gen_graph(rng):
    """A random (E, 2) edge array: a disjoint union of random shapes, then perturbed.

    Shapes: paths, cycles, stars, spiders, trees (with extra edges), lollipops,
    figure eights, theta graphs, grids and G(n, p). Perturbations: self-loops,
    isolated edges, duplicates in either orientation, shuffled order, flipped
    edges, sparse or negative ids and several integer dtypes.
    """
    comps = []
    for _ in range(rng.integers(1, 5)):
        kind = rng.choice(['path', 'cycle', 'star', 'spider', 'tree', 'tree+', 'lollipop',
                           'eight', 'theta', 'grid', 'gnp', 'edge'])
        n = int(rng.integers(1, 40))
        if kind == 'path':
            e = [(i, i + 1) for i in range(n)]
        elif kind == 'cycle':
            n = max(n, 3)
            e = [(i, (i + 1) % n) for i in range(n)]
        elif kind == 'star':
            e = [(0, i) for i in range(1, n + 1)]
        elif kind == 'spider':
            e, nxt = [], 1
            for _ in range(int(rng.integers(1, 6))):
                prev = 0
                for _ in range(int(rng.integers(1, 15))):
                    e.append((prev, nxt))
                    prev = nxt
                    nxt += 1
        elif kind in ('tree', 'tree+'):
            e = [(int(rng.integers(0, i)), i) for i in range(1, n + 1)]
            if kind == 'tree+':
                for _ in range(int(rng.integers(1, 5))):
                    a, b = rng.integers(0, n + 1, 2)
                    e.append((int(a), int(b)))
        elif kind == 'lollipop':
            k = max(3, n // 2)
            e = ([(i, (i + 1) % k) for i in range(k)] + [(0, k)]
                 + [(k + i, k + i + 1) for i in range(n)])
        elif kind == 'eight':
            k = max(3, n // 2)
            e = ([(i, (i + 1) % k) for i in range(k)] + [(0, k)]
                 + [(k + i, k + i + 1) for i in range(k)] + [(2 * k, 0)])
        elif kind == 'theta':
            e, nxt = [], 2
            for _ in range(int(rng.integers(2, 5))):
                prev = 0
                for _ in range(int(rng.integers(0, 12))):
                    e.append((prev, nxt))
                    prev = nxt
                    nxt += 1
                e.append((prev, 1))
        elif kind == 'grid':
            w, h = int(rng.integers(1, 7)), int(rng.integers(1, 7))
            e = ([(y * w + x, y * w + x + 1) for y in range(h) for x in range(w - 1)]
                 + [(y * w + x, (y + 1) * w + x) for y in range(h - 1) for x in range(w)])
        elif kind == 'gnp':
            p = float(rng.uniform(0.02, 0.2))
            e = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
        else:
            e = [(0, 1)]
        comps.append(e)
    # Disjoint union with random relabelling, then perturbations.
    edges, base = [], 0
    for e in comps:
        if not e:
            continue
        m = max(max(a, b) for a, b in e) + 1
        perm = rng.permutation(m) + base
        edges += [(int(perm[a]), int(perm[b])) for a, b in e]
        base += m
    nodes = sorted({x for ed in edges for x in ed}) or [0]
    for _ in range(int(rng.poisson(0.7))):  # self-loops
        x = int(rng.choice(nodes)) if rng.random() < 0.8 else base + int(rng.integers(0, 3))
        edges.append((x, x))
    for _ in range(int(rng.poisson(0.7))):  # isolated edges
        edges.append((base + 10, base + 11))
        base += 12
    if edges:
        for _ in range(int(rng.poisson(1.0))):  # duplicates, either orientation
            a, b = edges[int(rng.integers(0, len(edges)))]
            position = int(rng.integers(0, len(edges) + 1))
            edges.insert(position, (a, b) if rng.random() < 0.5 else (b, a))
    order = rng.permutation(len(edges)) if rng.random() < 0.5 else np.arange(len(edges))
    edges = [edges[i] if rng.random() < 0.7 else edges[i][::-1] for i in order]
    r = rng.random()
    if r < 0.1:  # sparse ids
        edges = [(a * 1_000_003 + 7, b * 1_000_003 + 7) for a, b in edges]
    elif r < 0.2:  # negative ids
        edges = [(a - 20, b - 20) for a, b in edges]
    if r < 0.2:
        dtype = np.int64
    else:  # the skeleton edge dtype, and dtypes that are converted first
        dtypes = [np.uint32, np.uint32, np.int32, np.int64, np.uint64, np.uint16]
        dtype = dtypes[int(rng.integers(0, 6))]
    return np.asarray(edges, dtype=dtype).reshape(-1, 2)


def _networkx_paths(edges):
    import networkx as nx
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return [[int(n) for n in path] for path in est.extract_inter_branch_paths(graph)]


class _Skeleton:
    """Stands in for ``osteoid.Skeleton``: float32 vertices and uint32 edges."""

    def __init__(self, vertices, edges):
        self.vertices = vertices.astype(np.float32)
        self.edges = edges.astype(np.uint32)


def _random_slab(rng):
    """Several random skeletons, some with chains long enough to become tracks."""
    skeletons = {}
    for label in range(1, int(rng.integers(1, 6)) + 1):
        while True:
            edges = gen_graph(rng)
            if edges.dtype == np.uint32:  # dense ids, as in skeleton edges
                break
        if len(edges) and rng.random() < 0.5:
            m = int(edges.max())
            chain = [(m + 1 + i, m + 2 + i) for i in range(12)]
            edges = np.concatenate([edges, np.asarray(chain, np.uint32)])
        n_vertices = (int(edges.max()) + 1 if len(edges) else 0) + int(rng.integers(0, 3))
        vertices = rng.integers(0, 4000, (n_vertices, 3)) + rng.random((n_vertices, 3))
        skeletons[label] = _Skeleton(vertices, edges)
    return skeletons


def _original_tracks(skeletons, offset_zyx):
    """The original per-slab loop (``get_skeleton_tracks``) on the given skeletons."""
    return est.get_skeleton_tracks(
        None, offset_zyx, skeletonize=lambda labels, **kwargs: skeletons)


@pytest.mark.parametrize('engine', ENGINES)
def test_fast_paths_match_networkx(engine):
    import fast_tracks as ft
    rng = np.random.default_rng(12345)
    cases = [np.asarray(e, np.uint32).reshape(-1, 2) for e in FIXED.values()]
    cases += [gen_graph(rng) for _ in range(300)]
    for edges in cases:
        paths = ft.extract_inter_branch_paths_fast(edges, engine=engine)
        assert [path.tolist() for path in paths] == _networkx_paths(edges), edges.tolist()


@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('engine', ENGINES)
def test_fast_tracks_pickle_identical(engine, copy, monkeypatch):
    import fast_tracks as ft
    monkeypatch.setattr(est, 'path_mode', 'interjoint')
    rng = np.random.default_rng(7)
    for _ in range(40):
        skeletons = _random_slab(rng)
        offset = tuple(int(x) for x in rng.integers(0, 20000, 3))
        expected = pickle.dumps(_original_tracks(skeletons, offset))

        tracks = ft.tracks_from_skeletons(skeletons.values(), offset, engine=engine, copy=copy)
        assert pickle.dumps(tracks) == expected

        v_off = np.concatenate([[0], np.cumsum([len(s.vertices) for s in skeletons.values()])])
        e_off = np.concatenate([[0], np.cumsum([len(s.edges) for s in skeletons.values()])])
        vertices = np.concatenate([s.vertices for s in skeletons.values()])
        edges = np.concatenate([s.edges.reshape(-1, 2) for s in skeletons.values()])
        tracks = ft.tracks_from_packed(
            v_off, e_off, vertices, edges, offset, engine=engine, copy=copy)
        assert pickle.dumps(tracks) == expected


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('bad', [np.array([[0, 5]], np.uint32), np.array([[-1, 0]], np.int64)])
def test_fast_tracks_reject_invalid_edges(engine, bad):
    import fast_tracks as ft
    with pytest.raises(ValueError):
        ft.tracks_from_packed(
            [0, 3], [0, 1], np.zeros((3, 3), np.float32), bad, (0, 0, 0), engine=engine)


# ------------------------------------------------------------------------------------------
# Backend selection and command line

def _fake_gpu_module(ok, reason, calls=None):
    module = types.ModuleType('surface_tracks_gpu')

    def check_available(gpus=None):
        if calls is not None:
            calls.append(gpus)
        return ok, reason
    module.check_available = check_available
    return module


def test_auto_selects_gpu_when_available(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, 'surface_tracks_gpu', _fake_gpu_module(True, '1 GPU', calls))
    assert est.select_backend('auto', gpus=(1,)) == 'gpu'
    assert est.select_backend('gpu') == 'gpu'
    assert calls == [(1,), None]


def test_auto_falls_back_to_cpu_without_brook(monkeypatch, capsys):
    import surface_tracks_gpu  # noqa: F401  (the real availability check)
    monkeypatch.setitem(sys.modules, 'brook', None)  # makes brook unimportable
    monkeypatch.setattr(est, '_have_kimimaro', lambda: True)
    assert est.select_backend('auto') == 'cpu'
    assert 'cpu (Kimimaro)' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='--backend gpu'):
        est.select_backend('gpu')


def test_auto_uses_cpu_for_maximal_chain(monkeypatch):
    monkeypatch.setitem(sys.modules, 'surface_tracks_gpu', _fake_gpu_module(True, '1 GPU'))
    monkeypatch.setattr(est, 'path_mode', 'maximal_chain')
    monkeypatch.setattr(est, '_have_kimimaro', lambda: True)
    assert est.select_backend('auto') == 'cpu'
    with pytest.raises(SystemExit, match='interjoint'):
        est.select_backend('gpu')


def test_gpu_requested_but_unavailable(monkeypatch):
    reason = 'no NVIDIA GPU of compute capability 8.0 or newer is visible'
    monkeypatch.setitem(sys.modules, 'surface_tracks_gpu', _fake_gpu_module(False, reason))
    monkeypatch.setattr(est, '_have_kimimaro', lambda: True)
    with pytest.raises(SystemExit, match=reason):
        est.select_backend('gpu')
    # --gpus asks for the GPU backend: auto does not fall back to the CPU
    with pytest.raises(SystemExit, match='--gpus was given'):
        est.select_backend('auto', gpus=(3,))
    assert est.select_backend('auto') == 'cpu'


def test_cpu_backend_without_kimimaro(monkeypatch):
    monkeypatch.setitem(sys.modules, 'kimimaro', None)  # makes kimimaro unimportable
    with pytest.raises(SystemExit, match='uv sync --extra cpu'):
        est.select_backend('cpu')
    # auto, when the GPU backend cannot run either: both reasons are given
    monkeypatch.setitem(
        sys.modules, 'surface_tracks_gpu', _fake_gpu_module(False, 'nvidia-smi was not found'))
    with pytest.raises(SystemExit, match='nvidia-smi was not found') as excinfo:
        est.select_backend('auto')
    assert 'uv sync --extra cpu' in str(excinfo.value)


def test_unknown_backend_is_rejected():
    with pytest.raises(SystemExit, match='unknown backend'):
        est.select_backend('tpu')


def test_placeholder_paths_are_rejected(monkeypatch, capsys):
    monkeypatch.setattr(est, 'predictions_zarr_path', 's3://<bucket>/<volpkg>/predictions.zarr/0')
    monkeypatch.setattr(est, 'tracks_dbm_path', '/path/to/<dataset>/tracks/<name>.dbm')
    with pytest.raises(SystemExit) as excinfo:
        est.main([])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--predictions 's3://<bucket>" in err and 'still contains a placeholder' in err
    with pytest.raises(SystemExit):
        est.main(['--predictions', 'path/to/predictions.zarr/0'])
    err = capsys.readouterr().err
    assert "--out '/path/to/<dataset>" in err and 'still contains a placeholder' in err


def test_invalid_z_range_is_rejected(capsys, tmp_path):
    with pytest.raises(SystemExit):
        est.main(['--predictions', 'p.zarr', '--out', str(tmp_path / 't.dbm'),
                  '--z-min', '200', '--z-max', '100'])
    assert '--z-min must be' in capsys.readouterr().err


def test_gpus_with_cpu_backend_is_rejected(capsys):
    with pytest.raises(SystemExit):
        est.parse_args(['--predictions', 'p.zarr', '--out', 't.dbm', '--backend', 'cpu',
                        '--gpus', '0'])
    assert '--gpus applies to the GPU backend only' in capsys.readouterr().err


def test_import_and_parse_args_stay_lazy():
    # A GPU-only install has no Kimimaro: importing the script and parsing its arguments must
    # not import the CPU backend's packages (or vesuvius and PyTorch). A fresh interpreter,
    # because other tests import them into this one.
    code = ("import sys, extract_surface_tracks as e; "
            "e.parse_args(['--predictions', 'p', '--out', 'o']); "
            "bad = {'kimimaro', 'cc3d', 'networkx', 'tqdm', 'vesuvius', 'torch'} & set(sys.modules); "
            "assert not bad, bad")
    subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                   check=True, timeout=120)


def test_command_line_reaches_the_backend(monkeypatch, tmp_path):
    runs = []
    monkeypatch.setattr(est, 'z_min', est.z_min)  # main() sets the z range; restore it after
    monkeypatch.setattr(est, 'z_max', est.z_max)
    monkeypatch.setattr(est, 'write_native_packed_store', True)
    monkeypatch.setattr(est, 'select_backend', lambda requested, gpus=None: requested)
    monkeypatch.setattr(
        est, 'run_cpu', lambda *args: runs.append(('cpu', args, est.z_min, est.z_max)))
    monkeypatch.setattr(
        est, 'run_gpu', lambda *args, gpus=None: runs.append(('gpu', args, gpus)))
    monkeypatch.setattr(est, 'write_packed_store', lambda path: runs.append(('packed', path)))
    out = tmp_path / 'tracks' / 't.dbm'
    est.main(['--predictions', 'p.zarr', '--out', str(out), '--z-min', '64', '--z-max', '128',
              '--backend', 'cpu', '--no-packed-store'])
    assert runs == [('cpu', ('p.zarr', str(out)), 64, 128)]
    assert out.parent.is_dir()
    runs.clear()
    est.main(['--predictions', 'p.zarr', '--out', str(out), '--backend', 'gpu', '--gpus', '2,0'])
    assert runs == [('gpu', ('p.zarr', str(out)), (2, 0)), ('packed', str(out))]


# ------------------------------------------------------------------------------------------
# Synthetic predictions

def synthetic_predictions(shape=(176, 232, 244), seed=0):
    """A uint8 stand-in for surface predictions.

    Open, wavy, slightly tilted cylindrical sheets (arcs in yx) joined by two
    radial walls, which give branch points; a gently tilted sheet; specks below
    the dust threshold; and a block above the maximum component size. Sheet
    voxels take random values in 1..255, since any value > 0 is a surface.
    """
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    z, y, x = np.ogrid[:Z, :Y, :X]
    cy, cx = 0.45 * Y, 0.5 * X
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    theta = np.arctan2(y - cy, x - cx)
    half = 1.6
    sheets = np.zeros(shape, bool)
    for k, radius in enumerate(range(22, 104, 14)):
        rz = radius + 0.05 * (k % 3 - 1) * (z - Z / 2) + 2.5 * np.sin(3 * theta + z / 18 + k)
        gap = np.abs(np.angle(np.exp(1j * (theta - 1.1 * k)))) < 0.5
        sheets |= (np.abs(r - rz) < half) & ~gap & (z >= 6) & (z < Z - 12)
    sheets |= ((np.abs(np.angle(np.exp(1j * (theta - 2.3)))) * r < half)
               & (r > 20) & (r < 52) & (z < 120))
    sheets |= ((np.abs(np.angle(np.exp(1j * (theta + 0.8)))) * r < half)
               & (r > 60) & (r < 96) & (z >= 40))
    sheets |= (np.abs(y - (200 + 0.1 * x + 0.1 * z)) < half) & (x > 10) & (x < 140) & (z >= 10)
    volume = np.zeros(shape, np.uint8)
    volume[sheets] = rng.integers(1, 256, int(sheets.sum()), dtype=np.uint8)
    for _ in range(60):
        a, b, c = rng.integers(0, Z - 3), rng.integers(0, Y - 3), rng.integers(0, X - 3)
        volume[a:a + 2, b:b + 2, c:c + 2] = 200
    volume[30:150, 196:230, 150:240] = 255
    return volume


def _small_configuration(monkeypatch):
    """A layout for the small synthetic volume: more slabs, and size limits it reaches."""
    for name, value in dict(z_min=8, z_max=168, z_chunk_stride_h=8, yx_stride_v=24,
                            dust_threshold=128, max_area_threshold=24_000,
                            path_mode='interjoint').items():
        monkeypatch.setattr(est, name, value)


def _read_dbm(path):
    with dbm.open(str(path), 'r') as database:
        return {key.decode(): bytes(database[key]) for key in database.keys()}


def _write_order(path):
    """Keys in the order they were written, or None for a DBM backend other than sqlite3.

    ``keys()`` of the sqlite3 backend (the default since Python 3.13) returns
    sorted keys, so the write order is read from the table's rowids.
    """
    if dbm.whichdb(str(path)) != 'dbm.sqlite3':
        return None
    with contextlib.closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as connection:
        return [bytes(key).decode()
                for (key,) in connection.execute('SELECT key FROM Dict ORDER BY rowid')]


def test_cpu_backend_writes_the_track_format(monkeypatch, tmp_path):
    kimimaro = pytest.importorskip('kimimaro')
    pytest.importorskip('cc3d')
    _small_configuration(monkeypatch)
    volume = synthetic_predictions()
    monkeypatch.setattr(est, 'open_predictions', lambda path: volume)

    # Kimimaro's worker processes would dominate the run time of this small volume.
    def skeletonize(labels, **kwargs):
        return kimimaro.skeletonize(labels, **{**kwargs, 'parallel': 1})
    monkeypatch.setattr(est, 'get_skeleton_tracks',
                        functools.partial(est.get_skeleton_tracks, skeletonize=skeletonize))

    out = tmp_path / 'tracks.dbm'
    est.run_cpu('synthetic', str(out))
    entries = _read_dbm(out)
    # h keys by z, then vy keys by y, then vx keys by x, over the yx range occupied in
    # the 20 sampled z slices (every 8th here)
    ys, xs = np.nonzero((volume[8:168:8] > 0).any(axis=0))
    order = ([f'h:{z}' for z in range(8, 168, 8)]
             + [f'vy:{w}' for w in range(ys.min(), ys.max(), 24)]
             + [f'vx:{w}' for w in range(xs.min(), xs.max(), 24)])
    assert sorted(entries) == sorted(order)
    assert _write_order(out) in (None, order)
    n_tracks = 0
    for value in entries.values():
        tracks = pickle.loads(value)
        assert isinstance(tracks, list)
        for track in tracks:
            assert track.dtype == np.int32 and track.ndim == 2 and track.shape[1] == 3
            assert track.flags.c_contiguous and len(track) >= 10
        n_tracks += len(tracks)
    assert n_tracks > 50
    assert pickle.dumps([]) in entries.values()

    # Resume: keys already present are not recomputed.
    with dbm.open(str(out), 'w') as database:
        database['h:16'] = pickle.dumps(['kept'])
    est.run_cpu('synthetic', str(out))
    assert _read_dbm(out) == {**entries, 'h:16': pickle.dumps(['kept'])}


# ------------------------------------------------------------------------------------------
# GPU backend, end to end

def _gpu_device():
    """nvidia-smi index of the GPU to test on: the first CUDA_VISIBLE_DEVICES entry when that is
    set (the in-process Brook reference runs on that GPU too), else the first GPU of compute
    capability 8.0 or newer; None when there is none."""
    import surface_tracks_gpu as stg
    try:
        gpus = stg._nvidia_smi_gpus()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        first = visible.split(',')[0].strip()
        candidates = [str(int(first))] if first.isdigit() else []
    else:
        candidates = sorted(gpus, key=int)
    for index in candidates:
        capability = gpus.get(index, {}).get('cc')
        if capability is not None and capability >= stg.MIN_COMPUTE_CAPABILITY:
            return int(index)
    return None


def _write_zarr_v2(path, volume, chunks, separator):
    import zarr
    from numcodecs import Blosc
    array = zarr.create_array(
        store=str(path), shape=volume.shape, chunks=chunks, dtype='uint8', fill_value=0,
        zarr_format=2, compressors=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE),
        chunk_key_encoding={'name': 'v2', 'separator': separator})
    array[...] = volume
    return str(path)


def test_gpu_backend_matches_per_slab_brook(monkeypatch, tmp_path):
    brook = pytest.importorskip('brook')
    pytest.importorskip('cupy')
    # Only look cuCIM up: importing it here would start its cuFile driver, which writes
    # cufile.log into the working directory. The GPU workers import it.
    if importlib.util.find_spec('cucim') is None:
        pytest.skip('cucim is not installed')
    pytest.importorskip('cc3d')
    zarr = pytest.importorskip('zarr')
    device = _gpu_device()
    if device is None:
        pytest.skip('needs an NVIDIA GPU of compute capability 8.0 or newer '
                    '(the first CUDA_VISIBLE_DEVICES entry when set)')

    _small_configuration(monkeypatch)
    monkeypatch.setattr(est, 'open_predictions', functools.partial(zarr.open_array, mode='r'))
    volume = synthetic_predictions()
    # '/' keys, blosc and an x chunk that is a multiple of 8: the direct chunk reader.
    direct = _write_zarr_v2(tmp_path / 'direct.zarr', volume, (64, 64, 64), '/')
    # '.' keys and an x chunk of 50: read through open_predictions.
    fallback = _write_zarr_v2(tmp_path / 'fallback.zarr', volume, (48, 50, 50), '.')

    # The original per-slab loop with Brook in place of Kimimaro.
    with monkeypatch.context() as patch:
        patch.setattr(est, 'get_skeleton_tracks', functools.partial(
            est.get_skeleton_tracks, skeletonize=brook.skeletonize))
        est.run_cpu(direct, str(tmp_path / 'per_slab.dbm'))
    expected = _read_dbm(tmp_path / 'per_slab.dbm')
    assert sum(len(pickle.loads(value)) for value in expected.values()) > 50

    est.run_gpu(direct, str(tmp_path / 'direct.dbm'), gpus=(device,))
    assert _read_dbm(tmp_path / 'direct.dbm') == expected
    assert _write_order(tmp_path / 'direct.dbm') == _write_order(tmp_path / 'per_slab.dbm')

    # Resume: a key already in the DBM is kept as it is.
    with dbm.open(str(tmp_path / 'fallback.dbm'), 'c') as database:
        database['h:16'] = pickle.dumps(['kept'])
    est.run_gpu(fallback, str(tmp_path / 'fallback.dbm'), gpus=(device,))
    assert _read_dbm(tmp_path / 'fallback.dbm') == {**expected, 'h:16': pickle.dumps(['kept'])}
