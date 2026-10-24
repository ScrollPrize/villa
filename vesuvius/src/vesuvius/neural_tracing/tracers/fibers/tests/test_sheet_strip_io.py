"""Boundary cases for strip inference outputs, indexes and evaluation caches."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.fibers.sheet_strip import evaluate, infer
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import StripIndex


def test_scorer_searches_past_empty_splits(tmp_path, monkeypatch):
    (tmp_path/'meta.json').write_text('{}')
    (tmp_path/'patches.json').write_text('[]')
    arrays = dict(runs_split=[0], runs_fiber=[42], runs_patch=[0], runs_start=[0.], runs_end=[10.],
                  run_samples=[0, 1], sample_rc=[[0, 0]], sample_xyz=[[0., 0., 0.]])
    for name, values in arrays.items():
        np.save(tmp_path/f'{name}.npy', np.asarray(values))
    empty = StripIndex(tmp_path, 'validation')
    assert len(empty) == 0 and empty.unit_runs.shape == (0, 2)
    assert empty.unit_spans == [] and empty.unit_length.size == 0
    train = StripIndex(tmp_path, 'train')
    assert train.units.tolist() == [42] and list(train.runs(0)) == [0]
    monkeypatch.setattr(infer, 'RunFibers', lambda *args: None)
    monkeypatch.setattr(infer, 'StripBuilder', lambda *args, **kwargs: None)
    cfg = SimpleNamespace(fine=SimpleNamespace(width=32, spacing=1.))
    spec = SimpleNamespace(ct_grid_scale=4., grid_scale=2.)
    scorer = infer.Scorer(tmp_path, None, 42, cfg, spec, {})
    assert scorer.unit == 0 and scorer.split == 'train'
    assert infer.Scorer(tmp_path, None, 99, cfg, spec, {}).unit is None


@pytest.mark.parametrize('existing', ['batches/00002.pt', 'meta.npz', 'results/old.npz'])
def test_cache_build_refuses_existing_contents(tmp_path, existing):
    path = tmp_path/existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'previous cache')
    # Rejected before attempting to load a checkpoint or write any cache files.
    with pytest.raises(ValueError, match='must be new or empty'):
        evaluate.build(SimpleNamespace(out=tmp_path, checkpoint='missing.pt', batches=2))
    assert path.read_bytes() == b'previous cache'
    assert [p for p in tmp_path.rglob('*') if p.is_file()] == [path]


@pytest.fixture
def inference_setup(monkeypatch):
    fiber = SimpleNamespace(length=200.)

    class Fibers:
        id_to_index = {42: 0}

        def __getitem__(self, index):
            return fiber

    class Predictor:
        source = 'test-checkpoint'
        cfg, ck, data = SimpleNamespace(), {}, {}
        spec = SimpleNamespace(grid_scale=1.)
        trace_per_patch, vol = 1., None
        f, v = np.array([-8., 0., 8.]), np.array([-4., 0., 4.])

        def predict(self, fiber, arcs, on_items=None, log=None):
            if on_items:
                on_items([dict(t=t) for t in arcs])
            n = len(arcs)
            positions = np.column_stack((np.full(n, 100.), np.full(n, 100.), 100.+arcs))
            return np.zeros((n, 3, 3)), None, np.tile(np.eye(3), (n, 1, 1)), positions

    predictor = Predictor()
    monkeypatch.setattr(infer, 'StripPredictor', lambda *args, **kwargs: predictor)
    monkeypatch.setattr(infer, 'AFVFibers', lambda *args, **kwargs: Fibers())
    # Exercise render_saved and its score handling without a CT volume or plotting.
    monkeypatch.setattr(infer, 'render_strip', lambda vol, rows, keep, *args: np.where(keep, 1., np.nan))
    renders = []
    monkeypatch.setattr(infer, 'save_render', lambda *args: renders.append(args))
    return fiber, renders


def test_empty_geometry_fails_before_writing(tmp_path, inference_setup):
    fiber, renders = inference_setup
    fiber.length = 100.
    out = tmp_path/'output'
    with pytest.raises(SystemExit, match='No strip nodes remain'):
        infer.main(['--fiber', '42', '--afv', 'test.afv', '--index', 'none', '--out', str(out)])
    assert not out.exists() and not renders


@pytest.mark.parametrize('coverage', ['none', 'trimmed_end', 'full'])
def test_inference_saves_with_partial_or_missing_scores(tmp_path, monkeypatch, inference_setup, coverage):
    class Scorer:
        unit, split = 0, 'train'

        def __call__(self, item):
            covered = coverage == 'full' or (coverage == 'trimmed_end' and item['t'] == 12.)
            return np.ones((3, 3)), np.full((3, 3), covered, dtype=bool)

    monkeypatch.setattr(infer, 'Scorer', lambda *args: Scorer())
    out = tmp_path/'output'
    infer.main(['--fiber', '42', '--afv', 'test.afv', '--index', 'test-index', '--out', str(out)])
    summary = json.loads((out/'summary.json').read_text())
    score = summary['patch_score']
    assert (out/'strip'/'x.tif').exists()
    with np.load(out/'strip.npz') as saved:
        assert saved['masks'].any() == (coverage != 'none')
        assert np.isfinite(saved['decision_error_vox']).any() == (coverage != 'none')
    assert ('strip_mae_vox' in score) == (coverage != 'none')
    assert ('stitched_node_distance_vox' in score) == (coverage == 'full')
    assert (out/'patch_sheet').exists() == (coverage != 'none')
    if coverage == 'none':
        assert score['supervised_points'] == score['scored_decisions'] == 0
    else:
        assert score['strip_mae_vox'] == score['p95_vox'] == 1.
    if coverage == 'full':
        assert score['stitched_node_distance_vox'] == dict(median=1., p95=1.)
    assert len(inference_setup[1]) == 1
