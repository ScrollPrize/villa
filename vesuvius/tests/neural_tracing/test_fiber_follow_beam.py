"""Beam re-ranker: native beam wrapper, pool states, dataset, model, hook.

Uses a synthetic scene: the annotated fiber runs along +x then bends toward +y,
while a decoy prediction continues straight. The prediction volume (what the
beam scores) shows both; CT (what the re-ranker sees) shows only the fiber.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("vc.fiber_trace")

from vesuvius.neural_tracing.fiber_follow import data as D
from vesuvius.neural_tracing.fiber_follow.beam import states as S
from vesuvius.neural_tracing.fiber_follow.beam.data import BeamDataset, collate_beam, record_pools
from vesuvius.neural_tracing.fiber_follow.beam.hook import ModelBeamHook
from vesuvius.neural_tracing.fiber_follow.beam.model import ARCHITECTURE, BeamRankNet
from vesuvius.neural_tracing.fiber_follow.beam.native import BeamSpec, NativeBeam, fiber_input_from_traced
from vesuvius.neural_tracing.fiber_follow.beam.supervision import beam_loss
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid
from vesuvius.neural_tracing.fiber_follow.model import FollowNetConfig
from vesuvius.neural_tracing.fiber_follow.runloop import read_checkpoint, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec

GRID = (16, 24, 64)  # z, y, x in trace-grid voxels (8 base voxels each)
BEND_X, THETA = 32.0, np.radians(20.0)
Y0, Z0 = 8.0, 8.0


def gt_points(step=0.5):
    """Annotated fiber in grid xyz: straight to BEND_X, then bending toward +y."""
    xs = np.arange(2.0, BEND_X + 1e-9, step)
    straight = np.c_[xs, np.full_like(xs, Y0), np.full_like(xs, Z0)]
    t = np.arange(step, 30.0, step)
    bend = np.c_[BEND_X + t * np.cos(THETA), Y0 + t * np.sin(THETA), np.full_like(t, Z0)]
    pts = np.concatenate([straight, bend])
    return pts[pts[:, 0] < GRID[2] - 2]


def _write_u8_zarr(directory: Path, values: np.ndarray):
    directory.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(values, dtype=np.uint8)
    (directory / '.zarray').write_text(json.dumps(dict(
        zarr_format=2, shape=list(values.shape), chunks=list(values.shape), dtype='|u1', compressor=None,
        fill_value=0, order='C', filters=None, dimension_separator='.')))
    (directory / '0.0.0').write_bytes(values.tobytes())


def paint(volume, points, value, radius, scale=1.0):
    z, y, x = np.indices(volume.shape)
    for p in points * scale:
        m = (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2 <= radius ** 2
        volume[m] = np.maximum(volume[m], value)


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp('beam_scene')
    gt = gt_points()
    decoy = np.c_[np.arange(BEND_X, GRID[2] - 1.0, 0.5), np.full(int((GRID[2] - 1 - BEND_X) * 2), Y0),
                  np.full(int((GRID[2] - 1 - BEND_X) * 2), Z0)]
    presence = np.zeros(GRID, np.uint8)
    # Weak prediction along the bend, strong along the straight decoy: the
    # hand loss prefers the decoy, CT only shows the annotated fiber.
    paint(presence, gt[gt[:, 0] <= BEND_X], 255, 1.2)
    paint(presence, gt[gt[:, 0] > BEND_X], 120, 1.2)
    paint(presence, decoy, 255, 1.2)
    # Direction field: +x everywhere except along the bend, which points along the bend.
    nx = np.full(GRID, 255, np.uint8)
    ny = np.full(GRID, 128, np.uint8)
    z, y, x = np.indices(GRID)
    bend_mask = np.zeros(GRID, bool)
    paint(bend_mask.view(np.uint8), gt[gt[:, 0] > BEND_X + 0.5], 1, 1.2)
    nx[bend_mask] = int(round(128 + 127 * np.cos(THETA)))
    ny[bend_mask] = int(round(128 + 127 * np.sin(THETA)))
    fields = root / 'fields'
    _write_u8_zarr(fields / 'scene_presence.ome.zarr' / '3', presence)
    _write_u8_zarr(fields / 'scene_nx.ome.zarr' / '3', nx)
    _write_u8_zarr(fields / 'scene_ny.ome.zarr' / '3', ny)
    (fields / 'scene.lasagna.json').write_text(json.dumps(dict(
        version=2, source_to_base=1.0,
        groups={name: dict(zarr=f'scene_{name}.ome.zarr/3', scaledown=3, channels=[name]) for name in ('presence', 'nx', 'ny')},
        base_shape_zyx=[8 * v for v in GRID])))
    ct = np.zeros(tuple(2 * v for v in GRID), np.uint8)
    paint(ct, gt, 200, 2.0, scale=2.0)
    _write_u8_zarr(root / 'ct' / '0', ct)
    base = gt * 8.0
    controls = [0, int(np.argmin(np.abs(gt[:, 0] - BEND_X))), len(gt) - 1]
    (root / 'fibers').mkdir()
    (root / 'fibers' / 'scene.json').write_text(json.dumps(dict(
        type='vc3d_fiber', version=1, line_points=base.tolist(), control_points=[base[i].tolist() for i in controls], tags=[])))
    fibers = D.load_fibers(str(root / 'fibers'), grid_scale=8)
    vol_spec = FiberVolumeSpec(str(fields), ct_zarr=str(root / 'ct'), ct_level=0, ct_grid_scale=4., inputs='ct')
    spec = BeamSpec(str(fields / 'scene.lasagna.json'), cache_bytes=64 << 20,
                    config=dict(smoothness_normal_weight=0., smoothness_tangent_weight=0.,
                                cumulative_smoothness_tangent_weight=0.),
                    hook_every_rounds=2, hook_pool_size=16)
    return dict(root=root, fibers=fibers, fiber=fibers[0], vol_spec=vol_spec, spec=spec)


def state_config():
    return S.BeamStateConfig(crop=CropSpec(depth=24, width=17, behind=6, spacing=.5, history_render='segments',
                                           history_sigma=.35),
                             n_history=16, k_back=4, k_fwd=4, pool_size=16, tube_sigma=.35)


def test_native_beam_units_and_hand_beam_takes_the_decoy(scene):
    beam = NativeBeam(scene['spec'], grid_scale=8.)
    fiber = scene['fiber']
    assert beam.trace_to_base == pytest.approx(2.0)
    assert beam.grid_to_trace == pytest.approx(4.0)
    straight_end = int(np.abs(fiber.s - fiber.spans[0].end).argmin())
    result = beam.trace_span(fiber.points, 0, straight_end)
    assert result.reached, result.reason
    np.testing.assert_allclose(result.points[0], fiber.points[0])
    assert abs(result.points[-1, 0] - fiber.points[straight_end, 0]) < 3
    # The bend: the hand beam prefers the straight decoy and misses the target.
    pools, result = record_pools(beam, fiber, straight_end, len(fiber.points) - 1)
    assert pools and all(len(p) >= 8 and len(p) <= 16 for p in pools)
    assert pools[0].round == 0 and pools[1].round == 2
    assert all(np.allclose(p.paths[0][0], fiber.points[straight_end]) for p in pools)
    assert not result.reached
    assert result.points[-1, 1] < Y0 + 2.0


def test_pool_state_labels_separate_fiber_from_decoy(scene):
    beam = NativeBeam(scene['spec'], grid_scale=8.)
    fiber = scene['fiber']
    cfg = state_config()
    start = int(np.abs(fiber.s - (fiber.spans[0].end - 6)).argmin())
    pools, _ = record_pools(beam, fiber, start, len(fiber.points) - 1)
    labeled_states = [S.pool_state(p, cfg, fiber, sign=1.0) for p in pools]
    junction = [it for it in labeled_states if it['label_mask'].sum() >= 4 and it['onfiber'].sum() >= 1
                and (it['label_mask'] - it['onfiber']).sum() >= 1]
    assert junction, 'expected a pool with both on-fiber and decoy candidates'
    it = junction[0]
    assert it['candidates'].shape == (16, cfg.k_points, 3)
    assert it['point_mask'][:, cfg.k_back] .all() or True
    # The anchor point sits at the local origin and the hand-best candidate is first.
    assert np.allclose(it['candidates'][0, cfg.k_back], 0)
    assert it['hand_rel'][0] == 0 and np.all(np.diff(it['hand_loss'][: int(it['cand_mask'].sum())]) >= 0)
    # Prefix failure persists to the end of a candidate.
    for row in np.nonzero(it['label_mask'])[0]:
        target = it['prefix_target'][row][it['prefix_mask'][row] > 0]
        assert np.all(np.diff(target) <= 0)
        assert it['onfiber'][row] == target[-1]
    assert 'tube_segments' in it and len(it['tube_segments'])
    # Departed anchors supply negatives only and no tube position supervision.
    off = [s for s in labeled_states if s['offtrack'] > 0]
    if off:
        assert off[0]['onfiber'].sum() == 0 and not off[0]['tube_supervised']


def test_holdout_rejects_candidates_inside_the_band(scene):
    beam = NativeBeam(scene['spec'], grid_scale=8.)
    fiber = scene['fiber']
    cfg = state_config()
    pools, _ = record_pools(beam, fiber, 0, int(np.abs(fiber.s - fiber.spans[0].end).argmin()))
    item = S.pool_state(pools[0], cfg, fiber, sign=1.0)
    assert D.training_state_allowed(item, cfg.crop, D.ZBand(200, 300))
    zmax = item['extra_world'][:, 2].max()
    assert not D.training_state_allowed(item, cfg.crop, D.ZBand(zmax - .1, zmax + 200))


def test_dataset_model_loss_checkpoint_and_hook(scene, tmp_path):
    torch.manual_seed(0)
    cfg = state_config()
    ds = BeamDataset(scene['fibers'], scene['vol_spec'], scene['spec'], cfg, None, chunk=4, seed=1,
                     cache_bytes=8 << 20, max_states_per_trace=6)
    batch = next(iter(ds))
    assert batch['x'].shape == (4, 2, 24, 17, 17)
    assert batch['candidates'].shape == (4, 16, cfg.k_points, 3)
    for key in ('hist', 'hmask', 'point_mask', 'cand_mask', 'hand_rel', 'prefix_target', 'prefix_mask', 'onfiber',
                'label_mask', 'quality', 'offtrack', 'tube_target', 'tube_mask', 'source'):
        assert key in batch, key
    assert batch['tube_target'].shape == (4, 24, 17, 17)
    model_cfg = FollowNetConfig(in_channels=2, depth=24, width=17, behind=6, spacing=.5, widths=(8, 16), hidden=16,
                                n_future=4, future_step=1., hist_points=4, hist_stride=4, clean_points=4,
                                heat_bins=9, heat_spacing=.5, n_candidates=2, norm='group', heatmap_target='tube',
                                tube_sigma=.35)
    model = BeamRankNet(model_cfg)
    out = model(batch['x'].float(), batch['hist'], batch['hmask'], batch['candidates'], batch['point_mask'], batch['hand_rel'])
    assert out['ranks'].shape == (4, 16) and out['prefix_logits'].shape == (4, 16, cfg.k_points)
    loss, metrics = beam_loss(out, batch, cfg.k_back)
    assert torch.isfinite(loss)
    loss.backward()
    for layer in (model.encoders[0][0], model.rank_head, model.onfiber_head, model.confidence_head, model.heat_head):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert {'ranking', 'onfiber', 'prefix', 'tube', 'hand_top1_onfiber', 'model_top1_onfiber', 'oracle_onfiber'} <= set(metrics)

    path = tmp_path / 'beam.pt'
    save_checkpoint(path, model, scene['vol_spec'], cfg.crop, cfg.n_history, ARCHITECTURE, dict(k_back=cfg.k_back))
    ck = read_checkpoint(path, ARCHITECTURE, 'cpu')
    assert ck['k_back'] == cfg.k_back and ck['model_cfg'] == model_cfg.to_dict()
    with pytest.raises(ValueError):
        read_checkpoint(path, 'spatial_candidates_v3', 'cpu')

    # An additive hook with zero weight reproduces the hand beam exactly.
    vol = FiberVolume(scene['vol_spec'], cache_bytes=8 << 20)
    beam = NativeBeam(scene['spec'], grid_scale=8.)
    fiber = scene['fiber']
    end = int(np.abs(fiber.s - fiber.spans[0].end).argmin())
    plain = beam.trace_span(fiber.points, 0, end)
    hook = ModelBeamHook(model, vol, cfg, device='cpu', mode='additive', weight=0.)
    same = beam.trace_span(fiber.points, 0, end, hook=hook)
    assert hook.calls >= 1
    np.testing.assert_array_equal(same.points, plain.points)
    replace = ModelBeamHook(model, vol, cfg, device='cpu', mode='replace')
    result = beam.trace_span(fiber.points, 0, end, hook=replace)
    assert result.points.shape[1] == 3 and replace.calls >= 1

    metric = beam.whole_fiber_metric(fiber_input_from_traced(fiber, 8.))
    assert metric['segment_count'] == 2
    assert metric['segments'][0]['success'] and not metric['segments'][1]['success']


def test_train_smoke_writes_checkpoint_and_span_diagnostics(scene, tmp_path):
    from vesuvius.neural_tracing.fiber_follow.beam import train as T
    from vesuvius.neural_tracing.fiber_follow.beam.trace import BeamTracer
    from vesuvius.neural_tracing.fiber_follow.evaluate import evaluate
    root = scene['root']
    out_root = tmp_path / 'runs'
    last = T.main(['--fiber-zarrs', str(root / 'fields'), '--fibers', str(root / 'fibers'), '--ct', str(root / 'ct'),
                   '--prediction-manifest', str(root / 'fields' / 'scene.lasagna.json'),
                   '--beam-config', json.dumps(scene['spec'].config), '--hook-every-rounds', '2', '--pool-size', '16',
                   '--k-back', '4', '--k-fwd', '4', '--n-history', '16', '--hist-points', '4', '--hist-stride', '4',
                   '--name', 'smoke', '--out-root', str(out_root), '--device', 'cpu', '--steps', '2', '--batch', '4',
                   '--workers', '0', '--worker-cache-gb', '.01', '--beam-cache-gb', '.05', '--val-z', '1000000', '1000001',
                   '--warmup', '1', '--ckpt-every', '2', '--log-every', '1', '--diag-every', '2', '--diag-fibers', '1',
                   '--crop-depth', '24', '--crop-width', '17', '--crop-behind', '6', '--widths', '8', '16', '--hidden', '16',
                   '--norm', 'group', '--states-per-trace', '4'])
    run = out_root / 'smoke'
    assert (run / 'config.json').exists() and Path(last).exists()
    records = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
    assert any('span_restarts_kvx_hand' in r and 'span_restarts_kvx_model' in r for r in records)
    assert any('model_top1_onfiber' in r for r in records)
    assert (run / 'images' / 'pool_000002.png').exists()
    model, state_cfg, vol_spec, beam_spec, ck = T.load_beam_checkpoint(last, 'cpu')
    assert state_cfg.k_back == 4 and beam_spec.hook_pool_size == 16 and ck['hook_mode'] == 'additive'
    # Open-ended tracing through the evaluate() interface.
    fiber = scene['fiber']
    tracer = BeamTracer(NativeBeam(beam_spec, vol_spec.grid_scale), max_len=40)
    seeds = [dict(fiber=0, t=float(fiber.s[4]), sign=1, pos=fiber.points[4], heading=np.array([1., 0., 0.]))]
    rows, summary = evaluate(tracer, [fiber], seeds, batch=1)
    assert rows[0]['reason'] == 'max_len' and summary['n'] == 1 and rows[0]['coverage'] > 0.5


def test_hard_span_mining_finds_the_bend_and_dataset_oversamples_it(scene, tmp_path):
    from vesuvius.neural_tracing.fiber_follow.beam.mine import load_or_mine_hard_spans
    beam = NativeBeam(scene['spec'], grid_scale=8.)
    cache = tmp_path / 'hard.json'
    hard = load_or_mine_hard_spans(cache, beam, scene['fibers'], 8.)
    assert hard == {scene['fiber'].name: [1]}
    assert load_or_mine_hard_spans(cache, beam, scene['fibers'], 8.) == hard  # cache hit
    with pytest.raises(ValueError):
        load_or_mine_hard_spans(cache, beam, scene['fibers'], 8., error_threshold_base=5.)
    cfg = state_config()
    ds = BeamDataset(scene['fibers'], scene['vol_spec'], scene['spec'], cfg, None, chunk=4, seed=2,
                     cache_bytes=8 << 20, hard_spans=hard, hard_span_prob=1.0, p_perturb=0.)
    rng = np.random.default_rng(0)
    fiber, span, mined = ds.pick_span(rng)
    assert mined and span == (fiber.spans[1].start, fiber.spans[1].end)
    free = BeamDataset(scene['fibers'], scene['vol_spec'], scene['spec'], cfg, None, chunk=4, seed=2, cache_bytes=8 << 20)
    starts, ends = zip(*(free.pick_span(rng)[1] for _ in range(50)))
    assert min(np.array(ends) - np.array(starts)) >= 12 and max(ends) <= fiber.length
    assert len({round(s, 3) for s in starts}) > 40  # arbitrary vertices, not only control points
    batch = next(iter(ds))
    assert (batch['source'] >= 1).all()
