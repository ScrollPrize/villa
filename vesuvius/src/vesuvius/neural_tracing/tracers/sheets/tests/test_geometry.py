"""Geometry, sampling, data and rollout checks on analytic surfaces and tiny generated data (no data files needed)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import interp_at
from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.geometry import (
    point_frames, carry_stopped, chord_tangents, grid_eval, hermite_dense, onto_confident_curve, project, respace,
    sheet_distance)
from vesuvius.neural_tracing.tracers.sheets.losses import landing, step_losses, targets
from vesuvius.neural_tracing.tracers.sheets.model import build_model
from vesuvius.neural_tracing.tracers.sheets.tracing.rollout import Tracer
from vesuvius.neural_tracing.tracers.sheets.sampling import crop_offsets, sample_crops
from vesuvius.neural_tracing.tracers.sheets.tracing.seeding import seed_rows, sheet_jump
from vesuvius.neural_tracing.tracers.sheets.data.patches import GridSpec


def cylinder_grid(spec, radius=300., center=(400., 400.), z0=300.):
    """GT grid on a cylinder: u along z (advance), v along the arc (front)."""
    u, v = np.meshgrid(spec.u_vals, spec.v_vals, indexing='ij')
    theta = v/radius
    xyz = np.stack([center[0]+radius*np.cos(theta), center[1]+radius*np.sin(theta), z0+u], -1)
    return torch.tensor(xyz, dtype=torch.float32)[None], torch.ones(1, spec.Gu, spec.Gv, dtype=torch.bool)


def _write_plane_patch(patch: Path, rows: int, cols: int) -> None:
    """A tifxyz patch on the plane x = 60.5 with 2 vx cells: y = 20 + 2 col, z = 20 + 2 row."""
    patch.mkdir(parents=True)
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    for name, values in dict(x=np.full(r.shape, 60.5), y=20.+2*c, z=20.+2*r).items():
        tifffile.imwrite(patch/f'{name}.tif', values.astype(np.float32))
    (patch/'meta.json').write_text(json.dumps(dict(scale=[.5, .5])))


def _script(name: str):
    """A module of scripts/ (not a package)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'scripts'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_volume(root, level=1, shape=(64, 64, 64), chunk=32, sheet_x=30):
    """Tiny uncompressed zarr level with a bright sheet x = const (level voxels)."""
    root = root/str(level)
    root.mkdir(parents=True)
    (root/'.zarray').write_text(json.dumps(dict(shape=list(shape), chunks=[chunk]*3, dtype='|u1', fill_value=0,
                                                 order='C', filters=None, dimension_separator='.', compressor=None,
                                                 zarr_format=2)))
    vol = np.zeros(shape, np.uint8)
    vol[:, :, sheet_x] = 255
    for k in np.ndindex(*(s//chunk for s in shape)):
        piece = vol[tuple(slice(i*chunk, (i+1)*chunk) for i in k)]
        (root/'.'.join(map(str, k))).write_bytes(np.ascontiguousarray(piece).tobytes())


# A small model of the supported architecture (level 1 so a 224 x 360 x 360 block covers the test cylinder).
SMALL = ['train.amp=none', 'model.dim=32', 'model.heads=2', 'model.layers=2', 'model.mlp_ratio=4', 'data.volume_level=1',
         'data.front_points=5', 'data.point_spacing=55', 'model.crop=[8,16,8]', 'model.channels=[4,8,8]',
         'model.strides=[1,2,2]', 'model.point_bottleneck=4', 'model.token_crop=[4,4,4]', 'model.token_channels=[4,8,8]',
         'model.token_strides=[1,2,2]']


def test_respace_spaces_the_curve_evenly():
    t = torch.linspace(0, 1, 5)**2  # unevenly spaced points on a line
    line = torch.stack([240*t, 0*t, 0*t], -1)[None]
    out = respace(hermite_dense(line, chord_tangents(line), 6), 5)
    gaps = (out[:, 1:]-out[:, :-1]).norm(dim=-1)
    assert torch.allclose(gaps, torch.full_like(gaps, 60.), atol=.5)


def test_projection_on_cylinder():
    data = C.DataConfig()
    spec = GridSpec.for_episode(data, 6)
    gt, valid = cylinder_grid(spec)
    S, _, _, _ = grid_eval(gt, valid, torch.tensor([[[10.3, 20.7]]]))
    q = S+torch.tensor([[[3., 0., 0.]]])  # roughly along the outward normal near theta ~ 0
    uv, foot, frame, ok = project(gt, valid, q)
    assert ok.all()
    assert abs(sheet_distance(q, foot, frame).abs().item()-3.) < .3
    assert torch.allclose(uv, torch.tensor([[[10.3, 20.7]]]), atol=.3)


def test_point_frames_orthonormal():
    g = torch.Generator().manual_seed(0)
    pos = (torch.tensor([10., 0., 0.])+torch.randn(2, 8, 3, generator=g)).cumsum(1)  # a front along x
    prev = pos-torch.tensor([0., 0., 5.])  # advancing along z
    f = point_frames(pos, prev)
    eye = torch.eye(3).expand(2, 8, 3, 3)
    assert torch.allclose(f.transpose(-1, -2)@f, eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(f), torch.ones(2, 8), atol=1e-5)


def test_crop_sampling_finds_bright_plane():
    """A bright plane z = 100 sampled from an point at z = 104 peaks 4 voxels below the crop centre."""
    block = torch.zeros(1, 1, 64, 64, 64)
    block[0, 0, 50] = 255.  # level-1 voxel 50 is centred on level-0 z = 100.5
    origin = torch.zeros(1, 3, dtype=torch.long)
    pos = torch.tensor([[[64., 64., 104.5]]])
    frames = torch.eye(3)[None, None]  # t = x, a = y, n = z
    offsets = crop_offsets([17, 5, 5], 1., 0., 'cpu')
    crops = sample_crops(block, origin, 1, pos, frames, offsets)
    profile = crops[0, 0, 0, :, 2, 2]
    assert int(profile.argmax()) == 8-4


def test_same_sheet_label_and_jump():
    """A run pushed one sheet spacing off is labelled off-sheet and gets no position target."""
    cfg = C.load(None, ['train.amp=none'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    pos = seed_rows(gt, valid, cfg, spec, noise=False)[:, -1]
    pushed, run = sheet_jump(pos, gt, valid, cfg, torch.tensor([True]))
    assert 1 <= int(run.sum()) <= cfg.data.front_points//2
    _, dist, same, ok = landing(pushed, gt, valid, cfg.loss.sheet_tolerance)
    assert ok.all()
    assert torch.equal(same, ~run)
    middle = (cfg.train.jump_min+cfg.train.jump_max)/2  # no volume given: push to the middle of the range
    assert torch.allclose(dist[run], torch.full_like(dist[run], middle), atol=.5)
    d = cfg.data
    columns = (torch.arange(d.front_points)*d.point_spacing+d.gt_margin)/d.gt_resolution
    _, _, applies = targets(pushed, gt, valid, d.step/d.gt_resolution, cfg.loss.sheet_tolerance, columns)
    assert torch.equal(applies, ~run)
    unchanged, none = sheet_jump(pos, gt, valid, cfg, torch.tensor([False]))
    assert not none.any() and torch.equal(unchanged, pos)


def test_hermite_curve_interpolates_and_follows_arc():
    theta = torch.linspace(0, 1., 5)
    arc = torch.stack([300*torch.cos(theta), 300*torch.sin(theta), 0*theta], -1)[None]
    dense = hermite_dense(arc, chord_tangents(arc), 6)
    assert torch.allclose(dense[:, ::6], arc, atol=1e-4)  # passes through every control point
    radius = dense[0, :, :2].norm(dim=-1)
    assert (radius-300).abs().max() < .1  # chord tangents keep a 75 vx-spaced arc on the circle
    line = torch.stack([60*torch.arange(5.), torch.zeros(5), torch.zeros(5)], -1)[None]
    dl = hermite_dense(line, chord_tangents(line), 6)
    assert torch.allclose(dl[0, :, 0], torch.linspace(0, 240, 25), atol=1e-3)


def test_carry_stopped_interpolates_neighbours():
    disp = torch.tensor([[[0., 0., 0.], [9., 9., 9.], [9., 9., 9.], [3., 0., 0.], [5., 0., 0.]]])
    confident = torch.tensor([[True, False, False, True, False]])
    out = carry_stopped(disp, confident)
    assert torch.allclose(out[0, 1], torch.tensor([1., 0., 0.]))
    assert torch.allclose(out[0, 2], torch.tensor([2., 0., 0.]))
    assert torch.allclose(out[0, 4], torch.tensor([3., 0., 0.]))  # beyond the last confident point
    assert torch.equal(carry_stopped(disp, torch.zeros_like(confident)), disp)


def test_straight_advance_on_cylinder_axis():
    """The zero-initialised head predicts a straight advance: along a cylinder's axis it stays on the sheet."""
    cfg = C.load(None, SMALL+['trace.confidence_threshold=.4'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    tracer = Tracer(build_model(cfg), cfg, 'cpu')
    front = tracer.initial(seed_rows(gt, valid, cfg, spec, noise=False))
    block = torch.zeros(1, 1, 8, 8, 8)
    for _ in range(4):
        out = tracer.step(front, block, torch.zeros(1, 3, dtype=torch.long))
        front = tracer.advance(front, out)
    _, foot, frame, ok = project(gt, valid, out.dense)
    assert ok.all()
    assert sheet_distance(out.dense, foot, frame).abs().max() < .1


def test_sheet_jump_snaps_to_neighbouring_sheet():
    """With a volume, pushed front_points land on the bright neighbouring sheet, not at a fixed distance."""
    cfg = C.load(None, ['train.amp=none', 'data.volume_level=1'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    u, v = np.meshgrid(spec.u_vals, spec.v_vals, indexing='ij')
    plane = np.stack([100+v, np.full_like(u, 200.), 100+u], -1)  # the sheet y = 200, normal along y
    gt = torch.tensor(plane, dtype=torch.float32)[None]
    valid = torch.ones(1, spec.Gu, spec.Gv, dtype=torch.bool)
    block = torch.zeros(1, 1, 256, 256, 256)
    for i in (94, 106):  # level-1 voxels centred on y = 2i + .5 = 188.5 and 212.5: neighbours ~12 vx away
        block[0, 0, :, i] = 255.
    pos = seed_rows(gt, valid, cfg, spec, noise=False)[:, -1]
    pushed, run = sheet_jump(pos, gt, valid, cfg, torch.tensor([True]), None, block, torch.zeros(1, 3, dtype=torch.long))
    offset = (pushed-pos)[run].norm(dim=-1)
    assert run.any() and torch.allclose(offset, torch.full_like(offset, 12.), atol=1.)


def test_onto_confident_curve_returns_stopped_points_to_the_sheet():
    """A stopped point a sheet away (12 vx off an arc) lands back on the curve through its confident neighbours."""
    theta = torch.linspace(0, 1.2, 6)
    arc = torch.stack([300*torch.cos(theta), 300*torch.sin(theta), 0*theta], -1)[None]
    off = arc.clone()
    off[0, 2] *= 312/300  # pushed radially onto the next sheet
    off[0, 3] *= 312/300
    off[0, 5] *= 312/300  # beyond the last confident point: left alone
    confident = torch.tensor([[True, True, False, False, True, False]])
    out = onto_confident_curve(off, confident)
    radius = out[0, :, :2].norm(dim=-1)
    assert (radius[2:4]-300).abs().max() < .5  # back on the sheet (chord sag handled by the tangents)
    assert torch.equal(out[0, confident[0]], off[0, confident[0]])
    assert torch.equal(out[0, 5], off[0, 5])
    none = torch.zeros_like(confident)
    assert torch.equal(onto_confident_curve(off, none), off)


def test_front_length_loss_penalises_stretch():
    """The front-length term is zero for a front at its seed length and grows when it stretches."""
    from vesuvius.neural_tracing.tracers.sheets.tracing.rollout import StepOutput
    cfg = C.load(None, SMALL+['loss.front_length=1.'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec, radius=10000.)  # nearly flat: arc length ~ chord length
    seeds = seed_rows(gt, valid, cfg, spec, noise=False)
    front = Tracer(build_model(cfg), cfg, 'cpu').initial(seeds)
    def stats_for(scale):
        pos = front.pos[:, :1]+(front.pos-front.pos[:, :1])*scale
        dense = hermite_dense(pos, chord_tangents(pos), cfg.trace.spline_samples)
        out = StepOutput(pos, torch.zeros(1, 5), pos, dense, torch.zeros(1, 5, 1), torch.eye(3).expand(1, 5, 3, 3), None)
        return step_losses(out, front, gt, valid, cfg)[1]['gap_excess']
    assert stats_for(1.) < .1
    assert abs(float(stats_for(1.2))-11.) < .5  # 20% stretch of 55 vx gaps


def test_connected_keep_and_row_reach():
    """Only confident points connected to the seed row are written; an unreachable row means stop."""
    from vesuvius.neural_tracing.tracers.sheets.tracing.inference import _reach_row, connected_keep
    conf = np.array([[1, 1, 1, 1, 1, 1],   # seed
                     [.9, .9, 0, 0, 0, 0],
                     [0, 0, 0, 0, .9, .9],  # confident but not touching row 1's kept columns
                     [.9, .9, .9, .9, .9, .9]])
    keep = connected_keep(conf, .5)
    assert keep[0].all() and keep[1, :2].all() and not keep[2].any() and not keep[3].any()
    reach = np.ones(6, bool)
    reach = _reach_row(conf[1] > .5, reach)
    assert reach.tolist() == [True, True, False, False, False, False]
    assert not _reach_row(conf[2] > .5, reach).any()  # tracing stops here
    assert _reach_row(np.array([True, True, True, False]), np.array([False, False, True, False])).tolist() == \
        [True, True, True, False]  # a contiguous run is reached through any of its columns


def test_trace_end_to_end_on_a_small_volume(tmp_path):
    """infer.trace runs a real model (parameters requiring grad) on a tiny on-disk volume and returns arrays."""
    from vesuvius.neural_tracing.tracers.sheets.tracing.inference import connected_keep, trace
    level = 1
    _write_volume(tmp_path/'vol.zarr', level)  # a bright sheet at level-1 x 30 ~ level-0 x 60.5
    cfg = C.load(None, ['train.amp=none', 'model.dim=32', 'model.channels=[4,8]', 'model.strides=[2,2]',
                        'model.heads=2', 'model.token_channels=[4,8]', 'model.token_strides=[2,2]',
                        f'data.volume="{tmp_path/"vol.zarr"}"', f'data.volume_level={level}'])
    model = build_model(cfg)
    seed = np.stack([np.full(5, 60.5), np.linspace(20, 100, 5), np.full(5, 20.)], 1)  # on the sheet, along y
    direction = np.tile([0., 0., 1.], (5, 1))  # advance along z
    rows, confs = trace(model, cfg, seed, direction, 3, 'cpu', stop_on_disconnect=False)
    assert isinstance(rows, np.ndarray) and rows.shape == (4, (5-1)*cfg.trace.spline_samples+1, 3)
    assert confs.shape == rows.shape[:2] and connected_keep(confs, .5)[0].all()


def test_history_warm_up_rollout_trains():
    from vesuvius.neural_tracing.tracers.sheets.train.train import train_rollout
    cfg = C.load(None, SMALL+['train.history_rows=3', 'data.history=4', 'data.steps=3'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    b = dict(gt=gt, valid=valid, block=torch.rand(1, 1, 224, 360, 360)*255, origin=torch.zeros(1, 3, dtype=torch.long))
    model = build_model(cfg)
    torch.nn.init.normal_(model.head.weight, std=1e-2)
    generator = torch.Generator().manual_seed(1)
    loss, stats, _, _ = train_rollout(Tracer(model, cfg, 'cpu'), b, cfg, spec, generator)  # backpropagates per step
    assert torch.isfinite(loss) and stats['supervised'] > .9
    assert model.token_geometry[0].weight.grad.abs().sum() > 0  # the grid memory the warm-up filled is read


def test_real_switch_needs_several_points_or_a_far_one():
    from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import switch_steps
    off = torch.zeros(6, 10, dtype=torch.bool)
    dist = torch.full((6, 10), 6.5)
    off[1, 4] = True  # one borderline point: a switch, but not a real one
    off[3, 2:5] = True  # three in one row: real
    assert switch_steps(off, dist) == (2, 4)
    dist[1, 4] = 10.  # one point far past the gap midpoint: real
    assert switch_steps(off, dist) == (2, 2)
    assert switch_steps(torch.zeros_like(off), dist) == (None, None)


def test_live_chain_continuation_window(tmp_path):
    """A chain's next episode is the same patch window moved along the advance by the front's progress."""
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import EpisodeSampler
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    _write_plane_patch(tmp_path/'patches'/'p0', 100, 40)  # 2 vx cells: z along rows, y along cols
    _write_volume(tmp_path/'vol.zarr')
    cfg = C.load(None, [f'data.patches_dir="{tmp_path/"patches"}"', f'data.volume="{tmp_path/"vol.zarr"}"', 'data.volume_level=1',
                        'data.front_points=3', 'data.point_spacing=20', 'data.step=10', 'data.steps=2',
                        'data.gt_margin=10', 'data.gt_resolution=2', 'data.roi=512',
                        'data.val_fraction=0', 'data.min_valid_fraction=.5'])
    sampler = EpisodeSampler(cfg.data, build_index(cfg.data.patches_dir, tmp_path/'cache'), 'train', 2, 512, 10.)
    rng = np.random.default_rng(0)
    assert any(sampler.sample(rng) is not None for _ in range(20))  # fresh episodes still draw
    meta = dict(patch='p0', root=str(tmp_path/'patches'), axis=0, sign_u=1, sign_v=1, anchor=[20., 5.])
    extent = dict(front_lo=np.full(3, 50.), front_hi=np.full(3, 70.))
    first = sampler.continuation(dict(meta=meta, shift=0., **extent), .5)
    nxt = sampler.continuation(dict(meta=meta, shift=20., **extent), .5)
    assert first is not None and nxt is not None and nxt['meta']['anchor'] == [30., 5.]  # 20 vx = 10 cells
    k = int(20/cfg.data.gt_resolution)  # the shift in GT grid rows
    both = first['valid'][k:] & nxt['valid'][:-k]
    assert both.sum() > 20 and np.allclose(first['gt'][k:][both], nxt['gt'][:-k][both], atol=1e-3)
    u0 = int(round(-sampler.spec.u_lo/sampler.spec.res))
    assert abs(float(nxt['gt'][u0, nxt['valid'][u0]][:, 2].mean())-80.) < 1e-3  # z = 20 + 2 * row 30
    assert nxt['block'].shape[0] == 1 and nxt['block'].max() == 255  # the sheet is in the block
    assert sampler.continuation(dict(meta=meta, shift=400., **extent), .5) is None  # past the patch end


def test_live_chains_resume_where_the_model_left_off():
    import queue as _queue
    from vesuvius.neural_tracing.tracers.sheets.train.live import LiveChains
    from vesuvius.neural_tracing.tracers.sheets.train.train import train_rollout
    cfg = C.load(None, SMALL+['train.live_chains=true', 'data.steps=3', 'train.chain_steps=[3,30]', 'train.batch=1',
                              'train.chain_on_sheet=0', 'trace.confidence_threshold=.4', 'train.free_fraction=1',
                              'train.jump_probability=0'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    b = dict(gt=gt, valid=valid, block=torch.rand(1, 1, 224, 360, 360)*255, origin=torch.zeros(1, 3, dtype=torch.long),
             meta=[dict(patch='p', root='r', axis=0, sign_u=1, sign_v=1, anchor=[5., 5.])])
    model = build_model(cfg)
    tracer = Tracer(model, cfg, 'cpu')
    jobs = _queue.Queue(4)
    chains = LiveChains(cfg, spec, tracer, jobs, torch.Generator().manual_seed(0))
    _, chain = chains.begin(b, 0)
    chain['limit'][:] = 30
    _, _, front, confs = train_rollout(tracer, b, cfg, spec, torch.Generator().manual_seed(1))
    chains.finish(b, chain, front, confs, 0)
    job = jobs.get_nowait()
    assert len(job['chains']) == 1 and job['chains'][0]['shift'] > 0
    _, resumed = chains.begin(dict(b, chains=[(job['id'], 0)]), 1)
    assert torch.equal(resumed['front'].pos, front.pos) and int(resumed['steps'][0]) == 3
    _, _, front2, _ = train_rollout(tracer, b, cfg, spec, torch.Generator().manual_seed(2), resumed['front'])
    assert len(front2.grid) == 3  # memory is full after a rollout
    assert chains.begin(dict(b, chains=[(job['id'], 0)]), 2)[1] is None  # consumed
    # Two jobs' chains share a batch; a chain whose state is gone is dropped with its episode.
    chains.finish(b, resumed, front2, confs, 3)
    chains.finish(b, resumed, front2, confs, 3)
    ids = [jobs.get_nowait()['id'] for _ in range(2)]
    del chains.pending[ids[0]]
    two = dict(b, gt=b['gt'].expand(2, -1, -1, -1), valid=b['valid'].expand(2, -1, -1), block=b['block'].expand(2, -1, -1, -1, -1),
               origin=b['origin'].expand(2, -1), meta=b['meta']*2, chains=[(ids[0], 0), (ids[1], 0)])
    kept, merged = chains.begin(two, 4)
    assert len(kept['meta']) == 1 and kept['gt'].shape[0] == 1 and kept['chains'] == [(ids[1], 0)]
    assert torch.equal(merged['front'].pos, front2.pos) and int(merged['steps'][0]) == 6
    # Survivors pool until a batch of them is ready: then one job holds both rollouts' chains.
    cfg.train.batch = 2
    chains.finish(b, resumed, front2, confs, 5)
    assert jobs.empty() and len(chains.pool) == 1
    chains.finish(b, resumed, front2, confs, 6)
    job = jobs.get_nowait()
    assert len(job['chains']) == 2 and not chains.pool and chains.pending[job['id']]['front'].pos.shape[0] == 2


def test_ema_weights_average_and_are_what_evaluation_loads(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.checkpoint import load_checkpoint
    from vesuvius.neural_tracing.tracers.sheets.train.train import EMA, save
    cfg = C.load(None, SMALL+['train.ema=0.9'])
    model = build_model(cfg)
    ema = EMA(model, cfg.train.ema)
    before = {k: v.clone() for k, v in ema.model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.)
    ema.update(model)  # first update: decay min(0.9, 2/11)
    w = 1-2/11
    assert all(torch.allclose(ema.model.state_dict()[k], before[k]+w) for k in before if before[k].is_floating_point())
    opt = torch.optim.AdamW(model.parameters())
    save(tmp_path/'c.pt', model, opt, cfg, 1, 0., 0., ema)
    loaded, _, checkpoint = load_checkpoint(tmp_path/'c.pt', 'cpu')
    raw, _, _ = load_checkpoint(tmp_path/'c.pt', 'cpu', raw=True)
    k = 'head.weight'
    assert torch.equal(loaded.state_dict()[k], ema.model.state_dict()[k])
    assert torch.equal(raw.state_dict()[k], model.state_dict()[k]) and checkpoint['ema_updates'] == 1


def test_hard_spot_episodes_start_at_the_recorded_spot(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import EpisodeSampler
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    _write_plane_patch(tmp_path/'patches'/'p0', 100, 40)
    _write_volume(tmp_path/'vol.zarr')
    spots = tmp_path/'hard.json'
    spots.write_text(json.dumps(dict(records=[dict(dir=0, patch='p0', axis=0, sign_u=1, sign_v=1, anchor=[20., 5.],
                                                    shift=20., step=8, kind='stop')])))
    overrides = [f'data.patches_dir=["{tmp_path/"patches"}"]', f'data.volume="{tmp_path/"vol.zarr"}"', 'data.volume_level=1',
                 'data.front_points=3', 'data.point_spacing=20', 'data.step=10', 'data.steps=2', 'data.gt_margin=10',
                 'data.gt_resolution=2', 'data.roi=512', 'data.min_valid_fraction=.5',
                 f'data.hard_windows="{spots}"', 'data.hard_fraction=1']
    index = build_index(C.load(None, overrides).data.patches_dir, tmp_path/'cache')
    cfg = C.load(None, overrides+['data.val_fraction=0'])
    rng = np.random.default_rng(0)
    sampler = EpisodeSampler(cfg.data, index, 'train', 2, 512, 10.)
    episodes = [sampler.sample(rng) for _ in range(10)]
    assert all(e is not None and e['meta']['hard'] for e in episodes)
    # the spot's anchor (row 20) moved 20 vx = 10 rows along the advance, jittered by up to 2 steps (10 rows)
    assert all(20. <= e['meta']['anchor'][0] <= 40. and e['meta']['axis'] == 0 for e in episodes)
    val = EpisodeSampler(C.load(None, overrides+['data.val_fraction=1']).data, index, 'val', 2, 512, 10.)
    assert not val.hard and all('hard' not in e['meta'] for e in (val.sample(rng) for _ in range(5)) if e)


def test_index_uses_measured_cell_when_meta_scale_is_wrong(tmp_path):
    """Some patch families state twice their real cell size in meta.json; the index measures it instead."""
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    r, c = np.meshgrid(np.arange(30), np.arange(20), indexing='ij')
    for name, scale in (('right', .5), ('wrong', .25), ('close', .45)):  # nodes 2 vx apart; claims 2, 4, 2.2 vx
        patch = tmp_path/'p'/name
        patch.mkdir(parents=True)
        for axis, values in dict(x=np.full(r.shape, 5.), y=2.*c, z=2.*r).items():
            tifffile.imwrite(patch/f'{axis}.tif', values.astype(np.float32))
        (patch/'meta.json').write_text(json.dumps(dict(scale=[scale, scale])))
    infos = {i.name: i for i in build_index(tmp_path/'p', tmp_path/'cache')}
    assert infos['right'].cell == 2. and not infos['right'].stated_cell
    assert abs(infos['wrong'].cell-2.) < 1e-6 and infos['wrong'].stated_cell == 4.
    assert abs(infos['close'].cell-1/.45) < 1e-6 and not infos['close'].stated_cell  # within tolerance: as stated


def test_model_rollout_trains_every_part():
    cfg = C.load(None, SMALL)
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    block = torch.rand(1, 1, 224, 360, 360)*255
    origin = torch.zeros(1, 3, dtype=torch.long)
    model = build_model(cfg)
    torch.nn.init.normal_(model.head.weight, std=1e-2)
    tracer = Tracer(model, cfg, 'cpu')
    front = tracer.initial(seed_rows(gt, valid, cfg, spec, noise=False))
    total = 0.
    for _ in range(3):
        out = tracer.step(front, block, origin)
        loss, _ = step_losses(out, front, gt, valid, cfg)
        total = total+loss
        front = tracer.advance(front, out)
    total.backward()
    assert len(front.grid) == 3 and front.grid[-1][0].shape == (1, tracer.columns, 32)
    for name in ('encoder', 'token_encoder', 'blocks', 'token_geometry', 'point_geometry'):
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in getattr(model, name).named_parameters() if n.endswith('weight')), name
    assert model.token_type.grad.abs().sum() > 0


def test_config_rejects_bad_crops_and_removed_model_types():
    with pytest.raises(ValueError):
        C.load(None, SMALL+['model.crop=[8,16,10]'])  # 10 is not divisible by 4 (two stride-2 stages)
    with pytest.raises(ValueError):
        C.load(None, SMALL+['model.residual=false'])  # a removed model type: rejected, not silently dropped


def test_patch_directories_on_their_own_volumes_and_weights(tmp_path):
    """data.patch_volumes: each directory's episodes read its own volume; data.dir_weights: its sampling share."""
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import EpisodeSampler
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    dirs = [tmp_path/'a', tmp_path/'b']
    for d in dirs:
        _write_plane_patch(d/'p0', 100, 40)
    _write_volume(tmp_path/'vol_a.zarr', shape=(128, 64, 64))  # the sheet at the patches (level-1 x 30)
    _write_volume(tmp_path/'vol_b.zarr', shape=(128, 64, 64), sheet_x=5)  # far from them
    base = [f'data.patches_dir=["{dirs[0]}", "{dirs[1]}"]', f'data.volume="{tmp_path/"vol_a.zarr"}"', 'data.volume_level=1',
            f'data.patch_volumes=["", "{tmp_path/"vol_b.zarr"}"]', 'data.front_points=3', 'data.point_spacing=20',
            'data.step=10', 'data.steps=2', 'data.gt_margin=10', 'data.gt_resolution=2', 'data.roi=512',
            'data.val_fraction=0', 'data.min_valid_fraction=.5']
    index = build_index(C.load(None, base).data.patches_dir, tmp_path/'cache')
    rng = np.random.default_rng(0)
    for weights, root, peak in (('[1, 0]', dirs[0], 255), ('[0, 1]', dirs[1], 0)):
        cfg = C.load(None, base+[f'data.dir_weights={weights}'])
        sampler = EpisodeSampler(cfg.data, index, 'train', 2, 512, 10.)
        episodes = [e for e in (sampler.sample(rng) for _ in range(8)) if e is not None]
        assert episodes and all(Path(e['meta']['root']) == root and e['block'].max() == peak for e in episodes)
        nxt = sampler.continuation(dict(meta=episodes[0]['meta'], shift=10., front_lo=np.full(3, 50.),
                                        front_hi=np.full(3, 70.)), .5)
        assert nxt is not None and nxt['block'].max() == peak  # continuations stay on their directory's volume
    with pytest.raises(ValueError):
        C.load(None, base+['data.patch_volumes=["x"]'])  # one entry per directory


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_device_batches_match_to_device():
    from vesuvius.neural_tracing.tracers.sheets.evaluation.rollouts import to_device
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import pack_blocks
    from vesuvius.neural_tracing.tracers.sheets.train.train import device_batches
    g = torch.Generator().manual_seed(0)
    blocks = lambda: [np.random.default_rng(i).integers(0, 255, (1, 8-i, 6+i, 7), dtype=np.uint8) for i in range(2)]
    batches = [dict(gt=torch.rand(2, 5, 4, 3, generator=g), valid=torch.rand(2, 5, 4, generator=g) > .5,
                    block=pack_blocks(blocks()), origin=torch.randint(0, 9, (2, 3), generator=g),
                    meta=[{}, {}]) for _ in range(3)]
    device = torch.device('cuda')
    got = list(device_batches(iter(batches), device))
    assert len(got) == 3
    for b, want in zip(got, (to_device(b, device) for b in batches)):
        assert all(torch.equal(b[k], want[k]) for k in ('gt', 'valid', 'block', 'origin'))

    def failing():
        yield batches[0]
        raise RuntimeError('loader died')
    with pytest.raises(RuntimeError, match='loader died'):
        list(device_batches(failing(), device))  # loader errors reach the trainer instead of hanging it


def test_seeds_from_reuses_earlier_seeds_with_another_front(tmp_path, monkeypatch):
    """--seeds-from: the same patch, orientation and anchor give the same windows; a front too long for the patch
    drops the seed."""
    from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds, seeds_from
    monkeypatch.setattr(C, 'OUTPUT', tmp_path)
    _write_plane_patch(tmp_path/'patches'/'p0', 100, 60)  # 2 vx cells: 200 vx by 120 vx
    cfg = C.load(None, [f'data.patches_dir="{tmp_path/"patches"}"', 'data.step=10', 'data.gt_margin=10',
                        'data.gt_resolution=2'])
    seeds, _, _ = draw_seeds(cfg, 12, 3, 40., .5, all_val=True, max_per_patch=0)
    assert seeds
    report = tmp_path/'report.json'
    report.write_text(json.dumps(dict(seeds=[w['meta'] for w in seeds])))
    again, _, _, kept = seeds_from(report, cfg, 3, 40., .5, all_val=True)
    assert kept == list(range(len(seeds)))
    assert all(np.array_equal(a['gt'], w['gt']) and np.array_equal(a['valid'], w['valid']) for a, w in zip(again, seeds))
    # 120 vx fits along the rows (198 vx) but not the columns (118 vx): seeds with the front along the columns
    # (axis 0) are dropped, the others keep their patch, orientation and anchor where the longer row still fits
    longer, _, _, kept = seeds_from(report, cfg, 3, 120., .5, all_val=True)
    assert 0 < len(longer) < len(seeds) and all(seeds[i]['meta']['axis'] == 1 for i in kept)
    assert all(w['meta'] == seeds[i]['meta'] for w, i in zip(longer, kept))


def test_mixed_front_batches_vary_the_point_count(tmp_path):
    """data.front_points_range: each batch has one front size N, its GT window and batch size follow N."""
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import MixedFrontStream
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    from vesuvius.neural_tracing.tracers.sheets.train.train import front_variants
    _write_plane_patch(tmp_path/'patches'/'p0', 100, 100)
    _write_volume(tmp_path/'vol.zarr', shape=(128, 128, 64))
    cfg = C.load(None, [f'data.patches_dir="{tmp_path/"patches"}"', f'data.volume="{tmp_path/"vol.zarr"}"', 'data.volume_level=1',
                        'data.front_points=3', 'data.point_spacing=20', 'data.step=10', 'data.steps=2',
                        'data.gt_margin=10', 'data.gt_resolution=2', 'data.roi=512',
                        'data.val_fraction=0', 'data.min_valid_fraction=.5',
                        'data.front_points_range=[3,5]', 'train.points_per_batch=12'])
    variants = front_variants(cfg, build_index(cfg.data.patches_dir, tmp_path/'cache'), 10.)
    assert sorted(variants) == [3, 4, 5] and [v[2] for v in variants.values()] == [4, 3, 2]
    stream = iter(MixedFrontStream({n: v[1] for n, v in variants.items()}, {n: v[2] for n, v in variants.items()}, 0))
    seen = set()
    for _ in range(12):
        b = next(stream)
        n = b['front_points']
        spec = variants[n][1].spec
        seen.add(n)
        assert b['gt'].shape == (variants[n][2], spec.Gu, spec.Gv, 3) and variants[n][0].data.front_points == n
    assert seen == {3, 4, 5}
    with pytest.raises(ValueError):
        C.load(None, ['data.front_points_range=[2,5]'])


def test_packed_blocks_unpack_padded_or_listed_and_sample_alike():
    """Packed blocks ship no padding; unpacked, the zero-padded batch is the blocks padded at the far end, and crops
    from the list of blocks (data.unpadded_blocks) match those from the padded batch."""
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import pack_blocks, unpack_blocks
    rng = np.random.default_rng(0)
    blocks = [rng.integers(0, 255, shape, dtype=np.uint8) for shape in ((1, 20, 24, 18), (1, 16, 30, 22))]
    packed = pack_blocks(blocks)
    assert packed['data'].numel() == sum(b.size for b in blocks)
    want = np.zeros((2, 1, 20, 30, 22), np.uint8)
    for i, b in enumerate(blocks):
        want[i, :, :b.shape[1], :b.shape[2], :b.shape[3]] = b
    assert torch.equal(unpack_blocks(packed, 'cpu'), torch.from_numpy(want).float())
    origin = torch.tensor([[3, 5, 7], [1, 2, 4]])
    pos = torch.tensor([[[12., 14., 13.], [15., 12., 11.]], [[10., 16., 9.], [14., 18., 12.]]])  # inside each block
    frames = torch.eye(3).expand(2, 2, 3, 3)
    offsets = crop_offsets((4, 6, 5), 1., 0., 'cpu')
    padded = sample_crops(unpack_blocks(packed, 'cpu'), origin, 0, pos, frames, offsets)
    listed = sample_crops(unpack_blocks(packed, 'cpu', pad=False), origin, 0, pos, frames, offsets)
    assert listed.shape == padded.shape and torch.allclose(listed, padded, atol=1e-4)


def test_held_out_scoring_seeds_and_mask(tmp_path, monkeypatch):
    """--held-out-scoring: seeds need only their seed rows in val cells (their reach may cross train cells), and
    in_split marks exactly the points in val cells."""
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import split_of
    from vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark import draw_seeds, in_split
    monkeypatch.setattr(C, 'OUTPUT', tmp_path)
    _write_plane_patch(tmp_path/'patches'/'p0', 200, 200)
    cfg = C.load(None, [f'data.patches_dir="{tmp_path/"patches"}"', 'data.step=10', 'data.gt_margin=10',
                        'data.gt_resolution=2', 'data.split_block=48', 'data.val_fraction=.5'])
    seeds, spec, data = draw_seeds(cfg, 20, 6, 40., .5, max_per_patch=0, seed_split=True)
    assert seeds
    iu0 = int(round(-spec.u_lo/spec.res))
    v0, v1 = int(round(-spec.v_lo/spec.res)), int(round((40.-spec.v_lo)/spec.res))
    crossing = 0
    for w in seeds:
        row = w['gt'][iu0, v0:v1+1][w['valid'][iu0, v0:v1+1]]  # the seed row: v = 0 .. front
        assert in_split(torch.from_numpy(row), data).all()  # the seed row is held out
        crossing += not in_split(torch.from_numpy(w['gt'][w['valid']]), data).all()
    assert crossing  # some windows reach into training cells, which scoring then ignores
    pts = torch.tensor([[1., 2., 3.], [50., 2., 3.], [100., 200., 7.]])
    want = [split_of(tuple(int(v)//48 for v in p.tolist()), .5, 0) == 'val' for p in pts]
    assert in_split(pts, data).tolist() == want


def test_resample_isotropic_squares_the_grid(tmp_path):
    mod = _script('resample_isotropic')
    src = tmp_path/'src'/'p0'
    src.mkdir(parents=True)
    r, c = np.meshgrid(np.arange(30), np.arange(80), indexing='ij')  # rows 20 vx apart, columns 5 vx apart
    xyz = dict(x=np.full(r.shape, 60.5), y=10.+5*c, z=10.+20*r)
    xyz['x'][3, 8] = -1  # the node new column 2 starts from
    for name, values in xyz.items():
        tifffile.imwrite(src/f'{name}.tif', values.astype(np.float32))
    (src/'meta.json').write_text(json.dumps(dict(scale=[.05, .05])))
    mod.main([str(tmp_path/'src'), str(tmp_path/'out')])
    out = np.stack([tifffile.imread(tmp_path/'out'/'p0'/f'{n}.tif') for n in 'xyz'], -1)
    ok = np.all(out != -1, -1)
    assert out.shape[:2] == (30, 20) and np.allclose(mod.steps(out, ok), (20., 20.))  # 79 * 5 / 20 + 1 columns
    assert not ok[3, 2] and ok.sum() == 30*20-1  # only the new node built from the invalid one is dropped
    assert json.loads((tmp_path/'out'/'p0'/'meta.json').read_text())['scale'] == [.05, .05]


def test_written_tifxyz_records_its_area(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.tracing.inference import surface_area_vx2, write_surface
    r, c = np.meshgrid(np.arange(5), np.arange(4), indexing='ij')
    rows = np.stack([np.full(r.shape, 7.), 10.*c, 20.*r], -1).astype(np.float32)  # 4 x 3 quads of 10 x 20 vx
    assert surface_area_vx2(rows) == pytest.approx(12*200.)
    keep = np.ones(r.shape, bool)
    keep[0, 0] = False  # one corner node: its quad goes
    write_surface(tmp_path/'s', rows, keep, 20., 10., voxel_size_um=10.)
    meta = json.loads((tmp_path/'s'/'meta.json').read_text())
    assert meta['area_vx2'] == pytest.approx(11*200.) and meta['area_cm2'] == pytest.approx(11*200.*100/1e8)


def test_fiber_seed_directions_follow_the_tensor_normal():
    """trace_from_fibers: the 2/8 structure-tensor normal of a tilted bright sheet, and an advance direction in the
    sheet, perpendicular to the fiber, pointing up."""
    mod = _script('trace_from_fibers')
    n = np.array([1., 0., .5])/np.linalg.norm([1., 0., .5])  # sheet normal (xyz), tilted from horizontal x
    z, y, x = np.meshgrid(*(np.arange(160.),)*3, indexing='ij')
    vol = (255*np.exp(-.5*((x*n[0]+y*n[1]+z*n[2]-100.)/1.5)**2)).astype(np.uint8)

    class Volume:
        shape = vol.shape

        def read(self, start, size):
            return vol[tuple(slice(a, a+b) for a, b in zip(start, size))]
    pts = np.stack([np.full(5, 80.), 60.+10*np.arange(5), np.full(5, (100.-80.*n[0])/n[2])], 1)  # on it, along y
    normal = mod.sheet_normal(Volume(), pts[2])
    assert normal is not None and abs(normal@n) > .99
    d = mod.advance_directions(pts, 1., Volume())
    assert d is not None and np.all(d[:, 2] > 0) and np.allclose(d@n, 0, atol=.05) and np.allclose(d[:, 1], 0, atol=.05)


def test_fiber_patch_matching_keeps_fibers_on_or_inside_the_sheet():
    """match_fibers_to_patches: a fiber touching or slightly inward of a patch (towards the umbilicus) matches it;
    one outward of it (the patch would be a sheet further in) does not."""
    from scipy.spatial import cKDTree
    mod = _script('match_fibers_to_patches')
    r, c = np.meshgrid(np.arange(40), np.arange(40), indexing='ij')  # patch x = 100, umbilicus at x = 0: outward is +x
    pos = np.stack([np.full(r.size, 100.), 20.*c.ravel(), 20.*r.ravel()], 1).astype(np.float32)
    nrm = np.tile(np.array([[1., 0., 0.]], np.float32), (r.size, 1))

    class Fibers:
        def __init__(self, x):
            self.catalog = [(7,)]
            self.x = x

        def geometry(self, i):
            from types import SimpleNamespace
            return SimpleNamespace(points=np.stack([np.full(200, self.x), 100.+2*np.arange(200.), np.full(200, 300.)], 1))
    mod._STATE.update(tree=cKDTree(pos), pos=pos, nrm=nrm, rc=np.stack([r.ravel(), c.ravel()], 1), patch=np.zeros(r.size, int),
                      cell=np.array([20.], np.float32), sample=5., candidates=8, max_inward=4., max_outward=1.,
                      max_tangential=.75, min_run=240., reach=20., roots=['root'], names=['p0'])
    for x, matches in ((100., True), (97.5, True), (102.5, False), (94., False)):
        mod._STATE['fibers'] = Fibers(x)
        runs = mod.match_fiber(0)
        assert bool(runs) == matches, x
        if runs:
            assert runs[0]['patch'] == 'p0' and runs[0]['end']-runs[0]['start'] >= 240 and abs(runs[0]['offset']-(x-100)) < 1e-3


def test_rotated_window_matches_axis_window_and_follows_a_diagonal(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.data.patches import PatchStore, build_index
    _write_plane_patch(tmp_path/'patches'/'p0', 120, 120)
    info = build_index(tmp_path/'patches', tmp_path/'cache')[0]
    cfg = C.load(None, ['data.front_points=3', 'data.point_spacing=20', 'data.step=10', 'data.steps=2',
                        'data.gt_margin=10', 'data.gt_resolution=2'])
    spec = GridSpec.for_episode(cfg.data, 2)
    store = PatchStore()
    a, va = store.window(info, spec, 0, 1, 1, 40., 30.)
    b, vb = store.window_rotated(info, spec, (40., 30.), (1., 0.), (0., 1.))
    assert np.array_equal(va, vb) and np.allclose(a, b)
    d = np.array([1., 1.])/np.sqrt(2)  # front along the grid diagonal, advance perpendicular to it
    g, v = store.window_rotated(info, spec, (40., 40.), (-d[1], d[0]), d)
    iu0, iv0 = int(round(-spec.u_lo/spec.res)), int(round(-spec.v_lo/spec.res))
    k = int(round(20./spec.res))
    assert v.all() and np.allclose(np.linalg.norm(g[iu0, iv0+k]-g[iu0, iv0]), 20., atol=1e-3)
    assert np.allclose(g[iu0, iv0+k]-g[iu0, iv0], 20.*np.array([0., d[1], d[0]]), atol=1e-3)  # along (y=col, z=row)


def test_fiber_seeded_episodes_start_from_the_fiber(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import EpisodeSampler, collate
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    _write_plane_patch(tmp_path/'patches'/'p0', 150, 150)  # plane x = 60.5: y = 20 + 2 col, z = 20 + 2 row
    _write_volume(tmp_path/'vol.zarr', shape=(256, 256, 64))
    t = np.arange(0., 120., 5.)  # a fiber 2 vx inside the sheet, running diagonally (row and col both grow)
    xyz = np.stack([np.full(len(t), 58.5), 100.+t*.8, 120.+t*.6], 1)
    rc = np.stack([(xyz[:, 2]-20.)/2, (xyz[:, 1]-20.)/2], 1)
    (tmp_path/'runs.jsonl').write_text(json.dumps(dict(fiber='f1', root=str(tmp_path/'patches'), patch='p0', start=0.,
                                                        end=float(t[-1]), offset=-2., rc=rc.tolist(), xyz=xyz.tolist()))+'\n')
    cfg = C.load(None, [f'data.patches_dir="{tmp_path/"patches"}"', f'data.volume="{tmp_path/"vol.zarr"}"', 'data.volume_level=1',
                        'data.front_points=3', 'data.point_spacing=20', 'data.step=10', 'data.steps=2', 'data.gt_margin=10',
                        'data.gt_resolution=2', 'data.roi=512', 'data.val_fraction=0',
                        'data.min_valid_fraction=.5', f'data.fiber_runs="{tmp_path/"runs.jsonl"}"',
                        'data.fiber_fraction=1'])
    sampler = EpisodeSampler(cfg.data, build_index(cfg.data.patches_dir, tmp_path/'cache'), 'train', 2, 512, 10.)
    assert len(sampler.fiber_runs) == 1
    rng = np.random.default_rng(0)
    eps = [e for e in (sampler.sample(rng) for _ in range(6)) if e is not None]
    assert eps and all('seed' in e and e['meta']['fiber'] == 'f1' for e in eps)
    for e in eps:
        seed = e['seed']
        assert seed.shape == (cfg.data.history+1, 3, 3)
        assert np.allclose(seed[-1, :, 0], 58.5) and np.allclose(np.linalg.norm(np.diff(seed[-1], axis=0), axis=1), 20., atol=.1)
        step = seed[-1]-seed[-2]  # one step along the sheet, perpendicular to the fiber
        assert np.allclose(np.linalg.norm(step, axis=1), 10., atol=1e-3) and np.allclose(step[:, 0], 0, atol=1e-3)
        assert np.allclose((step*np.array([0., .8, .6])).sum(1), 0, atol=1e-3)
    b = collate(eps+[{k: v for k, v in eps[0].items() if k != 'seed'}])  # plus one GT-seeded episode
    assert b['seed'].shape[0] == len(eps)+1 and b['seed_mask'].tolist() == [True]*len(eps)+[False]


def test_train_rollout_starts_from_provided_seed_rows():
    from vesuvius.neural_tracing.tracers.sheets.tracing.seeding import seed_rows
    from vesuvius.neural_tracing.tracers.sheets.train.train import train_rollout
    cfg = C.load(None, SMALL+['data.steps=1'])
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    gt, valid = gt.expand(2, -1, -1, -1).contiguous(), valid.expand(2, -1, -1).contiguous()
    rows = seed_rows(gt, valid, cfg, spec, noise=False)
    given = rows+torch.tensor([0., 0., 1.5])  # fiber seeds: 1.5 vx up from the GT rows
    b = dict(gt=gt, valid=valid, block=torch.rand(2, 1, 224, 360, 360)*255, origin=torch.zeros(2, 3, dtype=torch.long),
             seed=given, seed_mask=torch.tensor([True, False]))
    tracer = Tracer(build_model(cfg), cfg, 'cpu')
    started = []
    real = tracer.initial
    tracer.initial = lambda seeds: started.append(seeds) or real(seeds)
    train_rollout(tracer, b, cfg, spec, torch.Generator().manual_seed(0))
    assert torch.allclose(started[0][0], given[0])  # the fiber-seeded episode starts on its seed rows
    assert not torch.allclose(started[0][1], given[1], atol=1e-3)  # the other on (noisy) GT rows


def test_fiber_follow_episodes_put_the_middle_column_on_the_fiber(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import EpisodeSampler
    from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
    _write_plane_patch(tmp_path/'patches'/'p0', 200, 200)  # plane x = 60.5: y = 20 + 2 col, z = 20 + 2 row
    t = np.arange(0., 200., 5.)  # a gently curving horizontal fiber on the sheet: along y, z bending slowly
    xyz = np.stack([np.full(len(t), 60.5), 60.+t, 200.+20*np.sin(t/120.)], 1)
    rc = np.stack([(xyz[:, 2]-20.)/2, (xyz[:, 1]-20.)/2], 1)
    (tmp_path/'runs.jsonl').write_text(json.dumps(dict(fiber='f1', root=str(tmp_path/'patches'), patch='p0', start=0.,
                                                        end=float(t[-1]), offset=0., rc=rc.tolist(), xyz=xyz.tolist()))+'\n')
    cfg = C.load(None, [f'data.patches_dir="{tmp_path/"patches"}"', 'data.front_points=3', 'data.point_spacing=20',
                        'data.step=10', 'data.steps=4', 'data.gt_margin=10', 'data.gt_resolution=2', 'data.roi=512',
                        'data.val_fraction=0', f'data.fiber_runs="{tmp_path/"runs.jsonl"}"',
                        'data.fiber_follow=true'])
    sampler = EpisodeSampler(cfg.data, build_index(cfg.data.patches_dir, tmp_path/'cache'), 'train', 4, 512, 10.,
                             region_split=True)
    rng = np.random.default_rng(0)
    eps = [e for e in (sampler.window(rng) for _ in range(6)) if e is not None]
    assert eps and all(e['meta']['follow'] for e in eps)
    spec = sampler.spec
    mid = int(round((20.-spec.v_lo)/spec.res))  # the middle point's column
    for e in eps:
        g, v = e['gt'], e['valid']
        col = g[:, mid][v[:, mid]]
        fine = np.arange(0., t[-1], .25)
        line = interp_at(xyz, t, fine)
        dist = np.min(np.linalg.norm(col[:, None]-line[None], axis=-1), axis=1)
        assert dist.max() < 1.  # the middle column follows the fiber
        rows = np.flatnonzero(v[:, mid] & v[:, mid-10] & v[:, mid+10])
        assert np.allclose(np.linalg.norm(g[rows, mid+10]-g[rows, mid], axis=1), 20., atol=1.5)  # 20 vx across it


def test_mode_embedding_starts_neutral_and_reaches_the_tokens():
    cfg = C.load(None, SMALL+['model.fiber_follow_mode=true'])
    model = build_model(cfg)
    assert torch.equal(model.mode_embed, torch.zeros_like(model.mode_embed))
    plain = build_model(C.load(None, SMALL))
    from vesuvius.neural_tracing.tracers.sheets.model import init_model_state
    init_model_state(model, plain.state_dict())  # an older checkpoint without the embedding loads
    with torch.no_grad():
        model.mode_embed[1] = torch.randn(model.mode_embed.shape[1])  # not constant: LayerNorm removes that
        torch.nn.init.normal_(model.head.weight, std=1e-2)  # the head starts at zero: make the output depend on input
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    from vesuvius.neural_tracing.tracers.sheets.tracing.seeding import seed_rows
    seeds = seed_rows(gt, valid, cfg, spec, noise=False)
    block, origin = torch.rand(1, 1, 224, 360, 360)*255, torch.zeros(1, 3, dtype=torch.long)
    outs = []
    for follow in (False, True):
        c = C.load(None, SMALL+['model.fiber_follow_mode=true', f'data.fiber_follow={str(follow).lower()}'])
        tracer = Tracer(model, c, 'cpu')
        with torch.no_grad():
            outs.append(tracer.step(tracer.initial(seeds), block, origin).pred)
    assert not torch.allclose(outs[0], outs[1])  # mode 1 changes the prediction, mode 0 is the plain model


def test_fiber_pool_region_and_episode_instances(tmp_path):
    from vesuvius.neural_tracing.tracers.sheets.data.fibers import episode_fibers, fiber_pool
    mod = _script('build_fiber_pool')
    fibers = [('json:a', 0, *mod.resample(np.array([[100., 100., 100.], [100., 400., 100.]]), 5.)),
              ('afv:7', 1, *mod.resample(np.array([[110., 200., 0.], [110., 200., 300.]]), 5.)),
              ('afv:8', 0, *mod.resample(np.array([[900., 900., 900.], [900., 1000., 900.]]), 5.))]
    pts = np.concatenate([f[2] for f in fibers]).astype(np.float32)
    keys = mod.cell_key(pts)
    order = np.argsort(keys, kind='stable')
    cells, starts = np.unique(keys[order], return_index=True)
    out = tmp_path/'pool'
    out.mkdir()
    for name, a in dict(points=pts[order], tangent=np.concatenate([f[3] for f in fibers]).astype(np.float32)[order],
                        arc=np.concatenate([f[4] for f in fibers]).astype(np.float32)[order],
                        instance=np.concatenate([np.full(len(f[2]), k, np.int32) for k, f in enumerate(fibers)])[order],
                        family=np.concatenate([np.full(len(f[2]), f[1], np.int8) for f in fibers])[order],
                        cell_keys=cells, cell_starts=np.append(starts, len(pts)).astype(np.int64)).items():
        np.save(out/f'{name}.npy', a)
    (out/'keys.json').write_text(json.dumps([f[0] for f in fibers]))
    pool = fiber_pool(str(out))
    got = episode_fibers(pool, [0., 0., 0.], [300., 500., 300.], 'json:a', lambda arc: arc <= 150.)
    seed = got['instance'] == 0
    assert seed.any() and got['points'][seed][:, 1].max() <= 250.+1e-3  # the seed fiber, cut 150 vx along it
    assert set(got['instance'].tolist()) == {0, 1} and (got['family'][got['instance'] == 1] == 1).all()  # afv:7 only
    assert np.all(got['points'] <= 500.)  # afv:8 is outside the box


def test_fiber_tokens_reach_the_model_and_padding_is_inert():
    from vesuvius.neural_tracing.tracers.sheets.data.episodes import FIBER_CODE_DIM
    from vesuvius.neural_tracing.tracers.sheets.tracing.seeding import seed_rows
    cfg = C.load(None, SMALL+['model.fiber_tokens=4'])
    torch.manual_seed(0)
    model = build_model(cfg)
    torch.nn.init.normal_(model.head.weight, std=1e-2)
    spec = GridSpec.for_episode(cfg.data, cfg.data.steps)
    gt, valid = cylinder_grid(spec)
    seeds = seed_rows(gt, valid, cfg, spec, noise=False)
    block, origin = torch.rand(1, 1, 224, 360, 360, generator=torch.Generator().manual_seed(1))*255, torch.zeros(1, 3, dtype=torch.long)
    tracer = Tracer(model, cfg, 'cpu')
    front = tracer.initial(seeds)
    near = front.pos[0, :3]+torch.tensor([0., 0., 5.])
    codes = torch.nn.functional.normalize(torch.randn(1, 2, FIBER_CODE_DIM, generator=torch.Generator().manual_seed(2)), dim=-1)
    fibers = dict(fiber_points=torch.cat([near, torch.zeros(2, 3)])[None], fiber_tangent=torch.tensor([[[0., 1., 0.]]*5]),
                  fiber_family=torch.zeros(1, 5, dtype=torch.long), fiber_instance=torch.tensor([[0, 0, 1, -1, -1]]),
                  fiber_codes=codes)
    empty = dict(fibers, fiber_instance=torch.full((1, 5), -1))
    with torch.no_grad():
        plain = tracer.step(front, block, origin).pred
        none = tracer.step(front, block, origin, empty).pred
        some = tracer.step(front, block, origin, fibers).pred
        swapped = tracer.step(front, block, origin, dict(fibers, fiber_codes=codes.flip(1))).pred
    assert torch.allclose(plain, none, atol=1e-5)  # only padding: as without fiber tokens
    assert not torch.allclose(plain, some, atol=1e-4)  # fibers change the prediction
    assert not torch.allclose(some, swapped, atol=1e-5)  # and which instance a point belongs to matters


def test_extended_fiber_json_is_a_valid_vc3d_fiber():
    """trace_from_fibers.extended_fiber: the source line up to the start, then the kept middle points; control
    points are an ordered subset of the line (VC3D's validateFiberInputControlPoints)."""
    mod = _script('trace_from_fibers')
    line = np.stack([np.linspace(0., 400., 81), np.zeros(81), np.zeros(81)], 1)  # model voxels, along x
    T, C = 6, 11
    rows = np.zeros((T+1, C, 3))
    for t in range(T+1):  # the trace starts at x = 380 and grows past the end, the middle column on the fiber line
        rows[t, :, 0] = 380.+20*t
        rows[t, :, 2] = np.linspace(-60., 60., C)
    confs = np.ones((T+1, C))
    confs[5:, C//2] = 0.  # the middle point is lost from row 5 on
    fib = mod.extended_fiber(dict(filename='src.json'), line, 380., 1., rows, confs, .4, .25, 'x')
    pts = np.asarray(fib['line_points'])*.25
    assert fib['type'] == 'vc3d_fiber' and fib['optimization_mode'] == 'lasagna' and fib['extension_points'] == 4
    assert np.allclose(pts[-1], [460., 0., 0.]) and np.all(np.diff(pts[:, 0]) > 0)  # ordered, ends at row 4
    assert fib['control_points'] == [fib['line_points'][0], fib['line_points'][-1]]
    back = mod.extended_fiber(dict(), line, 20., -1., rows*np.array([-1., 1., 1.])+np.array([400., 0., 0.]), confs, .4, .25, 'y')
    assert np.allclose(np.asarray(back['line_points'][-1])*.25, line[-1])  # the extension comes first
