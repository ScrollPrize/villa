"""Long held-out traces through the inference path, several checkpoints on identical seeds.

Seeds are drawn from the validation split: a GT window whose seed row and the ``--steps`` rows ahead lie in
val split cells, with at least ``--min-valid`` of that region valid. Every checkpoint traces
the same seed polyline (``--front`` voxels of the GT seed row, re-sampled to that model's point spacing) with
the same advance direction, reading the volume around the front each step (``infer.trace``; no early stop).
Each traced point (front point or Hermite curve sample) is scored against the GT window like ``evaluate``:

  same_sheet / kept / coverage / precision   as in evaluate, per step over all seeds
  dist_p50, advance_p50                      on-sheet distance and |advance error| of kept on-sheet points
  first_real_switch                          per trace: first step with a real switch, i.e. a row with at least
                                             REAL_SWITCH_POINTS kept points off the seed's sheet, or any kept point
                                             farther than REAL_SWITCH_DISTANCE from it (the next sheet is ~12 vx away)
  first_switch                               per trace: first step with any kept point off the seed's sheet (also
                                             counts single points just past the same-sheet tolerance)
  off_kept_points                            kept points off the seed's sheet, all traces
  lost                                       per trace: first step where under half the points are on the sheet

``--rescore DIR ...`` recomputes the switch counts of existing benchmark folders from their tifxyz (no tracing) and
adds them to their report.json.

Outputs in ``output/trace_bench/TAG``: report.json, per_step.png, seed_NN.png, and tifxyz/ with ``seedNN_gt`` (the
GT window) and ``seedNN_MODEL`` (each trace; stopped points written as -1) for inspection in VC3D.

Usage:
  python -m vesuvius.neural_tracing.sheet_tracing.trace_benchmark --checkpoints output/A/best.pt output/B/best.pt \\
      --seeds 12 --steps 30 --tag ab
  python -m vesuvius.neural_tracing.sheet_tracing.trace_benchmark --rescore output/trace_bench/ab
  # checkpoints trained elsewhere: map their recorded data paths (volumes, patch directories)
  python -m vesuvius.neural_tracing.sheet_tracing.trace_benchmark --checkpoints remote/A/last.pt \\
      --path-map /remote/data/s1_ds2.zarr=/local/s1_ds2.zarr --path-map /remote/patches=/local/patches --out DIR
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray
from vesuvius.neural_tracing.sheet_tracing import config as C
from vesuvius.neural_tracing.sheet_tracing.data.episodes import EpisodeSampler, split_of
from vesuvius.neural_tracing.sheet_tracing.data.patches import build_index
from vesuvius.neural_tracing.sheet_tracing.evaluate import load_checkpoint
from vesuvius.neural_tracing.sheet_tracing.geometry import grid_eval, surface_frame
from vesuvius.neural_tracing.sheet_tracing.infer import connected_keep, read_block, resample, trace, write_surface
from vesuvius.neural_tracing.sheet_tracing.losses import landing
from vesuvius.neural_tracing.sheet_tracing.sampling import sample_plane


def _seed_sampler(cfg, steps: int, front: float, min_valid: float, split: str, all_val: bool, seed_split: bool):
    """Sampler of benchmark seed windows (20 vx point spacing for checks) and its data config. ``seed_split``: only
    the seed rows must lie in ``split`` cells (``--held-out-scoring``)."""
    data = copy.deepcopy(cfg.data)
    if all_val:
        data.val_fraction = 1.
    data.point_spacing = 20.
    data.front_points = int(round(front/data.point_spacing))+1
    data.min_valid_fraction = min_valid
    data.max_tries = 2000
    sampler = EpisodeSampler(data, build_index(data.patches_dir, C.OUTPUT/'cache'), split, steps,
                             roi=0, pad=0, region_split=True, seed_split=seed_split)
    return sampler, data


def draw_seeds(cfg, count: int, steps: int, front: float, min_valid: float, seed: int = 0, split: str = 'val',
               all_val: bool = False, max_per_patch: int = 1, patience: int = 500, seed_split: bool = False):
    """GT windows of ``split`` for traces of ``steps`` steps and a ``front``-voxel seed row. At most
    ``max_per_patch`` seeds per patch (0: any number); fewer than ``count`` if not enough patches fit, giving up
    after ``patience`` draws in a row add nothing. ``all_val``: every split cell is a validation cell (patches of a
    scroll never trained on)."""
    max_per_patch = max_per_patch or count
    sampler, data = _seed_sampler(cfg, steps, front, min_valid, split, all_val, seed_split)
    rng = np.random.default_rng([seed, 7])
    seeds, per_patch, misses = [], {}, 0
    for _ in range(count*20):
        w = sampler.window(rng)
        key = None if w is None else (w['meta'].get('root'), w['meta']['patch'])
        if w is not None and per_patch.get(key, 0) < max_per_patch:
            per_patch[key] = per_patch.get(key, 0)+1
            seeds.append(w)
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break
        if len(seeds) == count:
            break
    return seeds, sampler.spec, data


def seeds_from(report: Path, cfg, steps: int, front: float, min_valid: float, split: str = 'val',
               all_val: bool = False, seed_split: bool = False):
    """The seeds of an earlier benchmark (same patch, orientation and anchor: the seed row starts at the same place
    and runs the same way) as windows for this ``steps`` and ``front``, e.g. a longer front with more points. Seeds
    whose window no longer passes the validity and split checks are dropped. Returns the windows, spec, data
    config and the kept seeds' indices in the earlier report."""
    sampler, data = _seed_sampler(cfg, steps, front, min_valid, split, all_val, seed_split)
    by_key = {(info.root, info.name): info for info in sampler.index}
    seeds, kept = [], []
    for i, meta in enumerate(json.loads(Path(report).read_text())['seeds']):
        info = by_key.get((meta.get('root'), meta['patch']))
        w = None if info is None else sampler._episode(info, meta['axis'], meta['sign_u'], meta['sign_v'],
                                                       meta['anchor'])
        if w is not None:
            seeds.append(w)
            kept.append(i)
    return seeds, sampler.spec, data, kept


def seed_polyline(w, spec, data):
    """GT seed row (every grid node from v = 0 to the front length) and the per-point advance direction."""
    iu0 = int(round((0-spec.u_lo)/spec.res))
    back = int(round(data.step/spec.res))
    v0 = int(round((0-spec.v_lo)/spec.res))
    v1 = int(round(((data.front_points-1)*data.point_spacing-spec.v_lo)/spec.res))
    row, behind = w['gt'][iu0, v0:v1+1], w['gt'][iu0-back, v0:v1+1]
    return row.astype(np.float64), (row-behind).astype(np.float64)


REAL_SWITCH_POINTS = 3  # kept off-sheet points in one row that make a real switch
REAL_SWITCH_DISTANCE = 9.  # ... or one kept point this far from the seed's sheet (3/4 of the way to the next)


def switch_steps(off_kept: torch.Tensor, dist: torch.Tensor):
    """First step (1-based) with any kept off-sheet point, and with a real switch, or None; (T, C) inputs."""
    real = (off_kept.sum(1) >= REAL_SWITCH_POINTS) | (off_kept & (dist > REAL_SWITCH_DISTANCE)).any(1)
    first = lambda rows: int(rows.nonzero()[0, 0])+1 if rows.any() else None
    return first(off_kept.any(1)), first(real)


def in_split(points: torch.Tensor, data, split: str = 'val') -> torch.Tensor:
    """Which level-0 points (..., 3) lie in ``split`` cells of ``data``'s spatial split."""
    cells = torch.floor(points/data.split_block).long()
    flat = cells.reshape(-1, 3)
    uniq, inverse = torch.unique(flat, dim=0, return_inverse=True)
    keep = torch.tensor([split_of(tuple(c), data.val_fraction, data.split_seed) == split for c in uniq.tolist()],
                        device=points.device)
    return keep[inverse].reshape(points.shape[:-1])


def score(rows, confs, w, spec, cfg, device, traced: int | None = None, held=None):
    """Per-step arrays (steps,) of counts for one trace, from step 1 on.

    ``traced``: steps actually traced (the rest is padding after an early stop). Padded rows count against
    coverage (no output there) but are excluded from same-sheet / caught / lost, which describe the trace itself.
    """
    gt = torch.from_numpy(w['gt']).to(device)[None]
    valid = torch.from_numpy(w['valid']).to(device)[None]
    pts = torch.from_numpy(rows[1:].astype(np.float32)).to(device)
    T, Ccols = pts.shape[:2]
    uv, dist, same, ok = landing(pts.reshape(1, -1, 3), gt, valid, cfg.loss.sheet_tolerance)
    uv, dist, same, ok = uv.reshape(T, Ccols, 2), dist.reshape(T, Ccols), same.reshape(T, Ccols), ok.reshape(T, Ccols)
    if held is not None:  # --held-out-scoring: only points whose GT foot point lies in a val cell of ``held`` count
        foot = grid_eval(gt, valid, uv.reshape(1, -1, 2))[0].reshape(T, Ccols, 3)
        ok = ok & in_split(foot, held)
    expected = (0-spec.u_lo)/spec.res+torch.arange(1, T+1, device=device)[:, None]*cfg.data.step/spec.res
    advance = ((uv[..., 0]-expected)*spec.res).abs()
    kept = torch.from_numpy(connected_keep(confs, cfg.trace.confidence_threshold)[1:]).to(device)
    alive = (torch.arange(1, T+1, device=device) <= (T if traced is None else traced))[:, None]
    seen = ok & alive  # traced rows with GT
    out = dict(ok=ok.sum(1), same=(seen & same).sum(1), seen=seen.sum(1), kept=(ok & kept).sum(1),
               cover=(ok & kept & same).sum(1), off=(seen & ~same).sum(1), caught=(seen & ~same & ~kept).sum(1))
    out = {k: v.cpu().numpy() for k, v in out.items()}
    good = ok & kept & same
    out['dist'] = [dist[t][good[t]].cpu().numpy() for t in range(T)]
    out['advance'] = [advance[t][good[t]].cpu().numpy() for t in range(T)]
    off_kept = ok & kept & ~same
    out['first_switch'], out['first_real_switch'] = switch_steps(off_kept, dist)
    out['off_kept_points'] = int(off_kept.sum())
    frac = (seen & same).sum(1).float()/seen.sum(1).clamp_min(1)
    lost = ((frac < .5) & (seen.sum(1) > 0)).nonzero()
    out['lost'] = int(lost[0, 0])+1 if len(lost) else None
    out['uv'] = uv.cpu().numpy()
    out['dist_all'] = np.where(ok.cpu().numpy(), dist.cpu().numpy(), np.nan)
    out['kept_mask'] = kept.cpu().numpy()
    out['stopped'] = None if traced is None or traced >= T else traced+1  # first step not traced
    return out


def summarize(scores, steps):
    tot = {k: np.sum([s[k] for s in scores], 0) for k in ('ok', 'seen', 'same', 'kept', 'cover', 'off', 'caught')}
    div = lambda a, b: np.where(b > 0, a/np.maximum(b, 1), np.nan)
    q = lambda key, p: np.array([float(np.quantile(x, p)) if len(x) else np.nan for x in
                                 [np.concatenate([s[key][t] for s in scores]) for t in range(steps)]])
    per_step = dict(same_sheet=div(tot['same'], tot['seen']), kept=div(tot['kept'], tot['ok']),
                    coverage=div(tot['cover'], tot['ok']), precision=div(tot['cover'], tot['kept']),
                    caught=div(tot['caught'], tot['off']), dist_p50=q('dist', .5), advance_p50=q('advance', .5))
    switches = [s['first_switch'] for s in scores]
    reals = [s['first_real_switch'] for s in scores]
    losts = [s['lost'] for s in scores]
    summary = {k+'_mean': float(np.nanmean(v)) for k, v in per_step.items()}
    summary.update({k+'_last': float(v[-1]) for k, v in per_step.items()})
    summary.update(switch_summary(switches, reals, sum(s['off_kept_points'] for s in scores), steps))
    summary.update(points_scored=int(tot['ok'].sum()), points_kept=int(tot['kept'].sum()),
                   points_covered=int(tot['cover'].sum()))  # counts behind coverage / precision, all steps
    summary['traces_lost'] = int(sum(x is not None for x in losts))
    stops = [s['stopped'] for s in scores]
    summary['traces_stopped'] = int(sum(x is not None for x in stops))
    summary['median_stop'] = float(np.median([x for x in stops if x is not None])) if any(stops) else None
    return per_step, summary, switches, reals, losts


def switch_summary(switches, reals, off_points: int, steps: int) -> dict:
    return dict(traces_with_switch=int(sum(x is not None for x in switches)),
                traces_real_switch=int(sum(x is not None for x in reals)),
                median_first_switch=float(np.median([x if x is not None else steps+1 for x in switches])),
                median_first_real_switch=float(np.median([x if x is not None else steps+1 for x in reals])),
                off_kept_points=int(off_points))


def _read_xyz(path: Path) -> np.ndarray:
    return np.stack([tifffile.imread(path/f'{c}.tif') for c in 'xyz'], -1).astype(np.float32)


def rescore(folder: Path, tolerance: float = C.LossConfig.sheet_tolerance) -> None:
    """Switch counts of an existing benchmark folder from its tifxyz (kept points are the written ones), added to
    its report.json; checks that the recomputed ``first_switch`` matches the stored one."""
    report = json.loads((folder/'report.json').read_text())
    if report.get('held_out_scoring'):
        raise ValueError(f'{folder}: --rescore does not support --held-out-scoring reports')
    steps = report['steps']
    for name, entry in report['models'].items():
        switches, reals, off_points = [], [], 0
        for i in range(len(report['seeds'])):
            gt_dir = folder/'tifxyz'/f'seed{i:02d}_gt'
            gt = _read_xyz(gt_dir if gt_dir.is_dir() else folder/'tifxyz'/'gt'/gt_dir.name)  # GT moved aside
            rows = _read_xyz(folder/'tifxyz'/f'seed{i:02d}_{name}')[1:]
            kept = torch.from_numpy((rows != -1).all(-1))
            _, dist, same, ok = landing(torch.from_numpy(rows).reshape(1, -1, 3), torch.from_numpy(gt)[None],
                                        torch.from_numpy((gt != -1).all(-1))[None], tolerance)
            dist, same, ok = (x.reshape(kept.shape) for x in (dist[0], same[0], ok[0]))
            off_kept = ok & kept & ~same
            first, real = switch_steps(off_kept, dist)
            switches.append(first)
            reals.append(real)
            off_points += int(off_kept.sum())
        if switches != entry['first_switch']:
            raise ValueError(f'{folder.name}/{name}: recomputed first switches {switches} differ from the report\'s '
                             f'{entry["first_switch"]}')
        entry['summary'].update(switch_summary(switches, reals, off_points, steps))
        entry['first_real_switch'] = reals
        s = entry['summary']
        print(f"{folder.name} {name}: real switches {s['traces_real_switch']}/{len(switches)} (any off-sheet point "
              f"{s['traces_with_switch']}, {off_points} points) lost {s['traces_lost']}/{len(switches)} coverage "
              f"{s['coverage_mean']:.3f} precision {s['precision_mean']:.4f}", flush=True)
    (folder/'report.json').write_text(json.dumps(report, indent=2))


def seed_figure(path, w, spec, cfg0, results, names, steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt, valid = torch.from_numpy(w['gt'])[None], torch.from_numpy(w['valid'])[None]
    n_models = len(names)
    fig, axes = plt.subplots(1, n_models+1, figsize=(4.6*n_models+9, 5.2),
                             gridspec_kw=dict(width_ratios=[1.]*n_models+[2.2]))
    ext = (spec.v_lo, spec.v_lo+spec.res*(spec.Gv-1), spec.u_lo, spec.u_lo+spec.res*(spec.Gu-1))
    for ax, name in zip(axes, names):
        r = results[name]
        ax.imshow(w['valid'], cmap='Greys_r', vmin=-1, vmax=1.5, origin='lower', extent=ext, aspect='auto')
        uv, d, kept = r['uv'], r['dist_all'], r['kept_mask']
        v, u = spec.v_lo+spec.res*uv[..., 1], spec.u_lo+spec.res*uv[..., 0]
        sc = ax.scatter(v[kept], u[kept], c=d[kept], cmap='plasma', vmin=0, vmax=12, s=6, marker='o')
        ax.scatter(v[~kept], u[~kept], c=d[~kept], cmap='plasma', vmin=0, vmax=12, s=14, marker='x')
        sw = r['first_switch']
        ax.set_title(f'{name}\nfirst switch {sw if sw else "-"}, lost {r["lost"] or "-"}', fontsize=8)
        ax.set_xlabel('v (vx)')
    axes[0].set_ylabel('u advance (vx)')
    fig.colorbar(sc, ax=axes[n_models-1], fraction=.046, label='distance to GT sheet (vx)')
    # CT along the advance through the seed's middle column, GT curve and each model's middle column.
    d = cfg0.data
    v_mid = ((d.front_points-1)*d.point_spacing/2-spec.v_lo)/spec.res
    iu = torch.arange(spec.Gu, dtype=torch.float32)
    S, Su, Sv, ok = grid_eval(gt, valid, torch.stack([iu, torch.full_like(iu, v_mid)], -1)[None])
    mid_u = ((steps*d.step/2)-spec.u_lo)/spec.res
    Sm, Sum, Svm, _ = grid_eval(gt, valid, torch.tensor([[[mid_u, v_mid]]]))
    t, a, n = surface_frame(Sum, Svm)[0, 0].T
    centre = Sm[0, 0]
    span = steps*d.step+120
    volume = ChunkedArray(f'{d.volume}/{d.volume_level}', 256 << 20)
    pts_all = np.concatenate([S[0][ok[0]].numpy()]+[results[nm]['rows'].reshape(-1, 3) for nm in names])
    block, origin = read_block(volume, pts_all, 60, d.volume_level)
    img = sample_plane(torch.from_numpy(block).float()[None], torch.from_numpy(origin), d.volume_level, centre,
                       torch.stack([n, a]), (100., span), 1.)
    ax = axes[-1]
    ax.imshow(img.numpy(), cmap='gray', extent=(-span/2, span/2, 50, -50), aspect='auto')
    plane = lambda p: (((p-centre)*a).sum(-1).numpy(), ((p-centre)*n).sum(-1).numpy())
    x, y = plane(S[0][ok[0]])
    ax.plot(x, y, '-', color='lime', lw=1.2, label='GT sheet')
    colors = ('red', 'cyan', 'orange', 'magenta', 'yellow', 'deepskyblue', 'white')
    for name, col in zip(names, colors):
        rows = torch.from_numpy(results[name]['rows'].astype(np.float32))
        mid = rows.shape[1]//2
        x, y = plane(rows[:, mid])
        ax.plot(x, y, '-', color=col, lw=.8)
        kept = connected_keep(results[name]['confs'], cfg0.trace.confidence_threshold)[:, mid]
        ax.plot(x[kept], y[kept], 'o', color=col, ms=3, label=name)
        ax.plot(x[~kept], y[~kept], 'x', color=col, ms=5)
    ax.set_xlim(-span/2, span/2)
    ax.set_ylim(50, -50)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f"{w['meta']['patch'][:50]}: CT along the advance (middle column)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoints', nargs='+')
    ap.add_argument('--rescore', nargs='+', metavar='DIR', help='recompute switch counts of existing benchmark folders')
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW',
                    help='replace the path prefix OLD by NEW in every checkpoint\'s data paths (volumes, '
                         'patch directories), e.g. for checkpoints trained on another machine; repeatable')
    ap.add_argument('--out', help='output directory (default: output/trace_bench/TAG)')
    ap.add_argument('--patches-dir', nargs='+', help='draw seeds from these patch directories (default: the first '
                                                     'checkpoint\'s), e.g. segments of an unseen scroll')
    ap.add_argument('--all-val', action='store_true', help='no train/val split: every patch region is a seed candidate '
                                                          '(for scrolls never trained on)')
    ap.add_argument('--max-per-patch', type=int, default=1, help='at most this many seeds per patch (0: any number)')
    ap.add_argument('--raw', nargs='*', default=[], metavar='NAME',
                    help='use the raw instead of the EMA weights for these models (checkpoints trained with train.ema)')
    ap.add_argument('--names', nargs='+')
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--front', type=float, default=240., help='seed row length (voxels); the number of front points '
                    'is this over the model\'s point spacing, plus one')
    ap.add_argument('--voxel-size', type=float, help='level-0 voxel size (um) for area_cm2 in the written tifxyz '
                    '(default: the first checkpoint\'s data.voxel_size_um; set it with --volume of another scan)')
    ap.add_argument('--seeds-anywhere', action='store_true', help='draw seeds from every patch region, training ones '
                    'too; with --held-out-scoring only points over held-out (val) cells of the real split are scored')
    ap.add_argument('--held-out-scoring', action='store_true', help='seeds need only their seed row in held-out '
                    '(val) cells; traces run on regardless and only points whose GT foot point lies in a val cell are '
                    'scored (no effect with --all-val)')
    ap.add_argument('--seeds-from', metavar='REPORT', help='reuse the seeds of an earlier report.json (same patch, '
                    'orientation and anchor) with this --front and --steps, dropping those that no longer fit')
    ap.add_argument('--min-valid', type=float, default=.95)
    ap.add_argument('--tag', default='bench')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threshold', type=float, help='confidence threshold for every model (default: each model\'s own)')
    ap.add_argument('--volume', help='volume pyramid for every model (default: the first checkpoint\'s), e.g. when '
                                     'a checkpoint trained elsewhere records a path that does not exist here; '
                                     'each model still reads its own pyramid level')
    args = ap.parse_args(argv)
    if args.rescore:
        for folder in args.rescore:
            rescore(Path(folder))
        return
    if not args.checkpoints:
        ap.error('--checkpoints is required (or --rescore)')
    names = args.names or [Path(c).parent.name for c in args.checkpoints]
    models = {name: load_checkpoint(path, args.device, raw=name in args.raw)[:2]
              for name, path in zip(names, args.checkpoints)}
    apply = C.path_mapper(args.path_map)
    for _, cfg in models.values():
        apply(cfg)
    cfg0 = models[names[0]][1]
    volume = args.volume or cfg0.data.volume
    for name, (_, cfg) in models.items():
        cfg.data.volume = volume
        if args.threshold is not None:
            cfg.trace.confidence_threshold = args.threshold
    cfg0.data.volume = volume
    seed_cfg = cfg0
    if args.patches_dir:
        seed_cfg.data.patches_dir = args.patches_dir
    origin = None
    if args.seeds_from:
        seeds, spec, seed_data, origin = seeds_from(Path(args.seeds_from), seed_cfg, args.steps, args.front,
                                                    args.min_valid, all_val=args.all_val or args.seeds_anywhere,
                                                    seed_split=args.held_out_scoring)
        print(f'{len(seeds)} of the seeds of {args.seeds_from} fit a {args.front:g} vx front', flush=True)
    else:
        seeds, spec, seed_data = draw_seeds(seed_cfg, args.seeds, args.steps, args.front, args.min_valid,
                                            all_val=args.all_val or args.seeds_anywhere,
                                            max_per_patch=args.max_per_patch, seed_split=args.held_out_scoring)
    print(f'{len(seeds)} held-out seeds', flush=True)
    out = Path(args.out) if args.out else C.OUTPUT/'trace_bench'/args.tag
    (out/'tifxyz').mkdir(parents=True, exist_ok=True)
    report = dict(checkpoints=dict(zip(names, args.checkpoints)), steps=args.steps, front=args.front,
                  seeds=[s['meta'] for s in seeds], models={})
    if origin is not None:
        report.update(seeds_from=args.seeds_from, seeds_from_index=origin)
    held = None  # the split whose val cells are scored: the seed config's own (seed_data may have all cells val)
    if args.held_out_scoring and not args.all_val:
        held = seed_cfg.data if args.seeds_anywhere else seed_data
    voxel_size = args.voxel_size or cfg0.data.voxel_size_um
    if args.held_out_scoring:
        report['held_out_scoring'] = True
    per_model = {}
    all_results = [dict() for _ in seeds]
    for name in names:
        model, cfg = models[name]
        scores = []
        for i, w in enumerate(seeds):
            points, direction = seed_polyline(w, spec, seed_data)
            seed = resample(points, cfg.data.point_spacing)
            idx = np.linspace(0, len(points)-1, len(seed))
            dirs = np.stack([np.interp(idx, np.arange(len(points)), direction[:, k]) for k in range(3)], 1)
            rows, confs = trace(model, cfg, seed, dirs, args.steps, args.device)
            traced = len(rows)-1
            if len(rows) < args.steps+1:  # left the volume: pad with the last row, unsupervised beyond
                pad = args.steps+1-len(rows)
                rows = np.concatenate([rows, np.repeat(rows[-1:], pad, 0)])
                confs = np.concatenate([confs, np.zeros((pad, confs.shape[1]), confs.dtype)])
            sc = score(rows, confs, w, spec, cfg, args.device, traced, held)
            sc.update(rows=rows, confs=confs)
            keep = connected_keep(confs, cfg.trace.confidence_threshold)
            column = cfg.data.point_spacing/cfg.trace.spline_samples
            write_surface(out/'tifxyz'/f'seed{i:02d}_{name}', rows, keep, cfg.data.step, column,
                          dict(meta=w['meta'], checkpoint=args.checkpoints[names.index(name)],
                               first_switch=sc['first_switch'], first_real_switch=sc['first_real_switch'],
                               lost=sc['lost']), overwrite=True, voxel_size_um=voxel_size)
            if name == names[0]:
                write_surface(out/'tifxyz'/f'seed{i:02d}_gt', w['gt'], w['valid'], spec.res, spec.res,
                              dict(meta=w['meta']), overwrite=True, voxel_size_um=voxel_size)
            scores.append(sc)
            all_results[i][name] = sc
        per_step, summary, switches, reals, losts = summarize(scores, args.steps)
        per_model[name] = per_step
        report['models'][name] = dict(summary=summary, first_switch=switches, first_real_switch=reals, lost=losts,
                                      per_step={k: v.tolist() for k, v in per_step.items()})
        s = summary
        print(f"{name}: coverage {s['coverage_mean']:.3f} (last {s['coverage_last']:.3f}) same_sheet "
              f"{s['same_sheet_mean']:.3f} (last {s['same_sheet_last']:.3f}) precision {s['precision_mean']:.3f} "
              f"kept {s['kept_mean']:.3f} caught {s['caught_mean']:.3f} dist_p50 {s['dist_p50_mean']:.2f} | "
              f"real switches {s['traces_real_switch']}/{len(seeds)} (any off-sheet point "
              f"{s['traces_with_switch']}, {s['off_kept_points']} points) lost {s['traces_lost']}/{len(seeds)} "
              f"median first real switch {s['median_first_real_switch']:.0f} | stopped {s['traces_stopped']}/{len(seeds)}"
              + (f" (median step {s['median_stop']:.0f})" if s['median_stop'] else ''), flush=True)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ('same_sheet', 'coverage', 'precision', 'kept', 'caught', 'dist_p50', 'advance_p50')
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1*len(keys), 3))
    for name, ps in per_model.items():
        for ax, k in zip(axes, keys):
            ax.plot(np.arange(1, args.steps+1), ps[k], '-', label=name)
            ax.set_title(k, fontsize=9)
            ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/'per_step.png', dpi=100)
    plt.close(fig)
    for i, (w, res) in enumerate(zip(seeds, all_results)):
        seed_figure(out/f'seed_{i:02d}.png', w, spec, seed_data_cfg(cfg0, seed_data), res, names, args.steps)
    print(f'-> {out}')


def seed_data_cfg(cfg, data):
    cfg = copy.deepcopy(cfg)
    cfg.data = data
    return cfg


if __name__ == '__main__':
    main()
