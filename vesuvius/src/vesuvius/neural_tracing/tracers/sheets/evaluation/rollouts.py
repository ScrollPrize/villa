"""Free rollouts (no resets, no noise) on held-out episodes, with per-step metrics and diagnostic figures.

Per step, over front_points whose committed point has GT supervision:
  same_sheet  within ``loss.sheet_tolerance`` of the seed's GT sheet
  confident   confidence above ``trace.confidence_threshold``
  coverage    same_sheet and confident (the output a user would keep)
  precision   same_sheet among confident (low = confidently on another sheet)
  caught      not confident among front_points off the sheet (sheet switches that were flagged)
  kept_off    confident and off the sheet, over all supervised front_points (wrong points a user would keep)
  score       coverage - kept_off (checkpoint selection: keeping everything is not rewarded)
  rejoined    coverage among front_points that were stopped at some earlier step (stop, then reconnect)
  accurate    within ``loss.good_distance`` among kept points (coverage)
  curve_same_sheet  same_sheet among dense curve samples whose gap has two confident front_points
  curve_kept  fraction of supervised dense curve samples whose gap has two confident front_points
  dist_p50/p90, advance_p90  on-sheet distance and |advance error| of kept points

Usage:
  python -m vesuvius.neural_tracing.tracers.sheets.evaluation.rollouts --checkpoint output/RUN/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.data.episodes import (
    FIBER_KEYS, EpisodeSampler, batches, episode_pad, fixed_episodes, unpack_blocks)
from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
from vesuvius.neural_tracing.tracers.sheets.geometry import grid_eval, project, surface_frame
from vesuvius.neural_tracing.tracers.sheets.losses import front_metrics, gap_mask, landing
from vesuvius.neural_tracing.tracers.sheets.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import connected_keep, write_surface
from vesuvius.neural_tracing.tracers.sheets.tracing.rollout import Tracer
from vesuvius.neural_tracing.tracers.sheets.sampling import sample_plane
from vesuvius.neural_tracing.tracers.sheets.tracing.seeding import seed_rows


def to_device(batch, device, pad_blocks: bool = True):
    """A collated batch on ``device``, blocks padded there (``unpack_blocks``) as float, or a list of unpadded blocks
    with ``pad_blocks=False``."""
    return dict(gt=batch['gt'].to(device, non_blocking=True), valid=batch['valid'].to(device, non_blocking=True),
                block=unpack_blocks(batch['block'], device, pad=pad_blocks),
                origin=batch['origin'].to(device, non_blocking=True), meta=batch['meta'],
                # live-chain continuation batches; the front size of data.front_points_range batches
                **{k: batch[k] for k in ('chains', 'front_points') if k in batch},
                **{k: batch[k].to(device, non_blocking=True) for k in ('seed', 'seed_mask')+FIBER_KEYS if k in batch})


@torch.no_grad()
def _record(b, i, seeds, rows, confs, mets, cfg, spec, last_curve):
    """CPU copy of one episode's rollout plus two CT slices for the figure."""
    gt, valid, block, origin = b['gt'][i:i+1], b['valid'][i:i+1], b['block'][i], b['origin'][i]
    N, mid = cfg.data.front_points, cfg.data.front_points//2
    pts = torch.stack(rows)[:, i]
    uv_rows = project(gt, valid, pts.reshape(1, -1, 3))[0].reshape(len(rows), N, 2)
    uv_seed = project(gt, valid, seeds[i].reshape(1, -1, 3))[0].reshape(-1, N, 2)
    seed_mid = seeds[i, -1, mid]
    uv_mid = uv_seed[-1, mid][None, None]
    _, Su, Sv, _ = grid_eval(gt, valid, uv_mid)
    t, a, n = surface_frame(Su, Sv)[0, 0].T
    d = cfg.data
    span_a = (d.history+len(rows))*d.step+40
    centre_a = seed_mid+a*(len(rows)-d.history)*d.step/2
    adv = sample_plane(block, origin, d.volume_level, centre_a, torch.stack([n, a]), (80., span_a), 1.)
    last = pts[-1].mean(0)
    span_t = (N-1)*d.point_spacing+60
    front_axes = torch.stack([n, torch.nn.functional.normalize(pts[-1, -1]-pts[-1, 0], dim=0)])
    front = sample_plane(block, origin, d.volume_level, last, front_axes, (80., span_t), 1.)
    # GT curves: along u through the middle point's v, and along v at the last row's u.
    u_line = torch.arange(spec.Gu, device=gt.device, dtype=gt.dtype)
    v_mid = uv_seed[-1, mid, 1]
    S_u, _, _, ok_u = grid_eval(gt, valid, torch.stack([u_line, v_mid.expand_as(u_line)], -1)[None])
    v_line = torch.arange(spec.Gv, device=gt.device, dtype=gt.dtype)
    u_last = uv_rows[-1, :, 0].median()
    S_v, _, _, ok_v = grid_eval(gt, valid, torch.stack([u_last.expand_as(v_line), v_line], -1)[None])
    plane = lambda p, c, axes: torch.stack([((p-c)*axes[0]).sum(-1), ((p-c)*axes[1]).sum(-1)], -1)
    np_ = lambda x: x.detach().float().cpu().numpy()
    return dict(
        meta=b['meta'][i], valid=np_(valid[0]), uv_rows=np_(uv_rows), uv_seed=np_(uv_seed),
        conf=np_(torch.stack(confs)[:, i]), dist=np_(torch.stack([m['dist'][i] for m in mets])),
        ok=np_(torch.stack([m['ok'][i] for m in mets])),
        adv_img=np_(adv), adv_gt=np_(plane(S_u[0][ok_u[0]], centre_a, torch.stack([n, a]))),
        adv_trace=np_(plane(pts[:, mid], centre_a, torch.stack([n, a]))),
        adv_seed=np_(plane(seeds[i, :, mid], centre_a, torch.stack([n, a]))),
        front_img=np_(front), front_gt=np_(plane(S_v[0][ok_v[0]], last, front_axes)),
        front_trace=np_(plane(pts[-1], last, front_axes)), front_conf=np_(confs[-1][i]),
        front_curve=np_(plane(last_curve[i], last, front_axes)),
        spec=dict(u_lo=spec.u_lo, v_lo=spec.v_lo, res=spec.res), extents=dict(adv=(80., span_a), front=(80., span_t)))


def write_episode_surfaces(out_dir: Path, index: int, rows, keep, gt, valid, meta, tracer, cfg, spec, metrics):
    """``epNN_trace`` (rows = steps incl. the seed, columns as written by inference) and ``epNN_gt`` (the
    episode's canonical GT window, rows = u, columns = v, at ``gt_resolution``) tifxyz for VC3D."""
    provenance = dict(meta=meta, metrics=metrics, step=cfg.data.step, column_spacing=tracer.column_spacing(),
                      confidence_threshold=cfg.trace.confidence_threshold)
    write_surface(out_dir/f'ep{index:02d}_trace', rows, keep, cfg.data.step, tracer.column_spacing(), provenance,
                  overwrite=True, voxel_size_um=cfg.data.voxel_size_um)
    write_surface(out_dir/f'ep{index:02d}_gt', gt, valid, spec.res, spec.res, dict(meta=meta), overwrite=True,
                  voxel_size_um=cfg.data.voxel_size_um)


@torch.no_grad()
def rollout_metrics(tracer: Tracer, episodes, cfg, spec, steps: int, device, batch_size: int = 16,
                    records: int = 0, surfaces: Path | None = None, surface_count: int = 0):
    """Per-step metric arrays (``steps``,) and up to ``records`` episode records for figures. With ``surfaces``,
    the first ``surface_count`` episodes are also written there as trace and GT tifxyz."""
    tracer.model.eval()
    seed_u = -spec.u_lo/spec.res  # grid row of the seed (u = 0)
    keys = ('ok', 'same', 'conf', 'cover', 'off', 'caught', 'stopped_before', 'rejoined', 'accurate',
            'curve_ok', 'curve_kept', 'curve_same')
    sums = {k: np.zeros(steps) for k in keys}
    dists, advances = [[] for _ in range(steps)], [[] for _ in range(steps)]
    kept, n_written = [], 0
    if surfaces is not None:
        surfaces.mkdir(parents=True, exist_ok=True)
    for raw in batches(episodes, batch_size):
        b = to_device(raw, device)
        seeds = seed_rows(b['gt'], b['valid'], cfg, spec, noise=False)
        front = tracer.initial(seeds)
        stopped = torch.zeros(front.pos.shape[:2], dtype=torch.bool, device=device)
        rows, confs, mets, curves = [], [], [], []
        written = [tracer.output_columns(front.pos, front.conf)]
        for t in range(steps):
            out = tracer.step(front, b['block'], b['origin'])
            _, _, curve_same, curve_ok = landing(out.dense, b['gt'], b['valid'], cfg.loss.sheet_tolerance)
            gap_conf = gap_mask(torch.sigmoid(out.conf_logit) > cfg.trace.confidence_threshold,
                                out.dense.shape[1], cfg.trace.spline_samples)
            sums['curve_ok'][t] += curve_ok.sum().item()
            sums['curve_kept'][t] += (curve_ok & gap_conf).sum().item()
            sums['curve_same'][t] += (curve_ok & gap_conf & curve_same).sum().item()
            curves.append(out.dense)
            front = tracer.advance(front, out)
            written.append(tracer.output_columns(front.pos, front.conf, out.dense))
            m = front_metrics(front.pos, front.conf, seed_u, t+1, b['gt'], b['valid'], cfg)
            ok, same, conf = m['ok'], m['same'], m['confident']
            keep = ok & same & conf
            counts = dict(ok=ok, same=ok & same, conf=ok & conf, cover=keep, off=ok & ~same,
                          caught=ok & ~same & ~conf, stopped_before=ok & stopped, rejoined=keep & stopped,
                          accurate=keep & (m['dist'] < cfg.loss.good_distance))
            for k, mask in counts.items():
                sums[k][t] += mask.sum().item()
            dists[t].append(m['dist'][keep].cpu())
            advances[t].append(m['advance'][keep].abs().cpu())
            stopped |= ~conf
            rows.append(front.pos)
            confs.append(front.conf)
            mets.append(m)
        for i in range(min(records-len(kept), front.pos.shape[0])):
            kept.append(_record(b, i, seeds, rows, confs, mets, cfg, spec, curves[-1]))
        if surfaces is not None:
            pts = torch.stack([w[0] for w in written], 1).cpu().numpy()
            confs_w = torch.stack([w[1] for w in written], 1).cpu().numpy()
            keep = np.stack([connected_keep(c, cfg.trace.confidence_threshold) for c in confs_w])
            for i in range(front.pos.shape[0]):
                index = n_written+i
                if index >= surface_count:
                    break
                ok = torch.stack([m['ok'][i] for m in mets])
                metrics = dict(same_sheet_per_step=((torch.stack([m['same'][i] for m in mets]) & ok).sum(1)
                                                    / ok.sum(1).clamp_min(1)).tolist(),
                               kept_per_step=keep[i, 1:].mean(1).tolist())
                write_episode_surfaces(surfaces, index, pts[i], keep[i], raw['gt'][i].numpy(),
                                       raw['valid'][i].numpy(), raw['meta'][i], tracer, cfg, spec, metrics)
        n_written += front.pos.shape[0]
    tracer.model.train()
    total = len(episodes)*cfg.data.front_points
    div = lambda a, b: np.where(b > 0, a/np.maximum(b, 1), np.nan)
    flat = lambda xs: [torch.cat(x).numpy() if x else np.zeros(0) for x in xs]
    q = lambda xs, p: np.array([float(np.quantile(x, p)) if len(x) else float('nan') for x in flat(xs)])
    per_step = dict(same_sheet=div(sums['same'], sums['ok']), confident=div(sums['conf'], sums['ok']),
                    coverage=div(sums['cover'], sums['ok']), precision=div(sums['cover'], sums['conf']),
                    caught=div(sums['caught'], sums['off']), rejoined=div(sums['rejoined'], sums['stopped_before']),
                    kept_off=div(sums['conf']-sums['cover'], sums['ok']),
                    score=div(2*sums['cover']-sums['conf'], sums['ok']),
                    accurate=div(sums['accurate'], sums['cover']), dist_p50=q(dists, .5), dist_p90=q(dists, .9),
                    advance_p90=q(advances, .9), supervised=sums['ok']/total,
                    curve_same_sheet=div(sums['curve_same'], sums['curve_kept']),
                    curve_kept=div(sums['curve_kept'], sums['curve_ok']))
    return per_step, kept


def summarize(per_step: dict) -> dict:
    """Mean over steps and value at the last step of each metric."""
    out = {}
    for k, v in per_step.items():
        out[k+'_mean'] = float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')
        out[k+'_last'] = float(v[-1])
    return out


def eval_episodes(cfg, steps: int, roi: int, count: int):
    """``count`` fixed held-out episodes and their grid spec."""
    index = build_index(cfg.data.patches_dir, C.OUTPUT/'cache')
    sampler = EpisodeSampler(cfg.data, index, 'val', steps, roi, episode_pad(cfg.model, cfg.train))
    return fixed_episodes(sampler, count, 0), sampler.spec


def run(tracer, cfg, out_dir: Path, episodes, steps: int, roi: int, records: int = 6, tag: str = 'eval'):
    """Evaluate on ``episodes`` (an ``(episodes, spec)`` pair from ``eval_episodes``). Every episode is written as
    trace + GT tifxyz under ``out_dir/tifxyz_TAG``."""
    from vesuvius.neural_tracing.tracers.sheets.evaluation.figures import episode_figure, metric_figure
    device = next(tracer.model.parameters()).device
    episodes, spec = episodes
    count = len(episodes)
    per_step, kept = rollout_metrics(tracer, episodes, cfg, spec, steps, device, records=records,
                                     surfaces=out_dir/f'tifxyz_{tag}', surface_count=count)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(tag=tag, steps=steps, episodes=count, roi=roi, summary=summarize(per_step),
                  per_step={k: v.tolist() for k, v in per_step.items()})
    (out_dir/f'{tag}.json').write_text(json.dumps(report, indent=2))
    metric_figure({tag: per_step}, out_dir/f'{tag}_metrics.png')
    for i, record in enumerate(kept):
        episode_figure(record, cfg, out_dir/f'{tag}_episode_{i:02d}.png')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--roi', type=int)
    ap.add_argument('--episodes', type=int)
    ap.add_argument('--records', type=int, default=6)
    ap.add_argument('--tag')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    model, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    tracer = Tracer(model, cfg, args.device)
    e = cfg.eval
    tag = args.tag or 'eval'
    steps, roi, count = args.steps or e.steps, args.roi or e.roi, args.episodes or e.episodes
    t0 = time.time()
    episodes = eval_episodes(cfg, steps, roi, count)
    print(f'{count} episodes drawn in {time.time()-t0:.1f} s', flush=True)
    report = run(tracer, cfg, Path(args.checkpoint).parent/'eval', episodes, steps, roi, args.records, tag)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
