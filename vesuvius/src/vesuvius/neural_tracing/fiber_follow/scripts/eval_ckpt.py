"""Evaluate on controlled-span seeds using the current data contract.

  python scripts/eval_ckpt.py field --seeds-only
  python scripts/eval_ckpt.py output/controlled_v2/ckpt_001500.pt --tag controlled_v2
  python scripts/eval_ckpt.py hand --tag hand_beam            # VC beam, no model
  python scripts/eval_ckpt.py output/beam_run/last.pt --tag beam_run --params '{"confidence": 0.5}'

Beam re-ranker checkpoints (``beam_rerank_v1``) and ``hand`` trace open-ended with
the volume-cartographer beam through ``BeamTracer``; ``--params`` then accepts
``max_len``, ``confidence`` (on-fiber stop threshold), ``hook_mode`` and ``hook_weight``.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pickle
import time
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.data import (
    DATA_POLICY, ZBand, fiber_manifest, gt_presence, load_fibers, mark_breaks, split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.evaluate import evaluate, load_or_make_seeds
from vesuvius.neural_tracing.fiber_follow.trace import FieldTracer, ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec

FF = Path(__file__).resolve().parents[1]
LOCAL = '/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs'
DEFAULT_PREDICTION = LOCAL + '/PHercParis4-20260411134726-las-sd1-7ff0ce6c.lasagna.json'
DEFAULT_NORMALS = '/mnt/raid_nvme/volpkgs/s1_2um.volpkg/las_008_s1_full/las_008.lasagna.json'


def checkpoint_architecture(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)['architecture']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ckpt', help='checkpoint path, or field')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--n', type=int, default=0, help='0 = all eligible seeds')
    ap.add_argument('--tag', default='')
    ap.add_argument('--params', default='{}', help='TraceParams JSON (model only)')
    ap.add_argument('--history-audit', action='store_true', help='Audit observed/cleaned geometry on held-out follower decisions')
    ap.add_argument('--fibers', default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--fiber-zarrs', default=None)
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000.0, 48500.0))
    ap.add_argument('--out-dir', type=Path, default=FF / 'output' / 'eval')
    ap.add_argument('--seeds', type=Path, default=None)
    ap.add_argument('--rebuild-seeds', action='store_true')
    ap.add_argument('--seeds-only', action='store_true', help='prepare fixed v2 seeds without running a tracer')
    ap.add_argument('--prediction-manifest', default=DEFAULT_PREDICTION, help='hand beam only')
    ap.add_argument('--normal-manifest', default=DEFAULT_NORMALS, help='hand beam only')
    ap.add_argument('--ct', default=None, help='override the CT zarr of a beam checkpoint')
    args = ap.parse_args(argv)
    from vesuvius.neural_tracing.fiber_follow.beam.model import ARCHITECTURE as BEAM_ARCHITECTURE
    beam_ckpt = None
    if args.ckpt == 'field':
        spec = FiberVolumeSpec(args.fiber_zarrs or LOCAL)
    elif args.ckpt == 'hand':
        from vesuvius.neural_tracing.fiber_follow.beam.native import BeamSpec
        spec = FiberVolumeSpec(args.fiber_zarrs or LOCAL)
        beam_spec = BeamSpec(args.prediction_manifest, args.normal_manifest)
    elif checkpoint_architecture(args.ckpt) == BEAM_ARCHITECTURE:
        from vesuvius.neural_tracing.fiber_follow.beam.train import load_beam_checkpoint
        model, state_cfg, spec, beam_spec, beam_ckpt = load_beam_checkpoint(args.ckpt, 'cpu' if args.seeds_only else args.device)
        if args.fiber_zarrs:
            spec.fiber_zarr_dir = args.fiber_zarrs
        if args.ct:
            spec.ct_zarr = args.ct
    else:
        model, crop, nh, spec, checkpoint = load_checkpoint(args.ckpt, device='cpu' if args.seeds_only else args.device)
        if args.fiber_zarrs:
            spec.fiber_zarr_dir = args.fiber_zarrs
    vol = FiberVolume(spec, cache_bytes=8 << 30)
    fibers = load_fibers(args.fibers, grid_scale=spec.grid_scale)
    band = ZBand(args.val_z[0] / spec.grid_scale, args.val_z[1] / spec.grid_scale)
    _, val = split_fibers(fibers, band)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds_path = args.seeds or args.out_dir / 'val_seeds_v2.pkl'
    seeds = load_or_make_seeds(seeds_path, val, vol, band, rebuild=args.rebuild_seeds)
    if args.seeds_only:
        print(json.dumps(dict(data_policy=DATA_POLICY, seeds=len(seeds), fibers=len(val), path=str(seeds_path))))
        return
    if args.n:
        seeds = seeds[:args.n]
    pres = gt_presence(val, vol, str(args.out_dir / 'gt_presence_val_controlled_v2.npz'))
    mark_breaks(val, pres)
    params = json.loads(args.params)
    if args.ckpt == 'field':
        if params:
            ap.error('--params applies only to ModelTracer')
        tracer = FieldTracer(vol)
    elif args.ckpt == 'hand' or beam_ckpt is not None:
        from vesuvius.neural_tracing.fiber_follow.beam.native import NativeBeam
        from vesuvius.neural_tracing.fiber_follow.beam.trace import BeamTracer
        hook = None
        if beam_ckpt is not None:
            from vesuvius.neural_tracing.fiber_follow.beam.hook import ModelBeamHook
            hook = ModelBeamHook(model, vol, state_cfg, device=args.device,
                                 mode=params.get('hook_mode', beam_ckpt.get('hook_mode', 'additive')),
                                 weight=params.get('hook_weight', beam_ckpt.get('hook_weight', 1.)),
                                 stop_threshold=params.get('confidence'))
        tracer = BeamTracer(NativeBeam(beam_spec, spec.grid_scale), hook=hook, max_len=params.get('max_len', 6000.))
    else:
        tracer = ModelTracer(model, vol, crop, nh, TraceParams(**params), device=args.device)
    audit = None
    if args.history_audit:
        if not isinstance(tracer, ModelTracer):
            ap.error('--history-audit requires a spatial follower checkpoint')
        from vesuvius.neural_tracing.fiber_follow.history_audit import HistoryAudit
        audit = HistoryAudit(tracer, tolerance=checkpoint['tolerance'])
    start = time.time()
    try:
        rows, summary = evaluate(tracer, val, seeds, batch=args.batch, history_audit=audit)
    finally:
        if args.ckpt != 'field':
            tracer.close()
    if audit is not None:
        summary['history_audit'] = audit.summary()
    summary.update(seconds=time.time() - start, data_policy=DATA_POLICY,
                   checkpoint=args.ckpt,
                   trace_params=dataclasses.asdict(tracer.p) if isinstance(tracer, ModelTracer) else params,
                   reasons=dict(collections.Counter(r['reason'] for r in rows)),
                   seeds_path=str(seeds_path), volume=spec.to_dict(), val_band=dataclasses.asdict(band),
                   fiber_manifest=fiber_manifest(val))
    tag = args.tag or (args.ckpt if args.ckpt in ('field', 'hand') else Path(args.ckpt).parent.name)
    if Path(tag).name != tag:
        ap.error('--tag must be a filename component')
    with open(args.out_dir / f'rows_v2_{tag}.pkl', 'wb') as fh:
        pickle.dump(rows, fh)
    with open(args.out_dir / f'eval_v2_{tag}.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(args.ckpt, tag, json.dumps({k: v for k, v in summary.items() if k != 'fiber_manifest'}))


if __name__ == '__main__':
    main()
