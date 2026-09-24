"""VC's restart metric on held-out fibers: hand beam versus model-in-the-loop.

  python -m vesuvius.neural_tracing.fiber_follow.beam.evaluate_spans hand --tag hand
  python -m vesuvius.neural_tracing.fiber_follow.beam.evaluate_spans output/RUN/last.pt --tag RUN

For every held-out fiber the C++ tracer chains control point to control point,
restarting at the true control point after each miss (in-plane error above the
threshold). Reports restarts per 1000 trace-grid voxels and span success rate.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import time

from vesuvius.neural_tracing.fiber_follow.beam.diag import span_diag
from vesuvius.neural_tracing.fiber_follow.beam.native import BeamSpec, NativeBeam
from vesuvius.neural_tracing.fiber_follow.data import DATA_POLICY, ZBand, fiber_manifest, load_fibers, split_fibers

FF = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION = '/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs/PHercParis4-20260411134726-las-sd1-7ff0ce6c.lasagna.json'
DEFAULT_NORMALS = '/mnt/raid_nvme/volpkgs/s1_2um.volpkg/las_008_s1_full/las_008.lasagna.json'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ckpt', help='beam re-ranker checkpoint, or hand')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fibers', default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--val-z', type=float, nargs=2, default=(45000.0, 48500.0))
    ap.add_argument('--n', type=int, default=0, help='0 = every held-out fiber')
    ap.add_argument('--error-threshold-base', type=float, default=20.)
    ap.add_argument('--prediction-manifest', default=DEFAULT_PREDICTION, help='hand mode only')
    ap.add_argument('--normal-manifest', default=DEFAULT_NORMALS, help='hand mode only')
    ap.add_argument('--beam-config', default='{}', help='hand mode only: JSON TraceConfig overrides')
    ap.add_argument('--hook-mode', choices=('additive', 'replace'))
    ap.add_argument('--hook-weight', type=float)
    ap.add_argument('--threads', type=int, default=0, help='C++ candidate scoring threads; 0 = OpenMP default')
    ap.add_argument('--out-dir', type=Path, default=FF / 'output' / 'eval')
    args = ap.parse_args(argv)
    grid_scale = 8.
    hook = None
    if args.ckpt == 'hand':
        beam_spec = BeamSpec(args.prediction_manifest, args.normal_manifest, config=json.loads(args.beam_config),
                             parallel_threads=args.threads)
    else:
        from vesuvius.neural_tracing.fiber_follow.beam.hook import ModelBeamHook
        from vesuvius.neural_tracing.fiber_follow.beam.train import load_beam_checkpoint
        from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume
        model, state_cfg, vol_spec, beam_spec, ck = load_beam_checkpoint(args.ckpt, args.device)
        beam_spec = dataclasses.replace(beam_spec, parallel_threads=args.threads)
        grid_scale = vol_spec.grid_scale
        hook = ModelBeamHook(model, FiberVolume(vol_spec, cache_bytes=4 << 30), state_cfg, device=args.device,
                             mode=args.hook_mode or ck.get('hook_mode', 'additive'),
                             weight=args.hook_weight if args.hook_weight is not None else ck.get('hook_weight', 1.))
    fibers = load_fibers(args.fibers, grid_scale=grid_scale)
    band = ZBand(args.val_z[0] / grid_scale, args.val_z[1] / grid_scale)
    _, val = split_fibers(fibers, band)
    if args.n:
        val = val[:args.n]
    beam = NativeBeam(beam_spec, grid_scale)
    start = time.time()
    rows, summary = span_diag(beam, val, grid_scale, hook=hook, error_threshold_base=args.error_threshold_base)
    summary.update(seconds=time.time() - start, checkpoint=args.ckpt, data_policy=DATA_POLICY,
                   beam_spec=beam_spec.to_dict(), error_threshold_base=args.error_threshold_base,
                   fiber_manifest=fiber_manifest(val))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ('hand' if args.ckpt == 'hand' else Path(args.ckpt).parent.name)
    (args.out_dir / f'eval_spans_{tag}.json').write_text(json.dumps(summary, indent=2))
    with (args.out_dir / f'eval_spans_{tag}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ['fiber'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('beam_spec', 'fiber_manifest')}, indent=2))


if __name__ == '__main__':
    main()
