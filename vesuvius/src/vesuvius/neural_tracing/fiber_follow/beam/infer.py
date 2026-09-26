"""Trace with the volume-cartographer beam plus a trained step scorer.

Span mode re-traces every control-point span of an existing VC3D fiber JSON
the way the annotation window does (bidirectional trace + meeting fusion), with
the model scoring every proposed step, and writes a new fiber JSON with the same
control points:

  python -m vesuvius.neural_tracing.fiber_follow.beam.infer output/RUN/last.pt \\
      --fiber-json /path/fiber.json --out /tmp/retraced

Open mode traces both directions from base-voxel seeds for a fixed distance,
stopping where the model no longer trusts any candidate:

  python -m vesuvius.neural_tracing.fiber_follow.beam.infer output/RUN/last.pt \\
      --seed 12000,8000,46000 --confidence 0.5 --out /tmp/open

``hand`` instead of a checkpoint runs the plain beam (needs --prediction-manifest,
--normal-manifest, --fiber-zarrs and --ct).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.fiber_follow.beam.native import BeamSpec, NativeBeam, import_fiber_trace
from vesuvius.neural_tracing.fiber_follow.geometry import arclength, resample_polyline
from vesuvius.neural_tracing.fiber_follow.infer import make_fiber_json
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


def retrace_spans(beam: NativeBeam, fiber_json: str, hook=None):
    """Fused span traces between consecutive control points (base voxels).

    Spans whose trace is rejected keep the original annotated geometry, as the
    annotation window falls back to another optimizer there.
    """
    ft = import_fiber_trace()
    fiber = ft.load_fiber_json(fiber_json)
    line_base = np.asarray(fiber.line_points)
    indices = [int(i) for i in fiber.control_point_line_indices]
    line_grid = line_base / beam.grid_scale
    pieces, controls, report = [], [0], []
    for a, b in zip(indices[:-1], indices[1:]):
        try:
            result = beam.trace_segment(line_grid, a, b, hook=hook)
        except ValueError as exc:
            result = dict(points=None, accepted=False, reason=f'error:{exc}', detail='')
        if result['accepted']:
            piece = np.asarray(result['points']) * beam.grid_scale
        else:
            piece = line_base[a:b + 1]
        piece[0], piece[-1] = line_base[a], line_base[b]
        if pieces:
            piece = piece[1:]
        pieces.append(piece)
        controls.append(controls[-1] + len(piece) - (0 if len(pieces) == 1 else 0))
        report.append(dict(start=a, target=b, accepted=result['accepted'], reason=result['reason']))
    points = np.concatenate(pieces, 0)
    controls = [0] + list(np.cumsum([len(p) for p in pieces]) - 1)
    return points, controls, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint', help='beam step scorer checkpoint, or hand')
    ap.add_argument('--fiber-json', action='append', default=[], help='span mode input (repeatable)')
    ap.add_argument('--seed', action='append', default=[], help='open mode: base-voxel x,y,z (repeatable)')
    ap.add_argument('--heading', action='append', default=[], help='open mode: x,y,z per seed (required)')
    ap.add_argument('--max-len', type=float, default=6000., help='open mode per-direction limit, trace-grid voxels')
    ap.add_argument('--confidence', type=float, default=None, help='open mode on-fiber stop threshold')
    ap.add_argument('--prediction-manifest')
    ap.add_argument('--normal-manifest')
    ap.add_argument('--fiber-zarrs')
    ap.add_argument('--ct')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    if bool(args.fiber_json) == bool(args.seed):
        raise SystemExit('give either --fiber-json (span mode) or --seed (open mode)')

    hook = None
    if args.checkpoint == 'hand':
        if not (args.prediction_manifest and args.fiber_zarrs):
            raise SystemExit('hand mode needs --prediction-manifest and --fiber-zarrs')
        beam_spec = BeamSpec(args.prediction_manifest, args.normal_manifest)
        spec = FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=0, ct_grid_scale=4., inputs='ct' if args.ct else 'fiber')
    else:
        from vesuvius.neural_tracing.fiber_follow.beam.hook import ModelBeamHook
        from vesuvius.neural_tracing.fiber_follow.beam.train import load_beam_checkpoint
        model, state_cfg, spec, beam_spec, ck = load_beam_checkpoint(args.checkpoint, args.device)
        if args.fiber_zarrs:
            spec.fiber_zarr_dir = args.fiber_zarrs
        if args.ct:
            spec.ct_zarr = args.ct
        hook = ModelBeamHook(model, FiberVolume(spec, cache_bytes=4 << 30), state_cfg, device=args.device,
                             stop_threshold=args.confidence)
    beam = NativeBeam(beam_spec, spec.grid_scale)
    g = spec.grid_scale
    os.makedirs(args.out, exist_ok=True)
    meta = dict(username='fiber_follow_beam', fiber_manifest=beam_spec.prediction_manifest)
    written = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]
    for k, path in enumerate(args.fiber_json):
        points, controls, report = retrace_spans(beam, path, hook=hook)
        name = f'{Path(path).stem}_beam_{stamp}.json'
        obj = make_fiber_json(points, 800., dict(meta, filename=name, started_at=stamp, sequence=k,
                                                 fiber_follow_beam=dict(source=os.path.abspath(path), spans=report,
                                                                        checkpoint=os.path.abspath(args.checkpoint))),
                              control_indices=controls)
        (Path(args.out) / name).write_text(json.dumps(obj))
        written.append(dict(name=name, spans=report))
    if args.seed:
        if len(args.heading) != len(args.seed):
            raise SystemExit('open mode needs one --heading per --seed')
        for k, (seed, heading) in enumerate(zip(args.seed, args.heading)):
            pos = np.array([float(v) for v in seed.split(',')]) / g
            axis = np.array([float(v) for v in heading.split(',')])
            axis /= np.linalg.norm(axis)
            forward, reason_f, _ = beam.trace_open(pos, axis, args.max_len, hook=hook)
            backward, reason_b, _ = beam.trace_open(pos, -axis, args.max_len, hook=hook)
            poly = np.concatenate([backward[::-1], forward[1:]], 0)
            base = resample_polyline(poly * g, g)
            name = f'fiber_follow_beam_{stamp}_{k:06d}.json'
            obj = make_fiber_json(base, 800., dict(meta, filename=name, started_at=stamp, sequence=k,
                                                   fiber_follow_beam=dict(stop_reasons=[reason_b, reason_f],
                                                                          checkpoint=os.path.abspath(args.checkpoint))))
            (Path(args.out) / name).write_text(json.dumps(obj))
            written.append(dict(name=name, length_base=float(arclength(base)[-1]), reasons=[reason_b, reason_f]))
    print(json.dumps(dict(written=written, out=os.path.abspath(args.out)), indent=2))


if __name__ == '__main__':
    main()
