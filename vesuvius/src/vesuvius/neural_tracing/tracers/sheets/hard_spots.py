"""Find where a model fails on training-split traces, for oversampling those spots in training (``data.hard_windows``).

Traces ``--seeds`` seeds from the training split (``--steps`` steps, ``--front`` voxels, as the trace benchmark draws
them from the validation split, but several windows per patch are allowed) through the inference path without stopping, and records each trace's
first failure: the row where the inference stop rule would fire at ``--threshold`` (no confident point connected to
the surface), or the first real switch (``trace_benchmark.switch_steps``), whichever comes first. The record is the
seed's window moved ``--before`` steps short of the failure along the advance (``shift``, voxels), so a training
episode drawn there reaches the hard spot mid-rollout. Patch directories are referred to by their position in
``data.patches_dir`` (``dir``), so records written on one machine can be used with another machine's paths. The
output is rewritten every 100 traces, so a run can be stopped early and keep what it found.

  python -m vesuvius.neural_tracing.tracers.sheets.hard_spots --checkpoint output/remote_l0_points_ft/last.pt \\
      --seeds 2000 --out output/hard_windows_points_ft.json [--path-map OLD=NEW ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.sheets.evaluate import load_checkpoint
from vesuvius.neural_tracing.tracers.sheets.infer import _reach_row, resample, trace
from vesuvius.neural_tracing.tracers.sheets.config import path_mapper
from vesuvius.neural_tracing.tracers.sheets.trace_benchmark import draw_seeds, score, seed_polyline


def first_stop(confs: np.ndarray, threshold: float) -> int | None:
    """First row (1-based step) at which the inference stop rule fires, or None."""
    reach = np.ones(confs.shape[1], bool)
    for t in range(1, len(confs)):
        reach = _reach_row(confs[t] > threshold, reach)
        if not reach.any():
            return t
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seeds', type=int, default=2000)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--front', type=float, default=240.)
    ap.add_argument('--min-valid', type=float, default=.95)
    ap.add_argument('--threshold', type=float, default=.4)
    ap.add_argument('--before', type=int, default=6, help='steps before the failure where the episode starts')
    ap.add_argument('--out', required=True)
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    model, cfg = load_checkpoint(args.checkpoint, args.device)[:2]
    path_mapper(args.path_map)(cfg)
    cfg.trace.confidence_threshold = args.threshold
    t0 = time.time()
    seeds, spec, data = draw_seeds(cfg, args.seeds, args.steps, args.front, args.min_valid, split='train',
                                   max_per_patch=0)
    roots = [Path(p) for p in (cfg.data.patches_dir if isinstance(cfg.data.patches_dir, list) else [cfg.data.patches_dir])]
    print(f'{len(seeds)} training-split seeds in {time.time()-t0:.0f} s', flush=True)
    records, counts = [], dict(traced=0, stop=0, switch=0, none=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write():
        tmp = out.with_suffix('.partial')
        tmp.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), steps=args.steps, threshold=args.threshold,
                                       before=args.before, counts=counts, records=records), indent=1))
        tmp.replace(out)
    for i, w in enumerate(seeds):
        if i and i % 100 == 0:
            write()
            print(f'{i}/{len(seeds)} traced, {len(records)} hard spots ({counts}), {time.time()-t0:.0f} s', flush=True)
        points, direction = seed_polyline(w, spec, data)
        seed = resample(points, cfg.data.point_spacing)
        idx = np.linspace(0, len(points)-1, len(seed))
        dirs = np.stack([np.interp(idx, np.arange(len(points)), direction[:, k]) for k in range(3)], 1)
        rows, confs = trace(model, cfg, seed, dirs, args.steps, args.device, stop_on_disconnect=False)
        if len(rows) < args.steps+1:  # left the volume
            continue
        counts['traced'] += 1
        stop = first_stop(confs, args.threshold)
        switch = score(rows, confs, w, spec, cfg, args.device)['first_real_switch']
        failures = [(t, kind) for t, kind in ((stop, 'stop'), (switch, 'switch')) if t is not None]
        if not failures:
            counts['none'] += 1
            continue
        step, kind = min(failures)
        counts[kind] += 1
        meta = w['meta']
        records.append(dict(dir=roots.index(Path(meta['root'])), patch=meta['patch'], axis=meta['axis'],
                            sign_u=meta['sign_u'], sign_v=meta['sign_v'], anchor=meta['anchor'],
                            shift=float(max(step-args.before, 0)*cfg.data.step), step=int(step), kind=kind))
    write()
    print(f'{len(records)} hard spots from {counts["traced"]} traces ({counts}) -> {out}', flush=True)


if __name__ == '__main__':
    main()
