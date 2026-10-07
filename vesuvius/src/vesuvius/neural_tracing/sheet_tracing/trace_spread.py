"""Long traces from a spread of held-out seeds, for visual inspection (e.g. in VC3D).

Seed lines come from validation-split patches (the traced region's seed row lies in val split cells), one per
patch, chosen evenly across the scroll's z range. Each seed is ``--front`` voxels of a patch grid row with the
patch's own advance direction; tracing runs through the inference path for ``--steps`` steps (the volume block
is read around the front each step, so traces continue past the verified patch). Output in ``output/spread/TAG``:
one tifxyz per seed (``sNN_z<z>_<patch>``; stopped points written as -1, plus confidence.npy / trace.json), the
seed rows as tifxyz (``sNN_seed``), and summary.json (seed, patch, steps traced, kept fraction per step).

  python -m vesuvius.neural_tracing.sheet_tracing.trace_spread --checkpoint output/RUN/last.pt --seeds 10 \\
      --steps 120 --tag RUN_spread
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from vesuvius.neural_tracing.sheet_tracing import config as C
from vesuvius.neural_tracing.sheet_tracing.data.episodes import EpisodeSampler
from vesuvius.neural_tracing.sheet_tracing.data.patches import build_index
from vesuvius.neural_tracing.sheet_tracing.evaluate import load_checkpoint
from vesuvius.neural_tracing.sheet_tracing.infer import connected_keep, resample, trace, write_surface, write_trace
from vesuvius.neural_tracing.sheet_tracing.trace_benchmark import seed_polyline


def spread_seeds(cfg, count: int, front: float, pool: int = 400, seed: int = 0):
    """``count`` held-out seed windows, distinct patches, evenly spread over seed z."""
    data = copy.deepcopy(cfg.data)
    data.point_spacing = 20.
    data.front_points = int(round(front/data.point_spacing))+1
    data.min_valid_fraction = .9
    data.max_tries = 2000
    sampler = EpisodeSampler(data, build_index(data.patches_dir, C.OUTPUT/'cache'), 'val', 2, roi=0, pad=0,
                             region_split=True)
    rng = np.random.default_rng([seed, 11])
    windows, seen = [], set()
    for _ in range(pool*3):
        w = sampler.window(rng)
        if w is None:
            continue
        key = (w['meta'].get('root'), w['meta']['patch'])
        if key in seen:
            continue
        seen.add(key)
        points, _ = seed_polyline(w, sampler.spec, data)
        w['z'] = float(points[:, 2].mean())
        windows.append(w)
        if len(windows) >= pool:
            break
    windows.sort(key=lambda w: w['z'])
    picks = np.unique(np.linspace(0, len(windows)-1, count).round().astype(int))
    return [windows[i] for i in picks], sampler.spec, data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--front', type=float, default=240.)
    ap.add_argument('--threshold', type=float, help='confidence threshold (default: the checkpoint\'s)')
    ap.add_argument('--volume', help='volume path (default: the checkpoint\'s)')
    ap.add_argument('--patches-dir', nargs='+', help='patch directories (default: the checkpoint\'s)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW',
                    help='replace the path prefix OLD by NEW in the checkpoint\'s data paths; repeatable')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    model, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    C.path_mapper(args.path_map)(cfg)
    if args.volume:
        cfg.data.volume = args.volume
    if args.patches_dir:
        cfg.data.patches_dir = args.patches_dir
    if args.threshold is not None:
        cfg.trace.confidence_threshold = args.threshold
    windows, spec, seed_data = spread_seeds(cfg, args.seeds, args.front)
    out = C.OUTPUT/'spread'/args.tag
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(checkpoint=args.checkpoint, steps=args.steps, front=args.front,
                   threshold=cfg.trace.confidence_threshold, seeds=[])
    for i, w in enumerate(windows):
        points, direction = seed_polyline(w, spec, seed_data)
        seed = resample(points, cfg.data.point_spacing)
        idx = np.linspace(0, len(points)-1, len(seed))
        dirs = np.stack([np.interp(idx, np.arange(len(points)), direction[:, k]) for k in range(3)], 1)
        rows, confs = trace(model, cfg, seed, dirs, args.steps, args.device, min_confident=0., patience=10**9)
        name = f's{i:02d}_z{int(w["z"])}_{w["meta"]["patch"].removesuffix(".tifxyz")[:40]}'
        meta = dict(w['meta'], seed_z=w['z'], steps_traced=len(rows)-1)
        write_trace(out/name, rows, confs, cfg, cfg.trace.confidence_threshold, dict(meta, checkpoint=args.checkpoint))
        write_surface(out/f's{i:02d}_seed', points[None], np.ones((1, len(points)), bool), 1., spec.res,
                      dict(meta=w['meta']), overwrite=True, voxel_size_um=cfg.data.voxel_size_um)
        kept = connected_keep(confs, cfg.trace.confidence_threshold)[1:].mean(1)
        summary['seeds'].append(dict(meta, name=name, kept_per_step=[round(float(k), 3) for k in kept]))
        print(f'{name}: {len(rows)-1} steps, kept {kept.mean():.2f} (last 10 steps {kept[-10:].mean():.2f})',
              flush=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
