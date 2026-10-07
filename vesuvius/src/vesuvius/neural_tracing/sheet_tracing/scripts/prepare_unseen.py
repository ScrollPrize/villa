"""Prepare a scroll the models never trained on for the trace benchmark (open data, vesuvius-challenge-open-data).

Downloads the largest segments of ``--scroll`` (tifxyz) and, from the volume ``--volume-id``, only the 128^3 chunks
the benchmark's traces can reach, into a local OME-Zarr whose level 0 / 1 are the remote levels ``--levels`` (pick
the ones at 9.6 um / 19.2 um, the training volume's level 0 / 1; chunking and compression are kept as they are).
Segment coordinates are multiplied by ``--coord-scale`` to land in local level-0 voxels (e.g. 0.25 for segments in
full-resolution 2.4 um coordinates).
The seeds are drawn exactly as ``trace_benchmark.py --patches-dir OUT/patches --all-val --repeat-patches`` will draw
them with the same ``--checkpoint`` first, ``--seeds``, ``--steps`` and ``--front``.

  python scripts/prepare_unseen.py --metadata metadata.json --scroll PHerc1667 --volume-id 20251217075048 \\
      --segment-type tifxyz-transformed --levels 2 3 --coord-scale 0.25 --checkpoint output/remote_l0_points_ft/last.pt \\
      --out data/unseen/PHerc1667

The plain ``tifxyz`` segments are in the coordinates of whatever volume they were traced on, which need not be
``--volume-id``: use ``tifxyz-transformed`` (registered onto it) unless that is known.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from open_data import ROOT, chunk_name, download_chunks, mirror_array, read_segment, segment_path, write_segment


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metadata', required=True, help='open-data metadata.json (decompressed)')
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--volume-id', required=True)
    ap.add_argument('--segment-type', default='tifxyz', help="'tifxyz' (as traced) or 'tifxyz-transformed' (on --volume-id)")
    ap.add_argument('--coord-scale', type=float, default=1.)
    ap.add_argument('--levels', type=int, nargs=2, default=[2, 3], help='remote levels for local levels 0 and 1')
    ap.add_argument('--segments', type=int, default=12, help='largest segments to use')
    ap.add_argument('--checkpoint', required=True, help='the benchmark\'s first checkpoint (its config draws the seeds)')
    ap.add_argument('--seeds', type=int, default=60)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--front', type=float, default=240.)
    ap.add_argument('--pad', type=float, default=150., help='voxels around each seed\'s reachable GT region')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args(argv)
    from vesuvius.neural_tracing.sheet_tracing.evaluate import load_checkpoint
    from vesuvius.neural_tracing.sheet_tracing.trace_benchmark import draw_seeds

    meta = json.loads(Path(args.metadata).read_text())['samples'][args.scroll]
    volume = meta['volumes'][args.volume_id]
    vpath = volume['data'][0]['origins'][0]['path'].rstrip('/')
    out = Path(args.out)
    patches = out/'patches'
    patches.mkdir(parents=True, exist_ok=True)

    # 1. the largest segments, coordinates in local level-0 voxels
    segs = sorted(meta['segments'].values(), key=lambda g: -g['properties'].get('width', 0)*g['properties'].get('height', 0))
    chosen = []
    for g in segs:
        path = segment_path(g, args.segment_type, args.volume_id)
        if path is None:
            continue
        dst = patches/g['long_id']
        if not (dst/'z.tif').is_file():
            write_segment(dst, *read_segment(path, args.coord_scale))
        chosen.append(g['long_id'])
        if len(chosen) == args.segments:
            break
    print(f'{len(chosen)} segments -> {patches}', flush=True)

    # 2. the benchmark's seeds, and the chunks their traces can reach
    _, cfg, _ = load_checkpoint(args.checkpoint, 'cpu')
    cfg.data.patches_dir = [str(patches)]
    seeds, spec, data = draw_seeds(cfg, args.seeds, args.steps, args.front, .95, distinct=False, all_val=True)
    print(f'{len(seeds)} seeds from {len({s["meta"]["patch"] for s in seeds})} segments', flush=True)
    boxes = [(w['reach_lo']-args.pad, w['reach_hi']+args.pad) for w in seeds]  # xyz, local level 0

    # 3. download the chunks they cover into a local zarr (local level k = remote level args.levels[k])
    jobs = []
    for name, path in [('volume', vpath)]:
        zarr = out/f'{name}.zarr'
        for level, remote in enumerate(args.levels):
            zarray = mirror_array(f'{ROOT}/{path}/{remote}', zarr/str(level),
                                  dict(source=f'{ROOT}/{path}', levels=args.levels, scroll=args.scroll))
            size = np.asarray(zarray['chunks'])
            keys = set()
            for lo, hi in boxes:
                k0 = np.floor(np.maximum(lo[::-1], 0)/(1 << level)/size).astype(int)
                k1 = np.floor(hi[::-1]/(1 << level)/size).astype(int)
                keys.update(tuple(int(v) for v in k0+np.array(o)) for o in np.ndindex(*(k1-k0+1)))
            print(f'{name} level {level}: {len(keys)} chunks of {size.tolist()}', flush=True)
            for key in keys:
                chunk = chunk_name(zarray, key)
                dst = zarr/str(level)/chunk
                if not dst.is_file():
                    jobs.append((f'{ROOT}/{path}/{remote}/{chunk}', dst))

    got = download_chunks(jobs, args.workers)
    print(f'downloaded {sum(1 for g in got if g)} chunks ({sum(got)/2**30:.2f} GiB), {sum(1 for g in got if not g)} absent '
          f'-> {out}', flush=True)
    (out/'seeds.json').write_text(json.dumps(dict(checkpoint=args.checkpoint, seeds=args.seeds, steps=args.steps,
                                                  front=args.front, segments=chosen, metas=[s['meta'] for s in seeds])))


if __name__ == '__main__':
    main()
