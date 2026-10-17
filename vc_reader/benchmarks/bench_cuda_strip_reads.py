"""CUDA affine read latency on the sheet-strip trainer's crop geometry.

`export` turns a bench_fiber_strip.py capture into plain grids (requires the
trainer environment). `run` replays those grids through
CudaReader.read_affine_many only, with the GPU cache cleared before every batch
(outside timing), and hashes every output. It needs only vc_reader; run it via
run_native.py to compare builds. `--compare` enforces identical output hashes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from _common import environment
from bench_cuda_sample import stats


def export(args):
    import torch
    from bench_fiber_strip import grids_for, unpack
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume, CTVolumeSpec
    capture = torch.load(args.capture, weights_only=False)
    spec, crop = CTVolumeSpec.from_dict(capture['spec']), capture['crop']
    scale = CTVolume(spec).input_scale
    rows = [(b, *g) for b, req in enumerate(capture['requests']) for g in grids_for(unpack(req), crop, scale)]
    np.savez(args.output, batch=np.array([r[0] for r in rows]), shapes=np.array([r[1] for r in rows]),
             origins=np.array([r[2] for r in rows], np.float64), axes=np.array([r[3] for r in rows], np.float64),
             array=f'{spec.ct_zarr}/{spec.ct_level}')
    print(f'{len(rows)} crops in {len(capture["requests"])} batches -> {args.output}')


def run(args):
    from vc_reader.reader import ZarrReader
    g = np.load(args.grids)
    batches = [[(tuple(int(v) for v in g['shapes'][i]), g['origins'][i], g['axes'][i])
                for i in np.flatnonzero(g['batch'] == b)] for b in range(int(g['batch'].max()) + 1)]
    array = args.array or str(g['array'])
    times, uploads, hashes = [], [], []
    with ZarrReader.open(array, threads=args.threads, memory_cache=args.memory_cache or None) as reader, \
            reader.cuda(cache_bytes=int(args.cache_gb * (1 << 30))) as gpu:
        for rep in range(-args.warmup, args.reps):
            for b, grids in enumerate(batches):
                gpu.clear_cache()
                before = gpu.stats['uploaded_bytes']
                start = time.perf_counter()
                out = gpu.read_affine_many(grids, source='chunks')
                elapsed = (time.perf_counter() - start) * 1000
                if rep >= 0:
                    times.append(elapsed)
                    uploads.append(gpu.stats['uploaded_bytes'] - before)
                if rep == -args.warmup:
                    digest = hashlib.sha256()
                    for o in out:
                        digest.update(o.numpy().tobytes())
                    hashes.append(digest.hexdigest())
                del out
        cache = gpu.stats
    env = environment()
    report = dict(environment=env, command=sys.argv, array=array, grids=str(args.grids), reps=args.reps,
                  warmup=args.warmup, threads=args.threads, memory_cache=args.memory_cache,
                  binary_sha256=hashlib.sha256(Path(env['native_extension']).read_bytes()).hexdigest(),
                  protocol='GPU cache cleared before each batch outside timing; synchronous call including planning',
                  batch_crops=[len(b) for b in batches], timing=stats(times),
                  uploaded_mib_per_batch=float(np.mean(uploads)) / 2**20, cuda_stats=cache, sha256=hashes)
    if args.compare:
        reference = json.loads(args.compare.read_text())['sha256']
        report['matches'] = reference == hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    t = report['timing']
    print(f"mean/p50/p95 {t['mean_ms']:.2f}/{t['p50_ms']:.2f}/{t['p95_ms']:.2f} ms, "
          f"{report['uploaded_mib_per_batch']:.1f} MiB uploaded per batch")
    if args.compare and not report['matches']:
        raise SystemExit('output hashes differ from --compare')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    e = sub.add_parser('export')
    e.add_argument('--capture', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--grids', type=Path, required=True)
    r.add_argument('--array', help='override the CT array recorded in the grids')
    r.add_argument('--reps', type=int, default=5)
    r.add_argument('--warmup', type=int, default=1)
    r.add_argument('--threads', type=int, default=4)
    r.add_argument('--memory-cache', type=int, default=384 << 20)
    r.add_argument('--cache-gb', type=float, default=4.)
    r.add_argument('--compare', type=Path)
    r.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'export':
        export(args)
    else:
        if args.reps < 1 or args.warmup < 0 or args.threads < 1:
            ap.error('invalid counts')
        run(args)


if __name__ == '__main__':
    main()
