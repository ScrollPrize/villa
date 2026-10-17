"""Time public Elastic on the three Paris 4 crops; select a build with run_native.py."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from vc_reader import augment as A, _augment
from compare import ARRAY, CHUNKS, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--chunks', nargs='+', choices=list(CHUNKS), default=list(CHUNKS))
    parser.add_argument('--scales', nargs='+', type=float, default=[8, 16, 32])
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 0:
        parser.error('reps must be positive and warmup nonnegative')
    report = dict(array=str(ARRAY), reps=args.reps, warmup=args.warmup,
                  extension=str(_augment.__file__), extension_sha256=hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest(),
                  shape=[128]*3, seed=123, border=True, magnitude=8, rows=[])
    for name in args.chunks:
        chunk = CHUNKS[name]
        raw = np.fromfile(ARRAY.joinpath(*map(str, chunk)), dtype=np.uint8).reshape((128,)*3)
        image = raw.astype(np.float32)[None]/255
        original = image.copy()
        for scale in args.scales:
            for labels in (False, True):
                sample = dict(image=image)
                if labels: sample['mask'] = (raw[None]//64).astype(np.int16)
                pipeline = A.Pipeline([A.Elastic(sigma=scale, magnitude=8, border=True)])
                plan = pipeline.sample(sample, seed=123)
                totals, replays = [], []
                for rep in range(-args.warmup, args.reps):
                    start=time.perf_counter(); result=pipeline(sample, seed=123); elapsed=time.perf_counter()-start
                    start=time.perf_counter(); replay=plan.apply(sample); replay_elapsed=time.perf_counter()-start
                    for key in sample: np.testing.assert_array_equal(result[key], replay[key])
                    if rep>=0: totals.append(elapsed*1000); replays.append(replay_elapsed*1000)
                row=dict(chunk=name, scale=scale, labels=labels, total=stats(totals), replay=stats(replays),
                         hashes={key: hashlib.sha256(value).hexdigest() for key,value in result.items()},
                         plan_sha256=hashlib.sha256(plan._native.dump()).hexdigest())
                report['rows'].append(row)
                print(name,scale,'image+label' if labels else 'image',round(row['total']['p50_ms'],2),'ms',flush=True)
                del plan, result, replay
        np.testing.assert_array_equal(image, original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
