"""Compare a selected lab build (or original Python) on the saved GIF workload."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import _elastic_lab as lab
from compare import ARRAY, CHUNKS, EDGE, HALO, BOUND, EMPTY, read_crop, stats, reset, python_transform, python_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['bspline', 'python'], default='bspline')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--scales', type=float, nargs='+', default=[8, 16, 32])
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 0:
        parser.error('reps must be positive and warmup nonnegative')
    torch.set_num_threads(1)
    report = dict(array=str(ARRAY), method=args.method, reps=args.reps, warmup=args.warmup,
                  extension=str(lab.__file__),
                  extension_sha256=hashlib.sha256(Path(lab.__file__).read_bytes()).hexdigest(),
                  output_shape=[EDGE]*3, halo=HALO, seed=123,
                  strength='RMS 8, gradient cap 0.8' if args.method == 'bspline' else 'legacy magnitude 4', rows=[])
    for name, chunk in CHUNKS.items():
        raw, _ = read_crop(ARRAY, chunk)
        image = raw.astype(np.float32)/255
        tensor = torch.from_numpy(image[None])
        for scale in args.scales:
            transform = python_transform(scale) if args.method == 'python' else None
            samples = dict(total=[], field=[], warp=[])
            for rep in range(-args.warmup, args.reps):
                reset(123)
                start = time.perf_counter()
                record = (python_run(transform, tensor) if transform is not None else
                          lab.run(image, 'bspline', EDGE, float(HALO), scale, 8., BOUND, 123, EMPTY))
                elapsed = (time.perf_counter()-start)*1000
                if rep >= 0:
                    samples['total'].append(elapsed)
                    samples['field'].append(record['field_ms'])
                    samples['warp'].append(record['warp_ms'])
                if rep == args.reps-1:
                    hashes = {key: hashlib.sha256(record[key]).hexdigest()
                              for key in ['image', 'field'] if record[key] is not None}
                del record
            row = dict(chunk=name, scale=scale, hashes=hashes,
                       input_sha256=hashlib.sha256(raw).hexdigest(),
                       **{key: stats(value) for key, value in samples.items()})
            report['rows'].append(row)
            print(name, scale, round(row['total']['p50_ms'], 2), 'ms', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
