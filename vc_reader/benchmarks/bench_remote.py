"""Remote (S3) zarr reads: vc_reader vs zarr-python (and TensorStore if installed).

Workload: real Vesuvius Challenge data on the public open-data bucket, read
inside the 1024^3 voxel region starting at (z, y, x) = (8192, 3584, 3584),
which is fully populated for both arrays:

* ``ct``: PHercParis4 volume 20260411134726 (2.4 um, 78 keV), pyramid level 2.
  Uncompressed uint8, 128^3 chunks (2 MiB each).
* ``pred``: that volume's surface prediction computed on level 2
  (model 20260413222639). blosc-zstd uint8, 192^3 chunks.

Scenarios:

* ``block``: one aligned 256x512x512 block.
* ``crops128``: random unaligned 128^3 crops, one after another.
* ``batch128``: the same number of crops requested at once (``read_many``;
  a 32-thread pool of single reads for zarr-python).
* ``disk-fill`` / ``disk-warm``: vc_reader's disk cache, filled from S3 on
  the first pass and served from disk afterwards.

Results depend heavily on the network path to us-east-1; backends are
interleaved within each repetition so they see the same conditions.

    python benchmarks/bench_remote.py
    python benchmarks/bench_remote.py --arrays ct --reps 5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _common import Bench, have, random_crops, save, tensorstore, vc_reader, zarr_python

HTTPS = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
ARRAYS = {
    "ct": "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2",
    "pred": "PHercParis4/representations/predictions/surfaces/"
    "20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0",
}
REGION_START = (8192, 3584, 3584)
REGION_SIZE = (1024, 1024, 1024)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32, help="vc_reader worker threads / zarr-python concurrency")
    ap.add_argument("--crops", type=int, default=12)
    ap.add_argument("--arrays", default="ct,pred")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "vc_reader" / "bench-remote",
                    help="vc_reader disk cache for the disk-* scenarios (emptied first)")
    args = ap.parse_args(argv)

    w = args.workers
    backends = [
        ("zarr", zarr_python(threads=w)),
        (f"zarr (concurrency={w})", zarr_python(concurrency=w, threads=w)),
        ("vc_reader", vc_reader(workers=w)),
        ("vc_reader:native", vc_reader(workers=w, engine="native")),
    ]
    if have("tensorstore"):
        backends.append(("tensorstore", tensorstore()))

    hi = tuple(s + n for s, n in zip(REGION_START, REGION_SIZE))
    crops = random_crops(hi, 128, args.crops, 1, lambda v: True, lo=REGION_START, hi=hi)
    block = [(REGION_START, (256, 512, 512))]

    bench = Bench(args.reps)
    for name in args.arrays.split(","):
        s3 = f"s3://vesuvius-challenge-open-data/{ARRAYS[name]}"
        bench.run("block", name, s3, backends, block)
        bench.run("crops128", name, s3, backends, crops)
        bench.run("batch128", name, s3, backends, crops, batch=True)

        cache = args.cache_dir / name
        shutil.rmtree(cache, ignore_errors=True)
        native_cache = args.cache_dir / (name + "-native")
        shutil.rmtree(native_cache, ignore_errors=True)
        disk = [("vc_reader+disk", vc_reader(workers=w, cache_dir=cache)),
                ("vc_reader:native+disk", vc_reader(workers=w, engine="native", cache_dir=native_cache))]
        bench.run("disk-fill-batch128", name, s3, [backends[0], *disk], crops, batch=True, reps=1)
        bench.run("disk-warm-batch128", name, s3, [backends[0], *disk], crops, batch=True)

    save("remote", dict(arrays={k: f"{HTTPS}/{v}" for k, v in ARRAYS.items()}, region_start=REGION_START,
                        region_size=REGION_SIZE, workers=w), bench)


if __name__ == "__main__":
    main()
