"""Local on-disk zarr reads: vc_reader read modes vs zarr-python (and TensorStore if installed).

Default workload: the local copy of PHercParis4 volume 20260411134726
(2.4 um, 78 keV), pyramid level 2, at
``/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/...``: 382 GB of
uncompressed 128^3 uint8 chunks on an md RAID of NVMe drives, larger than RAM.
Pass ``--array`` for any other local zarr v2 array.

Crops are placed uniformly at random over the whole array, keeping those whose
centre chunk is stored (the volume is masked). ``cold`` scenarios evict every
touched chunk file from the page cache before each repetition
(posix_fadvise DONTNEED); ``warm`` ones run one untimed repetition first. The
array is only ever read.

    python benchmarks/bench_local.py
    python benchmarks/bench_local.py --array /path/to/array.zarr/0 --reps 5
    python benchmarks/bench_local.py --only crops64 --backends vc_reader:slab,vc_reader:mmap
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import (
    Bench,
    chunk_files,
    evict_files,
    filesystem_type,
    have,
    random_crops,
    save,
    tensorstore,
    vc_reader,
    zarr_python,
)

from vc_reader.reader import MemoryChunkCache, ZarrReader

DEFAULT_ARRAY = ("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/"
                 "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", help="comma-separated scenario name filters")
    ap.add_argument("--backends", help="comma-separated backend names to run (default: all)")
    ap.add_argument("--no-cold", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.array)
    probe = ZarrReader.open(root)
    shape, meta = probe.shape, probe.meta
    exists = lambda voxel: os.path.exists(probe.source.path(tuple(v // c for v, c in zip(voxel, meta.chunks))))  # noqa: E731
    print(f"array {root}\n  shape {shape} chunks {meta.chunks} {meta.dtype} compressor={meta.compressor}", flush=True)

    w = args.workers
    all_backends = [
        ("zarr", zarr_python(threads=w)),
        ("vc_reader:slab", vc_reader(workers=w, read_mode="slab")),
        ("vc_reader:pread", vc_reader(workers=w, read_mode="pread")),
        ("vc_reader:mmap", vc_reader(workers=w, read_mode="mmap")),
    ]
    if have("tensorstore"):
        all_backends.append(("tensorstore", tensorstore()))
    if args.backends:
        keep = set(args.backends.split(","))
        all_backends = [b for b in all_backends if b[0] in keep]

    cold_ok = not args.no_cold and filesystem_type(Path(root)) not in ("tmpfs", "ramfs")
    workloads = {
        "crops128": (random_crops(shape, 128, 200, 1, exists), False),
        "crops64": (random_crops(shape, 64, 500, 2, exists), False),
        "batch128": (random_crops(shape, 128, 256, 3, exists), True),
        "block512": (random_crops(shape, 512, 4, 4, exists), False),
    }
    bench = Bench(args.reps)
    for name, (ops, batch) in workloads.items():
        if args.only and not any(f in name for f in args.only.split(",")):
            continue
        files = chunk_files(root, ops)
        modes = ("cold", "warm") if cold_ok else ("warm",)
        for mode in modes:
            hook = (lambda f=files: evict_files(f)) if mode == "cold" else None
            bench.run(f"{name}-{mode}", "ct", root, all_backends, ops, batch=batch, before_rep=hook,
                      warmup=mode == "warm", reopen_each_rep=False)

    if not args.only or "memory" in args.only:
        # Hot loop over a small neighbourhood with vc_reader's memory cache.
        centre = random_crops(shape, 512, 1, 5, exists)[0][0]
        ops = random_crops(shape, 64, 500, 6, exists, lo=centre, hi=tuple(c + 512 for c in centre))
        cache = MemoryChunkCache(4 << 30)
        mem = [b for b in all_backends if b[0] in ("zarr", "vc_reader:slab")]
        mem.append(("vc_reader+memory", vc_reader(workers=w, memory_cache=cache)))
        bench.run("crops64-hot", "ct", root, mem, ops, warmup=True, reopen_each_rep=False)

    save("local", dict(array=root, shape=shape, chunks=meta.chunks, workers=w,
                       filesystem=filesystem_type(Path(root))), bench)


if __name__ == "__main__":
    main()
