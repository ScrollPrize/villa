"""Crop reads by size and access pattern: vc_reader vs zarr-python and TensorStore (local only).

For 64^3, 128^3 and 256^3 crops placed at random over the whole array, read
two ways:

* ``sequential``: one ``read`` after another (an inference loop, a viewer).
* ``batch``: data-loader style, ``read_many`` of a batch of crops at a time
  (zarr-python: a 32-thread pool of single reads; TensorStore: all reads of
  the batch issued, then awaited).

each ``cold`` (the touched chunk files dropped from the page cache before every
repetition) and ``warm``. Runs in about two minutes.

    python benchmarks/bench_crops.py
    python benchmarks/bench_crops.py --array /path/to/array.zarr/0 --reps 5
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

from vc_reader.reader import ZarrReader

DEFAULT_ARRAY = ("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/"
                 "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")

# crop edge -> (crops per repetition, crops per batch)
SIZES = {64: (256, 32), 128: (128, 32), 256: (24, 8)}


def summary(bench: Bench, backends: list[str]) -> str:
    rows = {}
    for r in bench.rows:
        rows.setdefault(r["scenario"], {})[r["backend"]] = r["seconds_median"]
    head = "| crops | pattern | cache | " + " | ".join(backends) + " | vc_reader vs zarr | vc_reader vs TensorStore |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for scenario, t in rows.items():
        size, pattern, cache = scenario.split("-")
        cells = [f"{t[b]:.3f} s" if b in t else "-" for b in backends]
        vs = [f"{t[b] / t['vc_reader']:.1f}x" if b in t else "-" for b in ("zarr", "tensorstore")]
        lines.append(f"| {size} | {pattern} | {cache} | " + " | ".join(cells) + " | " + " | ".join(vs) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-cold", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.array)
    meta = ZarrReader.open(root).meta
    exists = lambda voxel: os.path.exists(  # noqa: E731
        os.path.join(root, meta.chunk_key(tuple(v // c for v, c in zip(voxel, meta.chunks)))))
    print(f"array {root}\n  shape {meta.shape} chunks {meta.chunks} {meta.dtype}", flush=True)

    backends = [("zarr", zarr_python(threads=32)), ("vc_reader", vc_reader())]
    if have("tensorstore"):
        backends.append(("tensorstore", tensorstore()))
    caches = ("warm",) if args.no_cold or filesystem_type(Path(root)) in ("tmpfs", "ramfs") else ("cold", "warm")

    bench = Bench(args.reps)
    for size, (n, per_batch) in SIZES.items():
        ops = random_crops(meta.shape, size, n, seed=size, exists=exists)
        files = chunk_files(root, ops)
        for cache in caches:
            hook = (lambda f=files: evict_files(f)) if cache == "cold" else None
            for pattern in ("sequential", "batch"):
                bench.run(f"{size}^3-{pattern}-{cache}", "ct", root, backends, ops,
                          batch=pattern == "batch", batch_size=per_batch, before_rep=hook,
                          warmup=cache == "warm", reopen_each_rep=cache == "cold")

    save("crops", dict(array=root, shape=meta.shape, chunks=meta.chunks, sizes=SIZES), bench)
    print("\n" + summary(bench, [name for name, _ in backends]))


if __name__ == "__main__":
    main()
