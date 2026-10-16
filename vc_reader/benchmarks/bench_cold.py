"""Maximum cold-read throughput of whole 128^3 chunks: vc_reader vs zarr-python and TensorStore.

Reads only whole, aligned chunks (nothing is over-read) of a local array:

* ``random``: 1024 chunks chosen at random over the whole array, in batches of
  64 (``read_many``; a thread pool of single reads for zarr-python; all the
  batch's futures at once for TensorStore).
* ``blocks``: two contiguous aligned blocks of 512 x 1024 x 1024 voxels
  (256 chunks each), one read call per block.

Before every repetition of every backend, every touched chunk file is dropped
from the page cache (posix_fadvise DONTNEED) and the previous reader is
released, so every read comes from the drives. A ``raw pread`` row reads the
same files with plain parallel preads (no copy, no decode): the drives' own
ceiling for these reads.

    python benchmarks/bench_cold.py
    python benchmarks/bench_cold.py --array /path/to/array.zarr/0 --reps 5
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from _common import Bench, chunk_files, evict_files, filesystem_type, have, save, tensorstore, vc_reader, zarr_python

from vc_reader.reader import ZarrReader

DEFAULT_ARRAY = ("/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/"
                 "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
N_CHUNKS, BATCH = 1024, 64


def whole_grid(meta) -> tuple[int, ...]:
    """Chunk-grid extent of the chunks lying fully inside the array (edge chunks are partial)."""
    return tuple(s // c for s, c in zip(meta.shape, meta.chunks))


def stored_chunks(root: str, meta, n: int, seed: int) -> list[tuple[int, ...]]:
    """``n`` distinct stored whole chunks, uniformly at random over the array."""
    rng = random.Random(seed)
    grid = whole_grid(meta)
    found: set[tuple[int, ...]] = set()
    while len(found) < n:
        idx = tuple(rng.randrange(g) for g in grid)
        if os.path.exists(os.path.join(root, meta.chunk_key(idx))):
            found.add(idx)
    return sorted(found)


def full_blocks(root: str, meta, n: int, blocks: tuple[int, ...], seed: int) -> list[tuple[int, ...]]:
    """Origins (in chunks) of ``n`` aligned blocks of ``blocks`` chunks that are fully stored."""
    rng = random.Random(seed)
    out: list[tuple[int, ...]] = []
    while len(out) < n:
        o = tuple(rng.randrange(g - b + 1) for g, b in zip(whole_grid(meta), blocks))
        keys = [tuple(o[i] + d[i] for i in range(3)) for d in np.ndindex(*blocks)]
        if all(os.path.exists(os.path.join(root, meta.chunk_key(k))) for k in keys):
            out.append(o)
    return out


def raw_pread(files: list[str], threads: int, reps: int) -> float:
    """Median GB/s of plain parallel whole-file preads after eviction."""
    tls = threading.local()

    def rd(path):
        buf = getattr(tls, "buf", None)
        if buf is None:
            buf = tls.buf = np.empty(2 << 20, np.uint8)
        fd = os.open(path, os.O_RDONLY)
        try:
            return os.preadv(fd, [memoryview(buf)], 0)
        finally:
            os.close(fd)

    rates = []
    for _ in range(reps):
        evict_files(files)
        with ThreadPoolExecutor(threads) as ex:
            t = time.perf_counter()
            n = sum(ex.map(rd, files))
            rates.append(n / (time.perf_counter() - t) / 1e9)
    return statistics.median(rates)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", default=DEFAULT_ARRAY)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)

    root = os.path.abspath(args.array)
    meta = ZarrReader.open(root).meta
    if meta.chunks != (128, 128, 128) or not meta.is_raw:
        raise SystemExit(f"expected uncompressed 128^3 chunks, got {meta.chunks} {meta.compressor}")
    if filesystem_type(Path(root)) in ("tmpfs", "ramfs"):
        raise SystemExit(f"{root} is on tmpfs: there is no cold read to measure")
    print(f"array {root}\n  shape {meta.shape} chunks {meta.chunks} {meta.dtype}", flush=True)

    backends = [
        ("zarr (32 threads)", zarr_python(threads=32)),
        ("zarr (64 threads)", zarr_python(threads=64)),
        ("vc_reader (16 threads)", vc_reader(threads=16)),
        ("vc_reader (32 threads)", vc_reader(threads=32)),
    ]
    if have("tensorstore"):
        backends.append(("tensorstore", tensorstore()))

    c = 128
    keys = stored_chunks(root, meta, N_CHUNKS, seed=11)
    random_ops = [(tuple(i * c for i in k), (c, c, c)) for k in keys]
    origins = full_blocks(root, meta, 2, (4, 8, 8), seed=12)
    block_ops = [(tuple(o * c for o in org), (4 * c, 8 * c, 8 * c)) for org in origins]

    bench = Bench(args.reps)
    rows = []
    for pattern, ops, batch in (("random", random_ops, BATCH), ("blocks", block_ops, None)):
        files = chunk_files(root, ops)
        bench.run(f"cold-{pattern}", "ct", root, backends, ops, batch=batch is not None, batch_size=batch,
                  before_rep=lambda f=files: evict_files(f), reopen_each_rep=True)
        nbytes = sum(int(np.prod(s)) for _, s in ops)
        rows.append((pattern, len(files), nbytes, raw_pread(files, 32, args.reps)))

    save("cold", dict(array=root, n_chunks=N_CHUNKS, batch=BATCH), bench)
    names = [n for n, _ in backends]
    print("\nCold throughput, GB/s of output (median of %d), higher is better\n" % args.reps)
    print("| pattern | data | " + " | ".join(names) + " | raw pread (ceiling) |")
    print("|" + "---|" * (len(names) + 3))
    for pattern, n_files, nbytes, raw in rows:
        t = {r["backend"]: r["seconds_median"] for r in bench.rows if r["scenario"] == f"cold-{pattern}"}
        cells = [f"{nbytes / t[n] / 1e9:.2f}" for n in names]
        print(f"| {pattern} | {n_files} chunks, {nbytes / 2**30:.1f} GiB | " + " | ".join(cells) + f" | {raw:.2f} |")


if __name__ == "__main__":
    main()
