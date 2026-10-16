"""Shared benchmark machinery: backends, timing, correctness checks, page-cache eviction."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vc_reader.reader import ZarrReader  # noqa: E402

Region = tuple[tuple[int, ...], tuple[int, ...]]

# -- backends --------------------------------------------------------------------------
# opener(location) -> object with .read(start, size) and .read_many(regions).


class _Threaded:
    """read_many for backends without a batch API: concurrent single reads from a thread pool."""

    def __init__(self, read: Callable[[Sequence[int], Sequence[int]], np.ndarray], threads: int):
        self.read = read
        self._pool = ThreadPoolExecutor(threads)

    def read_many(self, regions: Iterable[Region]) -> list[np.ndarray]:
        return list(self._pool.map(lambda r: self.read(*r), regions))


def zarr_python(concurrency: int | None = None, threads: int = 32):
    def opener(loc: str):
        import zarr

        if concurrency is not None:
            zarr.config.set({"async.concurrency": concurrency})
        else:
            zarr.config.reset()
        if loc.startswith("s3://"):
            from zarr.storage import FsspecStore

            store = FsspecStore.from_url(loc, storage_options={"anon": True}, read_only=True)
        else:
            store = loc
        z = zarr.open_array(store, mode="r")
        return _Threaded(lambda start, size: z[tuple(slice(s, s + n) for s, n in zip(start, size))], threads)

    return opener


def vc_reader(**kwargs):
    return lambda loc: ZarrReader.open(loc, **kwargs)


def tensorstore():
    def opener(loc: str):
        import tensorstore as ts

        if loc.startswith("s3://"):
            bucket, _, path = loc[5:].partition("/")
            kv = {"driver": "s3", "bucket": bucket, "path": path + "/", "aws_region": "us-east-1",
                  "aws_credentials": {"type": "anonymous"}}
        else:
            kv = {"driver": "file", "path": loc + "/"}
        t = ts.open({"driver": "zarr", "kvstore": kv, "context": {"cache_pool": {"total_bytes_limit": 0}}},
                    read=True).result()

        class TS:
            @staticmethod
            def read(start, size):
                return t[tuple(slice(s, s + n) for s, n in zip(start, size))].read().result()

            @staticmethod
            def read_many(regions):
                futs = [t[tuple(slice(s, s + n) for s, n in zip(st, sz))].read() for st, sz in regions]
                return [f.result() for f in futs]

        return TS()

    return opener


def have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# -- page cache ----------------------------------------------------------------------


def evict_files(paths: Iterable[str]) -> int:
    """Drop files from the page cache (clean pages; no root needed). Returns files evicted."""
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("posix_fadvise unavailable: cold-cache scenarios need Linux")
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    return n


def evict_tree(root: Path) -> int:
    return evict_files(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)


def chunk_files(root: str, regions: Iterable[Region]) -> list[str]:
    """The chunk files of a local zarr v2 array that the regions touch."""
    r = ZarrReader.open(root)
    keys = sorted({k for start, size in regions for k in r.chunk_indices(start, size)})
    return [os.path.join(root, r.meta.chunk_key(k)) for k in keys]


def filesystem_type(path: Path) -> str | None:
    """Filesystem type of ``path`` from /proc/mounts (Linux), else None."""
    try:
        mounts = [line.split()[1:3] for line in open("/proc/mounts")]
    except OSError:
        return None
    p = str(path.resolve())
    best = max((m for m in mounts if p == m[0] or p.startswith(m[0].rstrip("/") + "/")), key=lambda m: len(m[0]))
    return best[1]


# -- timing ------------------------------------------------------------------------------


def digest(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=12).hexdigest()


class Bench:
    """Times backends on the same operations; only the reads are timed, never the checks."""

    def __init__(self, reps: int):
        self.reps = reps
        self.rows: list[dict[str, Any]] = []

    def run(
        self,
        scenario: str,
        array: str,
        location: str,
        backends: list[tuple[str, Callable]],
        ops: list[Region],
        *,
        batch: bool = False,
        before_rep: Callable[[], None] | None = None,
        reopen_each_rep: bool = True,
        warmup: bool = False,
        reps: int | None = None,
    ) -> None:
        """Run ``ops`` per backend ``reps`` times, interleaving backends; ``batch`` uses read_many.

        Every backend's output is compared with the first backend's (outside the
        timed region); a mismatch aborts the run. Cold scenarios must use
        ``reopen_each_rep`` so no reader holds pages across the eviction.
        """
        reps = reps or self.reps
        nbytes = sum(int(np.prod(s)) for _, s in ops)
        times: dict[str, list[float]] = {n: [] for n, _ in backends}
        per_op: dict[str, list[float]] = {n: [] for n, _ in backends}
        ref: list[str] | None = None
        readers: dict[str, Any] = {}
        for rep in range(reps + int(warmup)):
            timed = rep >= int(warmup)
            for name, opener in backends:
                if reopen_each_rep or name not in readers:
                    readers[name] = opener(location)
                reader = readers[name]
                if before_rep:
                    before_rep()
                total, outs = 0.0, []
                if batch:
                    t = time.perf_counter()
                    outs = reader.read_many(ops)
                    total = time.perf_counter() - t
                    op_times = [total]
                else:
                    op_times = []
                    for start, size in ops:
                        t = time.perf_counter()
                        a = reader.read(start, size)
                        op_times.append(time.perf_counter() - t)
                        outs.append(a)
                    total = sum(op_times)
                digests = [digest(a) for a in outs]
                del outs
                if reopen_each_rep:
                    # Release the reader now: a live reader may pin pages (e.g. persistent
                    # mmaps), which would keep the next backend's "cold" files cached.
                    dead = readers.pop(name)
                    getattr(dead, "close", lambda: None)()
                    del dead, reader
                if ref is None:
                    ref = digests
                elif digests != ref:
                    raise SystemExit(f"{scenario}/{array}: {name} returned different data than {backends[0][0]}")
                if timed:
                    times[name].append(total)
                    per_op[name].extend(op_times)
                    print(f"  {scenario:<24} {array:<5} {name:<26} rep {rep + 1 - int(warmup)}: {total:8.3f}s "
                          f"{nbytes / 2**20 / total:8.1f} MiB/s", flush=True)
        for name, ts in times.items():
            ops_t = sorted(per_op[name])
            self.rows.append(dict(
                scenario=scenario, array=array, backend=name, reps=len(ts), ops_per_rep=len(ops), batch=batch,
                bytes_per_rep=nbytes, seconds_min=min(ts), seconds_median=statistics.median(ts),
                seconds_max=max(ts), mib_s_median=nbytes / 2**20 / statistics.median(ts),
                op_ms_p50=1e3 * ops_t[len(ops_t) // 2],
                op_ms_p95=1e3 * ops_t[min(len(ops_t) - 1, int(0.95 * len(ops_t)))],
            ))

    def table(self, baseline: str = "zarr") -> str:
        lines = ["| scenario | array | backend | median s (min-max) | MiB/s | op p50 ms | op p95 ms | vs zarr |",
                 "|---|---|---|---|---|---|---|---|"]
        base = {(r["scenario"], r["array"]): r["seconds_median"] for r in self.rows if r["backend"] == baseline}
        for r in self.rows:
            b = base.get((r["scenario"], r["array"]))
            ratio = f"{b / r['seconds_median']:.2f}x" if b else "-"
            lines.append(
                f"| {r['scenario']} | {r['array']} | {r['backend']} | {r['seconds_median']:.3f} "
                f"({r['seconds_min']:.3f}-{r['seconds_max']:.3f}) | {r['mib_s_median']:.0f} | "
                f"{r['op_ms_p50']:.2f} | {r['op_ms_p95']:.2f} | {ratio} |"
            )
        return "\n".join(lines)


def environment() -> dict[str, Any]:
    env = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
               cpus=os.cpu_count())
    from importlib.metadata import PackageNotFoundError, version

    for m in ("vc-reader", "numpy", "zarr", "numcodecs", "s3fs", "fsspec", "aiohttp", "tensorstore"):
        try:
            env[m] = version(m)
        except PackageNotFoundError:
            env[m] = None
    return env


def save(name: str, record: dict[str, Any], bench: Bench) -> Path:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    record = dict(command=" ".join([sys.executable, *sys.argv]), environment=environment(), **record, rows=bench.rows)
    path.write_text(json.dumps(record, indent=1))
    print(bench.table())
    print(f"\nwrote {path}")
    return path


def random_crops(shape: Sequence[int], size: int, n: int, seed: int, exists: Callable[[tuple[int, ...]], bool],
                 lo: Sequence[int] | None = None, hi: Sequence[int] | None = None) -> list[Region]:
    """``n`` random ``size``^3 crops inside ``[lo, hi)`` whose centre voxel's chunk is stored."""
    rng = np.random.default_rng(seed)
    lo = lo or (0,) * len(shape)
    hi = hi or tuple(shape)
    out: list[Region] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 1000 * n:
            raise RuntimeError("could not find enough populated crops")
        start = tuple(int(rng.integers(a, b - size)) for a, b in zip(lo, hi))
        if exists(tuple(s + size // 2 for s in start)):
            out.append((start, (size,) * len(shape)))
    return out
