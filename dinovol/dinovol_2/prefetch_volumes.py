"""Cache coarse masks first, then mask-selected training chunks from a config.

Run from the repository root:
    python -m dinovol_2.prefetch_volumes --config CONFIG.json --connections 16

All occupancy levels are fetched before training chunks. Nonzero coarse voxels
select finer chunks with a boundary margin; --no-mask-filter fetches full levels.
Existing chunks and cached missing-chunk markers are reused by vc_reader.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from tqdm import tqdm

# Direct execution from dinovol_2/ must use this checkout's package, rather
# than an older non-editable copy installed in site-packages.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinovol_2.dataset.volume_reader import open_volume


@dataclass
class VolumeLevel:
    sample: str
    path: str
    level: int
    cache_dir: str
    auth: Any
    storage_options: dict | None

    def open(self, *, connections=16, cache_writes=True, compress_cache=False, offline=False):
        return open_volume(self.path, self.level, self.auth, self.storage_options,
                           cache_dir=self.cache_dir, connections=connections,
                           cache_writes=cache_writes, compress_cache=compress_cache, offline=offline)


@dataclass
class LevelPlan:
    source: VolumeLevel
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    itemsize: int
    selected: np.ndarray | None = None

    @property
    def grid(self):
        return tuple((size + chunk - 1) // chunk for size, chunk in zip(self.shape, self.chunks))

    @property
    def count(self):
        return math.prod(self.grid) if self.selected is None else int(self.selected.sum())

    @property
    def dense_bytes(self):
        # Includes padded edge chunks; excludes sparse missing chunks, compression,
        # filesystem overhead, and any subtraction for data already cached.
        return self.count * math.prod(self.chunks) * self.itemsize

    def regions(self):
        if self.selected is None:
            indices = itertools.product(*(range(size) for size in self.grid))
        else:
            indices = ((z, y, int(x)) for z in range(self.grid[0]) for y in range(self.grid[1])
                       for x in np.flatnonzero(self.selected[z, y]))
        for index in indices:
            start = tuple(i * chunk for i, chunk in zip(index, self.chunks))
            size = tuple(min(chunk, dim - offset)
                         for chunk, dim, offset in zip(self.chunks, self.shape, start))
            yield start, size


def configured_levels(config, *, cache_dir=None, samples=(), scope="all"):
    """Resolve flat/sectioned training configs without constructing a model/dataset."""
    if scope not in ("all", "training", "occupancy"):
        raise ValueError(f"Unknown scope: {scope}")
    data = config.get("data", config)
    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("datasets"), list):
        raise ValueError("Expected data.dataset.datasets (or legacy dataset.datasets)")
    root = cache_dir or dataset.get("volume_cache_dir")
    if not root:
        raise ValueError("Set data.dataset.volume_cache_dir or pass --cache-dir")
    root = str(Path(root).expanduser().resolve())
    requested = set(samples)
    found = set()
    levels = {}
    for entry in dataset["datasets"]:
        sample = str(entry.get("sample_id", entry["volume_path"]))
        if requested and sample not in requested:
            continue
        found.add(sample)
        path = str(entry["volume_path"]).rstrip("/")
        if not path.startswith(("s3://", "http://", "https://")):
            continue  # Local volumes already reside on disk.
        selected = []
        if scope != "training" and dataset.get("occupancy_level") is not None:
            selected.append(int(dataset["occupancy_level"]))
        if scope != "occupancy":
            selected.append(int(entry["volume_scale"]))
        for level in selected:
            if level < 0:
                raise ValueError(f"Negative pyramid level for {sample}: {level}")
            levels.setdefault((path, level), VolumeLevel(
                sample, path, level, root, dataset.get("volume_auth"), dataset.get("s3_storage_options")))
    if requested - found:
        raise ValueError(f"Unknown sample IDs: {', '.join(sorted(requested - found))}")
    return list(levels.values())


def staged_sources(config, *, cache_dir=None, samples=(), scope="all", mask_filter=True):
    """Include coarse prerequisites even with --scope training, globally first."""
    kwargs = dict(cache_dir=cache_dir, samples=samples)
    coarse = configured_levels(config, scope="occupancy", **kwargs)
    training = configured_levels(config, scope="training", **kwargs) if scope != "occupancy" else []
    if scope == "training" and not mask_filter:
        coarse = []
    keys = {(source.path, source.level) for source in coarse}
    return coarse, [source for source in training if (source.path, source.level) not in keys]


def select_mask_chunks(plan, coarse, *, padding=1, slab_bytes=64 << 20, progress=None):
    """Project nonzero coarse cells onto fine chunks, streaming bounded Z slabs.

    The mask is a prefetch heuristic, never evidence to write missing-chunk
    markers. Downsampling can hide thin structures; normal demand reads remain
    available for skipped chunks. Bounds use the same power-of-two pyramid
    scaling as training, including non-aligned chunk edges.
    """
    if padding < 0 or coarse.source.level < plan.source.level:
        raise ValueError("Mask padding must be nonnegative and mask level must be coarser than training")
    factor = 2 ** (coarse.source.level - plan.source.level)
    expected = tuple((dim + factor - 1) // factor for dim in plan.shape)
    if coarse.shape != expected:
        raise ValueError(f"Mask shape {coarse.shape} does not match pyramid shape {expected}: {plan.source.path}")
    bounds = []
    for dim, chunk, count, mask_dim in zip(plan.shape, plan.chunks, plan.grid, coarse.shape):
        starts = np.arange(count) * chunk
        lo = np.maximum(0, starts // factor - padding)
        hi = np.minimum(mask_dim, (np.minimum(starts + chunk, dim) + factor - 1) // factor + padding)
        bounds.append((lo, hi))
    selected = np.zeros(plan.grid, dtype=bool)
    zlo, zhi = bounds[0]
    ylo, yhi = (a[:, None] for a in bounds[1])
    xlo, xhi = (a[None, :] for a in bounds[2])
    slab_depth = max(1, min(coarse.chunks[0], slab_bytes // (math.prod(coarse.shape[1:]) * coarse.itemsize)))
    # Phase one has persisted these chunks. A partial/evicted mask must fail,
    # rather than silently become empty or trigger more downloads here.
    with coarse.source.open(cache_writes=False, offline=True) as reader:
        for start in range(0, coarse.shape[0], slab_depth):
            stop = min(start + slab_depth, coarse.shape[0])
            occupied = np.asarray(reader[start:stop]) > 0
            for z in np.flatnonzero((zlo < stop) & (zhi > start)):
                plane = occupied[max(0, zlo[z] - start):min(stop, zhi[z]) - start].any(axis=0)
                integral = np.pad(plane, ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1)
                counts = integral[yhi, xhi] - integral[ylo, xhi] - integral[yhi, xlo] + integral[ylo, xlo]
                selected[z] |= counts > 0
            if progress is not None:
                progress.update(stop - start)
    plan.selected = selected
    return plan


def inspect_level(source):
    # Metadata only: no chunk reads and no cache writes, even for --dry-run.
    with source.open(cache_writes=False) as reader:
        if reader.ndim != 3:
            raise ValueError(f"Expected a 3D volume: {source.path}, level {source.level}")
        return LevelPlan(source, tuple(reader.shape), tuple(reader.chunks), reader.dtype.itemsize)


def format_bytes(value):
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or suffix == "PiB":
            return f"{value:.2f} {suffix}"
        value /= 1024


def prefetch_level(plan, *, connections=16, batch_chunks=512, progress=None, compress_cache=False):
    """Bound the plan size and use the reader's cache-aware download scheduler."""
    if connections < 1 or batch_chunks < 1 or batch_chunks > 65536:
        raise ValueError("connections must be positive; batch_chunks must be in [1, 65536]")
    completed = 0
    with plan.source.open(connections=connections, compress_cache=compress_cache) as reader:
        if tuple(reader.shape) != plan.shape or tuple(reader.chunks) != plan.chunks:
            raise RuntimeError(f"Volume metadata changed after planning: {plan.source.path}")
        regions = iter(plan.regions())
        with reader.prefetch_window() as window:
            while batch := list(itertools.islice(regions, batch_chunks)):
                window.update([batch])
                while True:
                    try:
                        window.wait(timeout=0.25)
                        finished = True
                    except TimeoutError:
                        # A completed request may itself have failed with a timeout.
                        if window.done():
                            window.wait()
                            finished = True
                        else:
                            finished = False
                    current = reader.prefetch_stats()["completed"]
                    if progress is not None:
                        progress.update(current - completed)
                        stats = reader.stats()
                        progress.set_postfix(GETs=stats.get("remote_fetches", 0),
                                             fetched=format_bytes(stats.get("remote_bytes", 0)), refresh=False)
                    completed = current
                    if finished:
                        break
        reader.flush()
        stats = reader.stats()
        if stats.get("write_errors", 0):
            raise RuntimeError(f"Cache writes failed for {plan.source.path}, level {plan.source.level}")
        if completed != plan.count:
            raise RuntimeError(f"Incomplete prefetch: {completed} of {plan.count} chunk positions")
        return {"chunks": completed, **stats}


class LogProgress:
    """Periodic newline-delimited progress for files and tail -f."""

    def __init__(self, iterable=None, *, total=None, desc="", unit="item", interval=30, **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)
        self.desc, self.unit, self.interval = desc, unit, interval
        self.n = 0
        self.postfix = {}
        self.started = self.last = time.monotonic()
        self.closed = False
        self.emit()

    def emit(self):
        now = time.monotonic()
        elapsed = now - self.started
        rate = self.n / elapsed if elapsed else 0
        percent = 100 * self.n / self.total if self.total else 100
        eta = f"{max(0, self.total - self.n) / rate:.0f}s" if rate else "unknown"
        extra = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
              f"{self.desc}: {self.n:,}/{self.total:,} {self.unit} ({percent:.1f}%) "
              f"rate={rate:.1f} {self.unit}/s elapsed={elapsed:.0f}s ETA={eta} {extra}", flush=True)
        self.last = now

    def update(self, amount):
        self.n += amount
        if time.monotonic() - self.last >= self.interval:
            self.emit()

    def set_description(self, desc):
        self.desc = desc
        self.postfix = {}
        self.emit()

    def set_postfix(self, refresh=False, **values):
        self.postfix = values

    def __iter__(self):
        try:
            for value in self.iterable:
                yield value
                self.update(1)
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.emit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--connections", type=int, default=16, help="Maximum concurrent chunk connections (default: 16)")
    parser.add_argument("--cache-dir", help="Cache root before s3/<bucket>/; defaults to dataset.volume_cache_dir")
    parser.add_argument("--compress-cache", action="store_true",
                        help="Use lossless vc-delta3d for new stores (3D uint8/uint16); existing stores keep their codec")
    parser.add_argument("--sample-id", action="append", default=[], help="Limit to this sample ID; repeat for multiple samples")
    parser.add_argument("--scope", choices=("all", "training", "occupancy"), default="all",
                        help="Levels to prefetch; filtered training also includes coarse mask prerequisites")
    parser.add_argument("--no-mask-filter", action="store_true", help="Fetch full training levels without occupancy filtering")
    parser.add_argument("--mask-padding", type=int, default=1, help="Boundary margin in coarse voxels (default: 1)")
    parser.add_argument("--batch-chunks", type=int, default=512, help="Chunks scheduled at once (default: 512; maximum: 65536)")
    parser.add_argument("--dry-run", action="store_true", help="Read metadata and print sizes without downloading chunks or writing cache files")
    parser.add_argument("--log-file", type=Path,
                        help="Append output and timestamped progress to this file instead of terminal bars")
    parser.add_argument("--log-interval", type=float, default=30,
                        help="Seconds between file progress updates (default: 30)")
    args = parser.parse_args(argv)
    if args.connections < 1 or not 1 <= args.batch_chunks <= 65536:
        parser.error("connections must be positive; batch-chunks must be in [1, 65536]")
    if args.mask_padding < 0:
        parser.error("mask-padding must be nonnegative")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        parser.error("log-interval must be finite and positive")
    if args.log_file is None:
        return run(args)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} Starting prefetch: "
                  f"config={args.config.resolve()} connections={args.connections} "
                  f"compress_cache={args.compress_cache} mask_filter={not args.no_mask_filter}", flush=True)
            def progress_factory(*values, **options):
                return LogProgress(*values, interval=args.log_interval, **options)
            result = run(args, progress_factory)
            print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
                  f"Prefetch exited with status {result}", flush=True)
            return result


def run(args, progress_factory=tqdm):
    try:
        coarse_sources, training_sources = staged_sources(
            json.loads(args.config.read_text()), cache_dir=args.cache_dir,
            samples=args.sample_id, scope=args.scope, mask_filter=not args.no_mask_filter)
        sources = coarse_sources + training_sources
        if not sources:
            print("No remote volume levels selected.")
            return 0
        plans = []
        for source in progress_factory(sources, desc="Reading metadata", unit="level"):
            plans.append(inspect_level(source))
        for plan in plans:
            print(f"{plan.source.sample} level {plan.source.level}: shape={plan.shape}, "
                  f"chunks={plan.count:,}, dense cache={format_bytes(plan.dense_bytes)}")
        total = sum(plan.count for plan in plans)
        print(f"Unfiltered total: {len(plans)} levels, {total:,} chunks; "
              f"dense decoded size {format_bytes(sum(plan.dense_bytes for plan in plans))}.")
        print("Dense size includes cached data and absent chunks; actual additional disk use may be much smaller.")
        print(f"Cache root: {sources[0].cache_dir}", flush=True)
        if args.compress_cache:
            print("New stores use lossless vc-delta3d; existing stores retain their current codec.", flush=True)
        if args.dry_run:
            print("Metadata only: filtered counts are computed after caching the coarse masks.")
            return 0
        fetched = 0
        completed = 0
        coarse_plans = plans[:len(coarse_sources)]
        training_plans = plans[len(coarse_sources):]
        coarse_by_path = {plan.source.path: plan for plan in coarse_plans}
        with progress_factory(total=sum(p.count for p in coarse_plans), desc="Caching coarse masks", unit="chunk",
                  dynamic_ncols=True) as progress:
            for plan in coarse_plans:
                progress.set_description(f"{plan.source.sample} L{plan.source.level}")
                stats = prefetch_level(plan, connections=args.connections, batch_chunks=args.batch_chunks,
                                       progress=progress, compress_cache=args.compress_cache)
                fetched += stats.get("remote_bytes", 0)
                completed += plan.count
        for plan in training_plans:
            coarse = coarse_by_path.get(plan.source.path)
            if not args.no_mask_filter and coarse is not None:
                with progress_factory(total=coarse.shape[0], desc=f"Filtering {plan.source.sample}", unit="slice") as progress:
                    select_mask_chunks(plan, coarse, padding=args.mask_padding, progress=progress)
                print(f"{plan.source.sample} L{plan.source.level}: selected {plan.count:,} / "
                      f"{math.prod(plan.grid):,} chunks; skipped {math.prod(plan.grid) - plan.count:,}; "
                      f"selected dense size {format_bytes(plan.dense_bytes)}", flush=True)
            elif not args.no_mask_filter:
                print(f"{plan.source.sample}: no occupancy_level configured; fetching full training level.", flush=True)
            with progress_factory(total=plan.count, desc=f"{plan.source.sample} L{plan.source.level}", unit="chunk",
                      dynamic_ncols=True) as progress:
                stats = prefetch_level(plan, connections=args.connections, batch_chunks=args.batch_chunks,
                                       progress=progress, compress_cache=args.compress_cache)
            fetched += stats.get("remote_bytes", 0)
            completed += plan.count
        print(f"Complete: {completed:,} chunk positions cached or recorded absent; fetched {format_bytes(fetched)}.")
        return 0
    except KeyboardInterrupt:
        print("\nStopped. Completed cache entries are retained; rerun the same command to continue.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Prefetch failed: {exc}\nCompleted cache entries are retained; rerun to continue.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
