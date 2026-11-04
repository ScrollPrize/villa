from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch


_REMOTE_PREFIXES = ("http://", "https://", "s3://")


def volume_cache_dirs(volume_path: str, cache_root: str | None) -> str | list[str] | None:
    """``cache_dir`` for ``vc_reader.reader.open_zarr`` of a CT volume.

    Remote volumes use VC3D's configured cache root first, then the config's
    ``volume_cache_dir``; local volumes are read in place.
    """
    if not str(volume_path).startswith(_REMOTE_PREFIXES):
        return None
    return ["vc3d", str(cache_root)] if cache_root else "vc3d"


def level_points(
    coords_zyx_base: np.ndarray,
    valid_mask: np.ndarray,
    level: int,
    shape_zyx: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pyramid-level ``level`` positions of level-0 ZYX coordinates, and which of them are sampled.

    These scale-only pyramids place level-``k`` voxel ``i`` at level-0 coordinate
    ``i * 2**k``, so positions are ``base / 2**k`` with no half-voxel offset.
    A point is valid where ``valid_mask`` is set, its coordinates are finite and
    it lies in ``[0, shape)`` on every axis of the level array.
    """
    points = np.asarray(coords_zyx_base, dtype=np.float32) / np.float32(1 << int(level))
    shape = np.asarray(tuple(shape_zyx)[-3:], dtype=np.float32)
    inside = np.isfinite(points) & (points >= 0.0) & (points < shape)
    valid = np.asarray(valid_mask, dtype=bool) & np.all(inside, axis=-1)
    return points, valid


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{int(sec):02d}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def prefetch_samples(
    read_sample: Callable[[int], dict[str, int]],
    start_sample_index: int,
    sample_count: int,
    *,
    workers: int,
    readers: Iterable[Any],
    count_keys: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Run ``read_sample`` for samples ``start_sample_index ..+ sample_count`` on ``workers`` threads.

    ``read_sample(index)`` reads the sample's augmentation envelope through its
    vc_reader readers, blocking, and discards the values: the reads fetch and
    disk-cache exactly the chunks the sample can touch. It returns the
    ``count_keys`` counts, which are summed into the summary. A ValueError skips the sample, which still
    counts as finished; any other exception is an error and leaves the sample
    unfinished. ``max_exclusive_sample_index`` is the end of the contiguous run
    of finished samples from ``start_sample_index`` (0 while there is none).
    ``remote_fetches``/``remote_bytes`` sum the ``stats()`` changes of the
    distinct ``readers``, whose queued cache writes are flushed at the end.
    """
    total = int(sample_count)
    if total < 0:
        raise ValueError("sample_count must be >= 0")
    first_index = int(start_sample_index)
    end_index = first_index + total
    worker_count = max(1, int(workers))
    distinct = list({id(reader): reader for reader in readers}.values())
    before = [reader.stats() for reader in distinct]

    def remote(name: str) -> int:
        return sum(int(reader.stats()[name]) - int(stats[name]) for reader, stats in zip(distinct, before))

    totals = {key: 0 for key in count_keys}
    samples_done = skipped = errors = 0
    first_error = first_skip = ""
    finished: set[int] = set()
    next_unfinished = first_index
    started = time.perf_counter()
    print(f"{label}: samples={total} workers={worker_count}", flush=True)

    def progress(*, final: bool = False) -> None:
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        rate = samples_done / elapsed
        eta = (total - samples_done) / rate if rate > 0.0 else 0.0
        filled = int(24 * samples_done / max(total, 1))
        mib = remote("remote_bytes") / (1024.0 * 1024.0)
        line = (
            f"{label} samples[{'#' * filled}{'-' * (24 - filled)}] {samples_done}/{total} "
            f"eta={format_seconds(eta)} idx={next_unfinished if next_unfinished > first_index else 0} "
            f"skipped={skipped} errors={errors} fetches={remote('remote_fetches')} "
            f"{mib:.1f} MiB {mib / elapsed:.1f} MiB/s"
        )
        if first_error:
            line += f" first_error={first_error[:157]!r}"
        print(line, end="\n" if final else "\r", flush=True)

    previous_torch_threads = int(torch.get_num_threads())
    torch.set_num_threads(1)
    executor = ThreadPoolExecutor(max_workers=worker_count)
    pending: dict[Future[dict[str, int]], int] = {}
    completed = False
    try:
        next_index = first_index
        last_progress = 0.0
        while next_index < end_index or pending:
            while next_index < end_index and len(pending) < 2 * worker_count:
                pending[executor.submit(read_sample, next_index)] = next_index
                next_index += 1
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                samples_done += 1
                try:
                    counts = future.result()
                except ValueError as exc:
                    skipped += 1
                    first_skip = first_skip or f"sample={index}: {exc}"
                except Exception as exc:
                    errors += 1
                    first_error = first_error or f"sample={index}: {exc}"
                    continue
                else:
                    for key in totals:
                        totals[key] += int(counts[key])
                finished.add(index)
            while next_unfinished in finished:
                finished.discard(next_unfinished)
                next_unfinished += 1
            now = time.perf_counter()
            if now - last_progress >= 0.5:
                progress()
                last_progress = now
        for reader in distinct:
            reader.flush()
        progress(final=True)
        completed = True
    finally:
        executor.shutdown(wait=completed, cancel_futures=not completed)
        torch.set_num_threads(previous_torch_threads)
    return {
        "samples": samples_done,
        "skipped_samples": skipped,
        **totals,
        "errors": errors,
        "first_error": first_error,
        "first_sample_skip": first_skip,
        "workers": worker_count,
        "max_exclusive_sample_index": next_unfinished if next_unfinished > first_index else 0,
        "remote_fetches": remote("remote_fetches"),
        "remote_bytes": remote("remote_bytes"),
    }
