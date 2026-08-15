"""Export native phase-cache centre rays for fast Spiral supervision.

The native cache is deliberately lossless, but its slab-sized compression
chunks make random access unsuitable for an optimisation loop.  This module
decodes one canonical ray per slab into a small set of uncompressed NumPy
arrays which can be memory-mapped or copied to a GPU once at fit start.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from tqdm import tqdm


FORMAT_NAME = "winding_inference_crossings"
FORMAT_VERSION = 1
DEFAULT_EDGE_MARGIN = 8
DEFAULT_EDGE_TRIM = 2


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_array(directory: Path, name: str, value: np.ndarray) -> dict:
    path = directory / f"{name}.npy"
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": np.dtype(value.dtype).str,
        "bytes": path.stat().st_size,
        "sha256": _file_digest(path),
    }


def decode_center_ray(
    phase: np.ndarray,
    valid: np.ndarray,
    *,
    anchor: int,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
    edge_trim: int = DEFAULT_EDGE_TRIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained integer-passage positions and relative levels.

    Only the contiguous valid run containing ``anchor`` is trusted.  This
    prevents a long-baseline pair from silently spanning an invalid hole.
    """
    phase = np.asarray(phase, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if phase.ndim != 1 or valid.shape != phase.shape:
        raise ValueError("phase and valid must be equal-length one-dimensional arrays")
    if not 0 <= anchor < len(phase) or not valid[anchor]:
        return np.empty(0, np.float32), np.empty(0, np.int16)

    lo = anchor
    while lo > 0 and valid[lo - 1]:
        lo -= 1
    hi = anchor + 1
    while hi < len(valid) and valid[hi]:
        hi += 1
    lo = max(lo, int(edge_margin))
    hi = min(hi, len(phase) - int(edge_margin))
    if hi - lo < 2:
        return np.empty(0, np.float32), np.empty(0, np.int16)

    registered = phase - phase[anchor]
    lower = np.floor(registered[lo : hi - 1])
    counts = np.floor(registered[lo + 1 : hi]) - lower
    counts = np.maximum(counts.astype(np.int32), 0)
    segment_local = np.flatnonzero(counts)
    if not len(segment_local):
        return np.empty(0, np.float32), np.empty(0, np.int16)

    repetitions = counts[segment_local].astype(np.int64)
    segment = np.repeat(segment_local + lo, repetitions)
    starts = np.cumsum(repetitions) - repetitions
    within = np.arange(int(repetitions.sum()), dtype=np.int64) \
        - np.repeat(starts, repetitions)
    levels = np.repeat(lower[segment_local], repetitions) + within + 1
    base = registered[segment]
    step = np.maximum(registered[segment + 1] - base, np.float32(1e-9))
    positions = segment.astype(np.float32) + np.clip(
        (levels.astype(np.float32) - base) / step, 0.0, 1.0
    )

    trim = int(edge_trim)
    if trim < 0:
        raise ValueError("edge_trim cannot be negative")
    if len(positions) <= 2 * trim:
        return np.empty(0, np.float32), np.empty(0, np.int16)
    if trim:
        positions = positions[trim:-trim]
        levels = levels[trim:-trim]
    if np.any((levels < np.iinfo(np.int16).min)
              | (levels > np.iinfo(np.int16).max)):
        raise ValueError("relative crossing level does not fit int16")
    return positions.astype(np.float32), levels.astype(np.int16)


_EXPORT_WORKER_SOURCE = None
_EXPORT_WORKER_GROUP = None


def _init_export_worker(source_path: str) -> None:
    import zarr

    global _EXPORT_WORKER_SOURCE, _EXPORT_WORKER_GROUP
    _EXPORT_WORKER_SOURCE = source_path
    _EXPORT_WORKER_GROUP = zarr.open_group(source_path, mode="r")


def _get_export_group(source_path: str):
    import zarr

    if (_EXPORT_WORKER_GROUP is not None
            and _EXPORT_WORKER_SOURCE == source_path):
        return _EXPORT_WORKER_GROUP
    return zarr.open_group(source_path, mode="r")


def _source_signature(
    group, shard: dict, edge_margin: int, edge_trim: int, *,
    start: int | None = None, end: int | None = None,
) -> dict:
    name = str(shard["name"])
    result = {
        "root_identity": _canonical_digest(dict(group.attrs)),
        "name": name,
        "lo": int(shard["lo"]),
        "hi": int(shard["hi"]),
        "phase_shape": list(group["phase"][name].shape),
        "valid_shape": list(group["valid"][name].shape),
        "frame_shape": list(group["frame"][name].shape),
        "edge_margin": int(edge_margin),
        "edge_trim": int(edge_trim),
    }
    if start is not None:
        result["part_start"] = int(start)
        result["part_end"] = int(end)
    return result


def _arrays_are_valid(directory: Path, manifest: dict) -> bool:
    arrays = manifest.get("arrays", {})
    if len(arrays) != 6:
        return False
    for description in arrays.values():
        path = directory / description.get("file", "")
        if (not path.is_file()
                or path.stat().st_size != int(description.get("bytes", -1))
                or _file_digest(path) != description.get("sha256")):
            return False
    return True


def _export_part(task: tuple[str, str, dict, int, int, int, int]) -> dict:
    source_path, partial_path, shard, start, end, edge_margin, edge_trim = task
    group = _get_export_group(source_path)
    name = str(shard["name"])
    signature = _source_signature(
        group, shard, edge_margin, edge_trim, start=start, end=end)
    destination = Path(partial_path) / "parts" / name / f"{start:08d}_{end:08d}"
    part_manifest = destination / "manifest.json"
    if part_manifest.is_file():
        existing = json.loads(part_manifest.read_text())
        if (existing.get("source_signature") == signature
                and _arrays_are_valid(destination, existing)):
            return existing

    temporary = destination.parent / (
        f".{start:08d}_{end:08d}.tmp-{os.getpid()}")
    for stale in destination.parent.glob(
            f".{start:08d}_{end:08d}.tmp-*"):
        shutil.rmtree(stale)
    temporary.mkdir(parents=True)

    phase_array = group["phase"][name]
    valid_array = group["valid"][name]
    frame_array = group["frame"][name]
    available = np.asarray(
        group["available"][name][start:end], dtype=bool)
    ray_length = int(group.attrs["ray_length"])
    column_stride = int(group.attrs["column_stride"])
    columns = int(phase_array.shape[1])
    center = int(round((columns * column_stride - 1) / column_stride / 2))
    valid_center = min(
        int(valid_array.shape[1]) - 1, center * column_stride)
    transverse_center = center * column_stride
    anchor = int(round((ray_length - 1) / 2.0))
    spacing = float(group.attrs["spacing"])
    global_lo = int(shard["lo"]) + int(start)
    seed_winding_all = np.asarray(
        group["rays"]["seed_winding"][global_lo : global_lo + len(available)],
        dtype=np.int16,
    )

    origins = np.empty((len(available), 3), dtype=np.float32)
    steps = np.empty_like(origins)
    seed_windings = np.empty(len(available), dtype=np.int16)
    offsets = np.empty(len(available) + 1, dtype=np.int64)
    offsets[0] = 0
    retained_rays = 0
    position_batches: list[np.ndarray] = []
    level_batches: list[np.ndarray] = []
    pending_positions: list[np.ndarray] = []
    pending_levels: list[np.ndarray] = []
    started = time.monotonic()

    def flush_pending() -> None:
        if pending_positions:
            position_batches.append(np.concatenate(pending_positions))
            level_batches.append(np.concatenate(pending_levels))
            pending_positions.clear()
            pending_levels.clear()

    for part_index in np.flatnonzero(available):
        local_index = int(start) + int(part_index)
        phase = np.asarray(
            phase_array[local_index, center, center, :], dtype=np.float32
        )
        valid = np.asarray(
            valid_array[local_index, valid_center, valid_center, :], dtype=bool
        )
        positions, levels = decode_center_ray(
            phase, valid, anchor=anchor,
            edge_margin=edge_margin, edge_trim=edge_trim,
        )
        if len(positions) < 2:
            continue
        frame = np.asarray(frame_array[local_index], dtype=np.float64)
        origin_xyz = (
            frame[0]
            + spacing * transverse_center * (frame[1] + frame[2])
        )
        origins[retained_rays] = origin_xyz[::-1]
        steps[retained_rays] = (spacing * frame[3])[::-1]
        seed_windings[retained_rays] = seed_winding_all[int(part_index)]
        pending_positions.append(positions)
        pending_levels.append(levels)
        retained_rays += 1
        offsets[retained_rays] = offsets[retained_rays - 1] + len(positions)
        if len(pending_positions) >= 4096:
            flush_pending()
    flush_pending()

    crossing_t = (np.concatenate(position_batches)
                  if position_batches else np.empty(0, np.float32))
    crossing_level = (np.concatenate(level_batches)
                      if level_batches else np.empty(0, np.int16))
    arrays = {
        "ray_origin_zyx": _save_array(
            temporary, "ray_origin_zyx", origins[:retained_rays]),
        "ray_step_zyx": _save_array(
            temporary, "ray_step_zyx", steps[:retained_rays]),
        "crossing_offsets": _save_array(
            temporary, "crossing_offsets", offsets[: retained_rays + 1]),
        "crossing_t": _save_array(temporary, "crossing_t", crossing_t),
        "crossing_level": _save_array(
            temporary, "crossing_level", crossing_level),
        "seed_winding": _save_array(
            temporary, "seed_winding", seed_windings[:retained_rays]),
    }
    lengths = np.diff(offsets[: retained_rays + 1])
    result = {
        "name": name,
        "part_start": int(start),
        "part_end": int(end),
        "source_signature": signature,
        "num_source_rays": int(len(available)),
        "num_available_rays": int(available.sum()),
        "num_retained_rays": int(retained_rays),
        "num_crossings": int(len(crossing_t)),
        "crossings_per_ray": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "median": float(np.median(lengths)) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
        "elapsed_seconds": time.monotonic() - started,
        "arrays": arrays,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    return result


def _merge_shard_parts(
    partial: Path, group, shard: dict, parts: list[dict],
    edge_margin: int, edge_trim: int,
) -> dict:
    """Merge small restartable range products into one runtime shard."""
    name = str(shard["name"])
    destination = partial / name
    temporary = partial / f".{name}.merge-{os.getpid()}"
    for stale in partial.glob(f".{name}.merge-*"):
        shutil.rmtree(stale)
    temporary.mkdir(parents=True)
    parts = sorted(parts, key=lambda item: int(item["part_start"]))

    values = {key: [] for key in (
        "ray_origin_zyx", "ray_step_zyx", "crossing_t",
        "crossing_level", "seed_winding")}
    offsets = [np.array([0], dtype=np.int64)]
    crossing_base = 0
    for part in parts:
        part_root = (
            partial / "parts" / name
            / f"{int(part['part_start']):08d}_{int(part['part_end']):08d}")
        for key in values:
            values[key].append(np.load(
                part_root / part["arrays"][key]["file"], allow_pickle=False))
        local_offsets = np.load(
            part_root / part["arrays"]["crossing_offsets"]["file"],
            allow_pickle=False)
        offsets.append(local_offsets[1:] + crossing_base)
        crossing_base += int(local_offsets[-1])

    def concatenate(key, empty_shape, dtype):
        return (np.concatenate(values[key]).astype(dtype, copy=False)
                if values[key] else np.empty(empty_shape, dtype=dtype))

    merged = {
        "ray_origin_zyx": concatenate(
            "ray_origin_zyx", (0, 3), np.float32),
        "ray_step_zyx": concatenate("ray_step_zyx", (0, 3), np.float32),
        "crossing_offsets": np.concatenate(offsets),
        "crossing_t": concatenate("crossing_t", (0,), np.float32),
        "crossing_level": concatenate("crossing_level", (0,), np.int16),
        "seed_winding": concatenate("seed_winding", (0,), np.int16),
    }
    arrays = {
        key: _save_array(temporary, key, value)
        for key, value in merged.items()
    }
    lengths = np.diff(merged["crossing_offsets"])
    result = {
        "name": name,
        "source_signature": _source_signature(
            group, shard, edge_margin, edge_trim),
        "num_source_rays": int(sum(item["num_source_rays"] for item in parts)),
        "num_available_rays": int(sum(
            item["num_available_rays"] for item in parts)),
        "num_retained_rays": int(len(merged["ray_origin_zyx"])),
        "num_crossings": int(len(merged["crossing_t"])),
        "crossings_per_ray": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "median": float(np.median(lengths)) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
        "elapsed_seconds": float(sum(
            item.get("elapsed_seconds", 0.0) for item in parts)),
        "arrays": arrays,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    return result


def export_spiral_supervision(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
    edge_trim: int = DEFAULT_EDGE_TRIM,
    workers: int | None = None,
    rays_per_task: int = 256,
    show_progress: bool = True,
) -> dict:
    """Export ``source`` and return the completed root manifest."""
    import zarr

    source = Path(source).resolve()
    output = Path(output).resolve()
    group = zarr.open_group(str(source), mode="r")
    if group.attrs.get("artifact_type") != "winding_native_phase_cache":
        raise ValueError(f"not a native winding phase cache: {source}")
    if not bool(group.attrs.get("complete", False)):
        raise RuntimeError(f"native phase cache is incomplete: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    partial = Path(str(output) + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    shards = [dict(item) for item in group.attrs["phase_shards"]]
    if int(rays_per_task) < 1:
        raise ValueError("rays_per_task must be positive")
    # Native caches physically group 32 independently-compressed slab chunks
    # per file. Aligning work boundaries to that grouping prevents two
    # processes from repeatedly opening the same physical shard.
    task_size = max(32, ((int(rays_per_task) + 31) // 32) * 32)
    tasks = []
    for shard in shards:
        count = int(shard["hi"]) - int(shard["lo"])
        for start in range(0, count, task_size):
            end = min(count, start + task_size)
            tasks.append((
                str(source), str(partial), shard, start, end,
                int(edge_margin), int(edge_trim)))
    default_workers = min(32, os.cpu_count() or 1)
    worker_count = min(len(tasks), int(workers or default_workers))
    if worker_count < 1:
        raise ValueError("workers must be positive")
    started = time.monotonic()
    part_results = []
    retained = crossings = 0
    progress = tqdm(
        total=sum(task[4] - task[3] for task in tasks),
        desc=f"decoding ({worker_count} workers)", unit="ray", unit_scale=True,
        dynamic_ncols=True, disable=not show_progress,
    )

    def accept(result):
        nonlocal retained, crossings
        part_results.append(result)
        retained += int(result["num_retained_rays"])
        crossings += int(result["num_crossings"])
        progress.update(int(result["num_source_rays"]))
        elapsed = max(time.monotonic() - started, 1e-9)
        progress.set_postfix(
            retained=f"{retained:,}", crossings=f"{crossings:,}",
            source_rays_s=f"{progress.n / elapsed:,.0f}", refresh=False)

    try:
        if worker_count == 1:
            _init_export_worker(str(source))
            for task in tasks:
                accept(_export_part(task))
        else:
            executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_export_worker,
                    initargs=(str(source),))
            futures = []
            try:
                futures = [executor.submit(_export_part, task) for task in tasks]
                for future in as_completed(futures):
                    accept(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    finally:
        progress.close()

    results = []
    for shard in tqdm(
            shards, desc="merging compact shards", unit="shard",
            dynamic_ncols=True, disable=not show_progress):
        name = str(shard["name"])
        results.append(_merge_shard_parts(
            partial, group, shard,
            [item for item in part_results if item["name"] == name],
            int(edge_margin), int(edge_trim)))
    manifest = {
        "artifact_type": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "coordinate_order": "zyx",
        "coordinate_space": "reference zarr scale-0 voxels",
        "source_cache": str(source),
        "source_identity": _canonical_digest(dict(group.attrs)),
        "source_attributes": _jsonable(dict(group.attrs)),
        "center_native_column": int(round(
            (int(group["phase"][str(shards[0]["name"])].shape[1])
             * int(group.attrs["column_stride"]) - 1)
            / int(group.attrs["column_stride"]) / 2)),
        "center_transverse_sample": int(group.attrs["column_stride"]) * int(round(
            (int(group["phase"][str(shards[0]["name"])].shape[1])
             * int(group.attrs["column_stride"]) - 1)
            / int(group.attrs["column_stride"]) / 2)),
        "anchor_sample": int(round((int(group.attrs["ray_length"]) - 1) / 2)),
        "edge_margin": int(edge_margin),
        "edge_trim": int(edge_trim),
        "export_workers": int(worker_count),
        "rays_per_task": int(task_size),
        "num_rays": int(sum(item["num_retained_rays"] for item in results)),
        "num_crossings": int(sum(item["num_crossings"] for item in results)),
        "shards": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    identity_view = copy.deepcopy(manifest)
    identity_view.pop("elapsed_seconds")
    identity_view.pop("export_workers", None)
    identity_view.pop("rays_per_task", None)
    for shard in identity_view["shards"]:
        shard.pop("elapsed_seconds", None)
    manifest["fingerprint"] = _canonical_digest(identity_view)
    (partial / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    parts_directory = partial / "parts"
    if parts_directory.exists():
        shutil.rmtree(parts_directory)
    os.replace(partial, output)
    return manifest


def validate_spiral_supervision(
    path: str | os.PathLike[str], *, verify_hashes: bool = True,
) -> dict:
    """Validate a completed compact store and return its summary."""
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"compact supervision manifest is missing: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_type") != FORMAT_NAME:
        raise ValueError(f"not a compact winding supervision store: {root}")
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"unsupported compact supervision version: "
            f"{manifest.get('format_version')!r}")
    identity_view = copy.deepcopy(manifest)
    claimed_fingerprint = identity_view.pop("fingerprint", None)
    identity_view.pop("elapsed_seconds", None)
    identity_view.pop("export_workers", None)
    identity_view.pop("rays_per_task", None)
    for shard in identity_view.get("shards", []):
        shard.pop("elapsed_seconds", None)
    if claimed_fingerprint != _canonical_digest(identity_view):
        raise ValueError("compact supervision manifest fingerprint mismatch")
    ray_count = crossing_count = 0
    for shard in manifest.get("shards", []):
        shard_root = root / shard["name"]
        for description in shard.get("arrays", {}).values():
            array_path = shard_root / description["file"]
            value = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != list(description["shape"]):
                raise ValueError(f"array shape mismatch: {array_path}")
            if np.dtype(value.dtype).str != description["dtype"]:
                raise ValueError(f"array dtype mismatch: {array_path}")
            if verify_hashes and _file_digest(array_path) != description["sha256"]:
                raise ValueError(f"array checksum mismatch: {array_path}")
        ray_count += int(shard["num_retained_rays"])
        crossing_count += int(shard["num_crossings"])
    if ray_count != int(manifest["num_rays"]):
        raise ValueError("root and shard ray counts disagree")
    if crossing_count != int(manifest["num_crossings"]):
        raise ValueError("root and shard crossing counts disagree")
    return {
        "path": str(root),
        "num_rays": ray_count,
        "num_crossings": crossing_count,
        "fingerprint": manifest["fingerprint"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="complete native-phase cache Zarr")
    parser.add_argument("output", nargs="?", help="new compact supervision directory")
    parser.add_argument(
        "--validate", metavar="STORE",
        help="validate an existing compact store instead of exporting")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel range workers (default: up to 32)")
    parser.add_argument(
        "--rays-per-task", type=int, default=256,
        help="progress/restart granularity, rounded to a multiple of 32")
    parser.add_argument(
        "--no-progress", action="store_true",
        help="disable the aggregate tqdm progress bar")
    parser.add_argument("--edge-margin", type=int, default=DEFAULT_EDGE_MARGIN)
    parser.add_argument("--edge-trim", type=int, default=DEFAULT_EDGE_TRIM)
    args = parser.parse_args(argv)
    if args.validate:
        if args.source or args.output:
            parser.error("--validate cannot be combined with source/output")
        print(json.dumps(validate_spiral_supervision(args.validate), indent=2))
        return
    if not args.source or not args.output:
        parser.error("source and output are required when not using --validate")
    manifest = export_spiral_supervision(
        args.source, args.output, workers=args.workers,
        rays_per_task=args.rays_per_task,
        show_progress=not args.no_progress,
        edge_margin=args.edge_margin, edge_trim=args.edge_trim,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "num_rays": manifest["num_rays"],
        "num_crossings": manifest["num_crossings"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "fingerprint": manifest["fingerprint"],
    }, indent=2))


if __name__ == "__main__":
    main()
