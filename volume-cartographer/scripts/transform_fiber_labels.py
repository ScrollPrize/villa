#!/usr/bin/env python3
"""Transform legacy HZ/VT label volumes onto the 2.4 um, 78 keV scans.

The two built-in conversions are:

* PHercParis4: 7.91 um labels -> 20260411134726 (2.400 um, 78 keV)
* PHerc1667:   7.91 um labels -> 20251217075048 (2.399 um, 78 keV)

The registration ``transform.json`` matrices are in XYZ order and map target
(moving) coordinates to legacy (fixed) coordinates.  This script converts
them to ZYX array order and performs nearest-neighbour pull resampling.

PHerc1667's local label store is laid out as Z, X, Y.  For that dataset only,
the script treats the last Z slice as the extra trailing slice requested by
the caller, excludes it at level 0, and swaps the two in-plane axes before
using the published registration.

Output is sparse OME-Zarr v2.  Chunks which are guaranteed to be zero are not
written and therefore read back as the fill value.  Every written chunk uses
the numcodecs Zstd codec at level 3.

Example:

    python scripts/transform_fiber_labels.py --dry-run
    python scripts/transform_fiber_labels.py --workers 4

Interrupted conversions can be continued with ``--resume``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import shutil
import sqlite3
import sys
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import numpy as np
    import zarr
    from numcodecs import Zstd
    from scipy import ndimage
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - command-line dependency check
    raise SystemExit(
        "This script needs numpy, scipy, zarr, numcodecs, and tqdm. "
        "Install them in the active Python environment."
    ) from exc


DATA_ROOT = Path("/home/sean/Desktop/fiber_volumes")
OPEN_DATA = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
OUTPUT_CHUNKS = (128, 128, 128)
ZSTD_LEVEL = 3


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    source: Path
    target_root: str
    output_name: str
    expected_fixed_volume: str
    swap_source_yx: bool = False
    drop_trailing_z_level0: bool = False


SPECS = {
    "paris4": DatasetSpec(
        key="paris4",
        display_name="PHercParis4",
        source=DATA_ROOT / "to_transform/s1a-fibers-hzvt-05032025-ome.zarr",
        target_root=(
            f"{OPEN_DATA}/PHercParis4/volumes/"
            "20260411134726-2.400um-0.2m-78keV-masked.zarr"
        ),
        output_name="s1a-fibers-hzvt-2.400um-78keV.ome.zarr",
        expected_fixed_volume="PHercParis4-20230205180739_masked",
    ),
    "pherc1667": DatasetSpec(
        key="pherc1667",
        display_name="PHerc1667",
        source=DATA_ROOT / "to_transform/s4-hzvt-13032025-ome.zarr",
        target_root=(
            f"{OPEN_DATA}/PHerc1667/volumes/"
            "20251217075048-2.399um-0.2m-78keV-masked.zarr"
        ),
        output_name="s4-hzvt-2.399um-78keV.ome.zarr",
        expected_fixed_volume="PHerc1667-20231117161658_masked",
        swap_source_yx=True,
        drop_trailing_z_level0=True,
    ),
}


# Homogeneous coordinate-order conversions.
XYZ_TO_ZYX = np.array(
    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
ZYX_TO_ZXY = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class LevelInfo:
    name: str
    source_name: str
    target_shape: tuple[int, int, int]
    target_scale: tuple[float, float, float]
    source_shape: tuple[int, int, int]
    effective_source_shape: tuple[int, int, int]
    source_scale: tuple[float, float, float]
    matrix_target_to_local_zyx: np.ndarray


@dataclass(frozen=True)
class TileJob:
    level: int
    tile_id: int
    start: tuple[int, int, int]
    stop: tuple[int, int, int]


def _read_json(path_or_url: str | Path) -> dict[str, Any]:
    text = str(path_or_url)
    if text.startswith(("http://", "https://")):
        request = urllib.request.Request(text, headers={"User-Agent": "volume-cartographer/1"})
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    with Path(text).open() as stream:
        return json.load(stream)


def _url_join(root: str, child: str) -> str:
    return f"{root.rstrip('/')}/{child.lstrip('/')}"


def _dataset_entries(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("OME-Zarr metadata has no multiscales entry")
    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("OME-Zarr multiscales metadata has no datasets")
    return datasets


def _entry_scale(entry: dict[str, Any]) -> tuple[float, float, float]:
    for transform in entry.get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = tuple(float(value) for value in transform["scale"])
            if len(scale) != 3:
                raise ValueError(f"expected a 3-D scale, got {scale}")
            return scale  # type: ignore[return-value]
    raise ValueError(f"dataset {entry.get('path')!r} has no scale transformation")


def _homogeneous_scale(scale: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.diag(np.asarray(scale, dtype=np.float64))
    return result


def _load_level_info(
    spec: DatasetSpec,
    max_level: int,
) -> tuple[list[LevelInfo], dict[str, Any], dict[str, Any]]:
    source_attrs = _read_json(spec.source / ".zattrs")
    target_attrs = _read_json(_url_join(spec.target_root, ".zattrs"))
    transform = _read_json(_url_join(spec.target_root, "transform.json"))

    if transform.get("schema_version") != "1.0.0":
        raise ValueError(f"unsupported transform schema: {transform.get('schema_version')!r}")
    if transform.get("fixed_volume") != spec.expected_fixed_volume:
        raise ValueError(
            f"{spec.display_name}: transform fixed_volume is "
            f"{transform.get('fixed_volume')!r}, expected {spec.expected_fixed_volume!r}"
        )

    matrix_xyz = np.vstack(
        [np.asarray(transform["transformation_matrix"], dtype=np.float64), [0, 0, 0, 1]]
    )
    if matrix_xyz.shape != (4, 4):
        raise ValueError("transformation_matrix must be 3x4")

    # Published fixed ZYX -> local array coordinates.  Paris is already ZYX;
    # the local PHerc1667 labels are physically stored Z, X, Y.
    source_reorder = ZYX_TO_ZXY if spec.swap_source_yx else np.eye(4)
    matrix_l0 = source_reorder @ XYZ_TO_ZYX @ matrix_xyz @ XYZ_TO_ZYX

    source_entries = _dataset_entries(source_attrs)
    target_entries = _dataset_entries(target_attrs)
    count = min(len(source_entries), len(target_entries), max_level + 1)
    if count <= 0:
        raise ValueError("no common pyramid levels")

    source_root = zarr.open_group(str(spec.source), mode="r")
    levels: list[LevelInfo] = []
    for index, (source_entry, target_entry) in enumerate(
        zip(source_entries[:count], target_entries[:count])
    ):
        source_name = str(source_entry["path"])
        target_name = str(target_entry["path"])
        source_array = source_root[source_name]
        source_shape = tuple(int(value) for value in source_array.shape)
        if len(source_shape) != 3:
            raise ValueError(f"source level {source_name} is not 3-D")

        effective_shape = source_shape
        if spec.drop_trailing_z_level0 and index == 0:
            effective_shape = (source_shape[0] - 1, source_shape[1], source_shape[2])

        target_zarray = _read_json(_url_join(spec.target_root, f"{target_name}/.zarray"))
        target_shape = tuple(int(value) for value in target_zarray["shape"])
        source_scale = _entry_scale(source_entry)
        target_scale = _entry_scale(target_entry)

        # level coordinates -> L0 coordinates -> affine -> local level coordinates
        level_matrix = (
            np.linalg.inv(_homogeneous_scale(source_scale))
            @ matrix_l0
            @ _homogeneous_scale(target_scale)
        )
        levels.append(
            LevelInfo(
                name=target_name,
                source_name=source_name,
                target_shape=target_shape,  # type: ignore[arg-type]
                target_scale=target_scale,
                source_shape=source_shape,  # type: ignore[arg-type]
                effective_source_shape=effective_shape,  # type: ignore[arg-type]
                source_scale=source_scale,
                matrix_target_to_local_zyx=level_matrix,
            )
        )
    return levels, target_attrs, transform


def _corners(start: Sequence[int], stop: Sequence[int]) -> np.ndarray:
    axes = []
    for first, last in zip(start, stop):
        axes.append((float(first), float(max(first, last - 1))))
    return np.asarray([(*point, 1.0) for point in itertools.product(*axes)], dtype=np.float64)


def _mapped_input_bounds(
    matrix: np.ndarray,
    start: Sequence[int],
    stop: Sequence[int],
    input_shape: Sequence[int],
    margin: int = 1,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    mapped = (matrix @ _corners(start, stop).T).T[:, :3]
    lo = np.floor(mapped.min(axis=0)).astype(np.int64) - margin
    hi = np.ceil(mapped.max(axis=0)).astype(np.int64) + margin + 1
    shape = np.asarray(input_shape, dtype=np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape)
    if np.any(lo >= hi):
        return None
    return tuple(int(value) for value in lo), tuple(int(value) for value in hi)


def _target_bbox(level: LevelInfo) -> tuple[np.ndarray, np.ndarray]:
    source_corners = _corners((0, 0, 0), level.effective_source_shape)
    inverse = np.linalg.inv(level.matrix_target_to_local_zyx)
    mapped = (inverse @ source_corners.T).T[:, :3]
    lo = np.floor(mapped.min(axis=0)).astype(np.int64) - 2
    hi = np.ceil(mapped.max(axis=0)).astype(np.int64) + 3
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(level.target_shape, dtype=np.int64))
    return lo, hi


def _source_chunk_indices(level_dir: Path, separator: str) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    if separator == "/":
        for path in level_dir.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            parts = path.relative_to(level_dir).parts
            if len(parts) != 3:
                continue
            try:
                result.add(tuple(int(part) for part in parts))
            except ValueError:
                continue
    else:
        for path in level_dir.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            parts = path.name.split(separator)
            if len(parts) != 3:
                continue
            try:
                result.add(tuple(int(part) for part in parts))
            except ValueError:
                continue
    return result


def _has_source_chunk(
    occupied: set[tuple[int, int, int]],
    bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
    chunks: Sequence[int],
) -> bool:
    lo, hi = bounds
    ranges = [
        range(first // chunk, (last - 1) // chunk + 1)
        for first, last, chunk in zip(lo, hi, chunks)
    ]
    return any(index in occupied for index in itertools.product(*ranges))


def _plan_jobs(
    spec: DatasetSpec,
    level_index: int,
    level: LevelInfo,
    tile_size: int,
) -> tuple[list[TileJob], int]:
    zarray_meta = _read_json(spec.source / level.source_name / ".zarray")
    source_chunks = tuple(int(value) for value in zarray_meta["chunks"])
    if zarray_meta.get("fill_value", 0) != 0:
        raise ValueError("sparse planning requires source fill_value 0")
    separator = str(zarray_meta.get("dimension_separator", "."))
    occupied = _source_chunk_indices(spec.source / level.source_name, separator)

    lo, hi = _target_bbox(level)
    starts = [range((int(a) // tile_size) * tile_size, int(b), tile_size) for a, b in zip(lo, hi)]
    candidate_count = math.prod(len(axis) for axis in starts)
    grid = tuple(math.ceil(size / tile_size) for size in level.target_shape)

    jobs: list[TileJob] = []
    for z, y, x in itertools.product(*starts):
        start = (z, y, x)
        stop = tuple(min(first + tile_size, size) for first, size in zip(start, level.target_shape))
        bounds = _mapped_input_bounds(
            level.matrix_target_to_local_zyx,
            start,
            stop,
            level.effective_source_shape,
        )
        if bounds is None or not _has_source_chunk(occupied, bounds, source_chunks):
            continue
        iz, iy, ix = (first // tile_size for first in start)
        tile_id = (iz * grid[1] + iy) * grid[2] + ix
        jobs.append(TileJob(level=level_index, tile_id=tile_id, start=start, stop=stop))
    return jobs, candidate_count


def _zarr_major() -> int:
    return int(zarr.__version__.split(".", 1)[0])


def _create_output(
    output: Path,
    spec: DatasetSpec,
    levels: Sequence[LevelInfo],
    target_attrs: dict[str, Any],
    transform: dict[str, Any],
    tile_size: int,
) -> None:
    root_kwargs: dict[str, Any] = {}
    if _zarr_major() >= 3:
        root_kwargs["zarr_format"] = 2
    root = zarr.open_group(str(output), mode="w", **root_kwargs)
    compressor = Zstd(level=ZSTD_LEVEL)
    source_root = zarr.open_group(str(spec.source), mode="r")

    for level in levels:
        dtype = source_root[level.source_name].dtype
        kwargs: dict[str, Any] = dict(
            name=level.name,
            shape=level.target_shape,
            chunks=OUTPUT_CHUNKS,
            dtype=dtype,
            fill_value=0,
            overwrite=False,
        )
        if _zarr_major() >= 3:
            kwargs.update(
                compressors=compressor,
                chunk_key_encoding={"name": "v2", "separator": "/"},
            )
        else:  # pragma: no cover - retained for zarr-python 2 environments
            kwargs.update(compressor=compressor, dimension_separator="/")
        root.create_array(**kwargs)

    target_multiscale = target_attrs["multiscales"][0]
    root.attrs.update(
        {
            "multiscales": [
                {
                    "version": target_multiscale.get("version", "0.4"),
                    "name": spec.display_name + " HZVT labels",
                    "axes": target_multiscale["axes"],
                    "datasets": [
                        {
                            "path": level.name,
                            "coordinateTransformations": [
                                {"type": "scale", "scale": list(level.target_scale)}
                            ],
                        }
                        for level in levels
                    ],
                }
            ],
            "label_transform": {
                "source": str(spec.source),
                "target": spec.target_root,
                "transform": _url_join(spec.target_root, "transform.json"),
                "fixed_volume": transform["fixed_volume"],
                "transformation_matrix_xyz": transform["transformation_matrix"],
                "interpolation": "nearest",
                "source_axis_correction": "swap_yx" if spec.swap_source_yx else "none",
                "dropped_source_slices": (
                    {"axis": "z", "side": "trailing", "count": 1, "level": 0}
                    if spec.drop_trailing_z_level0
                    else None
                ),
                "tile_size": tile_size,
            },
            "compression": {"codec": "zstd", "level": ZSTD_LEVEL},
            "complete": False,
        }
    )


def _validate_existing_output(
    output: Path,
    spec: DatasetSpec,
    levels: Sequence[LevelInfo],
    tile_size: int,
) -> bool:
    root = zarr.open_group(str(output), mode="r")
    config = dict(root.attrs.get("label_transform", {}))
    expected = {
        "source": str(spec.source),
        "target": spec.target_root,
        "tile_size": tile_size,
        "source_axis_correction": "swap_yx" if spec.swap_source_yx else "none",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"cannot resume {output}: {key} is {config.get(key)!r}, expected {value!r}"
            )
    for level in levels:
        if level.name not in root or tuple(root[level.name].shape) != level.target_shape:
            raise ValueError(f"cannot resume {output}: level {level.name} is missing or mismatched")
    return bool(root.attrs.get("complete", False))


def _progress_path(output: Path) -> Path:
    return output.with_name(output.name + ".progress.sqlite3")


def _open_progress(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS completed "
        "(level INTEGER NOT NULL, tile_id INTEGER NOT NULL, "
        "PRIMARY KEY(level, tile_id)) WITHOUT ROWID"
    )
    connection.commit()
    return connection


def _completed_tiles(connection: sqlite3.Connection, level: int) -> set[int]:
    return {int(row[0]) for row in connection.execute(
        "SELECT tile_id FROM completed WHERE level = ?", (level,)
    )}


_WORKER_SOURCE: Any = None
_WORKER_OUTPUT: Any = None
_WORKER_LEVELS: Sequence[LevelInfo] = ()


def _init_worker(source: str, output: str, levels: Sequence[LevelInfo]) -> None:
    global _WORKER_SOURCE, _WORKER_OUTPUT, _WORKER_LEVELS
    _WORKER_SOURCE = zarr.open_group(source, mode="r")
    _WORKER_OUTPUT = zarr.open_group(output, mode="r+")
    _WORKER_LEVELS = levels
    try:
        import numcodecs.blosc

        numcodecs.blosc.set_nthreads(1)
    except Exception:
        pass


def _run_tile(job: TileJob) -> tuple[int, bool, int]:
    level = _WORKER_LEVELS[job.level]
    source = _WORKER_SOURCE[level.source_name]
    output = _WORKER_OUTPUT[level.name]
    bounds = _mapped_input_bounds(
        level.matrix_target_to_local_zyx,
        job.start,
        job.stop,
        level.effective_source_shape,
    )
    if bounds is None:
        return job.tile_id, False, 0

    lo, hi = bounds
    slab = np.asarray(source[tuple(slice(a, b) for a, b in zip(lo, hi))])
    matrix = level.matrix_target_to_local_zyx
    linear = matrix[:3, :3]
    offset = linear @ np.asarray(job.start, dtype=np.float64) + matrix[:3, 3]
    offset -= np.asarray(lo, dtype=np.float64)
    shape = tuple(b - a for a, b in zip(job.start, job.stop))
    transformed = ndimage.affine_transform(
        slab,
        linear,
        offset=offset,
        output_shape=shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    nonzero = int(np.count_nonzero(transformed))
    if nonzero:
        output[tuple(slice(a, b) for a, b in zip(job.start, job.stop))] = transformed
    return job.tile_id, bool(nonzero), nonzero


def _bounded_results(
    pool: ProcessPoolExecutor,
    jobs: Iterable[TileJob],
    workers: int,
) -> Iterator[tuple[int, bool, int]]:
    iterator = iter(jobs)
    pending: set[Future[tuple[int, bool, int]]] = set()
    exhausted = False
    limit = max(1, workers * 2)
    while pending or not exhausted:
        while not exhausted and len(pending) < limit:
            try:
                job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(pool.submit(_run_tile, job))
        if not pending:
            continue
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _remove_progress_files(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _convert_one(
    spec: DatasetSpec,
    output_dir: Path,
    *,
    workers: int,
    tile_size: int,
    max_level: int,
    overwrite: bool,
    resume: bool,
    dry_run: bool,
) -> None:
    if not spec.source.is_dir():
        raise FileNotFoundError(f"source does not exist: {spec.source}")
    levels, target_attrs, transform = _load_level_info(spec, max_level)
    output = output_dir / spec.output_name
    progress_path = _progress_path(output)

    print(f"\n{spec.display_name}")
    print(f"  source: {spec.source}")
    print(f"  target: {spec.target_root}")
    print(f"  output: {output}")
    if spec.swap_source_yx:
        print("  correction: drop trailing level-0 Z slice, then swap source Y/X")
    print(f"  transform fixed volume: {transform['fixed_volume']}")

    plans: list[tuple[list[TileJob], int]] = []
    for index, level in enumerate(levels):
        jobs, candidates = _plan_jobs(spec, index, level, tile_size)
        plans.append((jobs, candidates))
        raw_size = math.prod(level.target_shape) * np.dtype(
            zarr.open_group(str(spec.source), mode="r")[level.source_name].dtype
        ).itemsize
        print(
            f"  L{index}: target {level.target_shape}, source {level.source_shape}, "
            f"{len(jobs):,}/{candidates:,} tiles may contain labels, "
            f"full-grid raw size {_human_size(raw_size)}"
        )

    if dry_run:
        return

    if tile_size % OUTPUT_CHUNKS[0] != 0:
        raise ValueError(f"--tile-size must be a multiple of {OUTPUT_CHUNKS[0]}")
    if overwrite:
        if output.exists():
            shutil.rmtree(output)
        _remove_progress_files(progress_path)
    if output.exists() and not resume:
        raise FileExistsError(f"output exists: {output}; pass --resume or --overwrite")
    if resume and not output.exists():
        raise FileNotFoundError(f"cannot resume missing output: {output}")
    if not output.exists() and progress_path.exists():
        raise FileExistsError(
            f"stale resume journal exists: {progress_path}; pass --overwrite to discard it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if _validate_existing_output(output, spec, levels, tile_size):
            print("  already complete")
            return
        if not progress_path.exists():
            raise FileNotFoundError(f"resume journal is missing: {progress_path}")
    else:
        _create_output(output, spec, levels, target_attrs, transform, tile_size)

    progress = _open_progress(progress_path)
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(str(spec.source), str(output), levels),
        ) as pool:
            for level_index, ((jobs, _), level) in enumerate(zip(plans, levels)):
                completed = _completed_tiles(progress, level_index)
                pending = [job for job in jobs if job.tile_id not in completed]
                written_tiles = 0
                nonzero_voxels = 0
                since_commit = 0
                with tqdm(
                    total=len(jobs),
                    initial=len(jobs) - len(pending),
                    desc=f"{spec.display_name} L{level_index}",
                    unit="tile",
                    dynamic_ncols=True,
                ) as bar:
                    for tile_id, wrote, nonzero in _bounded_results(pool, pending, workers):
                        progress.execute(
                            "INSERT OR IGNORE INTO completed(level, tile_id) VALUES (?, ?)",
                            (level_index, tile_id),
                        )
                        written_tiles += int(wrote)
                        nonzero_voxels += nonzero
                        since_commit += 1
                        if since_commit >= 64:
                            progress.commit()
                            since_commit = 0
                        bar.update(1)
                progress.commit()
                print(
                    f"  L{level_index} complete: {written_tiles:,} newly written tiles, "
                    f"{nonzero_voxels:,} newly sampled nonzero voxels"
                )

        root = zarr.open_group(str(output), mode="r+")
        root.attrs["complete"] = True
        progress.close()
        _remove_progress_files(progress_path)
        print(f"  complete: {output}")
    except BaseException:
        progress.commit()
        progress.close()
        print(f"  interrupted; resume with --resume (journal: {progress_path})", file=sys.stderr)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *SPECS),
        default="all",
        help="conversion to run (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "transformed_78kev",
        help="directory for the transformed OME-Zarr stores",
    )
    parser.add_argument(
        "--paris4-source",
        type=Path,
        default=SPECS["paris4"].source,
        help="override the local Paris 4 label OME-Zarr",
    )
    parser.add_argument(
        "--pherc1667-source",
        type=Path,
        default=SPECS["pherc1667"].source,
        help="override the local PHerc1667 label OME-Zarr",
    )
    parser.add_argument("--workers", type=int, default=4, help="worker processes (default: 4)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="output tile edge; multiple of 128 (default: 512, about 128 MiB/worker)",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=5,
        help="highest OME pyramid level to create (default: 5)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue an interrupted conversion")
    mode.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate matches and count work without creating output",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.tile_size < OUTPUT_CHUNKS[0] or args.tile_size % OUTPUT_CHUNKS[0]:
        parser.error(f"--tile-size must be a positive multiple of {OUTPUT_CHUNKS[0]}")
    if not 0 <= args.max_level <= 5:
        parser.error("--max-level must be in 0..5")
    return args


def main() -> int:
    args = _parse_args()
    specs = {
        "paris4": replace(SPECS["paris4"], source=args.paris4_source.resolve()),
        "pherc1667": replace(SPECS["pherc1667"], source=args.pherc1667_source.resolve()),
    }
    selected = list(specs.values()) if args.dataset == "all" else [specs[args.dataset]]
    for spec in selected:
        _convert_one(
            spec,
            args.output_dir.resolve(),
            workers=args.workers,
            tile_size=args.tile_size,
            max_level=args.max_level,
            overwrite=args.overwrite,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
