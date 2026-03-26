import argparse
import json
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm.auto import tqdm

import aiohttp
import fsspec
import vesuvius.tifxyz as tifxyz

from koine_machines.common.common import load_volume_auth, open_zarr
from koine_machines.data.patch_finding import build_patch_index, resolve_patch_finding_type
from koine_machines.data.native_crop import compute_native_crop_bbox_from_patch_points
from koine_machines.data.segment import Segment


DEFAULT_PATCH_SIZE_ZYX = (512, 512, 512)
DEFAULT_OVERLAP_FRACTION = 0.25
DEFAULT_STORED_GRID_PAD = 40
DEFAULT_RECOMPRESS_PRESET = "balanced"
DEFAULT_PATCH_FILTER = "supervision"

_WORKER_SOURCE = None
_WORKER_DEST = None
_WORKER_CHUNK_SHAPE = None
_WORKER_ARRAY_SHAPE = None


@dataclass(frozen=True)
class DatasetSourceSpec:
    dataset_name: str
    volume_path: str
    volume_scale: int = 0
    volume_auth_json: str | None = None


@dataclass(frozen=True)
class DatasetChunkPlan:
    dataset_name: str
    dataset_dir: Path
    output_path: Path
    volume_path: str
    volume_scale: int
    volume_auth_json: str | None
    patch_count: int
    candidate_bbox_count: int
    chunk_shapes_by_scale: dict[int, tuple[int, int, int]]
    array_shapes_by_scale: dict[int, tuple[int, int, int]]
    chunk_ids_by_scale: dict[int, tuple[tuple[int, int, int], ...]]

    @property
    def chunk_bytes(self) -> int:
        return int(np.prod(self.chunk_shapes_by_scale[self.volume_scale], dtype=np.int64))

    @property
    def total_bytes(self) -> int:
        total = 0
        for scale, chunk_ids in self.chunk_ids_by_scale.items():
            total += int(len(chunk_ids) * np.prod(self.chunk_shapes_by_scale[scale], dtype=np.int64))
        return int(total)

    @property
    def requested_scale_chunk_count(self) -> int:
        return int(len(self.chunk_ids_by_scale[self.volume_scale]))


def open_zarr_group(path, auth=None):
    path_str = str(path)
    user, password = load_volume_auth(auth)
    use_https_auth = path_str.startswith("https://") and bool(user) and bool(password)
    if use_https_auth:
        fs = fsspec.filesystem(
            "https",
            client_kwargs={"auth": aiohttp.BasicAuth(user, password)},
        )
        store = zarr.storage.FSStore(
            path_str.rstrip("/"),
            fs=fs,
            mode="r",
            check=False,
            create=False,
            exceptions=(KeyError, FileNotFoundError, PermissionError, OSError, aiohttp.ClientResponseError),
        )
        return zarr.open(store, mode="r")
    return zarr.open(path_str, mode="r")


def compressor_from_recompress_preset(preset: str):
    preset_key = str(preset).strip().lower()
    if preset_key == "fast":
        return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    if preset_key == "balanced":
        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    raise ValueError(f"Unsupported recompress preset: {preset!r}")


def _normalize_mapping_entry(dataset_name: str, entry) -> DatasetSourceSpec:
    if isinstance(entry, str):
        return DatasetSourceSpec(dataset_name=dataset_name, volume_path=entry)

    if not isinstance(entry, dict):
        raise TypeError(
            f"Dataset entry for {dataset_name!r} must be a string or object, got {type(entry).__name__}."
        )

    volume_path = entry.get("volume_path")
    if not volume_path:
        raise ValueError(f"Dataset entry for {dataset_name!r} is missing volume_path.")

    return DatasetSourceSpec(
        dataset_name=dataset_name,
        volume_path=str(volume_path),
        volume_scale=int(entry.get("volume_scale", 0)),
        volume_auth_json=(
            None if entry.get("volume_auth_json") in (None, "") else str(entry["volume_auth_json"])
        ),
    )


def load_dataset_sources(mapping_json_path: str | Path) -> dict[str, DatasetSourceSpec]:
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "datasets" in raw:
        raw = raw["datasets"]

    if isinstance(raw, list):
        normalized = {}
        for entry in raw:
            if not isinstance(entry, dict) or "dataset" not in entry:
                raise ValueError("List-style JSON entries must contain a 'dataset' key.")
            dataset_name = str(entry["dataset"])
            normalized[dataset_name] = _normalize_mapping_entry(dataset_name, entry)
        return normalized

    if isinstance(raw, dict):
        return {
            str(dataset_name): _normalize_mapping_entry(str(dataset_name), entry)
            for dataset_name, entry in raw.items()
        }

    raise TypeError("Volumes JSON must be either an object mapping dataset names or a list of dataset objects.")


def _bbox_to_chunk_ids(
    world_bbox: tuple[int, int, int, int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> Iterable[tuple[int, int, int]]:
    clipped_bounds = _clip_world_bbox_to_array(world_bbox, array_shape_zyx)
    if clipped_bounds is None:
        return ()
    src_starts, src_ends = clipped_bounds

    z_chunk_start = src_starts[0] // int(chunk_shape_zyx[0])
    z_chunk_stop = (src_ends[0] - 1) // int(chunk_shape_zyx[0])
    y_chunk_start = src_starts[1] // int(chunk_shape_zyx[1])
    y_chunk_stop = (src_ends[1] - 1) // int(chunk_shape_zyx[1])
    x_chunk_start = src_starts[2] // int(chunk_shape_zyx[2])
    x_chunk_stop = (src_ends[2] - 1) // int(chunk_shape_zyx[2])

    return (
        (z_idx, y_idx, x_idx)
        for z_idx in range(z_chunk_start, z_chunk_stop + 1)
        for y_idx in range(y_chunk_start, y_chunk_stop + 1)
        for x_idx in range(x_chunk_start, x_chunk_stop + 1)
    )


def _clip_world_bbox_to_array(
    world_bbox: tuple[int, int, int, int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    z0, y0, x0, z1, y1, x1 = (int(v) for v in world_bbox)
    starts = (
        max(0, z0),
        max(0, y0),
        max(0, x0),
    )
    stops = (
        min(int(array_shape_zyx[0]), z1),
        min(int(array_shape_zyx[1]), y1),
        min(int(array_shape_zyx[2]), x1),
    )
    if any(start >= stop for start, stop in zip(starts, stops)):
        return None
    return starts, stops


def collect_unique_chunk_ids(
    patches: Iterable[dict],
    *,
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    chunk_ids = set()
    for patch in patches:
        world_bbox = patch["world_bbox"]
        chunk_ids.update(
            _bbox_to_chunk_ids(
                tuple(int(v) for v in world_bbox),
                chunk_shape_zyx=chunk_shape_zyx,
                array_shape_zyx=array_shape_zyx,
            )
    )
    return tuple(sorted(chunk_ids))


def _chunk_id_to_bounds(
    chunk_id: tuple[int, int, int],
    *,
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    starts = tuple(int(chunk_id[i]) * int(chunk_shape_zyx[i]) for i in range(3))
    stops = tuple(min(int(array_shape_zyx[i]), starts[i] + int(chunk_shape_zyx[i])) for i in range(3))
    if any(start >= stop for start, stop in zip(starts, stops)):
        return None
    return starts, stops


def _scaled_bounds_for_scale(
    starts: tuple[int, int, int],
    stops: tuple[int, int, int],
    *,
    source_scale: int,
    target_scale: int,
    target_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    if target_scale == source_scale:
        target_starts = starts
        target_stops = stops
    elif target_scale > source_scale:
        factor = 2 ** int(target_scale - source_scale)
        target_starts = tuple(int(start) // factor for start in starts)
        target_stops = tuple(int((stop + factor - 1) // factor) for stop in stops)
    else:
        factor = 2 ** int(source_scale - target_scale)
        target_starts = tuple(int(start) * factor for start in starts)
        target_stops = tuple(int(stop) * factor for stop in stops)

    clipped_starts = tuple(max(0, int(v)) for v in target_starts)
    clipped_stops = tuple(min(int(target_shape_zyx[i]), int(target_stops[i])) for i in range(3))
    if any(start >= stop for start, stop in zip(clipped_starts, clipped_stops)):
        return None
    return clipped_starts, clipped_stops


def _bounds_to_chunk_ids(
    starts: tuple[int, int, int],
    stops: tuple[int, int, int],
    *,
    chunk_shape_zyx: tuple[int, int, int],
) -> Iterable[tuple[int, int, int]]:
    return (
        (z_idx, y_idx, x_idx)
        for z_idx in range(starts[0] // int(chunk_shape_zyx[0]), ((stops[0] - 1) // int(chunk_shape_zyx[0])) + 1)
        for y_idx in range(starts[1] // int(chunk_shape_zyx[1]), ((stops[1] - 1) // int(chunk_shape_zyx[1])) + 1)
        for x_idx in range(starts[2] // int(chunk_shape_zyx[2]), ((stops[2] - 1) // int(chunk_shape_zyx[2])) + 1)
    )


def scale_chunk_ids_across_multiscale(
    requested_scale_chunk_ids: tuple[tuple[int, int, int], ...],
    *,
    requested_scale: int,
    chunk_shapes_by_scale: dict[int, tuple[int, int, int]],
    array_shapes_by_scale: dict[int, tuple[int, int, int]],
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    requested_chunk_shape = chunk_shapes_by_scale[int(requested_scale)]
    requested_array_shape = array_shapes_by_scale[int(requested_scale)]
    requested_bounds = []
    for chunk_id in requested_scale_chunk_ids:
        bounds = _chunk_id_to_bounds(
            chunk_id,
            chunk_shape_zyx=requested_chunk_shape,
            array_shape_zyx=requested_array_shape,
        )
        if bounds is not None:
            requested_bounds.append(bounds)

    scaled_chunk_ids = {}
    for scale in sorted(chunk_shapes_by_scale):
        target_chunk_shape = chunk_shapes_by_scale[scale]
        target_array_shape = array_shapes_by_scale[scale]
        scale_chunk_ids = set()
        for starts, stops in requested_bounds:
            target_bounds = _scaled_bounds_for_scale(
                starts,
                stops,
                source_scale=int(requested_scale),
                target_scale=int(scale),
                target_shape_zyx=target_array_shape,
            )
            if target_bounds is None:
                continue
            scale_chunk_ids.update(
                _bounds_to_chunk_ids(
                    target_bounds[0],
                    target_bounds[1],
                    chunk_shape_zyx=target_chunk_shape,
                )
            )
        scaled_chunk_ids[int(scale)] = tuple(sorted(scale_chunk_ids))
    return scaled_chunk_ids


def _build_output_arrays(source_group, output_path: Path, overwrite: bool, recompress: str):
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)

    root = zarr.open_group(str(output_path), mode="a" if output_path.exists() else "w")
    compressor = compressor_from_recompress_preset(recompress)
    try:
        root.attrs.update(source_group.attrs.asdict())
    except Exception:
        pass
    existing_array_keys = set(root.array_keys())
    for key in sorted(source_group.array_keys(), key=lambda v: int(v) if str(v).isdigit() else str(v)):
        source_array = source_group[key]
        if str(key) in existing_array_keys:
            out_array = root[str(key)]
        else:
            out_array = root.create_dataset(
                str(key),
                shape=tuple(int(v) for v in source_array.shape),
                chunks=tuple(int(v) for v in source_array.chunks),
                dtype=source_array.dtype,
                compressor=compressor,
                fill_value=source_array.fill_value,
                filters=source_array.filters,
                order=getattr(source_array, "order", "C"),
                overwrite=False,
            )
        try:
            out_array.attrs.update(source_array.attrs.asdict())
        except Exception:
            pass
    return root


def _missing_chunk_ids_for_scale(
    output_path: Path,
    *,
    scale: int,
    chunk_ids: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    if not chunk_ids or not output_path.exists():
        return chunk_ids

    try:
        output_array = zarr.open(str(output_path), path=str(int(scale)), mode="r")
    except Exception:
        return chunk_ids

    store = getattr(output_array, "chunk_store", output_array.store)
    missing_chunk_ids = []
    for chunk_id in chunk_ids:
        if output_array._chunk_key(tuple(int(v) for v in chunk_id)) not in store:
            missing_chunk_ids.append(tuple(int(v) for v in chunk_id))
    return tuple(missing_chunk_ids)


def _init_chunk_copy_worker(volume_path: str, volume_scale: int, volume_auth_json: str | None, output_path: str):
    global _WORKER_SOURCE, _WORKER_DEST, _WORKER_CHUNK_SHAPE, _WORKER_ARRAY_SHAPE
    _WORKER_SOURCE = open_zarr(volume_path, volume_scale, auth=volume_auth_json)
    _WORKER_DEST = zarr.open(str(output_path), path=str(int(volume_scale)), mode="r+")
    _WORKER_CHUNK_SHAPE = tuple(int(v) for v in _WORKER_SOURCE.chunks)
    _WORKER_ARRAY_SHAPE = tuple(int(v) for v in _WORKER_SOURCE.shape)


def _copy_one_chunk(chunk_id: tuple[int, int, int]) -> int:
    if _WORKER_SOURCE is None or _WORKER_DEST is None:
        raise RuntimeError("Chunk copy worker is not initialized.")

    z_idx, y_idx, x_idx = (int(v) for v in chunk_id)
    chunk_shape = _WORKER_CHUNK_SHAPE
    array_shape = _WORKER_ARRAY_SHAPE
    starts = (
        z_idx * int(chunk_shape[0]),
        y_idx * int(chunk_shape[1]),
        x_idx * int(chunk_shape[2]),
    )
    stops = (
        min(int(array_shape[0]), starts[0] + int(chunk_shape[0])),
        min(int(array_shape[1]), starts[1] + int(chunk_shape[1])),
        min(int(array_shape[2]), starts[2] + int(chunk_shape[2])),
    )
    if any(start >= stop for start, stop in zip(starts, stops)):
        return 0

    slices = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    _WORKER_DEST[slices] = np.asarray(_WORKER_SOURCE[slices])
    return 1


def copy_chunks_to_output(
    *,
    volume_path: str,
    volume_scale: int,
    volume_auth_json: str | None,
    output_path: Path,
    chunk_ids_by_scale: dict[int, tuple[tuple[int, int, int], ...]],
    worker_count: int,
    tqdm_desc: str,
    overwrite: bool,
    recompress: str,
) -> dict[str, dict[int, int]]:
    source_group = open_zarr_group(volume_path, auth=volume_auth_json)
    _build_output_arrays(
        source_group,
        output_path,
        overwrite=overwrite,
        recompress=recompress,
    )

    copied_counts_by_scale = {}
    existing_counts_by_scale = {}
    for scale in sorted(chunk_ids_by_scale):
        requested_chunk_ids = chunk_ids_by_scale[scale]
        chunk_ids = _missing_chunk_ids_for_scale(
            output_path,
            scale=scale,
            chunk_ids=requested_chunk_ids,
        )
        copied_counts_by_scale[int(scale)] = 0
        existing_counts_by_scale[int(scale)] = int(len(requested_chunk_ids) - len(chunk_ids))
        scale_desc = f"{tqdm_desc} [scale {scale}]"
        if not chunk_ids:
            continue
        if worker_count <= 1:
            _init_chunk_copy_worker(
                volume_path=volume_path,
                volume_scale=scale,
                volume_auth_json=volume_auth_json,
                output_path=str(output_path),
            )
            copied_count = 0
            for chunk_id in tqdm(chunk_ids, total=len(chunk_ids), desc=scale_desc, unit="chunk"):
                copied_count += _copy_one_chunk(chunk_id)
            copied_counts_by_scale[int(scale)] = int(copied_count)
            continue

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=worker_count,
            initializer=_init_chunk_copy_worker,
            initargs=(volume_path, scale, volume_auth_json, str(output_path)),
        ) as pool:
            iterator = pool.imap_unordered(_copy_one_chunk, chunk_ids, chunksize=8)
            copied_count = 0
            for copied in tqdm(iterator, total=len(chunk_ids), desc=scale_desc, unit="chunk"):
                copied_count += int(copied)
            copied_counts_by_scale[int(scale)] = int(copied_count)

    return {
        "copied_counts_by_scale": copied_counts_by_scale,
        "existing_counts_by_scale": existing_counts_by_scale,
    }


def _surface_patch_bbox(surface: int, y0: int, x0: int, patch_size) -> tuple[int, int, int, int, int, int]:
    depth, height, width = (int(v) for v in patch_size)
    z0 = int(surface - depth // 2)
    return (
        z0,
        int(y0),
        int(x0),
        z0 + depth,
        int(y0) + height,
        int(x0) + width,
    )


def _labeled_patch_coverage(label_patch) -> float:
    patch = np.asarray(label_patch)
    if patch.size == 0:
        return 0.0

    labeled_ys, labeled_xs = np.nonzero(patch)
    if labeled_ys.size == 0:
        return 0.0

    labeled_area = (
        (int(labeled_ys.max()) - int(labeled_ys.min()) + 1)
        * (int(labeled_xs.max()) - int(labeled_xs.min()) + 1)
    )
    return float(labeled_area) / float(patch.size)


def _iter_segment_dirs(dataset_dir: Path) -> Iterable[Path]:
    for segment_dir in sorted(dataset_dir.iterdir()):
        if not segment_dir.is_dir() or segment_dir.name == "unused":
            continue
        if not any(segment_dir.rglob("x.tif")):
            continue
        yield segment_dir


def _segment_patch_candidates(
    segment: Segment,
    *,
    patch_filter: str,
) -> tuple[list[tuple[int, int]], int]:
    patch_filter = str(patch_filter).strip().lower()
    if patch_filter not in {"positive", "supervision"}:
        raise ValueError(f"Unsupported patch filter: {patch_filter!r}")

    volume_auth = segment.config.get("volume_auth_json")
    supervision_mask = open_zarr(segment.supervision_mask, resolution=segment.scale, auth=volume_auth)
    inklabels = open_zarr(segment.inklabels, resolution=segment.scale, auth=volume_auth)

    surface = int(supervision_mask.shape[0] // 2)
    patch_size = tuple(int(v) for v in segment.patch_size)
    patch_finding_type = resolve_patch_finding_type(segment.config)

    if patch_finding_type == "default":
        ys, xs = np.nonzero(np.asarray(supervision_mask[surface]))
        if int(ys.size) == 0:
            raise ValueError(f"{segment.supervision_mask} contains no nonzero voxels")

        stride = int(patch_size[1] * float(segment.config["patch_overlap"]))
        if stride <= 0:
            raise ValueError(f"patch_overlap produced non-positive stride={stride}")

        patch_starts_yx = np.unique(
            np.stack([ys // stride * stride, xs // stride * stride], axis=1),
            axis=0,
        )
        candidates = [(int(y0), int(x0)) for y0, x0 in patch_starts_yx.tolist()]
        return candidates, int(len(candidates))

    patch_size_yx = int(patch_size[1])
    default_stride = int(patch_size_yx * float(segment.config["patch_overlap"]))
    tile_size = int(segment.config.get("patch_finding_tile_size", patch_size_yx))
    stride = int(segment.config.get("patch_finding_stride", default_stride))
    filter_empty_tile = bool(segment.config.get("patch_finding_filter_empty_tile", False))
    tile_mask = inklabels[surface] if patch_filter == "positive" else supervision_mask[surface]
    _, xyxys, _ = build_patch_index(
        mask=tile_mask,
        fragment_mask=supervision_mask[surface],
        size=patch_size_yx,
        tile_size=tile_size,
        stride=stride,
        filter_empty_tile=filter_empty_tile,
    )
    candidates = [(int(y1), int(x1)) for x1, y1, _, _ in xyxys.tolist()]
    return candidates, int(len(candidates))


def _build_segment_download_patches(
    segment: Segment,
    *,
    patch_filter: str,
) -> tuple[list[dict], int]:
    patch_filter = str(patch_filter).strip().lower()
    candidates, candidate_count = _segment_patch_candidates(segment, patch_filter=patch_filter)

    volume_auth = segment.config.get("volume_auth_json")
    supervision_mask = open_zarr(segment.supervision_mask, resolution=segment.scale, auth=volume_auth)
    inklabels = open_zarr(segment.inklabels, resolution=segment.scale, auth=volume_auth)
    validation_mask = None
    if segment.validation_mask is not None:
        validation_mask = open_zarr(segment.validation_mask, resolution=segment.scale, auth=volume_auth)

    surface = int(supervision_mask.shape[0] // 2)
    patch_size = tuple(int(v) for v in segment.patch_size)
    min_labeled_coverage = float(segment.config.get("patch_min_labeled_coverage", 0.0))
    patch_tifxyz = tifxyz.read_tifxyz(segment.segment_dir)
    patch_tifxyz.use_full_resolution()

    kept_patches: list[dict] = []
    for y0, x0 in candidates:
        patch_bbox_zyx = _surface_patch_bbox(surface, y0, x0, patch_size)
        flat_x, flat_y, flat_z, flat_valid = patch_tifxyz[
            int(y0):int(y0) + patch_size[1],
            int(x0):int(x0) + patch_size[2],
        ]
        patch_zyxs = np.stack([flat_z, flat_y, flat_x], axis=-1)
        native_world_bbox = compute_native_crop_bbox_from_patch_points(
            patch_zyxs,
            flat_valid,
            patch_size,
        )

        if validation_mask is not None:
            validation_patch = validation_mask[
                surface,
                int(y0):int(y0) + patch_size[1],
                int(x0):int(x0) + patch_size[2],
            ]
            if bool(validation_patch.size > 0 and np.any(validation_patch)):
                kept_patches.append(
                    {
                        "segment_relpath": str(segment.segment_relpath),
                        "world_bbox": native_world_bbox,
                        "flat_bbox": patch_bbox_zyx,
                        "is_validation": True,
                    }
                )
                continue

        if patch_filter == "supervision":
            supervision_patch = supervision_mask[
                surface,
                int(y0):int(y0) + patch_size[1],
                int(x0):int(x0) + patch_size[2],
            ]
            if bool(supervision_patch.size > 0 and np.any(supervision_patch)):
                kept_patches.append(
                    {
                        "segment_relpath": str(segment.segment_relpath),
                        "world_bbox": native_world_bbox,
                        "flat_bbox": patch_bbox_zyx,
                        "is_validation": False,
                    }
                )
            continue

        label_patch = inklabels[
            surface,
            int(y0):int(y0) + patch_size[1],
            int(x0):int(x0) + patch_size[2],
        ]
        if _labeled_patch_coverage(label_patch) >= min_labeled_coverage:
            kept_patches.append(
                {
                    "segment_relpath": str(segment.segment_relpath),
                    "world_bbox": native_world_bbox,
                    "flat_bbox": patch_bbox_zyx,
                    "is_validation": False,
                }
            )

    return kept_patches, int(candidate_count)


def _build_dataset_download_patches(
    *,
    dataset_dir: Path,
    source_spec: DatasetSourceSpec,
    patch_size_zyx: tuple[int, int, int],
    overlap_fraction: float,
    patch_filter: str,
    patch_finding_type: str,
    patch_min_labeled_coverage: float,
    patch_finding_tile_size: int | None,
    patch_finding_stride: int | None,
    patch_finding_filter_empty_tile: bool,
    label_version: str | None,
) -> tuple[list[dict], int]:
    config = {
        "patch_size": [int(v) for v in patch_size_zyx],
        "patch_overlap": float(overlap_fraction),
        "patch_finding_type": str(patch_finding_type),
        "patch_min_labeled_coverage": float(patch_min_labeled_coverage),
        "volume_auth_json": source_spec.volume_auth_json,
        "label_version": label_version,
    }
    if patch_finding_tile_size is not None:
        config["patch_finding_tile_size"] = int(patch_finding_tile_size)
    if patch_finding_stride is not None:
        config["patch_finding_stride"] = int(patch_finding_stride)
    if bool(patch_finding_filter_empty_tile):
        config["patch_finding_filter_empty_tile"] = True

    patches: list[dict] = []
    candidate_bbox_count = 0
    for segment_dir in _iter_segment_dirs(dataset_dir):
        segment = Segment(
            config=config,
            image_volume=Path(str(source_spec.volume_path)),
            scale=int(source_spec.volume_scale),
            dataset_idx=0,
            segment_relpath=segment_dir.relative_to(dataset_dir).as_posix(),
            segment_dir=segment_dir,
            segment_name=segment_dir.name,
        )
        segment.discover_labels(label_version=label_version, extension=".zarr")
        segment_patches, segment_candidate_count = _build_segment_download_patches(
            segment,
            patch_filter=patch_filter,
        )
        patches.extend(segment_patches)
        candidate_bbox_count += int(segment_candidate_count)

    return patches, int(candidate_bbox_count)


def _debug_patch_chunk_mapping(
    *,
    dataset_name: str,
    volume_path: str,
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    patches: list[dict],
    limit: int,
) -> None:
    limit = max(0, int(limit))
    if limit <= 0:
        return

    intersecting = 0
    for patch in patches:
        if _clip_world_bbox_to_array(tuple(int(v) for v in patch["world_bbox"]), array_shape_zyx) is not None:
            intersecting += 1

    print(
        f"[{dataset_name}] debug source={volume_path} array_shape={array_shape_zyx} "
        f"chunk_shape={chunk_shape_zyx} intersecting_patches={intersecting}/{len(patches)}"
    )
    for idx, patch in enumerate(patches[:limit]):
        world_bbox = tuple(int(v) for v in patch["world_bbox"])
        clipped_bounds = _clip_world_bbox_to_array(world_bbox, array_shape_zyx)
        chunk_ids = tuple(
            _bbox_to_chunk_ids(
                world_bbox,
                chunk_shape_zyx=chunk_shape_zyx,
                array_shape_zyx=array_shape_zyx,
            )
        )
        print(
            f"[{dataset_name}] debug patch[{idx}] "
            f"segment={patch.get('segment_relpath', '?')} "
            f"flat_bbox={patch.get('flat_bbox')} "
            f"native_bbox={world_bbox} "
            f"clipped={clipped_bounds} "
            f"chunks={len(chunk_ids)}"
        )


def build_dataset_chunk_plan(
    *,
    datasets_root: Path,
    output_root: Path,
    source_spec: DatasetSourceSpec,
    patch_size_zyx: tuple[int, int, int],
    overlap_fraction: float,
    stored_grid_pad: int,
    patch_finding_workers: int,
    patch_filter: str,
    patch_finding_type: str,
    patch_min_labeled_coverage: float,
    patch_finding_tile_size: int | None,
    patch_finding_stride: int | None,
    patch_finding_filter_empty_tile: bool,
    label_version: str | None,
    debug_patches: int = 0,
) -> DatasetChunkPlan:
    dataset_dir = datasets_root / source_spec.dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    source_group = open_zarr_group(source_spec.volume_path, auth=source_spec.volume_auth_json)
    scale_keys = sorted(
        int(key) for key in source_group.array_keys() if str(key).isdigit()
    )
    if int(source_spec.volume_scale) not in scale_keys:
        raise KeyError(
            f"Requested scale {source_spec.volume_scale} is missing from source group {source_spec.volume_path!r}."
        )

    del stored_grid_pad, patch_finding_workers

    patches, candidate_bbox_count = _build_dataset_download_patches(
        dataset_dir=dataset_dir,
        source_spec=source_spec,
        patch_size_zyx=patch_size_zyx,
        overlap_fraction=float(overlap_fraction),
        patch_filter=patch_filter,
        patch_finding_type=patch_finding_type,
        patch_min_labeled_coverage=float(patch_min_labeled_coverage),
        patch_finding_tile_size=patch_finding_tile_size,
        patch_finding_stride=patch_finding_stride,
        patch_finding_filter_empty_tile=bool(patch_finding_filter_empty_tile),
        label_version=label_version,
    )

    requested_scale_array = source_group[str(int(source_spec.volume_scale))]
    chunk_shapes_by_scale = {
        int(scale): tuple(int(v) for v in source_group[str(int(scale))].chunks)
        for scale in scale_keys
    }
    array_shapes_by_scale = {
        int(scale): tuple(int(v) for v in source_group[str(int(scale))].shape)
        for scale in scale_keys
    }
    _debug_patch_chunk_mapping(
        dataset_name=source_spec.dataset_name,
        volume_path=str(source_spec.volume_path),
        array_shape_zyx=tuple(int(v) for v in requested_scale_array.shape),
        chunk_shape_zyx=tuple(int(v) for v in requested_scale_array.chunks),
        patches=patches,
        limit=int(debug_patches),
    )
    requested_scale_chunk_ids = collect_unique_chunk_ids(
        patches,
        chunk_shape_zyx=tuple(int(v) for v in requested_scale_array.chunks),
        array_shape_zyx=tuple(int(v) for v in requested_scale_array.shape),
    )
    chunk_ids_by_scale = scale_chunk_ids_across_multiscale(
        requested_scale_chunk_ids,
        requested_scale=int(source_spec.volume_scale),
        chunk_shapes_by_scale=chunk_shapes_by_scale,
        array_shapes_by_scale=array_shapes_by_scale,
    )
    return DatasetChunkPlan(
        dataset_name=source_spec.dataset_name,
        dataset_dir=dataset_dir,
        output_path=output_root / f"{source_spec.dataset_name}.zarr",
        volume_path=str(source_spec.volume_path),
        volume_scale=int(source_spec.volume_scale),
        volume_auth_json=source_spec.volume_auth_json,
        patch_count=int(len(patches)),
        candidate_bbox_count=int(candidate_bbox_count),
        chunk_shapes_by_scale=chunk_shapes_by_scale,
        array_shapes_by_scale=array_shapes_by_scale,
        chunk_ids_by_scale=chunk_ids_by_scale,
    )


def _parse_patch_size(values: list[int]) -> tuple[int, int, int]:
    if len(values) == 1:
        values = values * 3
    if len(values) != 3 or any(int(v) <= 0 for v in values):
        raise ValueError(f"patch_size must be one int or three ints, got {values!r}")
    return tuple(int(v) for v in values)


def _format_bytes(num_bytes: int) -> str:
    gib = float(num_bytes) / float(1024**3)
    gb = float(num_bytes) / 1e9
    return f"{num_bytes} bytes ({gb:.2f} GB, {gib:.2f} GiB)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute tifxyz patch-driven chunk coverage and download only the required "
            "chunks into <dataset>/<dataset>.zarr."
        )
    )
    parser.add_argument(
        "--datasets-root",
        required=True,
        help="Root directory containing dataset folders such as 1667, 814, 841, etc.",
    )
    parser.add_argument(
        "--volumes-json",
        required=True,
        help=(
            "JSON mapping dataset folder names to source zarrs. "
            "Entries may be strings or objects with volume_path, volume_scale, volume_auth_json."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Parent directory where output dataset zarrs will be written as <dataset>.zarr.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="Dataset folder name to process. Repeat to restrict to multiple datasets.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs="+",
        default=list(DEFAULT_PATCH_SIZE_ZYX),
        help="Patch size in z y x. Pass one value for isotropic patches.",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=DEFAULT_OVERLAP_FRACTION,
        help="Patch overlap fraction used by tifxyz patchfinding.",
    )
    parser.add_argument(
        "--stored-grid-pad",
        type=int,
        default=DEFAULT_STORED_GRID_PAD,
        help="Stored grid padding passed into patchfinding.",
    )
    parser.add_argument(
        "--patch-finding-workers",
        type=int,
        default=4,
        help="Compatibility no-op kept for existing invocations.",
    )
    parser.add_argument(
        "--patch-finding-type",
        choices=("default", "subtiling"),
        default="default",
        help="Current shared patch-finding implementation to mirror.",
    )
    parser.add_argument(
        "--patch-min-labeled-coverage",
        type=float,
        default=0.0,
        help="Minimum labeled coverage when --patch-filter=positive.",
    )
    parser.add_argument(
        "--patch-finding-tile-size",
        type=int,
        default=None,
        help="Subtiling tile size. Leave unset to use the patch height.",
    )
    parser.add_argument(
        "--patch-finding-stride",
        type=int,
        default=None,
        help="Subtiling tile stride. Leave unset to derive it from patch overlap.",
    )
    parser.add_argument(
        "--patch-finding-filter-empty-tile",
        action="store_true",
        help="Mirror the subtiling empty-tile pruning used by training.",
    )
    parser.add_argument(
        "--label-version",
        default=None,
        help="Optional label version such as v2. Defaults to the latest labels in each segment.",
    )
    parser.add_argument(
        "--debug-patches",
        type=int,
        default=0,
        help="Print the first N flat-to-native bbox mappings and their source chunk intersections.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=max(1, (mp.cpu_count() or 1) - 1),
        help="Number of worker processes used for chunk copying.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute required chunk coverage and print the totals.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing <dataset>/<dataset>.zarr output before downloading.",
    )
    parser.add_argument(
        "--recompress",
        choices=("fast", "balanced"),
        default=DEFAULT_RECOMPRESS_PRESET,
        help=(
            "Output compression preset. "
            "'fast' = blosc lz4 clevel=5 bitshuffle; "
            "'balanced' = blosc zstd clevel=3 shuffle."
        ),
    )
    parser.add_argument(
        "--patch-filter",
        choices=("positive", "supervision"),
        default=DEFAULT_PATCH_FILTER,
        help=(
            "Which label coverage must be present for a patch to be downloaded. "
            "'positive' requires inklabels; 'supervision' requires supervision-mask coverage."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets_root = Path(args.datasets_root).expanduser().resolve()
    if not datasets_root.is_dir():
        raise FileNotFoundError(f"datasets root does not exist: {datasets_root}")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested_sources = load_dataset_sources(args.volumes_json)
    selected_dataset_names = (
        tuple(str(name) for name in args.datasets)
        if args.datasets
        else tuple(sorted(requested_sources))
    )
    patch_size_zyx = _parse_patch_size(args.patch_size)

    plans = []
    for dataset_name in selected_dataset_names:
        source_spec = requested_sources.get(dataset_name)
        if source_spec is None:
            raise KeyError(f"No source zarr mapping found for dataset {dataset_name!r}.")
        plan = build_dataset_chunk_plan(
            datasets_root=datasets_root,
            output_root=output_root,
            source_spec=source_spec,
            patch_size_zyx=patch_size_zyx,
            overlap_fraction=float(args.overlap_fraction),
            stored_grid_pad=int(args.stored_grid_pad),
            patch_finding_workers=int(args.patch_finding_workers),
            patch_filter=str(args.patch_filter),
            patch_finding_type=str(args.patch_finding_type),
            patch_min_labeled_coverage=float(args.patch_min_labeled_coverage),
            patch_finding_tile_size=args.patch_finding_tile_size,
            patch_finding_stride=args.patch_finding_stride,
            patch_finding_filter_empty_tile=bool(args.patch_finding_filter_empty_tile),
            label_version=None if args.label_version in (None, "") else str(args.label_version),
            debug_patches=int(args.debug_patches),
        )
        plans.append(plan)
        scale_counts = ", ".join(
            f"s{scale}={len(plan.chunk_ids_by_scale[scale])}" for scale in sorted(plan.chunk_ids_by_scale)
        )
        print(
            f"[{plan.dataset_name}] patches={plan.patch_count} candidates={plan.candidate_bbox_count} "
            f"requested_scale={plan.volume_scale} chunk_shape={plan.chunk_shapes_by_scale[plan.volume_scale]} "
            f"unique_chunks={plan.requested_scale_chunk_count} scales=({scale_counts}) "
            f"download={_format_bytes(plan.total_bytes)} output={plan.output_path}"
        )

    if args.dry_run:
        total_bytes = sum(plan.total_bytes for plan in plans)
        print(f"dry run total: {_format_bytes(total_bytes)} across {len(plans)} dataset(s)")
        return 0

    for plan in plans:
        copy_stats = copy_chunks_to_output(
            volume_path=plan.volume_path,
            volume_scale=plan.volume_scale,
            volume_auth_json=plan.volume_auth_json,
            output_path=plan.output_path,
            chunk_ids_by_scale=plan.chunk_ids_by_scale,
            worker_count=max(1, int(args.download_workers)),
            tqdm_desc=f"Downloading {plan.dataset_name}",
            overwrite=bool(args.overwrite),
            recompress=str(args.recompress),
        )
        copied_counts_by_scale = copy_stats["copied_counts_by_scale"]
        existing_counts_by_scale = copy_stats["existing_counts_by_scale"]
        print(
            f"[{plan.dataset_name}] copied "
            + ", ".join(
                f"s{scale}={copied_counts_by_scale.get(scale, 0)}"
                for scale in sorted(plan.chunk_ids_by_scale)
            )
            + " chunks"
            + ", reused "
            + ", ".join(
                f"s{scale}={existing_counts_by_scale.get(scale, 0)}"
                for scale in sorted(plan.chunk_ids_by_scale)
            )
            + f" existing chunks at {plan.output_path} with recompress={args.recompress}"
        )

    total_bytes = sum(plan.total_bytes for plan in plans)
    print(f"completed: {_format_bytes(total_bytes)} across {len(plans)} dataset(s)")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
