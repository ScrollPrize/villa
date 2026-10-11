import asyncio
import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import aiohttp
import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from fsspec.asyn import sync
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from dinovol_2.augmentation.pipelines import create_training_transforms
from dinovol_2.dataset.normalization import get_normalization
from dinovol_2.dataset.point_annotations import (
    load_point_collection,
    map_scale0_voxel_centers,
    xyz_to_zyx,
)


# Dropped connections, truncated bodies, timeouts and throttling from remote stores.
# Missing chunks surface as FileNotFoundError inside zarr and never reach this.
TRANSIENT_READ_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)


def read_with_retries(read, retries, backoff=1.0):
    """Call read(), retrying transient network errors with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return read()
        except TRANSIENT_READ_ERRORS as exc:
            if attempt == retries:
                raise
            delay = backoff * 2 ** attempt
            print(
                f"ssl_zarr_dataset: transient read error ({type(exc).__name__}: {exc}); "
                f"retry {attempt + 1}/{retries} in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


@dataclass
class Volume:
    usable_bbox: tuple  # (z0, y0, x0, z1, y1, x1)
    valid_crop_starts: int
    scale: int
    path: str
    weight: float = 0.0
    point_coordinates: np.ndarray | None = None
    point_type_ids: np.ndarray | None = None
    # Dilated nonzero mask of a coarse pyramid level over the usable bbox; cell
    # occupancy_origin + (i, j, k) covers occupancy_factor**3 selected-scale voxels.
    occupancy: np.ndarray | None = None
    occupancy_origin: tuple[int, int, int] | None = None
    occupancy_factor: int = 1


@dataclass(frozen=True)
class SourcePlan:
    volume: Volume
    array: Any
    starts: tuple[int, int, int]
    anchor_index: int | None


@dataclass
class ZarrHandle:
    array: Any
    fs: Any | None = None

    def close(self) -> None:
        if self.fs is None:
            return
        session = getattr(self.fs, "_session", None)
        if session is None:
            return
        loop = getattr(self.fs, "loop", None)
        if loop is not None and not loop.is_closed():
            try:
                sync(loop, session.close, timeout=1.0)
            except Exception:
                close_session = getattr(type(self.fs), "close_session", None)
                if callable(close_session):
                    close_session(loop, session)
        else:
            close_session = getattr(type(self.fs), "close_session", None)
            if callable(close_session):
                close_session(loop, session)
        connector = getattr(session, "_connector", None)
        if connector is not None and not connector.closed:
            connector._close()
        try:
            session._connector = None
        except AttributeError:
            pass
        try:
            self.fs._session = None
        except AttributeError:
            pass


@dataclass(frozen=True)
class SampledCropRegion:
    starts: tuple[int, int, int]
    shape: tuple[int, int, int]


def _as_3tuple(value):
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"Expected 3 values, got {result}")
    return result


def _as_float_pair(value, default):
    if value is None:
        return default
    return float(value[0]), float(value[1])


def _max_3tuple(*values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return tuple(max(int(value[axis]) for value in filtered) for axis in range(3))


def load_volume_auth(auth_json_path):
    if auth_json_path is None:
        return None, None

    with open(str(auth_json_path), "r", encoding="utf-8") as f:
        auth = json.load(f)
    return str(auth["username"]), str(auth["password"])


def open_zarr_handle(path, resolution, auth=None, s3_storage_options=None):
    path_str = str(path)
    user, password = load_volume_auth(auth)
    if path_str.startswith("s3://"):
        storage_options = {"anon": True}
        if s3_storage_options is not None:
            storage_options.update(dict(s3_storage_options))
        try:
            return ZarrHandle(
                array=zarr.open(
                    path_str,
                    path=str(resolution),
                    mode="r",
                    storage_options=storage_options,
                )
            )
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Opening s3:// zarr volumes requires the optional dependency `s3fs`."
            ) from exc
    use_https_auth = path_str.startswith("https://") and bool(user) and bool(password)
    if use_https_auth:
        fs = fsspec.filesystem(
            "https",
            asynchronous=True,
            client_kwargs={"auth": aiohttp.BasicAuth(user, password)},
        )
        if hasattr(zarr.storage, "FsspecStore"):
            store = zarr.storage.FsspecStore(
                fs=fs,
                path=path_str.rstrip("/"),
                read_only=True,
                allowed_exceptions=(
                    KeyError,
                    FileNotFoundError,
                    PermissionError,
                    OSError,
                    aiohttp.ClientResponseError,
                ),
            )
        else:
            store = zarr.storage.FSStore(
                path_str.rstrip("/"),
                fs=fs,
                mode="r",
                check=False,
                create=False,
                exceptions=(KeyError, FileNotFoundError, PermissionError, OSError, aiohttp.ClientResponseError),
            )
        return ZarrHandle(array=zarr.open(store, path=str(resolution), mode="r"), fs=fs)
    return ZarrHandle(array=zarr.open(path_str, path=str(resolution), mode="r"))


def open_zarr(path, resolution, auth=None, s3_storage_options=None):
    return open_zarr_handle(path, resolution, auth=auth, s3_storage_options=s3_storage_options).array


class SSLZarrDataset(Dataset):
    def __init__(self, config, do_augmentations=False, single_crop_only=False):
        self.config = config
        for key in ("global_view_size", "local_view_size", "gram_teacher_view_size"):
            if key in config:
                raise ValueError(f"{key} is no longer supported; configure the corresponding crop_size instead.")
        self.do_augmentations = do_augmentations
        self.single_crop_only = bool(self.config.get("single_crop_only", False) or single_crop_only)
        self.epoch_length = int(self.config["epoch_length"]) if "epoch_length" in self.config else None
        self.global_crop_size = _as_3tuple(
            self.config["global_crop_size"] if "global_crop_size" in self.config else self.config["crop_size"]
        )
        self.num_global_crops = self.config.get("num_global_crops", 2)
        self.local_crop_size = _as_3tuple(self.config["local_crop_size"] if "local_crop_size" in self.config else None)
        self.gram_teacher_crop_size = _as_3tuple(self.config.get("gram_teacher_crop_size"))
        if self.gram_teacher_crop_size is not None and self.single_crop_only:
            raise ValueError("gram_teacher_crop_size is not supported with single_crop_only datasets.")
        self.gram_teacher_no_augmentations = bool(self.config.get("gram_teacher_no_augmentations", True))
        self.paired_global_crop_size = _max_3tuple(self.global_crop_size, self.gram_teacher_crop_size)
        double_global = tuple(2 * int(d) for d in self.paired_global_crop_size)
        default_source_sampling_size = _max_3tuple(double_global, self.global_crop_size, self.local_crop_size)
        self.source_sampling_size = _as_3tuple(
            self.config.get("source_sampling_size", self.config.get("source_crop_size", default_source_sampling_size))
        )
        self.global_crop_scale = _as_float_pair(self.config.get("global_crop_scale"), (0.32, 1.0))
        self.local_crop_scale = _as_float_pair(self.config.get("local_crop_scale"), (0.05, 0.32))
        self.num_local_crops = self.config.get("num_local_crops", 8)
        self.volume_auth = self.config["volume_auth"] if "volume_auth" in self.config else None
        self.s3_storage_options = self.config.get("s3_storage_options")
        self.vol_trim_pct = self.config.get("vol_trim_pct", 0.60)
        self.normalizer = get_normalization(self.config.get("normalization_scheme", "robust"))
        point_config = dict(self.config.get("point_supervision") or {})
        self.point_supervision_enabled = bool(point_config.get("enabled", False)) and not self.single_crop_only
        self.point_sampling_probability = float(point_config.get("sampling_probability", 0.1))
        self.max_points_per_view = int(point_config.get("max_points_per_view", 64))
        if not 0.0 <= self.point_sampling_probability <= 1.0:
            raise ValueError(
                f"point sampling_probability must be in [0, 1], got {self.point_sampling_probability}."
            )
        if self.max_points_per_view <= 0:
            raise ValueError(f"max_points_per_view must be positive, got {self.max_points_per_view}.")
        configured_types = []
        if self.point_supervision_enabled:
            discovered_types = set()
            for dataset in self.config["datasets"]:
                entries = dataset.get("point_collections", [])
                if not isinstance(entries, list):
                    raise ValueError("point_collections must be a list of {path, type} objects.")
                for entry_index, collection in enumerate(entries):
                    if not isinstance(collection, dict) or "path" not in collection or "type" not in collection:
                        raise ValueError(
                            f"point_collections entry {entry_index} must contain path and type."
                        )
                    if not isinstance(collection["type"], str) or not collection["type"]:
                        raise ValueError("Point collection types must be non-empty strings.")
                    discovered_types.add(collection["type"])
            configured_types = sorted(discovered_types)
        if any(not type_name for type_name in configured_types):
            raise ValueError("Point collection types must be non-empty strings.")
        self.point_type_to_id = {type_name: index for index, type_name in enumerate(configured_types)}
        # occupancy_level: pyramid level whose nonzero voxels mark where source crops can
        # reach nonzero_threshold, so crops below it are rejected before any read.
        self.occupancy_level = self.config.get("occupancy_level")
        # read_ahead: source crops each process reads in background threads while it
        # augments the current sample. Their locations are drawn in the calling thread.
        self.read_ahead = int(self.config.get("read_ahead", 0))
        if self.read_ahead < 0:
            raise ValueError(f"read_ahead must be non-negative, got {self.read_ahead}")
        # read_retries: transient network errors are retried this many times with
        # backoff doubling from read_retry_backoff seconds; the same window is re-read.
        self.read_retries = int(self.config.get("read_retries", 5))
        self.read_retry_backoff = float(self.config.get("read_retry_backoff", 1.0))
        if self.read_retries < 0:
            raise ValueError(f"read_retries must be non-negative, got {self.read_retries}")
        self._volume_handles: dict[tuple[str, int], ZarrHandle] = {}
        self._handle_pid: int | None = None
        self._atexit_pid: int | None = None
        self._read_executor: ThreadPoolExecutor | None = None
        self._pending_reads: deque = deque()

        if not self.single_crop_only:
            if self.num_global_crops != 2:
                raise ValueError(
                    f"SSLZarrDataset currently expects exactly 2 global crops, got {self.num_global_crops}")
            if self.local_crop_size is not None and self.num_local_crops < 0:
                raise ValueError(
                    f"SSLZarrDataset expects a non-negative number of local crops, got {self.num_local_crops}")
        required_source_sampling_size = _max_3tuple(
            self.global_crop_size,
            self.local_crop_size,
            self.gram_teacher_crop_size,
        )
        if required_source_sampling_size is not None and any(
            source < required for source, required in zip(self.source_sampling_size, required_source_sampling_size)
        ):
            raise ValueError(
                "source_sampling_size must be at least as large as the largest requested crop size, "
                f"got source_sampling_size={self.source_sampling_size} and required={required_source_sampling_size}"
            )
        # gaussian_blur: false removes blur, which erases one-voxel structures.
        gaussian_blur = bool(self.config.get("gaussian_blur", True))
        self.global_transforms = [create_training_transforms(self.global_crop_size, gaussian_blur=gaussian_blur)
                                  for _ in range(self.num_global_crops)]
        self.local_transforms = (
            [create_training_transforms(self.local_crop_size, gaussian_blur=gaussian_blur)
             for _ in range(self.num_local_crops)]
            if self.local_crop_size is not None
            else []
        )

        self.volumes = []

        for dataset in self.config["datasets"]:
            volume_path = dataset["volume_path"]
            volume_scale = dataset["volume_scale"]
            handle = open_zarr_handle(volume_path, volume_scale, self.volume_auth, self.s3_storage_options)
            try:
                d_zarr = handle.array
                z, y, x = d_zarr.shape
                k_z = max(1, round(z * self.vol_trim_pct))
                k_y = max(1, round(y * self.vol_trim_pct))
                k_x = max(1, round(x * self.vol_trim_pct))
                z0 = (z - k_z) // 2
                y0 = (y - k_y) // 2
                x0 = (x - k_x) // 2
                z1 = (z0 + k_z)
                y1 = (y0 + k_y)
                x1 = (x0 + k_x)
                usable_bbox = (z0, y0, x0, z1, y1, x1)
                crop_z, crop_y, crop_x = self.source_sampling_size
                valid_z = max(0, (z1 - z0) - crop_z + 1)
                valid_y = max(0, (y1 - y0) - crop_y + 1)
                valid_x = max(0, (x1 - x0) - crop_x + 1)
                valid_crop_starts = valid_z * valid_y * valid_x

                point_coordinates, point_type_ids = self._load_volume_points(
                    dataset,
                    selected_shape=(z, y, x),
                    usable_bbox=usable_bbox,
                )
                self.volumes.append(Volume(
                    path=str(volume_path),
                    scale=volume_scale,
                    usable_bbox=usable_bbox,
                    valid_crop_starts=valid_crop_starts,
                    point_coordinates=point_coordinates,
                    point_type_ids=point_type_ids,
                ))
            finally:
                handle.close()

        if self.occupancy_level is not None:
            with ThreadPoolExecutor(max_workers=min(8, len(self.volumes))) as executor:
                list(executor.map(self._load_occupancy, self.volumes))

        self.total_valid_crop_starts = sum(vol.valid_crop_starts for vol in self.volumes)
        if self.total_valid_crop_starts <= 0:
            raise ValueError("No valid crop starts found across configured volumes for the selected source crop size.")

        for volume in self.volumes:
            volume.weight = volume.valid_crop_starts / self.total_valid_crop_starts

    def _load_volume_points(self, dataset_config, *, selected_shape, usable_bbox):
        entries = dataset_config.get("point_collections", [])
        if not self.point_supervision_enabled or not entries:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int64)
        if not isinstance(entries, list):
            raise ValueError("point_collections must be a list of {path, type} objects.")

        volume_path = dataset_config["volume_path"]
        volume_scale = int(dataset_config["volume_scale"])
        scale0_handle = None
        try:
            if volume_scale == 0:
                scale0_shape = selected_shape
            else:
                scale0_handle = open_zarr_handle(
                    volume_path,
                    0,
                    self.volume_auth,
                    self.s3_storage_options,
                )
                scale0_shape = tuple(int(value) for value in scale0_handle.array.shape)

            all_coordinates = []
            all_type_ids = []
            usable_start = np.asarray(usable_bbox[:3], dtype=np.float64)
            usable_stop = np.asarray(usable_bbox[3:], dtype=np.float64)
            for entry_index, entry in enumerate(entries):
                if not isinstance(entry, dict) or "path" not in entry or "type" not in entry:
                    raise ValueError(
                        f"point_collections entry {entry_index} for {volume_path} must contain path and type."
                    )
                type_name = entry["type"]
                if not isinstance(type_name, str) or not type_name:
                    raise ValueError("Point collection types must be non-empty strings.")
                points = map_scale0_voxel_centers(
                    xyz_to_zyx(load_point_collection(entry["path"])),
                    scale0_shape,
                    selected_shape,
                )
                usable = np.all((points >= usable_start) & (points < usable_stop), axis=1)
                points = points[usable]
                if points.shape[0] == 0:
                    raise ValueError(
                        f"Point collection {entry['path']} has no annotations usable in the trimmed bounds "
                        f"of volume {volume_path} at scale {volume_scale}."
                    )
                all_coordinates.append(points)
                all_type_ids.append(
                    np.full((points.shape[0],), self.point_type_to_id[type_name], dtype=np.int64)
                )
            return np.concatenate(all_coordinates), np.concatenate(all_type_ids)
        finally:
            if scale0_handle is not None:
                scale0_handle.close()

    def _load_occupancy(self, volume: Volume) -> None:
        level = int(self.occupancy_level)
        factor = 2 ** (level - int(volume.scale))
        if factor < 1:
            raise ValueError(f"occupancy_level {level} must not be finer than volume_scale {volume.scale}")
        bbox_lo = np.asarray(volume.usable_bbox[:3]) // factor
        bbox_hi = -(-np.asarray(volume.usable_bbox[3:]) // factor)
        handle = open_zarr_handle(volume.path, level, self.volume_auth, self.s3_storage_options)
        try:
            array = handle.array
            # Read one extra cell on each side for the dilation below.
            read_lo = np.maximum(bbox_lo - 1, 0)
            read_hi = np.minimum(bbox_hi + 1, array.shape)
            coarse = np.zeros(tuple(bbox_hi - bbox_lo + 2), dtype=bool)
            selection = tuple(slice(int(a), int(b)) for a, b in zip(read_lo, read_hi))
            values = read_with_retries(
                lambda: np.asarray(array[selection]), self.read_retries, self.read_retry_backoff
            )
            offset = read_lo - (bbox_lo - 1)
            coarse[tuple(slice(int(o), int(o) + n) for o, n in zip(offset, values.shape))] = values > 0
        finally:
            handle.close()
        # Downsampling can round sparse cells to zero; one cell of dilation keeps every
        # nonzero selected-scale voxel inside an occupied cell.
        coarse = binary_dilation(coarse, structure=np.ones((3, 3, 3), dtype=bool))
        volume.occupancy = np.ascontiguousarray(coarse[1:-1, 1:-1, 1:-1])
        volume.occupancy_origin = tuple(int(v) for v in bbox_lo)
        volume.occupancy_factor = factor

    def _occupied_fraction(self, volume: Volume, starts) -> float:
        """Fraction of the source window inside occupied cells, an upper bound on its nonzero fraction."""
        factor = volume.occupancy_factor
        slices, weights = [], []
        for start, size, origin in zip(starts, self.source_sampling_size, volume.occupancy_origin):
            stop = int(start) + int(size)
            first, last = int(start) // factor, (stop - 1) // factor
            edges = np.arange(first, last + 2) * factor
            weights.append((np.minimum(edges[1:], stop) - np.maximum(edges[:-1], int(start))).astype(np.float64))
            slices.append(slice(first - origin, last + 1 - origin))
        block = volume.occupancy[tuple(slices)].astype(np.float64)
        return float(np.einsum("ijk,i,j,k->", block, *weights)) / math.prod(self.source_sampling_size)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_volume_handles"] = {}
        state["_handle_pid"] = None
        state["_atexit_pid"] = None
        state["_read_executor"] = None
        state["_pending_reads"] = deque()
        return state

    def _register_close_hook(self) -> None:
        pid = os.getpid()
        if self._atexit_pid == pid:
            return
        atexit.register(self.close)
        self._atexit_pid = pid

    def _ensure_process_local_handles(self) -> None:
        pid = os.getpid()
        if self._handle_pid == pid:
            return
        self.close()
        self._read_executor = None
        self._pending_reads = deque()
        self._volume_handles = {}
        self._handle_pid = pid
        self._register_close_hook()

    def _get_volume_array(self, volume: Volume):
        self._ensure_process_local_handles()
        key = (volume.path, volume.scale)
        handle = self._volume_handles.get(key)
        if handle is None:
            handle = open_zarr_handle(volume.path, volume.scale, self.volume_auth, self.s3_storage_options)
            self._volume_handles[key] = handle
        return handle.array

    def close(self) -> None:
        executor = getattr(self, "_read_executor", None)
        if executor is not None and self._handle_pid == os.getpid():
            executor.shutdown(wait=True, cancel_futures=True)
            self._read_executor = None
            self._pending_reads.clear()
        for handle in self._volume_handles.values():
            handle.close()
        self._volume_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _sample_crop_shape(self, scale_range):
        ref_depth, ref_height, ref_width = self.source_sampling_size
        min_scale, max_scale = scale_range
        scale = np.random.uniform(min_scale, max_scale)
        scale_per_dim = scale ** (1.0 / 3.0)
        return (
            max(1, int(round(ref_depth * scale_per_dim))),
            max(1, int(round(ref_height * scale_per_dim))),
            max(1, int(round(ref_width * scale_per_dim))),
        )

    def _finalize_crop(self, crop, target_size):
        crop = np.asarray(crop, dtype=np.float32)
        if self.normalizer is not None:
            crop = self.normalizer.run(crop)
        crop = torch.from_numpy(crop).unsqueeze(0)
        if crop.shape[1:] == target_size:
            return crop.clone()
        resized = F.interpolate(
            crop.unsqueeze(0),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _sample_source_start(self, usable_bbox):
        z0, y0, x0, z1, y1, x1 = usable_bbox
        crop_d, crop_h, crop_w = self.source_sampling_size
        z_start = np.random.randint(z0, z1 - crop_d + 1)
        y_start = np.random.randint(y0, y1 - crop_h + 1)
        x_start = np.random.randint(x0, x1 - crop_w + 1)
        return z_start, y_start, x_start

    def _read_source_window(self, d_zarr, starts):
        z_start, y_start, x_start = starts
        crop_d, crop_h, crop_w = self.source_sampling_size
        selection = (
            slice(z_start, z_start + crop_d),
            slice(y_start, y_start + crop_h),
            slice(x_start, x_start + crop_w),
        )
        return read_with_retries(lambda: np.asarray(d_zarr[selection]), self.read_retries, self.read_retry_backoff)

    def _read_source_crop_3d(self, d_zarr, usable_bbox):
        return self._read_source_window(d_zarr, self._sample_source_start(usable_bbox))

    def _read_point_centered_source_crop_3d(self, d_zarr, usable_bbox, anchor):
        starts = self._sample_point_centered_source_start(usable_bbox, anchor)
        return self._read_source_window(d_zarr, starts), starts

    def _sample_point_centered_source_start(self, usable_bbox, anchor):
        starts = []
        for bbox_start, bbox_stop, crop_size, coordinate in zip(
            usable_bbox[:3],
            usable_bbox[3:],
            self.source_sampling_size,
            anchor,
        ):
            minimum = max(
                int(bbox_start),
                int(np.ceil(float(coordinate) - (int(crop_size) - 1))),
            )
            maximum = min(
                int(bbox_stop) - int(crop_size),
                int(np.floor(float(coordinate))),
            )
            if maximum < minimum:
                raise ValueError(
                    f"Annotated point {tuple(float(value) for value in anchor)} cannot fit in a source window "
                    f"of size {self.source_sampling_size} inside bounds {usable_bbox}."
                )
            starts.append(int(np.random.randint(minimum, maximum + 1)))
        return tuple(starts)

    def _sample_random_resized_crop_region(
        self,
        source_shape,
        scale_range,
        target_size,
        *,
        anchor=None,
        anchor_target_size=None,
    ):
        crop_shape = self._sample_crop_shape(scale_range)
        starts = []
        for axis, (source_dim, crop_dim) in enumerate(zip(source_shape, crop_shape)):
            low = 0
            high_inclusive = int(source_dim) - int(crop_dim)
            if anchor is not None:
                if anchor_target_size is None:
                    raise ValueError("anchor_target_size is required with an anchor.")
                # Keep the point inside the resized crop's voxel-center bounds.
                target_dim = float(anchor_target_size[axis])
                relative_min = 0.5 * int(crop_dim) / target_dim - 0.5
                relative_max = (target_dim - 0.5) * int(crop_dim) / target_dim - 0.5
                low = max(low, int(np.ceil(float(anchor[axis]) - relative_max - 1e-7)))
                high_inclusive = min(
                    high_inclusive, int(np.floor(float(anchor[axis]) - relative_min + 1e-7))
                )
            if high_inclusive < low:
                raise ValueError(
                    f"Cannot sample crop {crop_shape} inside source {source_shape} with anchor={anchor}"
                )
            starts.append(int(np.random.randint(low, high_inclusive + 1)))
        return SampledCropRegion(starts=tuple(starts), shape=tuple(int(size) for size in crop_shape))

    def _materialize_crop_from_region(self, source_crop, crop_region, target_size):
        z_start, y_start, x_start = crop_region.starts
        crop_d, crop_h, crop_w = crop_region.shape
        crop = source_crop[
            z_start:z_start + crop_d,
            y_start:y_start + crop_h,
            x_start:x_start + crop_w,
        ]
        return self._finalize_crop(crop, target_size)

    def _random_resized_crop_3d_from_array(self, source_crop, scale_range, target_size):
        crop_region = self._sample_random_resized_crop_region(
            source_crop.shape,
            scale_range,
            target_size,
        )
        return self._materialize_crop_from_region(source_crop, crop_region, target_size)

    def _read_random_resized_crop_3d(self, d_zarr, usable_bbox, scale_range, target_size):
        source_crop = self._read_source_crop_3d(d_zarr, usable_bbox)
        return self._random_resized_crop_3d_from_array(
            source_crop,
            scale_range,
            target_size,
        )

    @staticmethod
    def _capture_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _apply_paired_global_transform(self, transform, global_view, gram_teacher_view):
        if gram_teacher_view is None or self.gram_teacher_no_augmentations:
            transformed = transform(image=global_view, paired_spatial_image=gram_teacher_view)
            return transformed["image"], transformed["paired_spatial_image"]

        state_before = self._capture_rng_state()
        augmented_global = transform(image=global_view)["image"]
        state_after = self._capture_rng_state()
        self._restore_rng_state(state_before)
        augmented_gram_teacher = transform(image=gram_teacher_view)["image"]
        self._restore_rng_state(state_after)
        return augmented_global, augmented_gram_teacher

    def _apply_paired_global_transform_with_points(
        self,
        transform,
        global_view,
        gram_teacher_view,
        point_coordinates,
    ):
        transform_input = {
            "image": global_view,
            "keypoints": point_coordinates,
            "crop_shape": tuple(int(value) for value in self.global_crop_size),
        }
        if gram_teacher_view is None or self.gram_teacher_no_augmentations:
            transform_input["paired_spatial_image"] = gram_teacher_view
            transformed = transform(**transform_input)
            return transformed["image"], transformed["paired_spatial_image"], transformed["keypoints"]

        state_before = self._capture_rng_state()
        transformed = transform(**transform_input)
        state_after = self._capture_rng_state()
        self._restore_rng_state(state_before)
        augmented_gram_teacher = transform(image=gram_teacher_view)["image"]
        self._restore_rng_state(state_after)
        return transformed["image"], augmented_gram_teacher, transformed["keypoints"]

    @staticmethod
    def _map_points_to_view(points, crop_region, target_size):
        if points.shape[0] == 0:
            return torch.empty((0, 3), dtype=torch.float32)
        starts = np.asarray(crop_region.starts, dtype=np.float64)
        shape = np.asarray(crop_region.shape, dtype=np.float64)
        target = np.asarray(target_size, dtype=np.float64)
        mapped = (points - starts + 0.5) * (target / shape) - 0.5
        return torch.from_numpy(mapped.astype(np.float32, copy=False))

    def _visible_points_for_crop(self, volume, source_starts, crop_region, anchor_index):
        source_starts_array = np.asarray(source_starts, dtype=np.float64)
        source_points = volume.point_coordinates - source_starts_array
        crop_starts = np.asarray(crop_region.starts, dtype=np.float64)
        crop_stops = crop_starts + np.asarray(crop_region.shape, dtype=np.float64) - 1.0
        visible = np.all((source_points >= crop_starts) & (source_points <= crop_stops), axis=1)
        visible_indices = np.flatnonzero(visible)
        if anchor_index not in visible_indices:
            raise RuntimeError("The selected point anchor was not retained in its global crop.")
        ordered_indices = np.concatenate((
            np.asarray([anchor_index], dtype=np.int64),
            visible_indices[visible_indices != anchor_index],
        ))
        return (
            self._map_points_to_view(source_points[ordered_indices], crop_region, self.global_crop_size),
            torch.from_numpy(volume.point_type_ids[ordered_indices].copy()).long(),
        )

    def _filter_and_cap_view_points(self, coordinates, type_ids):
        maximum = coordinates.new_tensor(tuple(float(value - 1) for value in self.global_crop_size))
        visible = torch.all((coordinates >= 0.0) & (coordinates <= maximum), dim=1)
        if coordinates.shape[0] and not bool(visible[0]):
            raise RuntimeError("The selected point anchor fell outside the student output crop.")
        coordinates = coordinates[visible]
        type_ids = type_ids[visible]
        if coordinates.shape[0] <= self.max_points_per_view:
            return coordinates, type_ids

        selected = [0]
        remaining_budget = self.max_points_per_view - 1
        candidates_by_type = {}
        for type_id in torch.unique(type_ids, sorted=True).tolist():
            candidates = torch.nonzero(type_ids == type_id, as_tuple=False).flatten().tolist()
            candidates = [index for index in candidates if index != 0]
            random.shuffle(candidates)
            candidates_by_type[int(type_id)] = candidates
        type_order = list(candidates_by_type)
        while remaining_budget > 0 and type_order:
            next_order = []
            for type_id in type_order:
                candidates = candidates_by_type[type_id]
                if candidates and remaining_budget > 0:
                    selected.append(candidates.pop())
                    remaining_budget -= 1
                if candidates:
                    next_order.append(type_id)
            type_order = next_order
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        return coordinates[selected_tensor], type_ids[selected_tensor]

    def __len__(self):
        if self.epoch_length is not None:
            return self.epoch_length
        return self.total_valid_crop_starts

    @staticmethod
    def _sample_volume_anchor(volume):
        represented_types = np.unique(volume.point_type_ids)
        selected_type = int(np.random.choice(represented_types))
        candidates = np.flatnonzero(volume.point_type_ids == selected_type)
        return int(np.random.choice(candidates))

    def _draw_source_plan(self, nonzero_threshold):
        vol_weights = [vol.weight for vol in self.volumes]
        while True:
            vol_idx = np.random.choice(len(self.volumes), p=vol_weights)
            vol = self.volumes[vol_idx]
            anchor_index = None
            if (
                self.point_supervision_enabled
                and vol.point_coordinates is not None
                and vol.point_coordinates.shape[0] > 0
                and np.random.random() < self.point_sampling_probability
            ):
                anchor_index = self._sample_volume_anchor(vol)
                starts = self._sample_point_centered_source_start(
                    vol.usable_bbox,
                    vol.point_coordinates[anchor_index],
                )
            else:
                starts = self._sample_source_start(vol.usable_bbox)
            # The occupied fraction bounds the nonzero fraction from above, so this
            # skips only windows the check after reading would reject.
            if vol.occupancy is not None and self._occupied_fraction(vol, starts) < nonzero_threshold:
                continue
            return SourcePlan(vol, self._get_volume_array(vol), tuple(starts), anchor_index)

    def _next_source_crop(self, nonzero_threshold):
        while True:
            if self._pending_reads:
                plan, future = self._pending_reads.popleft()
                source_crop = future.result()
            else:
                plan = self._draw_source_plan(nonzero_threshold)
                source_crop = self._read_source_window(plan.array, plan.starts)
            if source_crop.size > 0 and (np.count_nonzero(source_crop) / source_crop.size) >= nonzero_threshold:
                break
        if self.read_ahead:
            if self._read_executor is None:
                self._read_executor = ThreadPoolExecutor(max_workers=self.read_ahead, thread_name_prefix="source-read")
            while len(self._pending_reads) < self.read_ahead:
                next_plan = self._draw_source_plan(nonzero_threshold)
                self._pending_reads.append(
                    (next_plan, self._read_executor.submit(self._read_source_window, next_plan.array, next_plan.starts))
                )
        return plan, source_crop

    def __getitem__(self, idx):
        nonzero_threshold = float(self.config.get("nonzero_threshold", 0.30))
        plan, source_crop = self._next_source_crop(nonzero_threshold)
        vol = plan.volume
        anchor_index = plan.anchor_index
        source_starts = plan.starts if anchor_index is not None else None

        if self.single_crop_only:
            crop = self._random_resized_crop_3d_from_array(
                source_crop,
                self.global_crop_scale,
                self.global_crop_size,
            )
            if self.do_augmentations:
                crop = self.global_transforms[0](image=crop)["image"]
            return crop

        global_views = []
        gram_teacher_views = []
        global_point_coordinates = []
        global_point_type_ids = []
        for transform in self.global_transforms:
            # Sample one shared region so the student and Gram teacher observe the
            # exact same 3D field of view, optionally at different resolutions.
            crop_region = self._sample_random_resized_crop_region(
                source_crop.shape,
                self.global_crop_scale,
                self.paired_global_crop_size,
                anchor=(
                    vol.point_coordinates[anchor_index] - np.asarray(source_starts, dtype=np.float64)
                    if anchor_index is not None
                    else None
                ),
                anchor_target_size=self.global_crop_size if anchor_index is not None else None,
            )
            global_view = self._materialize_crop_from_region(source_crop, crop_region, self.global_crop_size)
            gram_teacher_view = None
            if self.gram_teacher_crop_size is not None:
                gram_teacher_view = self._materialize_crop_from_region(
                    source_crop,
                    crop_region,
                    self.gram_teacher_crop_size,
                )
            if anchor_index is not None:
                point_coordinates, point_type_ids = self._visible_points_for_crop(
                    vol,
                    source_starts,
                    crop_region,
                    anchor_index,
                )
                if self.do_augmentations:
                    global_view, gram_teacher_view, point_coordinates = (
                        self._apply_paired_global_transform_with_points(
                            transform,
                            global_view,
                            gram_teacher_view,
                            point_coordinates,
                        )
                    )
                point_coordinates, point_type_ids = self._filter_and_cap_view_points(
                    point_coordinates,
                    point_type_ids,
                )
            else:
                point_coordinates = torch.empty((0, 3), dtype=torch.float32)
                point_type_ids = torch.empty((0,), dtype=torch.long)
                if self.do_augmentations:
                    global_view, gram_teacher_view = self._apply_paired_global_transform(
                        transform,
                        global_view,
                        gram_teacher_view,
                    )
            global_views.append(global_view)
            global_point_coordinates.append(point_coordinates)
            global_point_type_ids.append(point_type_ids)
            if gram_teacher_view is not None:
                gram_teacher_views.append(gram_teacher_view)

        local_views = []
        if self.local_crop_size is not None:
            local_views = [
                self._random_resized_crop_3d_from_array(
                    source_crop,
                    self.local_crop_scale,
                    self.local_crop_size,
                )
                for _ in range(self.num_local_crops)
            ]
            if self.do_augmentations:
                local_views = [
                    transform(image=view)["image"]
                    for transform, view in zip(self.local_transforms, local_views)
                ]

        sample = {
            "global_views": global_views,
            "local_views": local_views,
            "global_point_coordinates": global_point_coordinates,
            "global_point_type_ids": global_point_type_ids,
        }
        if gram_teacher_views:
            sample["gram_teacher_views"] = gram_teacher_views
        return sample
