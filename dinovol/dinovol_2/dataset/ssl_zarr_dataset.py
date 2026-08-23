import atexit
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from fsspec.asyn import sync
from torch.utils.data import Dataset

from dinovol_2.augmentation.pipelines import create_training_transforms
from dinovol_2.dataset.normalization import get_normalization
from dinovol_2.dataset.point_annotations import (
    load_point_collection,
    map_scale0_voxel_centers,
    xyz_to_zyx,
)


@dataclass
class Volume:
    usable_bbox: tuple  # (z0, y0, x0, z1, y1, x1)
    valid_crop_starts: int
    scale: int
    path: str
    weight: float = 0.0
    point_coordinates: np.ndarray | None = None
    point_type_ids: np.ndarray | None = None


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
        self.do_augmentations = do_augmentations
        self.single_crop_only = bool(self.config.get("single_crop_only", False) or single_crop_only)
        self.epoch_length = int(self.config["epoch_length"]) if "epoch_length" in self.config else None
        self.global_crop_size = _as_3tuple(
            self.config["global_crop_size"] if "global_crop_size" in self.config else self.config["crop_size"]
        )
        self.num_global_crops = self.config.get("num_global_crops", 2)
        self.local_crop_size = _as_3tuple(self.config["local_crop_size"] if "local_crop_size" in self.config else None)
        self.global_view_size = _as_3tuple(self.config.get("global_view_size", self.global_crop_size))
        self.local_view_size = _as_3tuple(self.config.get("local_view_size", self.local_crop_size))
        self.gram_teacher_crop_size = _as_3tuple(self.config.get("gram_teacher_crop_size"))
        if self.gram_teacher_crop_size is not None and self.single_crop_only:
            raise ValueError("gram_teacher_crop_size is not supported with single_crop_only datasets.")
        self.gram_teacher_view_size = _as_3tuple(
            self.config.get("gram_teacher_view_size", self.gram_teacher_crop_size)
        )
        self.gram_teacher_no_augmentations = bool(self.config.get("gram_teacher_no_augmentations", True))
        self.paired_global_crop_size = _max_3tuple(self.global_crop_size, self.gram_teacher_crop_size)
        self.paired_global_view_size = _max_3tuple(self.global_view_size, self.gram_teacher_view_size)
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
        self.global_point_crop_offset = tuple(
            (float(view) - float(crop)) / 2.0
            for view, crop in zip(self.global_view_size, self.global_crop_size)
        )
        self._volume_handles: dict[tuple[str, int], ZarrHandle] = {}
        self._handle_pid: int | None = None
        self._atexit_pid: int | None = None

        if not self.single_crop_only:
            if self.num_global_crops != 2:
                raise ValueError(
                    f"SSLZarrDataset currently expects exactly 2 global crops, got {self.num_global_crops}")
            if self.local_crop_size is not None and self.num_local_crops < 0:
                raise ValueError(
                    f"SSLZarrDataset expects a non-negative number of local crops, got {self.num_local_crops}")
        if any(view < crop for view, crop in zip(self.global_view_size, self.global_crop_size)):
            raise ValueError(
                f"global_view_size must be at least global_crop_size, got {self.global_view_size} < {self.global_crop_size}"
            )
        if self.local_crop_size is not None and self.local_view_size is not None:
            if any(view < crop for view, crop in zip(self.local_view_size, self.local_crop_size)):
                raise ValueError(
                    f"local_view_size must be at least local_crop_size, got {self.local_view_size} < {self.local_crop_size}"
                )
        if self.gram_teacher_crop_size is not None and self.gram_teacher_view_size is not None:
            if any(view < crop for view, crop in zip(self.gram_teacher_view_size, self.gram_teacher_crop_size)):
                raise ValueError(
                    "gram_teacher_view_size must be at least gram_teacher_crop_size, "
                    f"got {self.gram_teacher_view_size} < {self.gram_teacher_crop_size}"
                )
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
        self.source_read_window_size = self._required_read_window_size(self.source_sampling_size)
        required_read_window_size = _max_3tuple(self.paired_global_view_size, self.local_view_size)
        if required_read_window_size is not None and any(
            source < required for source, required in zip(self.source_read_window_size, required_read_window_size)
        ):
            raise ValueError(
                "computed source_read_window_size must be at least as large as the largest requested view size, "
                f"got source_read_window_size={self.source_read_window_size} and required={required_read_window_size}"
            )

        self.global_transforms = [create_training_transforms(self.global_view_size) for _ in
                                  range(self.num_global_crops)]
        self.local_transforms = (
            [create_training_transforms(self.local_view_size) for _ in range(self.num_local_crops)]
            if self.local_view_size is not None
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
                crop_z, crop_y, crop_x = self.source_read_window_size
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
            nominal_slices = self._nominal_source_slices(self.source_read_window_size)
            usable_start = np.asarray([
                int(bbox_start) + int(nominal_slice.start)
                for bbox_start, nominal_slice in zip(usable_bbox[:3], nominal_slices)
            ], dtype=np.float64)
            usable_stop = np.asarray([
                int(bbox_stop) - int(read_size) + int(nominal_slice.stop)
                for bbox_stop, read_size, nominal_slice in zip(
                    usable_bbox[3:], self.source_read_window_size, nominal_slices
                )
            ], dtype=np.float64)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_volume_handles"] = {}
        state["_handle_pid"] = None
        state["_atexit_pid"] = None
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

    def _read_source_crop_3d(self, d_zarr, usable_bbox):
        z0, y0, x0, z1, y1, x1 = usable_bbox
        crop_d, crop_h, crop_w = self.source_read_window_size
        z_start = np.random.randint(z0, z1 - crop_d + 1)
        y_start = np.random.randint(y0, y1 - crop_h + 1)
        x_start = np.random.randint(x0, x1 - crop_w + 1)
        return np.asarray(
            d_zarr[
                z_start:z_start + crop_d,
                y_start:y_start + crop_h,
                x_start:x_start + crop_w,
            ]
        )

    def _read_point_centered_source_crop_3d(self, d_zarr, usable_bbox, anchor):
        starts = []
        nominal_slices = self._nominal_source_slices(self.source_read_window_size)
        for bbox_start, bbox_stop, crop_size, coordinate, nominal_slice in zip(
            usable_bbox[:3],
            usable_bbox[3:],
            self.source_read_window_size,
            anchor,
            nominal_slices,
        ):
            minimum = max(
                int(bbox_start),
                int(np.ceil(float(coordinate) - (int(nominal_slice.stop) - 1))),
            )
            maximum = min(
                int(bbox_stop) - int(crop_size),
                int(np.floor(float(coordinate) - int(nominal_slice.start))),
            )
            if maximum < minimum:
                raise ValueError(
                    f"Annotated point {tuple(float(value) for value in anchor)} cannot fit in a source window "
                    f"of size {self.source_read_window_size} inside bounds {usable_bbox}."
                )
            starts.append(int(np.random.randint(minimum, maximum + 1)))
        z_start, y_start, x_start = starts
        crop_d, crop_h, crop_w = self.source_read_window_size
        crop = np.asarray(
            d_zarr[
                z_start:z_start + crop_d,
                y_start:y_start + crop_h,
                x_start:x_start + crop_w,
            ]
        )
        return crop, (z_start, y_start, x_start)

    @staticmethod
    def _expand_crop_shape(crop_shape, target_size, reference_size):
        return tuple(
            max(1, int(round(int(size) * int(target) / int(reference))))
            for size, target, reference in zip(crop_shape, target_size, reference_size)
        )

    @staticmethod
    def _centered_slices(outer_size, inner_size):
        slices = []
        for outer, inner in zip(outer_size, inner_size):
            delta = int(outer) - int(inner)
            if delta < 0:
                raise ValueError(
                    f"cannot place centered region of size {inner_size} inside {outer_size}"
                )
            start = delta // 2
            slices.append(slice(start, start + int(inner)))
        return tuple(slices)

    def _nominal_source_slices(self, source_shape):
        source_shape = tuple(int(dim) for dim in source_shape)
        return self._centered_slices(source_shape, self.source_sampling_size)

    def _extract_nominal_source_region(self, source_crop):
        return source_crop[self._nominal_source_slices(source_crop.shape)]

    def _required_read_window_size(self, source_sampling_size):
        required_sizes = [tuple(int(size) for size in source_sampling_size)]
        required_sizes.append(
            self._expand_crop_shape(source_sampling_size, self.paired_global_view_size, self.paired_global_crop_size)
        )
        if self.local_crop_size is not None and self.local_view_size is not None:
            required_sizes.append(
                self._expand_crop_shape(source_sampling_size, self.local_view_size, self.local_crop_size)
            )
        return _max_3tuple(*required_sizes)

    def _sample_random_resized_crop_region(
        self,
        source_shape,
        scale_range,
        target_size,
        *,
        reference_size=None,
        anchor=None,
        anchor_target_size=None,
        anchor_output_size=None,
    ):
        if reference_size is None:
            reference_size = target_size
        source_depth, source_height, source_width = source_shape
        nominal_crop_shape = self._sample_crop_shape(scale_range)
        # Sample the nominal DINO crop inside source_sampling_size, then expand around it when
        # a deeper-embed halo is requested.
        crop_d, crop_h, crop_w = self._expand_crop_shape(
            nominal_crop_shape,
            target_size,
            reference_size,
        )
        expanded_crop_shape = (int(crop_d), int(crop_h), int(crop_w))
        source_shape = (int(source_depth), int(source_height), int(source_width))
        if any(crop > source for crop, source in zip(expanded_crop_shape, source_shape)):
            raise ValueError(
                "expanded crop does not fit inside source_read_window_size; "
                f"expanded_crop_shape={expanded_crop_shape}, source_shape={source_shape}, "
                f"source_sampling_size={self.source_sampling_size}, target_size={target_size}, "
                f"reference_size={reference_size}"
            )

        nominal_source_slices = self._nominal_source_slices(source_shape)
        nominal_starts = []
        for axis, (nominal_slice, nominal_dim, raw_dim) in enumerate(zip(
            nominal_source_slices,
            nominal_crop_shape,
            expanded_crop_shape,
        )):
            low = int(nominal_slice.start)
            high_inclusive = int(nominal_slice.stop) - int(nominal_dim)
            if anchor is not None:
                if anchor_target_size is None or anchor_output_size is None:
                    raise ValueError("anchor_target_size and anchor_output_size are required with an anchor.")
                raw_offset = (int(raw_dim) - int(nominal_dim)) // 2
                output_offset = (float(anchor_target_size[axis]) - float(anchor_output_size[axis])) / 2.0
                relative_min = (
                    (output_offset + 0.5) * int(raw_dim) / float(anchor_target_size[axis]) - 0.5
                )
                relative_max = (
                    (output_offset + float(anchor_output_size[axis]) - 0.5)
                    * int(raw_dim)
                    / float(anchor_target_size[axis])
                    - 0.5
                )
                low = max(
                    low,
                    int(np.ceil(float(anchor[axis]) - relative_max + raw_offset - 1e-7)),
                )
                high_inclusive = min(
                    high_inclusive,
                    int(np.floor(float(anchor[axis]) - relative_min + raw_offset + 1e-7)),
                )
            if high_inclusive < low:
                raise ValueError(
                    "nominal crop cannot contain the requested annotation; "
                    f"source_sampling_size={self.source_sampling_size}, nominal_crop_shape={nominal_crop_shape}, "
                    f"anchor={anchor}"
                )
            nominal_starts.append(int(np.random.randint(low, high_inclusive + 1)))

        raw_starts = []
        for nominal_start, nominal_dim, raw_dim, source_dim in zip(
            nominal_starts,
            nominal_crop_shape,
            expanded_crop_shape,
            source_shape,
        ):
            raw_start = int(nominal_start) - (int(raw_dim) - int(nominal_dim)) // 2
            raw_stop = raw_start + int(raw_dim)
            if raw_start < 0 or raw_stop > int(source_dim):
                raise ValueError(
                    "expanded crop fell outside source_read_window_size; "
                    f"raw_start={raw_start}, raw_stop={raw_stop}, source_dim={source_dim}, "
                    f"nominal_crop_shape={nominal_crop_shape}, expanded_crop_shape={expanded_crop_shape}, "
                    f"source_sampling_size={self.source_sampling_size}, "
                    f"source_read_window_size={self.source_read_window_size}"
                )
            raw_starts.append(raw_start)

        z_start, y_start, x_start = raw_starts
        return SampledCropRegion(
            starts=(int(z_start), int(y_start), int(x_start)),
            shape=(int(crop_d), int(crop_h), int(crop_w)),
        )

    def _materialize_crop_from_region(self, source_crop, crop_region, target_size):
        z_start, y_start, x_start = crop_region.starts
        crop_d, crop_h, crop_w = crop_region.shape
        crop = source_crop[
            z_start:z_start + crop_d,
            y_start:y_start + crop_h,
            x_start:x_start + crop_w,
        ]
        return self._finalize_crop(crop, target_size)

    def _random_resized_crop_3d_from_array(self, source_crop, scale_range, target_size, *, reference_size=None):
        crop_region = self._sample_random_resized_crop_region(
            source_crop.shape,
            scale_range,
            target_size,
            reference_size=reference_size,
        )
        return self._materialize_crop_from_region(source_crop, crop_region, target_size)

    def _read_random_resized_crop_3d(self, d_zarr, usable_bbox, scale_range, target_size, *, reference_size=None):
        if reference_size is None:
            reference_size = target_size
        source_crop = self._read_source_crop_3d(d_zarr, usable_bbox)
        return self._random_resized_crop_3d_from_array(
            source_crop,
            scale_range,
            target_size,
            reference_size=reference_size,
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
            return transform(image=global_view)["image"], gram_teacher_view

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
            "crop_shape": tuple(int(value) for value in self.global_view_size),
        }
        if gram_teacher_view is None or self.gram_teacher_no_augmentations:
            transformed = transform(**transform_input)
            return transformed["image"], gram_teacher_view, transformed["keypoints"]

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
            self._map_points_to_view(source_points[ordered_indices], crop_region, self.global_view_size),
            torch.from_numpy(volume.point_type_ids[ordered_indices].copy()).long(),
        )

    def _filter_and_cap_view_points(self, coordinates, type_ids):
        offset = coordinates.new_tensor(self.global_point_crop_offset)
        coordinates = coordinates - offset
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

    def __getitem__(self, idx):
        vol_weights = [vol.weight for vol in self.volumes]
        nonzero_threshold = float(self.config.get("nonzero_threshold", 0.30))

        while True:
            vol_idx = np.random.choice(len(self.volumes), p=vol_weights)
            vol = self.volumes[vol_idx]
            d_zarr = self._get_volume_array(vol)
            anchor_index = None
            source_starts = None
            if (
                self.point_supervision_enabled
                and vol.point_coordinates is not None
                and vol.point_coordinates.shape[0] > 0
                and np.random.random() < self.point_sampling_probability
            ):
                anchor_index = self._sample_volume_anchor(vol)
                source_crop, source_starts = self._read_point_centered_source_crop_3d(
                    d_zarr,
                    vol.usable_bbox,
                    vol.point_coordinates[anchor_index],
                )
            else:
                source_crop = self._read_source_crop_3d(d_zarr, vol.usable_bbox)
            nominal_source = self._extract_nominal_source_region(source_crop)
            if nominal_source.size > 0 and (np.count_nonzero(nominal_source) / nominal_source.size) >= nonzero_threshold:
                break

        if self.single_crop_only:
            crop = self._random_resized_crop_3d_from_array(
                source_crop,
                self.global_crop_scale,
                self.global_view_size,
                reference_size=self.global_crop_size,
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
                self.paired_global_view_size,
                reference_size=self.paired_global_crop_size,
                anchor=(
                    vol.point_coordinates[anchor_index] - np.asarray(source_starts, dtype=np.float64)
                    if anchor_index is not None
                    else None
                ),
                anchor_target_size=self.global_view_size if anchor_index is not None else None,
                anchor_output_size=self.global_crop_size if anchor_index is not None else None,
            )
            global_view = self._materialize_crop_from_region(source_crop, crop_region, self.global_view_size)
            gram_teacher_view = None
            if self.gram_teacher_view_size is not None:
                gram_teacher_view = self._materialize_crop_from_region(
                    source_crop,
                    crop_region,
                    self.gram_teacher_view_size,
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
                    self.local_view_size,
                    reference_size=self.local_crop_size,
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
