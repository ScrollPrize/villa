import atexit
import json
import os
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


@dataclass
class Volume:
    usable_bbox: tuple  # (z0, y0, x0, z1, y1, x1)
    valid_crop_starts: int
    scale: int
    path: str
    weight: float = 0.0


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
        self.source_crop_size = _as_3tuple(self.config.get("source_crop_size", self.global_crop_size))
        self.global_crop_scale = _as_float_pair(self.config.get("global_crop_scale"), (0.32, 1.0))
        self.local_crop_scale = _as_float_pair(self.config.get("local_crop_scale"), (0.05, 0.32))
        self.num_local_crops = self.config.get("num_local_crops", 8)
        self.volume_auth = self.config["volume_auth"] if "volume_auth" in self.config else None
        self.s3_storage_options = self.config.get("s3_storage_options")
        self.vol_trim_pct = self.config.get("vol_trim_pct", 0.60)
        self.normalizer = get_normalization(self.config.get("normalization_scheme", "robust"))
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
        
        self.global_transforms = [create_training_transforms(self.global_crop_size) for _ in
                                  range(self.num_global_crops)]
        self.local_transforms = (
            [create_training_transforms(self.local_crop_size) for _ in range(self.num_local_crops)]
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
                crop_z, crop_y, crop_x = self.source_crop_size
                valid_z = max(0, (z1 - z0) - crop_z + 1)
                valid_y = max(0, (y1 - y0) - crop_y + 1)
                valid_x = max(0, (x1 - x0) - crop_x + 1)
                valid_crop_starts = valid_z * valid_y * valid_x
                
                self.volumes.append(Volume(
                    path=str(volume_path),
                    scale=volume_scale,
                    usable_bbox=usable_bbox,
                    valid_crop_starts=valid_crop_starts,
                ))
            finally:
                handle.close()
        
        self.total_valid_crop_starts = sum(vol.valid_crop_starts for vol in self.volumes)
        if self.total_valid_crop_starts <= 0:
            raise ValueError("No valid crop starts found across configured volumes for the selected source crop size.")
        
        for volume in self.volumes:
            volume.weight = volume.valid_crop_starts / self.total_valid_crop_starts

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
        ref_depth, ref_height, ref_width = self.source_crop_size
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
        crop_d, crop_h, crop_w = self.source_crop_size
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
    
    def _random_resized_crop_3d_from_array(self, source_crop, scale_range, target_size):
        source_depth, source_height, source_width = source_crop.shape
        crop_d, crop_h, crop_w = self._sample_crop_shape(scale_range)
        crop_d = min(crop_d, source_depth)
        crop_h = min(crop_h, source_height)
        crop_w = min(crop_w, source_width)
        
        z_start = np.random.randint(0, source_depth - crop_d + 1)
        y_start = np.random.randint(0, source_height - crop_h + 1)
        x_start = np.random.randint(0, source_width - crop_w + 1)
        crop = source_crop[
            z_start:z_start + crop_d,
            y_start:y_start + crop_h,
            x_start:x_start + crop_w,
        ]
        return self._finalize_crop(crop, target_size)
    
    def _read_random_resized_crop_3d(self, d_zarr, usable_bbox, scale_range, target_size):
        z0, y0, x0, z1, y1, x1 = usable_bbox
        bbox_depth = z1 - z0
        bbox_height = y1 - y0
        bbox_width = x1 - x0
        crop_d, crop_h, crop_w = self._sample_crop_shape(scale_range)
        crop_d = min(crop_d, bbox_depth)
        crop_h = min(crop_h, bbox_height)
        crop_w = min(crop_w, bbox_width)
        
        z_start = np.random.randint(z0, z1 - crop_d + 1)
        y_start = np.random.randint(y0, y1 - crop_h + 1)
        x_start = np.random.randint(x0, x1 - crop_w + 1)
        crop = d_zarr[
            z_start:z_start + crop_d,
            y_start:y_start + crop_h,
            x_start:x_start + crop_w,
        ]
        return self._finalize_crop(crop, target_size)
    
    def __len__(self):
        if self.epoch_length is not None:
            return self.epoch_length
        return self.total_valid_crop_starts
    
    def __getitem__(self, idx):
        vol_weights = [vol.weight for vol in self.volumes]
        nonzero_threshold = float(self.config.get("nonzero_threshold", 0.30))
        
        while True:
            vol_idx = np.random.choice(len(self.volumes), p=vol_weights)
            vol = self.volumes[vol_idx]
            d_zarr = self._get_volume_array(vol)
            source_crop = self._read_source_crop_3d(d_zarr, vol.usable_bbox)
            if source_crop.size > 0 and (np.count_nonzero(source_crop) / source_crop.size) >= nonzero_threshold:
                break
        
        if self.single_crop_only:
            crop = self._random_resized_crop_3d_from_array(
                source_crop,
                self.global_crop_scale,
                self.global_crop_size,
            )
            if self.do_augmentations:
                crop = self.global_transforms[0](image=crop)["image"]
            return crop
        
        global_views = [
            self._random_resized_crop_3d_from_array(
                source_crop,
                self.global_crop_scale,
                self.global_crop_size,
            )
            for _ in range(self.num_global_crops)
        ]
        if self.do_augmentations:
            global_views = [
                transform(image=view)["image"]
                for transform, view in zip(self.global_transforms, global_views)
            ]
        
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
        
        return {
            "global_views": global_views,
            "local_views": local_views,
        }
