import os.path as osp

import numpy as np
import zarr

from train_resnet3d_lib.config import CFG
from train_resnet3d_lib.data.image_readers import (
    _clip_intensity_inplace,
    _compute_selected_layer_indices,
    _require_dict,
)


def _looks_like_zarr_store(path: str) -> bool:
    if not osp.exists(path):
        return False
    if osp.isfile(path):
        return path.endswith(".zarr")
    if osp.isdir(path):
        if osp.exists(osp.join(path, ".zarray")):
            return True
        if osp.exists(osp.join(path, ".zgroup")):
            return True
        if osp.exists(osp.join(path, "0", ".zarray")):
            return True
    return False


def resolve_segment_zarr_path(fragment_id):
    fragment_id = str(fragment_id)
    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))
    if not fragment_id:
        raise ValueError("segment id must be a non-empty string")

    candidate = osp.normpath(osp.join(dataset_root, f"{fragment_id}.zarr"))
    if _looks_like_zarr_store(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not resolve zarr volume path for segment={fragment_id}. "
        f"Expected zarr store at {candidate!r}."
    )


def _ensure_zarr_v2():
    ver = str(getattr(zarr, "__version__", "") or "")
    major_str = ver.split(".", 1)[0].strip()
    if not major_str.isdigit():
        raise RuntimeError(f"Could not parse zarr version {ver!r}; expected major version integer.")
    major = int(major_str)
    if major >= 3:
        raise RuntimeError(
            f"zarr backend requires zarr v2, found version {ver!r}. "
            "Install a v2 release (e.g., `zarr<3`)."
        )


def _from_uint16_to_uint8(arr: np.ndarray, *, fragment_id: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8, copy=False)
    raise TypeError(
        f"{fragment_id}: unsupported zarr dtype for 8-bit pipeline: {arr.dtype}. "
        "Expected uint8 or uint16."
    )


class ZarrSegmentVolume:
    def __init__(
        self,
        fragment_id,
        seg_meta,
        *,
        layer_range,
        reverse_layers=False,
    ):
        _ensure_zarr_v2()

        self.fragment_id = str(fragment_id)
        _require_dict(seg_meta, name=f"segments[{self.fragment_id!r}]")
        self.path = resolve_segment_zarr_path(self.fragment_id)

        idxs = _compute_selected_layer_indices(self.fragment_id, layer_range=layer_range)
        if reverse_layers:
            idxs = list(reversed(idxs))
        self._requested_layer_indices = [int(i) for i in idxs]

        self._zarr_array = None

        meta = self._inspect_volume()
        self._depth_axis_first = bool(meta["depth_axis_first"])
        self._dtype = np.dtype(meta["dtype"])
        self._base_h = int(meta["base_h"])
        self._base_w = int(meta["base_w"])
        self._layer_indices = np.asarray(meta["layer_indices"], dtype=np.int64)
        self._layer_read_mode = str(meta["layer_read_mode"])
        self._z_slice_start = int(meta["z_slice_start"])
        self._z_slice_stop = int(meta["z_slice_stop"])

        pad_h = int((256 - (self._base_h % 256)) % 256)
        pad_w = int((256 - (self._base_w % 256)) % 256)
        self._padded_h = int(self._base_h + pad_h)
        self._padded_w = int(self._base_w + pad_w)
        self._out_h = int(self._padded_h)
        self._out_w = int(self._padded_w)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_zarr_array"] = None
        return state

    @staticmethod
    def _open_zarr_array(path):
        root = zarr.open(path, mode="r")
        if hasattr(root, "shape"):
            return root
        if "0" in root:
            return root["0"]
        raise ValueError(
            f"Expected a 3D zarr array or group key '0' at {path}; got group without key '0'."
        )

    def _inspect_volume(self):
        arr = self._open_zarr_array(self.path)
        raw_shape = tuple(int(x) for x in arr.shape)
        if len(raw_shape) != 3:
            raise ValueError(f"{self.fragment_id}: expected 3D zarr volume, got shape={raw_shape} at {self.path}")

        min_dim_idx = int(np.argmin(raw_shape))
        depth_axis_first = bool(min_dim_idx == 0)
        if depth_axis_first:
            base_h, base_w = raw_shape[1], raw_shape[2]
            n_layers = raw_shape[0]
        else:
            base_h, base_w = raw_shape[0], raw_shape[1]
            n_layers = raw_shape[2]

        layer_indices = [int(i) for i in self._requested_layer_indices]
        if len(layer_indices) == 0:
            raise ValueError(f"{self.fragment_id}: no selected layers for zarr volume")

        min_idx = int(min(layer_indices))
        max_idx = int(max(layer_indices))
        if min_idx < 0 or max_idx >= int(n_layers):
            raise ValueError(
                f"{self.fragment_id}: selected layer indices out of bounds for zarr depth={n_layers}. "
                f"expected 0-based indices in [0, {int(n_layers) - 1}], got min={min_idx}, max={max_idx}"
            )

        li = np.asarray(layer_indices, dtype=np.int64)
        layer_read_mode = "fancy"
        z_slice_start = int(li[0])
        z_slice_stop = int(li[-1]) + 1
        if li.size > 1 and np.all(np.diff(li) == 1):
            layer_read_mode = "slice_asc"
        elif li.size > 1 and np.all(np.diff(li) == -1):
            layer_read_mode = "slice_desc"
            z_slice_start = int(li[-1])
            z_slice_stop = int(li[0]) + 1

        return {
            "depth_axis_first": depth_axis_first,
            "dtype": arr.dtype,
            "base_h": int(base_h),
            "base_w": int(base_w),
            "layer_indices": li,
            "layer_read_mode": layer_read_mode,
            "z_slice_start": int(z_slice_start),
            "z_slice_stop": int(z_slice_stop),
        }

    @property
    def shape(self):
        return (int(self._out_h), int(self._out_w), int(CFG.in_chans))

    def _ensure_zarr_array(self):
        if self._zarr_array is None:
            self._zarr_array = self._open_zarr_array(self.path)
        return self._zarr_array

    def _read_raw_patch(self, y1, y2, x1, x2):
        z = self._ensure_zarr_array()
        if self._depth_axis_first:
            if self._layer_read_mode == "slice_asc":
                data = z[self._z_slice_start:self._z_slice_stop, y1:y2, x1:x2]
            elif self._layer_read_mode == "slice_desc":
                data = z[self._z_slice_start:self._z_slice_stop, y1:y2, x1:x2][::-1]
            else:
                data = z[self._layer_indices, y1:y2, x1:x2]
            data = np.asarray(data)
            if data.ndim != 3:
                raise ValueError(f"{self.fragment_id}: invalid zarr read shape={data.shape}")
            data = np.transpose(data, (1, 2, 0))
            return data

        if self._layer_read_mode == "slice_asc":
            data = z[y1:y2, x1:x2, self._z_slice_start:self._z_slice_stop]
        elif self._layer_read_mode == "slice_desc":
            data = z[y1:y2, x1:x2, self._z_slice_start:self._z_slice_stop][..., ::-1]
        else:
            data = z[y1:y2, x1:x2, self._layer_indices]
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"{self.fragment_id}: invalid zarr read shape={data.shape}")
        return data

    def _read_patch_unflipped(self, y1, y2, x1, x2):
        out_h = int(y2 - y1)
        out_w = int(x2 - x1)
        out = np.zeros((out_h, out_w, int(CFG.in_chans)), dtype=self._dtype)
        yy1 = max(0, int(y1))
        yy2 = min(int(self._base_h), int(y2))
        xx1 = max(0, int(x1))
        xx2 = min(int(self._base_w), int(x2))
        if yy2 > yy1 and xx2 > xx1:
            block = self._read_raw_patch(yy1, yy2, xx1, xx2)
            out[yy1 - int(y1):yy2 - int(y1), xx1 - int(x1):xx2 - int(x1), :] = block
        return out

    def _read_patch(self, y1, y2, x1, x2):
        return self._read_patch_unflipped(y1, y2, x1, x2)

    def read_patch(self, y1, y2, x1, x2):
        y1 = int(y1)
        y2 = int(y2)
        x1 = int(x1)
        x2 = int(x2)
        if y2 <= y1 or x2 <= x1:
            raise ValueError(f"{self.fragment_id}: invalid patch coords {(x1, y1, x2, y2)}")

        patch = self._read_patch(y1, y2, x1, x2)
        patch = _from_uint16_to_uint8(patch, fragment_id=self.fragment_id)
        _clip_intensity_inplace(patch)

        expected = (int(y2 - y1), int(x2 - x1), int(CFG.in_chans))
        if patch.shape != expected:
            raise ValueError(
                f"{self.fragment_id}: patch shape mismatch, got {patch.shape}, expected {expected} "
                f"for coords={(x1, y1, x2, y2)}"
            )
        return patch


