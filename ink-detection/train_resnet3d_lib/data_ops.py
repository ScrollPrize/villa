from train_resnet3d_lib.config import CFG

import os.path as osp

import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F

import albumentations as A
from torch.utils.data import Dataset
from scipy import ndimage
import PIL.Image
import zarr


def _require_dict(value, *, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def _read_gray(path):
    if not osp.exists(path):
        return None

    if path.lower().endswith(".png"):
        try:
            with PIL.Image.open(path) as im:
                return np.array(im.convert("L"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read PNG image via PIL: {path}") from exc

    try:
        img = cv2.imread(path, 0)
    except cv2.error as exc:
        raise RuntimeError(f"Could not read image via OpenCV: {path}") from exc
    if img is None:
        raise RuntimeError(f"Could not read image via OpenCV (returned None): {path}")
    return img


def _parse_layer_range(fragment_id, layer_range):
    if layer_range is None:
        raise KeyError(f"{fragment_id}: missing required segments metadata key 'layer_range'")
    if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
        raise TypeError(
            f"{fragment_id}: layer_range must be a [start_idx, end_idx] pair, got {layer_range!r}"
        )

    start_idx = int(layer_range[0])
    end_idx = int(layer_range[1])
    if end_idx <= start_idx:
        raise ValueError(f"{fragment_id}: layer_range must satisfy end_idx > start_idx, got {layer_range!r}")
    return start_idx, end_idx


def _compute_selected_layer_indices(fragment_id, layer_range):
    start_idx, end_idx = _parse_layer_range(fragment_id, layer_range)

    idxs = list(range(start_idx, end_idx))
    if len(idxs) < CFG.in_chans:
        raise ValueError(
            f"{fragment_id}: expected at least {CFG.in_chans} layers, got {len(idxs)} from range {start_idx}-{end_idx}"
        )
    if len(idxs) > CFG.in_chans:
        start = max(0, (len(idxs) - CFG.in_chans) // 2)
        idxs = idxs[start:start + CFG.in_chans]
    if len(idxs) != CFG.in_chans:
        raise ValueError(
            f"{fragment_id}: expected {CFG.in_chans} layers after cropping, got {len(idxs)} from range {start_idx}-{end_idx}"
        )
    return [int(i) for i in idxs]


def read_image_layers(
    fragment_id,
    *,
    layer_range,
):
    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))
    layers_dir = osp.join(dataset_root, fragment_id, "layers")
    layer_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _iter_layer_paths(layer_idx):
        # Support 00.tif, 000.tif, 0000.tif, etc.
        for fmt in (f"{layer_idx:02}", f"{layer_idx:03}", f"{layer_idx:04}", str(layer_idx)):
            for ext in layer_exts:
                yield osp.join(layers_dir, f"{fmt}{ext}")

    idxs = _compute_selected_layer_indices(fragment_id, layer_range)

    layer_read_workers = int(getattr(CFG, "layer_read_workers", 1) or 1)
    layer_read_workers = max(1, min(layer_read_workers, len(idxs)))

    first = None
    for image_path in _iter_layer_paths(idxs[0]):
        first = _read_gray(image_path)
        if first is not None:
            break
    if first is None:
        raise FileNotFoundError(
            f"Could not read layer for {fragment_id}: {layers_dir}/{idxs[0]}.[tif|tiff|png|jpg|jpeg]"
        )

    base_h, base_w = first.shape[:2]
    pad0 = (256 - base_h % 256)
    pad1 = (256 - base_w % 256)
    out_h = base_h + pad0
    out_w = base_w + pad1

    images = np.zeros((out_h, out_w, len(idxs)), dtype=first.dtype)
    np.clip(first, 0, 200, out=first)
    images[:base_h, :base_w, 0] = first

    def _load_and_write(task):
        chan, i = task
        img = None
        for image_path in _iter_layer_paths(i):
            img = _read_gray(image_path)
            if img is not None:
                break
        if img is None:
            raise FileNotFoundError(
                f"Could not read layer for {fragment_id}: {layers_dir}/{i}.[tif|tiff|png|jpg|jpeg]"
            )
        if img.shape[0] != base_h or img.shape[1] != base_w:
            raise ValueError(
                f"{fragment_id}: layer {i:02} has shape {img.shape} but expected {(base_h, base_w)}"
            )
        np.clip(img, 0, 200, out=img)
        images[:base_h, :base_w, chan] = img
        return None

    tasks = [(chan, i) for chan, i in enumerate(idxs[1:], start=1)]
    if tasks:
        if layer_read_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=layer_read_workers) as executor:
                list(executor.map(_load_and_write, tasks))
        else:
            for task in tasks:
                _load_and_write(task)

    return images


def read_image_mask(
    fragment_id,
    *,
    layer_range=None,
    reverse_layers=False,
    label_suffix="",
    mask_suffix="",
    images=None,
):
    if images is None:
        images = read_image_layers(
            fragment_id,
            layer_range=layer_range,
        )

    if reverse_layers:
        images = images[:, :, ::-1]

    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))
    label_base = osp.join(dataset_root, str(fragment_id), f"{fragment_id}_inklabels{label_suffix}")
    mask = _read_gray(f"{label_base}.png")
    if mask is None:
        mask = _read_gray(f"{label_base}.tiff")
    if mask is None:
        mask = _read_gray(f"{label_base}.tif")
    if mask is None:
        raise FileNotFoundError(f"Could not read label for {fragment_id}: {label_base}.png/.tif/.tiff")

    mask_base = osp.join(dataset_root, str(fragment_id), f"{fragment_id}_mask{mask_suffix}")
    fragment_mask = _read_gray(f"{mask_base}.png")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tiff")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tif")
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_base}.png/.tif/.tiff")

    def _assert_bottom_right_pad_compatible(a_name, a_hw, b_name, b_hw, multiple):
        _assert_bottom_right_pad_compatible_global(fragment_id, a_name, a_hw, b_name, b_hw, multiple)
    pad_multiple = 256
    _assert_bottom_right_pad_compatible("image", images.shape[:2], "label", mask.shape[:2], pad_multiple)
    _assert_bottom_right_pad_compatible("image", images.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)
    _assert_bottom_right_pad_compatible("label", mask.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)

    fragment_mask_padded = np.zeros((images.shape[0], images.shape[1]), dtype=fragment_mask.dtype)
    h = min(fragment_mask.shape[0], fragment_mask_padded.shape[0])
    w = min(fragment_mask.shape[1], fragment_mask_padded.shape[1])
    fragment_mask_padded[:h, :w] = fragment_mask[:h, :w]
    fragment_mask = fragment_mask_padded
    del fragment_mask_padded

    target_h = min(images.shape[0], mask.shape[0], fragment_mask.shape[0])
    target_w = min(images.shape[1], mask.shape[1], fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"{fragment_id}: empty shapes after alignment (images={images.shape}, label={mask.shape}, mask={fragment_mask.shape})"
        )

    images = images[:target_h, :target_w]
    mask = mask[:target_h, :target_w]
    fragment_mask = fragment_mask[:target_h, :target_w]

    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8, copy=False)
    if images.shape[0] != mask.shape[0] or images.shape[1] != mask.shape[1]:
        raise ValueError(f"{fragment_id}: label shape {mask.shape} does not match image shape {images.shape[:2]}")
    return images, mask, fragment_mask


def _assert_bottom_right_pad_compatible_global(fragment_id, a_name, a_hw, b_name, b_hw, multiple):
    a_h, a_w = [int(x) for x in a_hw]
    b_h, b_w = [int(x) for x in b_hw]

    def _check_dim(dim_name, a_dim, b_dim):
        small = min(a_dim, b_dim)
        big = max(a_dim, b_dim)
        ceil_to_multiple = ((small + multiple - 1) // multiple) * multiple
        if small % multiple == 0:
            allowed = {small, small + multiple}
        else:
            allowed = {small, ceil_to_multiple}

        if big not in allowed:
            raise ValueError(
                f"{fragment_id}: {a_name} {a_hw} vs {b_name} {b_hw} mismatch. "
                f"Only bottom/right padding to a multiple of {multiple} is allowed "
                f"(see inference_resnet3d.py). Got {dim_name}={a_dim} vs {b_dim}."
            )

    _check_dim("height", a_h, b_h)
    _check_dim("width", a_w, b_w)


def read_image_fragment_mask(
    fragment_id,
    *,
    layer_range=None,
    reverse_layers=False,
    mask_suffix="",
    images=None,
):
    if images is None:
        images = read_image_layers(
            fragment_id,
            layer_range=layer_range,
        )

    if reverse_layers:
        images = images[:, :, ::-1]

    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))
    mask_base = osp.join(dataset_root, str(fragment_id), f"{fragment_id}_mask{mask_suffix}")
    fragment_mask = _read_gray(f"{mask_base}.png")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tiff")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tif")
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_base}.png/.tif/.tiff")

    pad_multiple = 256
    _assert_bottom_right_pad_compatible_global(
        fragment_id, "image", images.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple
    )

    target_h = min(images.shape[0], fragment_mask.shape[0])
    target_w = min(images.shape[1], fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"{fragment_id}: empty shapes after alignment (image={images.shape}, mask={fragment_mask.shape})")

    images = images[:target_h, :target_w]
    fragment_mask = fragment_mask[:target_h, :target_w]

    return images, fragment_mask


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
        # Common OME-Zarr group layout.
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
        # Match the common 16-bit -> 8-bit downscale convention used by OpenCV grayscale reads.
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

        pad_h = int(256 - (self._base_h % 256))
        pad_w = int(256 - (self._base_w % 256))
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

        # Group-like store: training expects the canonical OME-Zarr base scale at key "0".
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
        np.clip(patch, 0, 200, out=patch)

        expected = (int(y2 - y1), int(x2 - x1), int(CFG.in_chans))
        if patch.shape != expected:
            raise ValueError(
                f"{self.fragment_id}: patch shape mismatch, got {patch.shape}, expected {expected} "
                f"for coords={(x1, y1, x2, y2)}"
            )
        return patch


def read_label_and_fragment_mask_for_shape(
    fragment_id,
    image_shape_hw,
    *,
    label_suffix="",
    mask_suffix="",
):
    image_h = int(image_shape_hw[0])
    image_w = int(image_shape_hw[1])
    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))

    label_base = osp.join(dataset_root, str(fragment_id), f"{fragment_id}_inklabels{label_suffix}")
    mask = _read_gray(f"{label_base}.png")
    if mask is None:
        mask = _read_gray(f"{label_base}.tiff")
    if mask is None:
        mask = _read_gray(f"{label_base}.tif")
    if mask is None:
        raise FileNotFoundError(f"Could not read label for {fragment_id}: {label_base}.png/.tif/.tiff")

    mask_base = osp.join(dataset_root, str(fragment_id), f"{fragment_id}_mask{mask_suffix}")
    fragment_mask = _read_gray(f"{mask_base}.png")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tiff")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tif")
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_base}.png/.tif/.tiff")

    pad_multiple = 256
    _assert_bottom_right_pad_compatible_global(
        str(fragment_id),
        "image",
        (image_h, image_w),
        "label",
        mask.shape[:2],
        pad_multiple,
    )
    _assert_bottom_right_pad_compatible_global(
        str(fragment_id),
        "image",
        (image_h, image_w),
        "mask",
        fragment_mask.shape[:2],
        pad_multiple,
    )
    _assert_bottom_right_pad_compatible_global(
        str(fragment_id),
        "label",
        mask.shape[:2],
        "mask",
        fragment_mask.shape[:2],
        pad_multiple,
    )

    fragment_mask_padded = np.zeros((image_h, image_w), dtype=fragment_mask.dtype)
    h = min(fragment_mask.shape[0], fragment_mask_padded.shape[0])
    w = min(fragment_mask.shape[1], fragment_mask_padded.shape[1])
    fragment_mask_padded[:h, :w] = fragment_mask[:h, :w]
    fragment_mask = fragment_mask_padded

    target_h = min(image_h, mask.shape[0], fragment_mask.shape[0])
    target_w = min(image_w, mask.shape[1], fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"{fragment_id}: empty shapes after alignment "
            f"(image={(image_h, image_w)}, label={mask.shape}, mask={fragment_mask.shape})"
        )

    mask = mask[:target_h, :target_w]
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8, copy=False)
    fragment_mask = fragment_mask[:target_h, :target_w]
    return mask, fragment_mask


def read_fragment_mask_for_shape(
    fragment_id,
    image_shape_hw,
    *,
    mask_suffix="",
):
    image_h = int(image_shape_hw[0])
    image_w = int(image_shape_hw[1])
    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))

    mask_base = osp.join(dataset_root, str(fragment_id), f"{fragment_id}_mask{mask_suffix}")
    fragment_mask = _read_gray(f"{mask_base}.png")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tiff")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tif")
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_base}.png/.tif/.tiff")

    pad_multiple = 256
    _assert_bottom_right_pad_compatible_global(
        str(fragment_id),
        "image",
        (image_h, image_w),
        "mask",
        fragment_mask.shape[:2],
        pad_multiple,
    )

    target_h = min(image_h, fragment_mask.shape[0])
    target_w = min(image_w, fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"{fragment_id}: empty shapes after alignment "
            f"(image={(image_h, image_w)}, mask={fragment_mask.shape})"
        )
    return fragment_mask[:target_h, :target_w]


def _label_tile_is_empty(label_tile) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def extract_patch_coordinates(
    mask,
    fragment_mask,
    *,
    filter_empty_tile,
):
    xyxys = []
    stride = CFG.stride
    x1_list = list(range(0, fragment_mask.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, fragment_mask.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if filter_empty_tile and mask is not None and _label_tile_is_empty(
                mask[a:a + CFG.tile_size, b:b + CFG.tile_size]
            ):
                continue
            if np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue
                    windows_dict[(y1, y2, x1, x2)] = True
                    xyxys.append([x1, y1, x2, y2])
    if len(xyxys) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(xyxys, dtype=np.int64)


def _component_bboxes(mask, *, connectivity=2):
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8, copy=False)
    if mask_u8.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape={tuple(mask_u8.shape)}")
    if not bool(mask_u8.any()):
        return np.zeros((0, 4), dtype=np.int32)

    cc_conn = 4 if int(connectivity) == 1 else 8
    n_all, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=cc_conn)
    bboxes = []
    for li in range(1, int(n_all)):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        w = int(stats[li, cv2.CC_STAT_WIDTH])
        h = int(stats[li, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        bboxes.append((y, y + h, x, x + w))
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    bboxes.sort(key=lambda b: (int(b[0]), int(b[2]), int(b[1]), int(b[3])))
    return np.asarray(bboxes, dtype=np.int32)


def _build_mask_store_and_patch_index(
    mask,
    fragment_mask,
    *,
    filter_empty_tile,
):
    mask_u8 = np.asarray(mask)
    if mask_u8.ndim != 2:
        raise ValueError(f"expected 2D label mask, got shape={tuple(mask_u8.shape)}")
    if mask_u8.dtype != np.uint8:
        mask_u8 = np.clip(mask_u8, 0, 255).astype(np.uint8, copy=False)

    fragment_mask = np.asarray(fragment_mask)
    if fragment_mask.ndim != 2:
        raise ValueError(f"expected 2D fragment mask, got shape={tuple(fragment_mask.shape)}")
    if fragment_mask.shape != mask_u8.shape:
        raise ValueError(
            f"label/fragment mask shape mismatch: {tuple(mask_u8.shape)} vs {tuple(fragment_mask.shape)}"
        )

    bboxes = _component_bboxes(fragment_mask, connectivity=2)
    if int(bboxes.shape[0]) == 0:
        xyxys = extract_patch_coordinates(mask_u8, fragment_mask, filter_empty_tile=bool(filter_empty_tile))
        bbox_idx = np.full((int(xyxys.shape[0]),), -1, dtype=np.int32)
        return (
            {"mode": "full", "shape": tuple(mask_u8.shape), "mask": mask_u8},
            xyxys,
            bbox_idx,
        )

    mask_crops = []
    kept_bboxes = []
    xy_chunks = []
    bbox_chunks = []
    seen_windows = set()
    for y0, y1, x0, x1 in bboxes.tolist():
        y0 = int(y0)
        y1 = int(y1)
        x0 = int(x0)
        x1 = int(x1)
        if y1 <= y0 or x1 <= x0:
            continue
        mask_c = np.asarray(mask_u8[y0:y1, x0:x1], dtype=np.uint8).copy()
        fragment_mask_c = fragment_mask[y0:y1, x0:x1]
        xy_local = extract_patch_coordinates(mask_c, fragment_mask_c, filter_empty_tile=bool(filter_empty_tile))
        if int(xy_local.shape[0]) == 0:
            continue

        xy_global_rows = []
        for x1_l, y1_l, x2_l, y2_l in np.asarray(xy_local, dtype=np.int64).tolist():
            gx1 = int(x1_l) + int(x0)
            gy1 = int(y1_l) + int(y0)
            gx2 = int(x2_l) + int(x0)
            gy2 = int(y2_l) + int(y0)
            key = (gx1, gy1, gx2, gy2)
            if key in seen_windows:
                continue
            seen_windows.add(key)
            xy_global_rows.append([gx1, gy1, gx2, gy2])
        if len(xy_global_rows) == 0:
            continue

        local_bbox_idx = int(len(mask_crops))
        xy_global = np.asarray(xy_global_rows, dtype=np.int64)

        mask_crops.append(mask_c)
        kept_bboxes.append((int(y0), int(y1), int(x0), int(x1)))
        xy_chunks.append(xy_global)
        bbox_chunks.append(np.full((int(xy_global.shape[0]),), local_bbox_idx, dtype=np.int32))

    if len(xy_chunks) == 0:
        return (
            {"mode": "full", "shape": tuple(mask_u8.shape), "mask": mask_u8},
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

    xyxys = np.concatenate(xy_chunks, axis=0)
    bbox_idx = np.concatenate(bbox_chunks, axis=0)
    return (
        {
            "mode": "bboxes",
            "shape": tuple(mask_u8.shape),
            "bboxes": np.asarray(kept_bboxes, dtype=np.int32),
            "mask_crops": mask_crops,
        },
        xyxys,
        bbox_idx,
    )


def _mask_store_shape(mask_store):
    if isinstance(mask_store, np.ndarray):
        if mask_store.ndim < 2:
            raise ValueError(f"expected at least 2D mask array, got shape={tuple(mask_store.shape)}")
        return (int(mask_store.shape[0]), int(mask_store.shape[1]))
    if isinstance(mask_store, dict):
        shape = mask_store.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("mask store missing valid 'shape'")
        return int(shape[0]), int(shape[1])
    raise TypeError(f"unsupported mask store type: {type(mask_store).__name__}")


def _read_mask_patch(mask_store, *, y1, y2, x1, x2, bbox_index=None):
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    if y2 <= y1 or x2 <= x1:
        raise ValueError(f"invalid patch coords: {(x1, y1, x2, y2)}")

    if isinstance(mask_store, np.ndarray):
        return np.asarray(mask_store[y1:y2, x1:x2])

    if not isinstance(mask_store, dict):
        raise TypeError(f"unsupported mask store type: {type(mask_store).__name__}")

    mode = str(mask_store.get("mode", "full"))
    if mode == "full":
        if "mask" not in mask_store:
            raise ValueError("mask store mode='full' is missing key 'mask'")
        mask_arr = np.asarray(mask_store["mask"])
        return np.asarray(mask_arr[y1:y2, x1:x2])

    if mode != "bboxes":
        raise ValueError(f"unsupported mask store mode: {mode!r}")

    bboxes = np.asarray(mask_store.get("bboxes"))
    mask_crops = list(mask_store.get("mask_crops", []))
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("mask store mode='bboxes' requires bboxes shape (N, 4)")
    if int(bboxes.shape[0]) != int(len(mask_crops)):
        raise ValueError("mask store mode='bboxes' requires matching bboxes and mask_crops lengths")

    idx = None
    if bbox_index is not None:
        idx_i = int(bbox_index)
        if idx_i >= 0:
            idx = idx_i
    if idx is None:
        for i, bbox in enumerate(bboxes.tolist()):
            by0, by1, bx0, bx1 = [int(v) for v in bbox]
            if y1 >= by0 and y2 <= by1 and x1 >= bx0 and x2 <= bx1:
                idx = int(i)
                break
    if idx is None:
        raise ValueError(f"could not resolve bbox for patch coords {(x1, y1, x2, y2)}")
    if idx < 0 or idx >= int(len(mask_crops)):
        raise ValueError(f"bbox index out of range: {idx}")

    by0, by1, bx0, bx1 = [int(v) for v in bboxes[idx].tolist()]
    ly1 = int(y1 - by0)
    ly2 = int(y2 - by0)
    lx1 = int(x1 - bx0)
    lx2 = int(x2 - bx0)
    if ly1 < 0 or lx1 < 0 or ly2 > (by1 - by0) or lx2 > (bx1 - bx0):
        raise ValueError(
            f"patch {(x1, y1, x2, y2)} is out of bbox bounds {(bx0, by0, bx1, by1)}"
        )
    crop = np.asarray(mask_crops[idx])
    return np.asarray(crop[ly1:ly2, lx1:lx2])


def extract_patches_infer(image, fragment_mask, *, include_xyxys=True):
    images = []
    xyxys = []

    stride = CFG.stride
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue

                    windows_dict[(y1, y2, x1, x2)] = True
                    images.append(image[y1:y2, x1:x2])
                    if include_xyxys:
                        xyxys.append([x1, y1, x2, y2])
                    assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)

    return images, xyxys


def build_group_mappings(fragment_ids, segments_metadata, group_key="base_path"):
    segments_metadata = _require_dict(segments_metadata, name="segments_metadata")
    fragment_to_group_name = {}
    for fragment_id in fragment_ids:
        if fragment_id not in segments_metadata:
            raise KeyError(f"segments_metadata missing segment id: {fragment_id!r}")
        seg_meta = _require_dict(segments_metadata[fragment_id], name=f"segments_metadata[{fragment_id!r}]")
        if group_key not in seg_meta:
            raise KeyError(f"segment {fragment_id!r} missing required group key {group_key!r}")
        group_name = seg_meta[group_key]
        fragment_to_group_name[fragment_id] = str(group_name)

    group_names = sorted(set(fragment_to_group_name.values()))
    group_name_to_idx = {name: i for i, name in enumerate(group_names)}
    fragment_to_group_idx = {fid: group_name_to_idx[g] for fid, g in fragment_to_group_name.items()}
    return group_names, group_name_to_idx, fragment_to_group_idx


def extract_patches(image, mask, fragment_mask, *, include_xyxys, filter_empty_tile):
    images = []
    masks = []
    xyxys = []

    stride = CFG.stride
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if filter_empty_tile and _label_tile_is_empty(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]):
                continue
            if np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue

                    windows_dict[(y1, y2, x1, x2)] = True
                    images.append(image[y1:y2, x1:x2])
                    masks.append(mask[y1:y2, x1:x2, None])
                    if include_xyxys:
                        xyxys.append([x1, y1, x2, y2])
                    assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)

    return images, masks, xyxys


def _downsample_bool_mask_any(mask: np.ndarray, ds: int) -> np.ndarray:
    ds = max(1, int(ds))
    if mask is None:
        raise ValueError("mask is None")
    mask_bool = (mask > 0)
    h = int(mask_bool.shape[0])
    w = int(mask_bool.shape[1])
    ds_h = (h + ds - 1) // ds
    ds_w = (w + ds - 1) // ds
    pad_h = int(ds_h * ds - h)
    pad_w = int(ds_w * ds - w)
    if pad_h or pad_w:
        mask_bool = np.pad(mask_bool, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    mask_bool = mask_bool.reshape(ds_h, ds, ds_w, ds)
    return mask_bool.any(axis=(1, 3))


def _mask_bbox_downsample(mask: np.ndarray, ds: int) -> tuple[int, int, int, int] | None:
    mask_ds = _downsample_bool_mask_any(mask, int(ds))
    if not mask_ds.any():
        return None
    ys, xs = np.where(mask_ds)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return (y0, y1, x0, x1)


def _mask_border(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool is None:
        raise ValueError("mask_bool is None")
    mask_bool = mask_bool.astype(bool, copy=False)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)
    thickness = 5
    eroded = ndimage.binary_erosion(
        mask_bool,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
        iterations=thickness,
    )
    return mask_bool & ~eroded


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug


def _resize_label_for_loss(label, cfg):
    if not torch.is_floating_point(label):
        label = label.float()
    else:
        label = label.to(dtype=torch.float32)
    if label.numel() > 0 and float(label.max().detach().item()) > 1.0:
        label = label / 255.0
    return F.interpolate(label.unsqueeze(0), (cfg.size // 4, cfg.size // 4)).squeeze(0)


def _apply_joint_transform(transform, image, label, cfg):
    if transform is None:
        return image, label
    data = transform(image=image, mask=label)
    image = data["image"].unsqueeze(0)
    label = _resize_label_for_loss(data["mask"], cfg)
    return image, label


def _apply_image_transform(transform, image):
    if transform is None:
        return image
    data = transform(image=image)
    return data["image"].unsqueeze(0)


def _xy_to_bounds(xy):
    x1, y1, x2, y2 = [int(v) for v in xy]
    return x1, y1, x2, y2


def _fourth_augment(image, cfg):
    in_chans = int(cfg.in_chans)
    if in_chans <= 0:
        raise ValueError(f"in_chans must be > 0 for fourth augment, got {in_chans}")
    if image.shape[-1] != in_chans:
        raise ValueError(
            f"fourth augment expected image with {in_chans} channels, got shape {tuple(image.shape)}"
        )

    min_crop_ratio = float(cfg.fourth_augment_min_crop_ratio)
    max_crop_ratio = float(cfg.fourth_augment_max_crop_ratio)
    if not (0.0 < min_crop_ratio <= max_crop_ratio <= 1.0):
        raise ValueError(
            "fourth augment crop ratios must satisfy 0 < min_crop_ratio <= max_crop_ratio <= 1, "
            f"got min={min_crop_ratio}, max={max_crop_ratio}"
        )

    image_tmp = np.zeros_like(image)
    min_crop = max(1, int(np.ceil(in_chans * min_crop_ratio)))
    max_crop = max(1, int(np.floor(in_chans * max_crop_ratio)))
    if min_crop > max_crop:
        raise ValueError(
            f"invalid fourth augment crop window for in_chans={in_chans}: min_crop={min_crop}, max_crop={max_crop}"
        )
    cropping_num = random.randint(min_crop, max_crop)

    max_start = max(0, in_chans - cropping_num)
    start_idx = random.randint(0, max_start)
    crop_indices = np.arange(start_idx, start_idx + cropping_num)

    start_paste_idx = random.randint(0, max_start)

    tmp = np.arange(start_paste_idx, start_paste_idx + cropping_num)
    np.random.shuffle(tmp)

    cutout_max_count = int(cfg.fourth_augment_cutout_max_count)
    if cutout_max_count < 0:
        raise ValueError(f"fourth_augment_cutout_max_count must be >= 0, got {cutout_max_count}")
    cutout_idx = random.randint(0, min(cutout_max_count, cropping_num))
    temporal_random_cutout_idx = tmp[:cutout_idx]

    image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

    cutout_p = float(cfg.fourth_augment_cutout_p)
    if not (0.0 <= cutout_p <= 1.0):
        raise ValueError(f"fourth_augment_cutout_p must be in [0, 1], got {cutout_p}")
    if random.random() < cutout_p:
        image_tmp[..., temporal_random_cutout_idx] = 0
    return image_tmp


def _maybe_fourth_augment(image, cfg):
    p = float(cfg.fourth_augment_p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"fourth_augment_p must be in [0, 1], got {p}")
    if random.random() < p:
        return _fourth_augment(image, cfg)
    return image


class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, groups=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.groups = groups

        self.transform = transform
        self.xyxys = xyxys
        self.rotate = CFG.rotate

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        group_id = 0
        if self.groups is not None:
            group_id = int(self.groups[idx])
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy = self.xyxys[idx]
            image, label = _apply_joint_transform(self.transform, image, label, self.cfg)
            return image, label, xy, group_id
        else:
            image = self.images[idx]
            label = self.labels[idx]
            # 3d rotate
            # image=image.transpose(2,1,0)#(c,w,h)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,h,w)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,w,h)
            # image=image.transpose(2,1,0)#(h,w,c)

            image = _maybe_fourth_augment(image, self.cfg)
            image, label = _apply_joint_transform(self.transform, image, label, self.cfg)
            return image, label, group_id


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        image = _apply_image_transform(self.transform, image)
        return image, xy


def _flatten_segment_patch_index(
    xyxys_by_segment,
    groups_by_segment=None,
    sample_bbox_indices_by_segment=None,
):
    xyxys_by_segment = _require_dict(xyxys_by_segment, name="xyxys_by_segment")
    if groups_by_segment is not None:
        groups_by_segment = _require_dict(groups_by_segment, name="groups_by_segment")
    if sample_bbox_indices_by_segment is not None:
        sample_bbox_indices_by_segment = _require_dict(
            sample_bbox_indices_by_segment,
            name="sample_bbox_indices_by_segment",
        )

    segment_ids = []
    seg_indices = []
    xy_chunks = []
    group_chunks = []
    bbox_idx_chunks = []

    for segment_id, xyxys in xyxys_by_segment.items():
        xy = np.asarray(xyxys, dtype=np.int64)
        if xy.ndim != 2 or xy.shape[1] != 4:
            raise ValueError(
                f"{segment_id}: expected xyxys shape (N, 4), got {tuple(xy.shape)}"
            )
        if xy.shape[0] == 0:
            continue

        seg_idx = len(segment_ids)
        segment_ids.append(str(segment_id))
        xy_chunks.append(xy)
        seg_indices.append(np.full((xy.shape[0],), seg_idx, dtype=np.int32))
        if sample_bbox_indices_by_segment is not None:
            seg_id_key = str(segment_id)
            if seg_id_key not in sample_bbox_indices_by_segment:
                raise KeyError(f"sample_bbox_indices_by_segment missing segment id: {seg_id_key!r}")
            bbox_idx = np.asarray(sample_bbox_indices_by_segment[seg_id_key], dtype=np.int32).reshape(-1)
            if bbox_idx.shape[0] != xy.shape[0]:
                raise ValueError(
                    f"{segment_id}: sample_bbox_indices length {bbox_idx.shape[0]} "
                    f"does not match xyxys length {xy.shape[0]}"
                )
            bbox_idx_chunks.append(bbox_idx)
        if groups_by_segment is not None:
            seg_id_key = str(segment_id)
            if seg_id_key not in groups_by_segment:
                raise KeyError(f"groups_by_segment missing segment id: {seg_id_key!r}")
            group_id = int(groups_by_segment[seg_id_key])
            group_chunks.append(np.full((xy.shape[0],), group_id, dtype=np.int64))

    if len(segment_ids) == 0:
        empty_xy = np.zeros((0, 4), dtype=np.int64)
        empty_seg = np.zeros((0,), dtype=np.int32)
        empty_bbox_idx = np.zeros((0,), dtype=np.int32) if sample_bbox_indices_by_segment is not None else None
        if groups_by_segment is None:
            return [], empty_seg, empty_xy, None, empty_bbox_idx
        return [], empty_seg, empty_xy, np.zeros((0,), dtype=np.int64), empty_bbox_idx

    flat_xy = np.concatenate(xy_chunks, axis=0)
    flat_seg = np.concatenate(seg_indices, axis=0)
    flat_bbox_idx = None
    if sample_bbox_indices_by_segment is not None:
        flat_bbox_idx = np.concatenate(bbox_idx_chunks, axis=0)
    if groups_by_segment is None:
        return segment_ids, flat_seg, flat_xy, None, flat_bbox_idx
    flat_groups = np.concatenate(group_chunks, axis=0)
    return segment_ids, flat_seg, flat_xy, flat_groups, flat_bbox_idx


def _init_flat_segment_index(
    xyxys_by_segment,
    groups_by_segment,
    dataset_name,
    *,
    sample_bbox_indices_by_segment=None,
):
    segment_ids, sample_segment_indices, sample_xyxys, sample_groups, sample_bbox_indices = _flatten_segment_patch_index(
        xyxys_by_segment,
        groups_by_segment,
        sample_bbox_indices_by_segment=sample_bbox_indices_by_segment,
    )
    if sample_xyxys.shape[0] == 0:
        raise ValueError(f"{dataset_name} has no samples")
    return segment_ids, sample_segment_indices, sample_xyxys, sample_groups, sample_bbox_indices


def _validate_segment_data(segment_ids, volumes, masks=None):
    for segment_id in segment_ids:
        if segment_id not in volumes:
            raise ValueError(f"Missing volume for segment={segment_id}")
        if masks is not None and segment_id not in masks:
            raise ValueError(f"Missing mask for segment={segment_id}")


class LazyZarrTrainDataset(Dataset):
    def __init__(
        self,
        volumes_by_segment,
        masks_by_segment,
        xyxys_by_segment,
        groups_by_segment,
        cfg,
        transform=None,
        sample_bbox_indices_by_segment=None,
    ):
        self.volumes = dict(_require_dict(volumes_by_segment, name="volumes_by_segment"))
        self.masks = dict(_require_dict(masks_by_segment, name="masks_by_segment"))
        self.cfg = cfg
        self.transform = transform
        self.rotate = CFG.rotate

        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            self.sample_groups,
            self.sample_bbox_indices,
        ) = _init_flat_segment_index(
            xyxys_by_segment,
            groups_by_segment,
            "LazyZarrTrainDataset",
            sample_bbox_indices_by_segment=sample_bbox_indices_by_segment,
        )
        _validate_segment_data(self.segment_ids, self.volumes, self.masks)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        x1, y1, x2, y2 = _xy_to_bounds(self.sample_xyxys[idx])
        group_id = int(self.sample_groups[idx])
        bbox_idx = int(self.sample_bbox_indices[idx]) if self.sample_bbox_indices is not None else None

        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        label = _read_mask_patch(self.masks[segment_id], y1=y1, y2=y2, x1=x1, x2=x2, bbox_index=bbox_idx)[..., None]
        image = _maybe_fourth_augment(image, self.cfg)
        image, label = _apply_joint_transform(self.transform, image, label, self.cfg)

        return image, label, group_id


class LazyZarrXyLabelDataset(Dataset):
    def __init__(
        self,
        volumes_by_segment,
        masks_by_segment,
        xyxys_by_segment,
        groups_by_segment,
        cfg,
        transform=None,
        sample_bbox_indices_by_segment=None,
    ):
        self.volumes = dict(_require_dict(volumes_by_segment, name="volumes_by_segment"))
        self.masks = dict(_require_dict(masks_by_segment, name="masks_by_segment"))
        self.cfg = cfg
        self.transform = transform
        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            self.sample_groups,
            self.sample_bbox_indices,
        ) = _init_flat_segment_index(
            xyxys_by_segment,
            groups_by_segment,
            "LazyZarrXyLabelDataset",
            sample_bbox_indices_by_segment=sample_bbox_indices_by_segment,
        )
        _validate_segment_data(self.segment_ids, self.volumes, self.masks)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        xy = self.sample_xyxys[idx]
        x1, y1, x2, y2 = _xy_to_bounds(xy)
        group_id = int(self.sample_groups[idx])
        bbox_idx = int(self.sample_bbox_indices[idx]) if self.sample_bbox_indices is not None else None

        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        label = _read_mask_patch(self.masks[segment_id], y1=y1, y2=y2, x1=x1, x2=x2, bbox_index=bbox_idx)[..., None]
        image, label = _apply_joint_transform(self.transform, image, label, self.cfg)
        return image, label, xy, group_id


class LazyZarrXyOnlyDataset(Dataset):
    def __init__(
        self,
        volumes_by_segment,
        xyxys_by_segment,
        cfg,
        transform=None,
    ):
        self.volumes = dict(_require_dict(volumes_by_segment, name="volumes_by_segment"))
        self.cfg = cfg
        self.transform = transform
        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            _,
            _,
        ) = _init_flat_segment_index(
            xyxys_by_segment,
            groups_by_segment=None,
            dataset_name="LazyZarrXyOnlyDataset",
            sample_bbox_indices_by_segment=None,
        )
        _validate_segment_data(self.segment_ids, self.volumes)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        xy = self.sample_xyxys[idx]
        x1, y1, x2, y2 = _xy_to_bounds(xy)
        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        image = _apply_image_transform(self.transform, image)
        return image, xy
