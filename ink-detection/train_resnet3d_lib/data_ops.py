from train_resnet3d_lib.config import CFG, REVERSE_LAYER_FRAGMENT_IDS_FALLBACK

import os.path as osp

import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from scipy import ndimage
import PIL.Image
import zarr


def _read_gray(path):
    if not osp.exists(path):
        return None

    # OpenCV can emit noisy libpng warnings (e.g. "chunk data is too large") for some PNGs.
    # Prefer PIL for PNGs to keep logs clean; fall back to OpenCV if PIL fails.
    if path.lower().endswith(".png"):
        try:
            with PIL.Image.open(path) as im:
                return np.array(im.convert("L"))
        except Exception:
            pass

    try:
        img = cv2.imread(path, 0)
        if img is not None:
            return img
    except cv2.error:
        pass

    try:
        with PIL.Image.open(path) as im:
            return np.array(im.convert("L"))
    except Exception as e:
        raise RuntimeError(f"Could not read image: {path}. Error: {e}") from e


def read_image_layers(
    fragment_id,
    start_idx=15,
    end_idx=45,
    *,
    layer_range=None,
):
    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))
    layers_dir = osp.join(dataset_root, fragment_id, "layers")
    layer_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _iter_layer_paths(layer_idx):
        # Support 00.tif, 000.tif, 0000.tif, etc.
        for fmt in (f"{layer_idx:02}", f"{layer_idx:03}", f"{layer_idx:04}", str(layer_idx)):
            for ext in layer_exts:
                yield osp.join(layers_dir, f"{fmt}{ext}")

    if layer_range is not None:
        start_idx, end_idx = layer_range

    idxs = list(range(int(start_idx), int(end_idx)))
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

    layer_read_workers = int(getattr(CFG, "layer_read_workers", 1) or 1)
    layer_read_workers = max(1, min(layer_read_workers, len(idxs)))

    if "frag" in fragment_id:
        images_list = []
        for i in idxs:
            image = None
            for image_path in _iter_layer_paths(i):
                image = _read_gray(image_path)
                if image is not None:
                    break
            if image is None:
                raise FileNotFoundError(
                    f"Could not read layer for {fragment_id}: {layers_dir}/{i}.[tif|tiff|png|jpg|jpeg]"
                )

            pad0 = (256 - image.shape[0] % 256)
            pad1 = (256 - image.shape[1] % 256)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            np.clip(image, 0, 200, out=image)
            if fragment_id == '20230827161846':
                image = cv2.flip(image, 0)
            images_list.append(image)

        images = np.stack(images_list, axis=2)
        del images_list
        return images

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
    if fragment_id == '20230827161846':
        first = cv2.flip(first, 0)
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
        if fragment_id == '20230827161846':
            img = cv2.flip(img, 0)
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
    start_idx=15,
    end_idx=45,
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
            start_idx=start_idx,
            end_idx=end_idx,
            layer_range=layer_range,
        )

    if reverse_layers or fragment_id in REVERSE_LAYER_FRAGMENT_IDS_FALLBACK:
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
    if fragment_id == '20230827161846':
        fragment_mask = cv2.flip(fragment_mask, 0)

    def _assert_bottom_right_pad_compatible(a_name, a_hw, b_name, b_hw, multiple):
        _assert_bottom_right_pad_compatible_global(fragment_id, a_name, a_hw, b_name, b_hw, multiple)
    if "frag" not in fragment_id:
        pad_multiple = 256
        _assert_bottom_right_pad_compatible("image", images.shape[:2], "label", mask.shape[:2], pad_multiple)
        _assert_bottom_right_pad_compatible("image", images.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)
        _assert_bottom_right_pad_compatible("label", mask.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)

    if "frag" in fragment_id:
        pad0 = max(0, images.shape[0] * 2 - fragment_mask.shape[0])
        pad1 = max(0, images.shape[1] * 2 - fragment_mask.shape[1])
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        fragment_mask_padded = np.zeros((images.shape[0], images.shape[1]), dtype=fragment_mask.dtype)
        h = min(fragment_mask.shape[0], fragment_mask_padded.shape[0])
        w = min(fragment_mask.shape[1], fragment_mask_padded.shape[1])
        fragment_mask_padded[:h, :w] = fragment_mask[:h, :w]
        fragment_mask = fragment_mask_padded
        del fragment_mask_padded

    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1] // 2, fragment_mask.shape[0] // 2), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA)

    target_h = min(images.shape[0], mask.shape[0], fragment_mask.shape[0])
    target_w = min(images.shape[1], mask.shape[1], fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"{fragment_id}: empty shapes after alignment (images={images.shape}, label={mask.shape}, mask={fragment_mask.shape})"
        )

    images = images[:target_h, :target_w]
    mask = mask[:target_h, :target_w]
    fragment_mask = fragment_mask[:target_h, :target_w]

    mask = mask.astype('float32')
    mask /= 255
    if images.shape[0] != mask.shape[0] or images.shape[1] != mask.shape[1]:
        raise ValueError(f"{fragment_id}: label shape {mask.shape} does not match image shape {images.shape[:2]}")
    return images, mask, fragment_mask


def _assert_bottom_right_pad_compatible_global(fragment_id, a_name, a_hw, b_name, b_hw, multiple):
    a_h, a_w = [int(x) for x in a_hw]
    b_h, b_w = [int(x) for x in b_hw]

    def _check_dim(dim_name, a_dim, b_dim):
        small = min(a_dim, b_dim)
        big = max(a_dim, b_dim)
        padded = ((small + multiple - 1) // multiple) * multiple
        allowed = {small, padded}
        if small % multiple == 0:
            allowed.add(small + multiple)  # supports the legacy "always pad one block" variant

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

    if reverse_layers or fragment_id in REVERSE_LAYER_FRAGMENT_IDS_FALLBACK:
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
    if fragment_id == '20230827161846':
        fragment_mask = cv2.flip(fragment_mask, 0)

    if "frag" not in fragment_id:
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


def _compute_selected_layer_indices(fragment_id, layer_range=None):
    start_idx = 15
    end_idx = 45
    if layer_range is not None:
        start_idx, end_idx = layer_range

    idxs = list(range(int(start_idx), int(end_idx)))
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


def _expand_relative_path_candidates(path: str, dataset_root: str) -> list[str]:
    path = str(path)
    if osp.isabs(path):
        return [path]
    out = [path]
    if dataset_root:
        out.append(osp.join(dataset_root, path))
    return out


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


def resolve_segment_zarr_path(fragment_id, seg_meta):
    fragment_id = str(fragment_id)
    seg_meta = seg_meta or {}
    dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))

    segment_roots = [osp.join(dataset_root, fragment_id)]
    original_path = seg_meta.get("original_path")
    if original_path:
        segment_roots.extend(_expand_relative_path_candidates(str(original_path), dataset_root))

    zarr_candidate_paths = []
    explicit_keys = (
        "zarr_path",
        "volume_zarr",
        "surface_volume_zarr",
        "layers_zarr",
        "zarr",
        "volume_path",
    )
    for key in explicit_keys:
        val = seg_meta.get(key)
        if not val:
            continue
        for candidate in _expand_relative_path_candidates(str(val), dataset_root):
            zarr_candidate_paths.append(candidate)
            for root in segment_roots:
                zarr_candidate_paths.append(osp.join(root, str(val)))

    implicit_names = ("layers.zarr", "surface_volume.zarr", "volume.zarr", "layers")
    for root in segment_roots:
        for name in implicit_names:
            zarr_candidate_paths.append(osp.join(root, name))

    seen = set()
    ordered = []
    for candidate in zarr_candidate_paths:
        c = osp.normpath(candidate)
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)

    for candidate in ordered:
        if _looks_like_zarr_store(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve zarr volume path for segment={fragment_id}. "
        f"Tried: {ordered[:12]}{' ...' if len(ordered) > 12 else ''}"
    )


def _ensure_zarr_v2():
    ver = str(getattr(zarr, "__version__", "") or "")
    try:
        major = int(ver.split(".")[0])
    except Exception:
        major = None
    if major is not None and major >= 3:
        raise RuntimeError(
            f"zarr backend requires zarr v2, found version {ver!r}. "
            "Install a v2 release (e.g., `zarr<3`)."
        )


class ZarrSegmentVolume:
    def __init__(
        self,
        fragment_id,
        seg_meta,
        *,
        layer_range=None,
        reverse_layers=False,
    ):
        _ensure_zarr_v2()

        self.fragment_id = str(fragment_id)
        self.seg_meta = seg_meta or {}
        self.path = resolve_segment_zarr_path(self.fragment_id, self.seg_meta)
        self.is_frag = "frag" in self.fragment_id
        self.flip_vertical = self.fragment_id == "20230827161846"

        idxs = _compute_selected_layer_indices(self.fragment_id, layer_range=layer_range)
        if reverse_layers or self.fragment_id in REVERSE_LAYER_FRAGMENT_IDS_FALLBACK:
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
        if self.is_frag:
            self._out_h = int(self._padded_h // 2)
            self._out_w = int(self._padded_w // 2)
        else:
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

        # Group-like store: prefer OME-Zarr's "0"; otherwise, use the only array key.
        if "0" in root:
            return root["0"]
        array_keys = list(root.array_keys()) if hasattr(root, "array_keys") else []
        if len(array_keys) == 1:
            return root[array_keys[0]]
        raise ValueError(
            f"Expected a 3D zarr array at {path}, got group with array keys={array_keys}"
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

        if max(layer_indices) >= int(n_layers):
            # Support 1-based layer ranges mapped onto 0-based zarr depth.
            if min(layer_indices) >= 1 and (max(layer_indices) - 1) < int(n_layers):
                layer_indices = [int(i - 1) for i in layer_indices]
            else:
                raise ValueError(
                    f"{self.fragment_id}: selected layer indices out of bounds for zarr depth={n_layers}. "
                    f"indices={layer_indices[:5]}...{layer_indices[-5:]}"
                )

        if min(layer_indices) < 0:
            raise ValueError(f"{self.fragment_id}: negative layer index in {layer_indices}")

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

    @property
    def base_shape(self):
        return (int(self._base_h), int(self._base_w))

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

    def _read_nonfrag_patch_unflipped(self, y1, y2, x1, x2):
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

    def _read_nonfrag_patch(self, y1, y2, x1, x2):
        if not self.flip_vertical:
            return self._read_nonfrag_patch_unflipped(y1, y2, x1, x2)

        src_y1 = int(self._base_h) - int(y2)
        src_y2 = int(self._base_h) - int(y1)
        patch = self._read_nonfrag_patch_unflipped(src_y1, src_y2, x1, x2)
        return np.flipud(patch)

    def _read_frag_patch_unflipped(self, y1, y2, x1, x2):
        # "frag" segments are resized by 2x downsampling after padding.
        py1 = int(y1) * 2
        py2 = int(y2) * 2
        px1 = int(x1) * 2
        px2 = int(x2) * 2

        raw = self._read_nonfrag_patch_unflipped(py1, py2, px1, px2)
        out_h = int(y2 - y1)
        out_w = int(x2 - x1)
        out = np.zeros((out_h, out_w, int(CFG.in_chans)), dtype=self._dtype)
        for c in range(int(CFG.in_chans)):
            out[:, :, c] = cv2.resize(raw[:, :, c], (out_w, out_h), interpolation=cv2.INTER_AREA)
        return out

    def _read_frag_patch(self, y1, y2, x1, x2):
        if not self.flip_vertical:
            return self._read_frag_patch_unflipped(y1, y2, x1, x2)

        src_y1 = int(self._out_h) - int(y2)
        src_y2 = int(self._out_h) - int(y1)
        patch = self._read_frag_patch_unflipped(src_y1, src_y2, x1, x2)
        return np.flipud(patch)

    def read_patch(self, y1, y2, x1, x2):
        y1 = int(y1)
        y2 = int(y2)
        x1 = int(x1)
        x2 = int(x2)
        if y2 <= y1 or x2 <= x1:
            raise ValueError(f"{self.fragment_id}: invalid patch coords {(x1, y1, x2, y2)}")

        if self.is_frag:
            patch = self._read_frag_patch(y1, y2, x1, x2)
        else:
            patch = self._read_nonfrag_patch(y1, y2, x1, x2)

        np.clip(patch, 0, 200, out=patch)
        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8, copy=False)

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
    is_frag=False,
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
    if str(fragment_id) == "20230827161846":
        fragment_mask = cv2.flip(fragment_mask, 0)

    if not bool(is_frag):
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

    if bool(is_frag):
        pad0 = max(0, image_h * 2 - fragment_mask.shape[0])
        pad1 = max(0, image_w * 2 - fragment_mask.shape[1])
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        fragment_mask = cv2.resize(
            fragment_mask,
            (fragment_mask.shape[1] // 2, fragment_mask.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA)
    else:
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

    mask = mask[:target_h, :target_w].astype("float32")
    mask /= 255.0
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
    if str(fragment_id) == "20230827161846":
        fragment_mask = cv2.flip(fragment_mask, 0)

    if "frag" not in str(fragment_id):
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
            if filter_empty_tile and mask is not None and np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.01):
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
    fragment_to_group_name = {}
    for fragment_id in fragment_ids:
        seg_meta = (segments_metadata or {}).get(fragment_id, {}) or {}
        group_name = seg_meta.get(group_key)
        if group_name is None:
            group_name = seg_meta.get("base_path", fragment_id)
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
            if filter_empty_tile and np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.01):
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


def _fourth_augment(image, in_chans):
    image_tmp = np.zeros_like(image)
    max_crop = min(62, int(in_chans))
    min_crop = min(56, max_crop)
    if min_crop <= 0:
        return image
    cropping_num = random.randint(min_crop, max_crop)

    max_start = max(0, int(in_chans) - cropping_num)
    start_idx = random.randint(0, max_start)
    crop_indices = np.arange(start_idx, start_idx + cropping_num)

    start_paste_idx = random.randint(0, max_start)

    tmp = np.arange(start_paste_idx, start_paste_idx + cropping_num)
    np.random.shuffle(tmp)

    cutout_idx = random.randint(0, 2)
    temporal_random_cutout_idx = tmp[:cutout_idx]

    image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

    if random.random() > 0.4:
        image_tmp[..., temporal_random_cutout_idx] = 0
    return image_tmp


def _maybe_fourth_augment(image, in_chans):
    if random.random() > 0.4:
        return _fourth_augment(image, in_chans)
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

    def cubeTranslate(self, y):
        x = np.random.uniform(0, 1, 4).reshape(2, 2)
        x[x < .4] = 0
        x[x > .633] = 2
        x[(x > .4) & (x < .633)] = 1
        mask = cv2.resize(x, (x.shape[1] * 64, x.shape[0] * 64), interpolation=cv2.INTER_AREA)

        x = np.zeros((self.cfg.size, self.cfg.size, self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x = np.where(np.repeat((mask == 0).reshape(self.cfg.size, self.cfg.size, 1), self.cfg.in_chans, axis=2), y[:, :, i:self.cfg.in_chans + i], x)
        return x

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

            image = _maybe_fourth_augment(image, self.cfg.in_chans)
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


def _flatten_segment_patch_index(xyxys_by_segment, groups_by_segment=None):
    segment_ids = []
    seg_indices = []
    xy_chunks = []
    group_chunks = []

    for segment_id, xyxys in (xyxys_by_segment or {}).items():
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
        if groups_by_segment is not None:
            group_id = int(groups_by_segment.get(str(segment_id), 0))
            group_chunks.append(np.full((xy.shape[0],), group_id, dtype=np.int64))

    if len(segment_ids) == 0:
        empty_xy = np.zeros((0, 4), dtype=np.int64)
        empty_seg = np.zeros((0,), dtype=np.int32)
        if groups_by_segment is None:
            return [], empty_seg, empty_xy, None
        return [], empty_seg, empty_xy, np.zeros((0,), dtype=np.int64)

    flat_xy = np.concatenate(xy_chunks, axis=0)
    flat_seg = np.concatenate(seg_indices, axis=0)
    if groups_by_segment is None:
        return segment_ids, flat_seg, flat_xy, None
    flat_groups = np.concatenate(group_chunks, axis=0)
    return segment_ids, flat_seg, flat_xy, flat_groups


def _init_flat_segment_index(xyxys_by_segment, groups_by_segment, dataset_name):
    segment_ids, sample_segment_indices, sample_xyxys, sample_groups = _flatten_segment_patch_index(
        xyxys_by_segment,
        groups_by_segment,
    )
    if sample_xyxys.shape[0] == 0:
        raise ValueError(f"{dataset_name} has no samples")
    return segment_ids, sample_segment_indices, sample_xyxys, sample_groups


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
    ):
        self.volumes = dict(volumes_by_segment or {})
        self.masks = dict(masks_by_segment or {})
        self.cfg = cfg
        self.transform = transform
        self.rotate = CFG.rotate

        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            self.sample_groups,
        ) = _init_flat_segment_index(xyxys_by_segment, groups_by_segment, "LazyZarrTrainDataset")
        _validate_segment_data(self.segment_ids, self.volumes, self.masks)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        x1, y1, x2, y2 = _xy_to_bounds(self.sample_xyxys[idx])
        group_id = int(self.sample_groups[idx])

        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        label = self.masks[segment_id][y1:y2, x1:x2, None]
        image = _maybe_fourth_augment(image, self.cfg.in_chans)
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
    ):
        self.volumes = dict(volumes_by_segment or {})
        self.masks = dict(masks_by_segment or {})
        self.cfg = cfg
        self.transform = transform
        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            self.sample_groups,
        ) = _init_flat_segment_index(xyxys_by_segment, groups_by_segment, "LazyZarrXyLabelDataset")
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

        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        label = self.masks[segment_id][y1:y2, x1:x2, None]
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
        self.volumes = dict(volumes_by_segment or {})
        self.cfg = cfg
        self.transform = transform
        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            _,
        ) = _init_flat_segment_index(xyxys_by_segment, groups_by_segment=None, dataset_name="LazyZarrXyOnlyDataset")
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
