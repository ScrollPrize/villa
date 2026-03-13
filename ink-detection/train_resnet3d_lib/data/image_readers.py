import os.path as osp

import cv2
import numpy as np
import PIL.Image

from train_resnet3d_lib.config import CFG
from train_resnet3d_lib.data.segment_layout import (
    resolve_segment_label_path,
    resolve_segment_layers_dir,
    resolve_segment_mask_path,
)


def _require_dict(value, *, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def parse_layer_range_value(layer_range, *, context):
    start_idx, end_idx = [int(v) for v in layer_range]
    if end_idx <= start_idx:
        raise ValueError(f"{context} requires end_idx > start_idx")
    return start_idx, end_idx


def get_segment_meta(segments_metadata, segment_id):
    return dict(segments_metadata[segment_id])


def get_segment_layer_range(seg_meta, segment_id):
    layer_range = seg_meta["layer_range"]
    return parse_layer_range_value(
        layer_range,
        context=f"segments[{segment_id!r}].layer_range",
    )


def get_segment_reverse_layers(seg_meta, segment_id):
    reverse_layers = seg_meta["reverse_layers"]
    if not isinstance(reverse_layers, bool):
        raise TypeError(
            f"segments[{segment_id!r}].reverse_layers must be boolean, got {type(reverse_layers).__name__}"
        )
    return reverse_layers


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
    return parse_layer_range_value(
        layer_range,
        context=f"{fragment_id}: layer_range",
    )


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


def _clip_intensity_inplace(image):
    normalization_mode = str(getattr(CFG, "normalization_mode", "clip_max_div255")).strip().lower()
    if normalization_mode == "clip_max_div255":
        if CFG.max_clip_value is None:
            return
        np.clip(image, 0, int(CFG.max_clip_value), out=image)
        return
    if normalization_mode == "train_fold_fg_clip_zscore":
        return
    if normalization_mode == "train_fold_fg_clip_robust_zscore":
        return
    raise ValueError(f"Unsupported normalization_mode: {normalization_mode!r}")


def read_image_layers(
    fragment_id,
    *,
    layer_range,
    seg_meta=None,
):
    layers_dir = resolve_segment_layers_dir(fragment_id, seg_meta=seg_meta)
    layer_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _iter_layer_paths(layer_idx):
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
    pad0 = (256 - base_h % 256) % 256
    pad1 = (256 - base_w % 256) % 256
    out_h = base_h + pad0
    out_w = base_w + pad1

    images = np.zeros((out_h, out_w, len(idxs)), dtype=first.dtype)
    _clip_intensity_inplace(first)
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
        _clip_intensity_inplace(img)
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


def read_image_mask(
    fragment_id,
    *,
    layer_range=None,
    reverse_layers=False,
    label_suffix="",
    mask_suffix="",
    images=None,
    seg_meta=None,
):
    if images is None:
        images = read_image_layers(
            fragment_id,
            layer_range=layer_range,
            seg_meta=seg_meta,
        )

    if reverse_layers:
        images = images[:, :, ::-1]

    label_path = resolve_segment_label_path(fragment_id, seg_meta=seg_meta, suffix=label_suffix)
    mask_path = resolve_segment_mask_path(fragment_id, seg_meta=seg_meta, suffix=mask_suffix)
    mask = _read_gray(label_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read label for {fragment_id}: {label_path}")

    fragment_mask = _read_gray(mask_path)
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_path}")

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


def read_image_fragment_mask(
    fragment_id,
    *,
    layer_range=None,
    reverse_layers=False,
    mask_suffix="",
    images=None,
    seg_meta=None,
):
    if images is None:
        images = read_image_layers(
            fragment_id,
            layer_range=layer_range,
            seg_meta=seg_meta,
        )

    if reverse_layers:
        images = images[:, :, ::-1]

    mask_path = resolve_segment_mask_path(fragment_id, seg_meta=seg_meta, suffix=mask_suffix)
    fragment_mask = _read_gray(mask_path)
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_path}")

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


def read_label_and_fragment_mask_for_shape(
    fragment_id,
    image_shape_hw,
    *,
    label_suffix="",
    mask_suffix="",
    seg_meta=None,
):
    image_h = int(image_shape_hw[0])
    image_w = int(image_shape_hw[1])
    label_path = resolve_segment_label_path(fragment_id, seg_meta=seg_meta, suffix=label_suffix)
    mask_path = resolve_segment_mask_path(fragment_id, seg_meta=seg_meta, suffix=mask_suffix)

    mask = _read_gray(label_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read label for {fragment_id}: {label_path}")

    fragment_mask = _read_gray(mask_path)
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_path}")

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
    seg_meta=None,
):
    image_h = int(image_shape_hw[0])
    image_w = int(image_shape_hw[1])
    mask_path = resolve_segment_mask_path(fragment_id, seg_meta=seg_meta, suffix=mask_suffix)
    fragment_mask = _read_gray(mask_path)
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_path}")

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
