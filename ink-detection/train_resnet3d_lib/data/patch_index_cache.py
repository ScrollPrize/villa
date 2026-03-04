import hashlib
import json
import os
import os.path as osp
import time

import numpy as np

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data.patching import _build_mask_store_and_patch_index


_PATCH_INDEX_CACHE_SCHEMA_VERSION = 1
_PATCH_INDEX_CACHE_REQUIRED_META_KEYS = {
    "schema_version",
    "fragment_id",
    "split",
    "filter_empty_tile",
    "size",
    "tile_size",
    "stride",
    "label_suffix",
    "mask_suffix",
    "mask_shape",
    "mask_store_mode",
}
_PATCH_INDEX_CACHE_HASH_META_KEYS = {
    "label_sha256",
    "fragment_mask_sha256",
}


def _require_2d_array(value, *, context):
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{context}: expected 2D array, got shape={tuple(arr.shape)}")
    return arr


def _label_mask_uint8(mask, *, context):
    mask_arr = _require_2d_array(mask, context=context)
    if mask_arr.dtype == np.uint8:
        return mask_arr
    if np.issubdtype(mask_arr.dtype, np.integer) or np.issubdtype(mask_arr.dtype, np.floating):
        return np.clip(mask_arr, 0, 255).astype(np.uint8, copy=False)
    raise TypeError(f"{context}: expected integer/float label mask dtype, got {mask_arr.dtype}")


def _sha256_array(arr, *, context):
    arr_2d = _require_2d_array(arr, context=context)
    arr_c = np.ascontiguousarray(arr_2d)
    hasher = hashlib.sha256()
    hasher.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    hasher.update(str(arr_c.dtype).encode("utf-8"))
    hasher.update(arr_c.tobytes(order="C"))
    return hasher.hexdigest()


def _cache_segment_slug(segment_id, *, max_len=80):
    raw = str(segment_id)
    normalized = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw)
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = "segment"
    return normalized[:max_len]


def _resolve_patch_index_cache_dir():
    cache_dir_value = getattr(CFG, "dataset_cache_dir", "./dataset_cache/train_resnet3d_patch_index")
    if not isinstance(cache_dir_value, str):
        raise TypeError(f"CFG.dataset_cache_dir must be a string, got {type(cache_dir_value).__name__}")
    cache_dir = cache_dir_value.strip()
    if not cache_dir:
        raise ValueError("CFG.dataset_cache_dir must be a non-empty string")
    if not osp.isabs(cache_dir):
        cache_dir = osp.abspath(cache_dir)
    return cache_dir


def _build_patch_index_cache_params(
    *,
    fragment_id,
    split_name,
    filter_empty_tile,
    label_suffix,
    mask_suffix,
):
    split = str(split_name)
    if split not in {"train", "val"}:
        raise ValueError(f"split_name must be 'train' or 'val', got {split_name!r}")
    return {
        "schema_version": int(_PATCH_INDEX_CACHE_SCHEMA_VERSION),
        "fragment_id": str(fragment_id),
        "split": split,
        "filter_empty_tile": bool(filter_empty_tile),
        "size": int(CFG.size),
        "tile_size": int(CFG.tile_size),
        "stride": int(CFG.stride),
        "label_suffix": str(label_suffix),
        "mask_suffix": str(mask_suffix),
    }


def _build_patch_index_expected_metadata(params, mask, fragment_mask, *, check_hash):
    mask_arr = _require_2d_array(mask, context="patch-index cache label")
    fragment_mask_arr = _require_2d_array(fragment_mask, context="patch-index cache fragment mask")
    if mask_arr.shape != fragment_mask_arr.shape:
        raise ValueError(
            "patch-index cache label/mask shape mismatch: "
            f"{tuple(mask_arr.shape)} vs {tuple(fragment_mask_arr.shape)}"
        )
    metadata = dict(params)
    metadata["mask_shape"] = [int(mask_arr.shape[0]), int(mask_arr.shape[1])]
    if bool(check_hash):
        metadata["label_sha256"] = _sha256_array(mask_arr, context="patch-index cache label")
        metadata["fragment_mask_sha256"] = _sha256_array(
            fragment_mask_arr,
            context="patch-index cache fragment mask",
        )
    return metadata


def _patch_index_cache_file_path(cache_dir, params):
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()[:16]
    segment_slug = _cache_segment_slug(params["fragment_id"])
    filename = f"{segment_slug}__{params['split']}__{params_hash}.npz"
    return osp.join(cache_dir, filename)


def _atomic_save_npz(path, payload):
    directory = osp.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.remove(tmp_path)


def _serialize_patch_index_cache_payload(
    *,
    metadata,
    mask_store,
    xyxys,
    sample_bbox_indices,
):
    if not isinstance(mask_store, dict):
        raise TypeError(f"mask_store must be a dict, got {type(mask_store).__name__}")

    xy = np.asarray(xyxys, dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 4:
        raise ValueError(f"xyxys must have shape (N, 4), got {tuple(xy.shape)}")
    bbox_idx = np.asarray(sample_bbox_indices, dtype=np.int32).reshape(-1)
    if bbox_idx.shape[0] != xy.shape[0]:
        raise ValueError(
            f"sample_bbox_indices length {bbox_idx.shape[0]} does not match xyxys length {xy.shape[0]}"
        )

    mode = str(mask_store.get("mode"))
    payload_meta = dict(metadata)
    payload_meta["mask_store_mode"] = mode
    if mode == "full":
        if xy.shape[0] > 0 and np.any(bbox_idx != -1):
            raise ValueError("mask_store mode='full' requires all sample_bbox_indices to be -1")
        bboxes = np.zeros((0, 4), dtype=np.int32)
    elif mode == "bboxes":
        bboxes = np.asarray(mask_store.get("bboxes"), dtype=np.int32)
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(f"mask_store mode='bboxes' requires bboxes shape (N, 4), got {tuple(bboxes.shape)}")
        if xy.shape[0] > 0:
            if int(bboxes.shape[0]) <= 0:
                raise ValueError("mask_store mode='bboxes' has patches but no bboxes")
            if np.any(bbox_idx < 0):
                raise ValueError("mask_store mode='bboxes' requires non-negative sample_bbox_indices")
            max_idx = int(bbox_idx.max())
            if max_idx >= int(bboxes.shape[0]):
                raise ValueError(
                    f"sample_bbox_indices max={max_idx} is out of range for {int(bboxes.shape[0])} bboxes"
                )
    else:
        raise ValueError(f"unsupported mask_store mode: {mode!r}")

    return {
        "metadata_json": np.asarray(json.dumps(payload_meta, sort_keys=True)),
        "xyxys": xy,
        "sample_bbox_indices": bbox_idx,
        "bboxes": bboxes.astype(np.int32, copy=False),
    }


def _save_patch_index_cache(
    *,
    cache_path,
    metadata,
    mask_store,
    xyxys,
    sample_bbox_indices,
):
    payload = _serialize_patch_index_cache_payload(
        metadata=metadata,
        mask_store=mask_store,
        xyxys=xyxys,
        sample_bbox_indices=sample_bbox_indices,
    )
    _atomic_save_npz(cache_path, payload)


def _rebuild_mask_store_from_cached_bboxes(mask, bboxes):
    mask_u8 = _label_mask_uint8(mask, context="patch-index cache label")
    bboxes_i32 = np.asarray(bboxes, dtype=np.int32)
    if bboxes_i32.ndim != 2 or bboxes_i32.shape[1] != 4:
        raise ValueError(f"cached bboxes must have shape (N, 4), got {tuple(bboxes_i32.shape)}")

    mask_h = int(mask_u8.shape[0])
    mask_w = int(mask_u8.shape[1])
    mask_crops = []
    for bbox in bboxes_i32.tolist():
        y0, y1, x0, x1 = [int(v) for v in bbox]
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"invalid cached bbox: {(y0, y1, x0, x1)}")
        if y0 < 0 or x0 < 0 or y1 > mask_h or x1 > mask_w:
            raise ValueError(f"cached bbox out of bounds for mask shape {tuple(mask_u8.shape)}: {(y0, y1, x0, x1)}")
        mask_crops.append(np.asarray(mask_u8[y0:y1, x0:x1], dtype=np.uint8).copy())

    return {
        "mode": "bboxes",
        "shape": tuple(mask_u8.shape),
        "bboxes": bboxes_i32,
        "mask_crops": mask_crops,
    }


def _load_patch_index_cache(
    *,
    cache_path,
    expected_metadata,
    mask,
):
    if not osp.exists(cache_path):
        return None

    with np.load(cache_path, allow_pickle=False) as npz_data:
        required_keys = {"metadata_json", "xyxys", "sample_bbox_indices", "bboxes"}
        missing_keys = required_keys - set(npz_data.files)
        if missing_keys:
            raise ValueError(f"cache file {cache_path!r} is missing keys: {sorted(missing_keys)!r}")

        raw_metadata = npz_data["metadata_json"]
        metadata_text = str(np.asarray(raw_metadata).item())
        metadata = json.loads(metadata_text)
        if not isinstance(metadata, dict):
            raise TypeError(f"cache metadata must be an object, got {type(metadata).__name__}")

        missing_meta_keys = _PATCH_INDEX_CACHE_REQUIRED_META_KEYS - set(metadata.keys())
        if missing_meta_keys:
            raise ValueError(
                f"cache metadata {cache_path!r} is missing keys: {sorted(missing_meta_keys)!r}"
            )
        if all(key in expected_metadata for key in _PATCH_INDEX_CACHE_HASH_META_KEYS):
            missing_hash_keys = _PATCH_INDEX_CACHE_HASH_META_KEYS - set(metadata.keys())
            if missing_hash_keys:
                return None

        for key, expected_value in expected_metadata.items():
            if key not in metadata:
                return None
            if metadata[key] != expected_value:
                return None

        mode = str(metadata["mask_store_mode"])
        xyxys = np.asarray(npz_data["xyxys"], dtype=np.int64)
        if xyxys.ndim != 2 or xyxys.shape[1] != 4:
            raise ValueError(f"cached xyxys must have shape (N, 4), got {tuple(xyxys.shape)}")

        bbox_idx = np.asarray(npz_data["sample_bbox_indices"], dtype=np.int32).reshape(-1)
        if bbox_idx.shape[0] != xyxys.shape[0]:
            raise ValueError(
                f"cached sample_bbox_indices length {bbox_idx.shape[0]} does not match xyxys length {xyxys.shape[0]}"
            )

        if mode == "full":
            if xyxys.shape[0] > 0 and np.any(bbox_idx != -1):
                raise ValueError("cached mode='full' requires all sample_bbox_indices to be -1")
            mask_u8 = _label_mask_uint8(mask, context="patch-index cache label")
            mask_store = {"mode": "full", "shape": tuple(mask_u8.shape), "mask": mask_u8}
        elif mode == "bboxes":
            bboxes = np.asarray(npz_data["bboxes"], dtype=np.int32)
            if bboxes.ndim != 2 or bboxes.shape[1] != 4:
                raise ValueError(f"cached bboxes must have shape (N, 4), got {tuple(bboxes.shape)}")
            if xyxys.shape[0] > 0:
                if int(bboxes.shape[0]) <= 0:
                    raise ValueError("cached mode='bboxes' has patches but no bboxes")
                if np.any(bbox_idx < 0):
                    raise ValueError("cached mode='bboxes' requires non-negative sample_bbox_indices")
                max_idx = int(bbox_idx.max())
                if max_idx >= int(bboxes.shape[0]):
                    raise ValueError(
                        f"cached sample_bbox_indices max={max_idx} is out of range for {int(bboxes.shape[0])} bboxes"
                    )
            mask_store = _rebuild_mask_store_from_cached_bboxes(mask, bboxes)
        else:
            raise ValueError(f"unsupported cached mask_store_mode: {mode!r}")

    return mask_store, xyxys, bbox_idx


def build_mask_store_and_patch_index_cached(
    mask,
    fragment_mask,
    *,
    fragment_id,
    split_name,
    filter_empty_tile,
    label_suffix,
    mask_suffix,
):
    if not bool(getattr(CFG, "dataset_cache_enabled", True)):
        return _build_mask_store_and_patch_index(
            mask,
            fragment_mask,
            filter_empty_tile=bool(filter_empty_tile),
        )

    params = _build_patch_index_cache_params(
        fragment_id=fragment_id,
        split_name=split_name,
        filter_empty_tile=filter_empty_tile,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    check_hash = bool(getattr(CFG, "dataset_cache_check_hash", True))
    expected_metadata = _build_patch_index_expected_metadata(
        params,
        mask,
        fragment_mask,
        check_hash=check_hash,
    )
    cache_dir = _resolve_patch_index_cache_dir()
    cache_path = _patch_index_cache_file_path(cache_dir, params)
    cached = _load_patch_index_cache(
        cache_path=cache_path,
        expected_metadata=expected_metadata,
        mask=mask,
    )
    if cached is not None:
        log(f"patch-index cache hit split={split_name} segment={fragment_id} path={cache_path}")
        return cached

    log(f"patch-index cache miss split={split_name} segment={fragment_id} path={cache_path}")
    mask_store, xyxys, sample_bbox_indices = _build_mask_store_and_patch_index(
        mask,
        fragment_mask,
        filter_empty_tile=bool(filter_empty_tile),
    )
    _save_patch_index_cache(
        cache_path=cache_path,
        metadata=expected_metadata,
        mask_store=mask_store,
        xyxys=xyxys,
        sample_bbox_indices=sample_bbox_indices,
    )
    return mask_store, xyxys, sample_bbox_indices


def extract_infer_patch_coordinates_cached(
    *,
    fragment_mask,
    fragment_id,
    mask_suffix,
    split_name="val",
):
    split = str(split_name)
    if split not in {"train", "val"}:
        raise ValueError(f"split_name must be 'train' or 'val', got {split_name!r}")

    fragment_mask_u8 = _label_mask_uint8(fragment_mask, context="inference patch-index fragment mask")
    _mask_store, xyxys, _sample_bbox_indices = build_mask_store_and_patch_index_cached(
        fragment_mask_u8,
        fragment_mask_u8,
        fragment_id=str(fragment_id),
        split_name=split,
        filter_empty_tile=False,
        label_suffix="__stitch_infer__",
        mask_suffix=str(mask_suffix),
    )
    if xyxys.ndim != 2 or xyxys.shape[1] != 4:
        raise ValueError(f"inference cached xyxys must have shape (N, 4), got {tuple(xyxys.shape)}")
    return xyxys
