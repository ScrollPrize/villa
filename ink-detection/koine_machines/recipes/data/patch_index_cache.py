from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

_PATCH_INDEX_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedPatchIndex:
    xyxys: np.ndarray
    bbox_rows: np.ndarray
    sample_bbox_indices: np.ndarray


def _cache_segment_slug(segment_id: str, *, max_len: int = 80) -> str:
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", str(segment_id).strip()).strip("._-")
    if not raw:
        raw = "segment"
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{raw[: max_len - 9]}-{digest}"


def _cache_metadata(
    *,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    mask_names,
    label_artifact: str = "",
    mask_artifacts=(),
) -> dict[str, object]:
    return {
        "schema_version": int(_PATCH_INDEX_CACHE_SCHEMA_VERSION),
        "segment_id": str(segment_id),
        "split": str(split_name),
        "patch_size": int(patch_size),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "label_suffix": str(label_suffix),
        "mask_suffix": str(mask_suffix),
        "mask_names": [str(mask_name) for mask_name in tuple(mask_names or ())],
        "label_artifact": str(label_artifact),
        "mask_artifacts": [str(mask_artifact) for mask_artifact in tuple(mask_artifacts or ())],
    }


def _cache_file_path(cache_dir: str | Path, *, metadata: dict[str, object]) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    slug = _cache_segment_slug(str(metadata["segment_id"]))
    return cache_dir / str(metadata["split"]) / f"{slug}-{digest}"


def _save_json_atomic(path: Path, *, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _save_npy_atomic(path: Path, *, array: np.ndarray, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, np.asarray(array, dtype=dtype), allow_pickle=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_cached_patch_index(
    *,
    cache_dir: str | Path | None,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    mask_names,
    label_artifact: str = "",
    mask_artifacts=(),
    log=None,
) -> CachedPatchIndex | None:
    if cache_dir is None:
        return None
    metadata = _cache_metadata(
        segment_id=segment_id,
        split_name=split_name,
        patch_size=patch_size,
        tile_size=tile_size,
        stride=stride,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
        mask_names=mask_names,
        label_artifact=label_artifact,
        mask_artifacts=mask_artifacts,
    )
    base_path = _cache_file_path(cache_dir, metadata=metadata)
    metadata_path = base_path.with_suffix(".json")
    xyxy_path = base_path.with_suffix(".npy")
    bbox_rows_path = base_path.with_name(f"{base_path.name}-bboxes").with_suffix(".npy")
    bbox_indices_path = base_path.with_name(f"{base_path.name}-bbox_idx").with_suffix(".npy")
    if not metadata_path.exists() or not xyxy_path.exists() or not bbox_rows_path.exists() or not bbox_indices_path.exists():
        if callable(log):
            log(f"[data] patch-index cache miss split={split_name} segment={segment_id}")
        return None
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != metadata:
            if callable(log):
                log(f"[data] patch-index cache stale split={split_name} segment={segment_id}")
            return None
        with xyxy_path.open("rb") as handle:
            xyxys = np.asarray(np.load(handle, allow_pickle=False), dtype=np.int64)
        with bbox_rows_path.open("rb") as handle:
            bbox_rows = np.asarray(np.load(handle, allow_pickle=False), dtype=np.int32)
        with bbox_indices_path.open("rb") as handle:
            sample_bbox_indices = np.asarray(np.load(handle, allow_pickle=False), dtype=np.int32)
    except Exception:
        if callable(log):
            log(f"[data] patch-index cache invalid split={split_name} segment={segment_id}")
        return None
    if xyxys.ndim != 2 or int(xyxys.shape[1]) != 4:
        if callable(log):
            log(f"[data] patch-index cache invalid split={split_name} segment={segment_id}")
        return None
    if bbox_rows.ndim != 2 or int(bbox_rows.shape[1]) != 4:
        if callable(log):
            log(f"[data] patch-index cache invalid split={split_name} segment={segment_id}")
        return None
    if sample_bbox_indices.ndim != 1 or int(sample_bbox_indices.shape[0]) != int(xyxys.shape[0]):
        if callable(log):
            log(f"[data] patch-index cache invalid split={split_name} segment={segment_id}")
        return None
    if callable(log):
        log(f"[data] patch-index cache hit split={split_name} segment={segment_id}")
    return CachedPatchIndex(
        xyxys=xyxys,
        bbox_rows=bbox_rows,
        sample_bbox_indices=sample_bbox_indices,
    )


def save_cached_patch_index(
    *,
    cache_dir: str | Path | None,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    mask_names,
    label_artifact: str = "",
    mask_artifacts=(),
    xyxys: np.ndarray,
    bbox_rows: np.ndarray,
    sample_bbox_indices: np.ndarray,
) -> None:
    if cache_dir is None:
        return
    metadata = _cache_metadata(
        segment_id=segment_id,
        split_name=split_name,
        patch_size=patch_size,
        tile_size=tile_size,
        stride=stride,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
        mask_names=mask_names,
        label_artifact=label_artifact,
        mask_artifacts=mask_artifacts,
    )
    base_path = _cache_file_path(cache_dir, metadata=metadata)
    _save_json_atomic(base_path.with_suffix(".json"), payload=metadata)
    _save_npy_atomic(base_path.with_suffix(".npy"), array=xyxys, dtype=np.int64)
    _save_npy_atomic(
        base_path.with_name(f"{base_path.name}-bboxes").with_suffix(".npy"),
        array=bbox_rows,
        dtype=np.int32,
    )
    _save_npy_atomic(
        base_path.with_name(f"{base_path.name}-bbox_idx").with_suffix(".npy"),
        array=sample_bbox_indices,
        dtype=np.int32,
    )


__all__ = [
    "CachedPatchIndex",
    "load_cached_patch_index",
    "save_cached_patch_index",
]
