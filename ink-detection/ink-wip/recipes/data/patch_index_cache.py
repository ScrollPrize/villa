from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

from ink.recipes.data.layout import NestedZarrLayout

_PATCH_INDEX_CACHE_SCHEMA_VERSION = 1


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
    layout: NestedZarrLayout,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
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
        "source_fingerprint": layout.label_mask_fingerprint(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
        ),
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


def _save_npy_atomic(path: Path, *, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.int64), allow_pickle=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_cached_patch_xyxys(
    *,
    cache_dir: str | Path | None,
    layout: NestedZarrLayout,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    log=None,
) -> np.ndarray | None:
    if cache_dir is None:
        return None
    metadata = _cache_metadata(
        layout=layout,
        segment_id=segment_id,
        split_name=split_name,
        patch_size=patch_size,
        tile_size=tile_size,
        stride=stride,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    base_path = _cache_file_path(cache_dir, metadata=metadata)
    metadata_path = base_path.with_suffix(".json")
    xyxy_path = base_path.with_suffix(".npy")
    if not metadata_path.exists() or not xyxy_path.exists():
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
    except Exception:
        if callable(log):
            log(f"[data] patch-index cache invalid split={split_name} segment={segment_id}")
        return None
    if xyxys.ndim != 2 or int(xyxys.shape[1]) != 4:
        if callable(log):
            log(f"[data] patch-index cache invalid split={split_name} segment={segment_id}")
        return None
    if callable(log):
        log(f"[data] patch-index cache hit split={split_name} segment={segment_id}")
    return xyxys


def save_cached_patch_xyxys(
    *,
    cache_dir: str | Path | None,
    layout: NestedZarrLayout,
    segment_id: str,
    split_name: str,
    patch_size: int,
    tile_size: int,
    stride: int,
    label_suffix: str,
    mask_suffix: str,
    xyxys: np.ndarray,
) -> None:
    if cache_dir is None:
        return
    metadata = _cache_metadata(
        layout=layout,
        segment_id=segment_id,
        split_name=split_name,
        patch_size=patch_size,
        tile_size=tile_size,
        stride=stride,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    base_path = _cache_file_path(cache_dir, metadata=metadata)
    _save_json_atomic(base_path.with_suffix(".json"), payload=metadata)
    _save_npy_atomic(base_path.with_suffix(".npy"), array=np.asarray(xyxys, dtype=np.int64))


__all__ = [
    "load_cached_patch_xyxys",
    "save_cached_patch_xyxys",
]
