"""
Simplified patch cache for OME-Zarr datasets.

This module provides caching utilities for valid patch positions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PatchCacheParams:
    """
    Cache parameters for OME-Zarr patch validation.

    These parameters define the cache key - if any change, the cache is invalidated.
    """

    data_path: str
    volume_ids: Tuple[str, ...]
    patch_size: Tuple[int, ...]
    min_labeled_ratio: float
    bbox_threshold: float
    valid_patch_find_resolution: int
    valid_patch_value: Optional[float] = None
    unlabeled_fg_enabled: bool = True
    unlabeled_fg_threshold: float = 0.05
    unlabeled_fg_bbox_threshold: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": SCHEMA_VERSION,
            "data_path": self.data_path,
            "volume_ids": list(self.volume_ids),
            "patch_size": list(self.patch_size),
            "min_labeled_ratio": float(self.min_labeled_ratio),
            "bbox_threshold": float(self.bbox_threshold),
            "valid_patch_find_resolution": int(self.valid_patch_find_resolution),
            "valid_patch_value": self.valid_patch_value,
            "unlabeled_fg_enabled": bool(self.unlabeled_fg_enabled),
            "unlabeled_fg_threshold": float(self.unlabeled_fg_threshold),
            "unlabeled_fg_bbox_threshold": float(self.unlabeled_fg_bbox_threshold),
        }


@dataclass
class PatchEntry:
    """Single patch with volume context."""

    volume_idx: int
    volume_name: str
    position: Tuple[int, ...]


@dataclass
class PatchCacheData:
    """Container for cached patch positions with volume info."""

    fg_patches: List[PatchEntry]
    bg_patches: List[PatchEntry]
    unlabeled_fg_patches: List[PatchEntry]


def build_cache_params(
    *,
    data_path: str | Path,
    volume_ids: Sequence[str],
    patch_size: Sequence[int],
    min_labeled_ratio: float,
    bbox_threshold: float,
    valid_patch_find_resolution: int,
    valid_patch_value: Optional[float] = None,
    unlabeled_fg_enabled: bool = True,
    unlabeled_fg_threshold: float = 0.05,
    unlabeled_fg_bbox_threshold: float = 0.15,
) -> PatchCacheParams:
    """Build cache params from individual arguments."""
    return PatchCacheParams(
        data_path=str(Path(data_path).resolve()),
        volume_ids=tuple(sorted(volume_ids)),
        patch_size=tuple(int(v) for v in patch_size),
        min_labeled_ratio=float(min_labeled_ratio),
        bbox_threshold=float(bbox_threshold),
        valid_patch_find_resolution=int(valid_patch_find_resolution),
        valid_patch_value=valid_patch_value,
        unlabeled_fg_enabled=bool(unlabeled_fg_enabled),
        unlabeled_fg_threshold=float(unlabeled_fg_threshold),
        unlabeled_fg_bbox_threshold=float(unlabeled_fg_bbox_threshold),
    )


def cache_filename(cache_params: PatchCacheParams) -> str:
    """Generate hash-based cache filename."""
    config = cache_params.to_dict()
    digest = _hash_config(config)
    return f"patches_v{SCHEMA_VERSION}_{digest}.json"


def load_patch_cache(
    cache_dir: Path,
    cache_params: PatchCacheParams,
) -> Optional[PatchCacheData]:
    """Load cached patches from file if valid."""
    cache_file = cache_dir / cache_filename(cache_params)
    if not cache_file.exists():
        return None

    try:
        with cache_file.open("r") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    schema_version = payload.get("schema_version")

    # Handle schema v2 (legacy format without volume info)
    if schema_version == 2:
        # Legacy format - positions only, assume single volume
        volume_name = cache_params.volume_ids[0] if cache_params.volume_ids else "unknown"
        legacy_fg = _decode_positions_legacy(payload.get("fg_patches", []))
        legacy_bg = _decode_positions_legacy(payload.get("bg_patches", []))
        legacy_unlabeled = _decode_positions_legacy(payload.get("unlabeled_fg_patches", []))
        fg = [PatchEntry(0, volume_name, pos) for pos in legacy_fg]
        bg = [PatchEntry(0, volume_name, pos) for pos in legacy_bg]
        unlabeled_fg = [PatchEntry(0, volume_name, pos) for pos in legacy_unlabeled]
        return PatchCacheData(fg_patches=fg, bg_patches=bg, unlabeled_fg_patches=unlabeled_fg)

    if schema_version != SCHEMA_VERSION:
        return None

    metadata = payload.get("metadata", {})
    cached_params = metadata.get("cache_params")
    expected_params = cache_params.to_dict()
    if cached_params != expected_params:
        return None

    fg = _decode_patches(payload.get("fg_patches", []))
    bg = _decode_patches(payload.get("bg_patches", []))
    unlabeled_fg = _decode_patches(payload.get("unlabeled_fg_patches", []))
    return PatchCacheData(fg_patches=fg, bg_patches=bg, unlabeled_fg_patches=unlabeled_fg)


def save_patch_cache(
    cache_dir: Path,
    cache_params: PatchCacheParams,
    cache_data: PatchCacheData,
) -> Path:
    """Save patch positions to cache file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_filename(cache_params)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "cache_params": cache_params.to_dict(),
            "counts": {
                "fg": len(cache_data.fg_patches),
                "bg": len(cache_data.bg_patches),
                "unlabeled_fg": len(cache_data.unlabeled_fg_patches),
            },
        },
        "fg_patches": _encode_patches(cache_data.fg_patches),
        "bg_patches": _encode_patches(cache_data.bg_patches),
        "unlabeled_fg_patches": _encode_patches(cache_data.unlabeled_fg_patches),
    }

    with cache_file.open("w") as fh:
        json.dump(payload, fh, indent=2)

    return cache_file


def _hash_config(config: Dict[str, Any]) -> str:
    """Generate hash from config dict."""
    payload = json.dumps(config, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _encode_patches(patches: Sequence[PatchEntry]) -> List[Dict[str, Any]]:
    """Encode patches for JSON serialization."""
    return [
        {
            "vol_idx": p.volume_idx,
            "vol_name": p.volume_name,
            "pos": [int(v) for v in p.position],
        }
        for p in patches
    ]


def _decode_patches(values: Sequence[Dict[str, Any]]) -> List[PatchEntry]:
    """Decode patches from JSON."""
    return [
        PatchEntry(
            volume_idx=int(v["vol_idx"]),
            volume_name=str(v["vol_name"]),
            position=tuple(int(x) for x in v["pos"]),
        )
        for v in values
    ]


def _decode_positions_legacy(values: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    """Decode legacy positions from schema v2 JSON."""
    return [tuple(int(v) for v in entry) for entry in values]


def try_load_patch_cache(
    cache_dir: Path,
    data_path: str | Path,
    volume_ids: Sequence[str],
    patch_size: Sequence[int],
    min_labeled_ratio: float,
    bbox_threshold: float,
    valid_patch_find_resolution: int,
    valid_patch_value: Optional[float] = None,
    unlabeled_fg_enabled: bool = True,
    unlabeled_fg_threshold: float = 0.05,
    unlabeled_fg_bbox_threshold: float = 0.15,
) -> Optional[PatchCacheData]:
    """
    Convenience function to build cache params and load cache if it exists.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files.
    data_path : str | Path
        Path to the dataset directory.
    volume_ids : Sequence[str]
        List of volume identifiers.
    patch_size : Sequence[int]
        Patch dimensions.
    min_labeled_ratio : float
        Minimum fraction of labeled voxels.
    bbox_threshold : float
        Minimum bounding box coverage.
    valid_patch_find_resolution : int
        Multi-resolution level for patch finding.
    valid_patch_value : Optional[float]
        Specific label value to match.
    unlabeled_fg_enabled : bool
        Enable unlabeled foreground collection.
    unlabeled_fg_threshold : float
        Min fraction of non-zero image voxels.
    unlabeled_fg_bbox_threshold : float
        Min bbox coverage for image data.

    Returns
    -------
    Optional[PatchCacheData]
        Cached patch data if found, None otherwise.
    """
    params = build_cache_params(
        data_path=data_path,
        volume_ids=volume_ids,
        patch_size=patch_size,
        min_labeled_ratio=min_labeled_ratio,
        bbox_threshold=bbox_threshold,
        valid_patch_find_resolution=valid_patch_find_resolution,
        valid_patch_value=valid_patch_value,
        unlabeled_fg_enabled=unlabeled_fg_enabled,
        unlabeled_fg_threshold=unlabeled_fg_threshold,
        unlabeled_fg_bbox_threshold=unlabeled_fg_bbox_threshold,
    )
    return load_patch_cache(cache_dir, params)
