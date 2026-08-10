"""Flat patch-index identity and JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Patch, Segment


_CACHE_VERSION = "v6"


def patch_finding_cache_token(config: InkDataConfig) -> str:
    """Return the v6 token including every patch-semantic input."""
    finding = config.patch_finding
    patch_y = config.patch_size[1]
    default_stride = int(patch_y * finding.overlap)
    if finding.kind == "subtiling":
        tile_size = patch_y if finding.tile_size is None else finding.tile_size
        stride = default_stride if finding.stride is None else finding.stride
        return (
            f"{config.discovery_mode}-subtiling-{_CACHE_VERSION}"
            f"-ts-{tile_size}_st-{stride}_fe-{int(finding.filter_empty_tile)}"
        )
    scan_scale = "" if finding.scan_scale is None else finding.scan_scale
    if config.discovery_mode == "unlabeled":
        # The reference spelled its unused option default here; keep token compatibility.
        return (
            f"unlabeled-default-{_CACHE_VERSION}"
            f"-po-{finding.overlap}"
            "-mdc-0.15"
            f"-pfs-{scan_scale}"
        )
    return (
        f"labeled-default-{_CACHE_VERSION}"
        f"-po-{finding.overlap}"
        f"-mlc-{finding.min_labeled_coverage}"
        f"-pfs-{scan_scale}"
    )


def patch_cache_path(config: InkDataConfig) -> Path:
    if config.patch_cache_filename is not None:
        return config.patch_cache_filename
    patch_size_key = "x".join(str(value) for value in config.patch_size)
    label_key = "auto" if config.label_version is None else config.label_version
    return config.out_dir / (
        f"flat_ink_patches_dm-{config.discovery_mode}_"
        f"pf-{patch_finding_cache_token(config)}_"
        f"ps-{patch_size_key}_labels-{label_key}.json"
    )


def save_patch_cache(path: str | Path, patches: Iterable[Patch]) -> None:
    """Write the complete patch identity needed to reject stale caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for patch in patches:
        segment = patch.segment
        records.append(
            {
                "dataset_idx": segment.dataset_idx,
                "segment_relpath": segment.segment_relpath,
                "scale": segment.scale,
                "inklabels_path": "" if segment.inklabels is None else str(segment.inklabels),
                "supervision_mask_path": (
                    "" if segment.supervision_mask is None else str(segment.supervision_mask)
                ),
                "validation_mask_path": (
                    "" if segment.validation_mask is None else str(segment.validation_mask)
                ),
                "active_supervision_mask_path": (
                    "" if patch.supervision_mask is None else str(patch.supervision_mask)
                ),
                "is_validation": patch.is_validation,
                "is_unlabeled": patch.is_unlabeled,
                "patch_finding_key": patch_finding_cache_token(
                    segment.data_config
                ),
                "bbox": list(patch.bbox),
            }
        )
    with path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream)


def load_patch_cache(
    path: str | Path,
    *,
    config: InkDataConfig,
    segments: Iterable[Segment],
) -> list[Patch] | None:
    """Return reconstructed patches, or None when any cache identity is stale."""
    with Path(path).open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"patch cache {path} must contain a JSON array")
    segments_by_key = {segment.cache_key: segment for segment in segments}
    expected_token = patch_finding_cache_token(config)
    patches: list[Patch] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"patch cache {path} contains a non-object record")
        if record.get("patch_finding_key") != expected_token:
            return None
        key = (
            int(record["dataset_idx"]),
            str(record["segment_relpath"]),
            int(record["scale"]),
            str(record.get("inklabels_path", "")),
            str(record.get("supervision_mask_path", "")),
            str(record.get("validation_mask_path", "")),
        )
        segment = segments_by_key.get(key)
        if segment is None:
            return None
        bbox = tuple(int(value) for value in record["bbox"])
        if len(bbox) != 6:
            raise ValueError(f"patch cache bbox must have six ZYX bounds, got {bbox!r}")
        patches.append(
            Patch(
                segment=segment,
                bbox=bbox,
                is_validation=bool(record.get("is_validation", False)),
                is_unlabeled=bool(record.get("is_unlabeled", False)),
                supervision_mask_override=(
                    record.get("active_supervision_mask_path") or None
                ),
            )
        )
    return patches
