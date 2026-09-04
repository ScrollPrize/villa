"""Flat patch-index identity and JSON serialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Patch, Segment


_CACHE_VERSION = "v6"


def label_asset_fingerprint(paths: Iterable[Path | str | None]) -> str:
    """Return a short digest of the label assets' on-disk layout.

    A cached split is identified by the label *paths* it was found under, which catches
    pointing a run at a different tree and misses regenerating a mask in place. The split
    is then reused against labels it no longer describes, and nothing looks wrong: the
    patch count and the bounding boxes are the old ones, so training continues over
    supervision that has been deleted.

    Hashing each asset's relative file names and sizes catches that, because rewriting a
    zarr changes chunk sizes and dropping annotation deletes chunks outright when the
    store does not write empty ones. It reads no chunk contents, so it costs one
    directory walk -- ``os.scandir`` carries the size on Windows and Linux alike -- and it
    cannot tell apart two different labels that compress to identical sizes under
    identical names. A byte-identical copy of a tree fingerprints the same as its source,
    which is the wanted behaviour.
    """

    digest = hashlib.sha256()
    for path in sorted(str(value) for value in paths if value):
        root = Path(path)
        digest.update(root.name.encode())
        if not root.exists():
            digest.update(b"\0missing")
            continue
        entries: list[tuple[str, int]] = []
        stack = [("", str(root))]
        while stack:
            relative, current = stack.pop()
            try:
                children = sorted(os.scandir(current), key=lambda entry: entry.name)
            except OSError:
                continue
            for child in children:
                name = f"{relative}/{child.name}" if relative else child.name
                if child.is_dir(follow_symlinks=False):
                    stack.append((name, child.path))
                    continue
                try:
                    size = child.stat().st_size
                except OSError:
                    size = -1
                entries.append((name, size))
        for name, size in sorted(entries):
            digest.update(name.encode())
            digest.update(str(size).encode())
    return digest.hexdigest()[:16]


def _segment_label_fingerprint(segment: Segment) -> str:
    return label_asset_fingerprint(
        (segment.inklabels, segment.supervision_mask, segment.validation_mask)
    )


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
    fingerprints: dict[tuple[int, str, int, str, str, str], str] = {}
    for patch in patches:
        segment = patch.segment
        fingerprint = fingerprints.get(segment.cache_key)
        if fingerprint is None:
            fingerprint = fingerprints.setdefault(
                segment.cache_key, _segment_label_fingerprint(segment)
            )
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
                "label_fingerprint": fingerprint,
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
    segments_by_key = {
        (*segment.cache_key, _segment_label_fingerprint(segment)): segment
        for segment in segments
    }
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
            str(record.get("label_fingerprint", "")),
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
