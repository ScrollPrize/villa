"""Flat patch-index identity and JSON serialization.

v6 (PR #2677, review 5, point 3) -- zero-patch sidecar:

``load_patch_cache`` in labeled modes requires a patch record for every
discovered segment, so a complete segment below the ink-coverage
threshold (legitimately zero patches) invalidated the cache on every
rerun and forced a full re-discovery. The fix records such segments
in a sidecar file next to the cache (``<cache>.zero_patch.json``):
``save_patch_cache`` writes one entry per segment that legitimately
yielded no patches, and ``load_patch_cache`` counts a sidecar entry
with a matching identity (paths + label fingerprint + token) as
coverage for that segment. The main cache file format is unchanged
(no token/version change -- the repo test pins ``labeled-default-v7``),
and main's loader never reads the sidecar, so an older loader reading
a v6 cache sees exactly the same records as before. A sidecar entry
whose identity no longer matches (labels arrived, segment removed)
invalidates the cache, exactly like a stale patch record.

v7 (PR #2677, review 6, point 3) -- sidecar selection re-validation:

A sidecar entry records only that the segment yielded no patches --
not the selection it was found under. A cached automatic scan of a
ready mirror and a complete below-threshold mirror, then both
allowlisted, returned the ready mirror's patches: the fresh run would
have raised on the low-ink mirror (a selected segment that yields no
patches fails patch finding loudly), but the cache hit never ran patch
finding. The fix re-validates each sidecar-covered segment against
the CURRENT selection: a sidecar entry must not cover a segment that
is now selected (allowlisted, explicitly mapped, held out) or
main-discovered (x.tif, explicit surface) -- such a segment
invalidates the cache, so fresh discovery runs and raises, matching a
fresh run. Patch records need no such check (a segment that yielded
patches yields them under any selection).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Patch, Segment


_CACHE_VERSION = "v6"
# Default discovery now measures coverage at the ink store's own mid-plane.
_LABELED_DEFAULT_CACHE_VERSION = "v7"
# Subtiling caches written before the patch Z window came from the image mid-plane hold label-based Z.
_SUBTILING_CACHE_VERSION = "v7"


def label_asset_fingerprint(paths: Iterable[Path | str | None]) -> str:
    """Return a short digest of the label assets' on-disk layout.

    A cached split is identified by the label *paths* it was found under, which catches
    pointing a run at a different tree and misses regenerating a mask in place. The split
    is then reused against labels it no longer describes, and nothing looks wrong: the
    patch count and the bounding boxes are the old ones, so training continues over
    supervision that has been deleted.

    Hashing each asset's relative file names, sizes and modification times catches that:
    rewriting a zarr changes chunk sizes and stamps every chunk it writes, and dropping
    annotation deletes chunks outright when the store does not write empty ones. It reads
    no chunk contents, so it costs one directory walk -- ``os.scandir`` carries the stat
    result on Windows and Linux alike, and the mtime rides along with the size for free.

    The trade is that a copy of a tree fingerprints the same as its source only when the
    copy preserved modification times; otherwise it misses once and the split is rebuilt.
    A label rewritten to the same bytes at a new time misses the same way. Both err
    towards rebuilding, which is the cheap side.
    """

    digest = hashlib.sha256()
    for path in sorted(str(value) for value in paths if value):
        root = Path(path)
        digest.update(root.name.encode())
        if not root.exists():
            digest.update(b"\0missing")
            continue
        entries: list[tuple[str, int, int]] = []
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
                    stat = child.stat()
                    size, mtime_ns = stat.st_size, stat.st_mtime_ns
                except OSError:
                    size, mtime_ns = -1, -1
                entries.append((name, size, mtime_ns))
        for name, size, mtime_ns in sorted(entries):
            digest.update(name.encode())
            digest.update(str(size).encode())
            digest.update(str(mtime_ns).encode())
    return digest.hexdigest()[:16]


def _segment_label_fingerprint(segment: Segment) -> str:
    return label_asset_fingerprint(
        (segment.inklabels, segment.supervision_mask, segment.validation_mask)
    )


def patch_finding_cache_token(config: InkDataConfig) -> str:
    """Return a versioned token including every patch-semantic input."""
    finding = config.patch_finding
    patch_y = config.patch_size[1]
    default_stride = int(patch_y * finding.overlap)
    if finding.kind == "subtiling":
        tile_size = patch_y if finding.tile_size is None else finding.tile_size
        stride = default_stride if finding.stride is None else finding.stride
        return (
            f"{config.discovery_mode}-subtiling-{_SUBTILING_CACHE_VERSION}"
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
        f"labeled-default-{_LABELED_DEFAULT_CACHE_VERSION}"
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


def zero_patch_cache_path(path: str | Path) -> Path:
    """Sidecar file listing segments that legitimately yielded no patches.

    Lives next to the patch cache (``<cache>.zero_patch.json``). A
    complete segment below the ink-coverage threshold has no patch
    record, so without this file the labeled coverage check in
    :func:`load_patch_cache` would invalidate the cache on every
    rerun. The sidecar carries the same identity as a patch record
    (paths + label fingerprint + token), so a changed segment
    invalidates the cache exactly like a stale patch record.
    """
    return Path(str(path) + ".zero_patch.json")


def _read_zero_patch_entries(path: str | Path) -> list | None:
    """Sidecar entries for ``path``, [] when absent, None when corrupt.

    A corrupt or non-list sidecar is treated as stale (the cache is
    invalidated), never as coverage: silently trusting it could reuse
    patches for a segment whose labels changed.
    """
    sidecar = zero_patch_cache_path(path)
    if not sidecar.exists():
        return []
    try:
        with sidecar.open("r", encoding="utf-8") as stream:
            entries = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    return entries


def save_patch_cache(path: str | Path, patches: Iterable[Patch], zero_patch_segments: Iterable[Segment] = ()) -> None:
    """Write the complete patch identity needed to reject stale caches.

    ``zero_patch_segments`` are segments that legitimately yielded no
    patches (complete but below the ink-coverage threshold): they get
    a sidecar entry instead of a patch record, so an unchanged rerun
    reuses the cache instead of re-discovering them (review 5, point
    3). Segments that FAILED patch finding must never be listed here:
    only segments whose stores were read successfully and yielded
    nothing qualify.
    """
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
                "exclude_validation_mask": patch.exclude_validation_mask,
                "patch_finding_key": patch_finding_cache_token(
                    segment.data_config
                ),
                "bbox": list(patch.bbox),
            }
        )
    with path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream)
    # Zero-patch sidecar (review 5, point 3): segments that legitimately
    # yielded no patches get an identity entry here instead of a patch
    # record. An empty list removes a stale sidecar from an earlier run
    # (e.g. every segment now yields patches).
    sidecar = zero_patch_cache_path(path)
    entries = [
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
            "label_fingerprint": _segment_label_fingerprint(segment),
            "patch_finding_key": patch_finding_cache_token(segment.data_config),
        }
        for segment in zero_patch_segments
    ]
    if entries:
        with sidecar.open("w", encoding="utf-8") as stream:
            json.dump(entries, stream)
    else:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass


def _is_selected_segment(segment) -> bool:
    """True when the user explicitly selected this segment.

    Mirror of ``dataset._is_selected_segment`` (duplicated here to
    avoid a circular import: dataset imports patch_cache). Keep in
    sync. A segment is selected when its name is in the source's
    allowlist, when it has an explicit surface volume (a single
    ``surface_volume_path`` or a ``surface_volume_paths`` entry for
    its relpath or name), or when its source is held out
    (``split == "validation"``).
    """
    source = segment.source
    if source.split == "validation":
        return True
    if segment.segment_name in set(source.segment_names):
        return True
    if source.surface_volume_path is not None:
        return True
    if (
        segment.segment_relpath in source.surface_volume_paths
        or segment.segment_name in source.surface_volume_paths
    ):
        return True
    return False


def _is_main_discovered_segment(segment) -> bool:
    """True when main would discover this segment without the PR.

    Mirror of ``dataset._is_main_discovered_segment`` (duplicated here
    to avoid a circular import: dataset imports patch_cache). Keep in
    sync. Main discovers a flat segment when it carries native tifxyz
    geometry (``x.tif`` in its directory) or when it has an explicit
    surface volume (a single ``surface_volume_path`` or a
    ``surface_volume_paths`` entry for its relpath or name).
    """
    if (Path(segment.segment_dir) / "x.tif").is_file():
        return True
    source = segment.source
    if source.surface_volume_path is not None:
        return True
    if (
        segment.segment_relpath in source.surface_volume_paths
        or segment.segment_name in source.surface_volume_paths
    ):
        return True
    return False


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
                exclude_validation_mask=bool(
                    record.get("exclude_validation_mask", False)
                ),
                supervision_mask_override=(
                    record.get("active_supervision_mask_path") or None
                ),
            )
        )
    if config.discovery_mode != "unlabeled":
        # Every labeled segment yields patches or fails discovery, so a segment with no
        # record means the cache predates it (a dataset added under the same out_dir).
        # A sidecar entry covers a segment that legitimately yielded no patches
        # (complete but below the ink-coverage threshold, review 5, point 3):
        # without it the cache would be invalidated on every rerun.
        covered = {patch.segment.cache_key for patch in patches}
        sidecar_entries = _read_zero_patch_entries(path)
        if sidecar_entries is None:
            return None
        for entry in sidecar_entries:
            if not isinstance(entry, Mapping):
                return None
            if entry.get("patch_finding_key") != expected_token:
                return None
            key = (
                int(entry["dataset_idx"]),
                str(entry["segment_relpath"]),
                int(entry["scale"]),
                str(entry.get("inklabels_path", "")),
                str(entry.get("supervision_mask_path", "")),
                str(entry.get("validation_mask_path", "")),
                str(entry.get("label_fingerprint", "")),
            )
            segment = segments_by_key.get(key)
            if segment is None:
                return None
            # (review 6, point 3) A sidecar entry covers a segment that
            # legitimately yielded no patches -- but only under the
            # selection that produced it. A segment that is now
            # selected (allowlisted, explicitly mapped, held out) or
            # main-discovered would RAISE on a fresh run (no patches
            # from a selected segment is a patch-finding failure, and
            # failures on selected/main-discovered segments re-raise),
            # so the sidecar must not cover it: invalidate the cache
            # so fresh discovery runs and raises, matching a fresh
            # run. An auto-discovered PR mirror below the threshold
            # keeps its coverage (the W9/R6 rerun behaviour).
            if _is_selected_segment(segment) or _is_main_discovered_segment(segment):
                return None
            covered.add(segment.cache_key)
        if any(segment.cache_key not in covered for segment in segments_by_key.values()):
            return None
    return patches
