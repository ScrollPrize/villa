"""Flat patch-index identity and JSON serialization.

v8 (PR #2677, review 7, point 6) -- cache-first discovery and store fingerprint:

``load_patch_cache`` is now reached from a LIGHT gather (segment.py
``gather_segments(light=True)``): the dataset gathers lightly, tries
the cache, and only on a miss re-gathers with full content
validation. A cache hit therefore decodes no image/label chunks.
Each patch record and sidecar entry now also carries a
``store_fingerprint``: the exact chunk inventory (relative paths,
sizes, mtimes, hashed per store) of the image AND the label stores,
recorded at cache-write time and verified on a hit by listing only.
The label fingerprint alone covers labels, so an image chunk removed
after caching would not have invalidated it; the store fingerprint
closes that hole. Each record also carries a ``record_checksum``
(adversarial self-review A12): a hash of the record itself, so an
edit to a bbox (or any field) after caching invalidates the cache
instead of being served. A record without a fingerprint or a checksum
(written before v8) never matches and misses, which is the safe
direction. Residual limitation: a chunk replaced by a different file
with the same name, size and mtime is undetectable without reading
contents (deliberate forgery, not a sync issue). The selection helpers
(``_is_selected_segment``, ``_is_main_discovered_segment``) now live
in segment.py as a single definition; this module imports them.

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
from vesuvius.ink_detection.data.segment import (
    _is_main_discovered_segment,
    _is_selected_segment,
    _segment_store_fingerprint,
)
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


def _record_checksum(record: Mapping) -> str:
    """Checksum of a patch record / sidecar entry (all fields but itself).

    The store/label fingerprints verify the STORES; this verifies the
    RECORD: an edit to a bbox (or any other field) after caching
    changes the checksum, so the cache misses and fresh discovery runs
    instead of serving the edited record. A record without a checksum
    (written before v8) never matches and misses, the safe direction.
    """
    content = {k: v for k, v in record.items() if k != "record_checksum"}
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


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
    store_fingerprints: dict[tuple[int, str, int, str, str, str], str] = {}
    for patch in patches:
        segment = patch.segment
        fingerprint = fingerprints.get(segment.cache_key)
        if fingerprint is None:
            fingerprint = fingerprints.setdefault(
                segment.cache_key, _segment_label_fingerprint(segment)
            )
        store_fingerprint = store_fingerprints.get(segment.cache_key)
        if store_fingerprint is None:
            store_fingerprint = store_fingerprints.setdefault(
                segment.cache_key, _segment_store_fingerprint(segment)
            )
        record = {
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
            # Review 7, point 6: the exact chunk inventory of the
            # image AND the label stores, so a chunk removed after
            # caching invalidates the cache (the label fingerprint
            # alone covers labels only).
            "store_fingerprint": store_fingerprint,
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
        # Adversarial self-review (A12): a checksum of the record
        # itself, so an edit to the bbox (or any field) after caching
        # is detected instead of served.
        record["record_checksum"] = _record_checksum(record)
        records.append(record)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream)
    # Zero-patch sidecar (review 5, point 3): segments that legitimately
    # yielded no patches get an identity entry here instead of a patch
    # record. An empty list removes a stale sidecar from an earlier run
    # (e.g. every segment now yields patches).
    sidecar = zero_patch_cache_path(path)
    entries = []
    for segment in zero_patch_segments:
        entry = {
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
            "store_fingerprint": _segment_store_fingerprint(segment),
            "patch_finding_key": patch_finding_cache_token(segment.data_config),
        }
        entry["record_checksum"] = _record_checksum(entry)
        entries.append(entry)
    if entries:
        with sidecar.open("w", encoding="utf-8") as stream:
            json.dump(entries, stream)
    else:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass


# Selection helpers live in segment.py (single definition, review 7,
# point 6); dataset.py and this module import them from there.
# (The mirrors that used to live here were removed in v8.)


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
    seen_patch_keys: set[tuple] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"patch cache {path} contains a non-object record")
        # Adversarial self-review (A12): the record checksum must
        # match, so an edit to the bbox (or any field) after caching
        # invalidates the cache instead of being served. A record
        # without a checksum (written before v8) never matches.
        if record.get("record_checksum") != _record_checksum(record):
            return None
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
        # Review 7, point 6: the store fingerprint (exact chunk
        # inventory of the image AND the label stores) must match the
        # current stores. A chunk removed after caching changes the
        # inventory, so the cache is invalidated and the fresh run
        # detects the incomplete input. A record without a fingerprint
        # (written before v8) never matches and misses, which is the
        # safe direction.
        if record.get("store_fingerprint") != _segment_store_fingerprint(segment):
            return None
        bbox = tuple(int(value) for value in record["bbox"])
        if len(bbox) != 6:
            raise ValueError(f"patch cache bbox must have six ZYX bounds, got {bbox!r}")
        # Red-team R6c: a duplicated record is served twice -- the
        # per-record checksum cannot see duplication (an exact copy
        # has a valid checksum). Dedupe by (segment, bbox) at load.
        patch_key = (segment.cache_key, bbox)
        if patch_key in seen_patch_keys:
            continue
        seen_patch_keys.add(patch_key)
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
    # Coverage: every discovered segment must be accounted for by a
    # patch record or a sidecar entry, in ALL modes. (Red-team R7:
    # the check used to run only in labeled mode, so in unlabeled
    # mode a segment added after caching was silently ignored and a
    # deleted record was silently served -- review 7, point 3's
    # principle, "cache hits apply the same selection rules as fresh
    # discovery", covers unlabeled too. Main has the same staleness
    # in unlabeled mode, inherited, but the guard itself was v8
    # code.) A segment with no record means the cache predates it
    # (a dataset added under the same out_dir). A sidecar entry
    # covers a segment that legitimately yielded no patches
    # (complete but below the ink-coverage threshold, review 5,
    # point 3): without it the cache would be invalidated on every
    # rerun.
    covered = {patch.segment.cache_key for patch in patches}
    sidecar_entries = _read_zero_patch_entries(path)
    if sidecar_entries is None:
        return None
    for entry in sidecar_entries:
        if not isinstance(entry, Mapping):
            return None
        # Adversarial self-review (A12): same record checksum as
        # for patch records -- an edited sidecar entry invalidates.
        if entry.get("record_checksum") != _record_checksum(entry):
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
        # Review 7, point 6: same store-fingerprint check as for
        # patch records -- a chunk removed after caching must
        # invalidate the sidecar coverage too.
        if entry.get("store_fingerprint") != _segment_store_fingerprint(segment):
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
        # keeps its coverage (the W9/R6 rerun behaviour). That
        # RAISE rationale holds only in labeled mode: in unlabeled
        # mode find_segment_unlabeled_patches returns empty lists
        # (with a warning) instead of raising, so a zero-patch
        # segment is a legitimate outcome there and its valid
        # sidecar entry covers it. Without this carve-out, every
        # unlabeled rerun containing a main-discovered zero-patch
        # segment would invalidate the cache (review 4 blocked
        # exactly that rescan when it removed the v3 unlabeled
        # coverage check; R6 unlabeled run2 is the regression
        # case). The fingerprint/checksum/token checks above still
        # apply in unlabeled mode, so a changed segment still
        # invalidates (Y4).
        if config.discovery_mode != "unlabeled" and (
            _is_selected_segment(segment) or _is_main_discovered_segment(segment)
        ):
            return None
        covered.add(segment.cache_key)
    if any(segment.cache_key not in covered for segment in segments_by_key.values()):
        return None
    return patches
