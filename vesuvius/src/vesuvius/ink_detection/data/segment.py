"""Discover segment directories and select versioned label assets.

v2 — targeted flat-discovery fix for PR #2677 (review: villa-review, 24/11/2026).

What v1 (PR #2677, rglob("*.tif")) got wrong, and what v2 changes
-----------------------------------------------------------------
1. Any folder containing a .tif became a segment candidate. A sibling folder
   holding only a preview TIFF then raised a missing-label error and aborted
   the whole run, even when a valid segment was present. v2 drops the
   wildcard TIFF gate entirely: a folder is a candidate only because it
   carries the assets a segment needs, never because it happens to contain
   a TIFF.
2. A complete flat mirror (image volume + inklabels + supervision Zarr
   stores, no TIFF at all) was still skipped, even allowlisted. v2
   recognises flat mirrors by their Zarr image/label paths, so no TIFF is
   required for flat discovery. (Main already supported that layout with an
   explicit surface_volume_path and one allowlisted segment; the useful
   addition here is AUTOMATIC discovery of the layout.)
3. Native discovery must keep its geometry requirement: a folder with usable
   labels and patches but no tifxyz geometry passed the v1 gate and failed
   later when the sample reader opened its missing coordinates. v2 restores
   the native geometry check — x.tif must exist where the reader looks for
   it, i.e. directly in the segment directory (main used a recursive
   rglob("x.tif"); a nested x.tif never worked at read time, so the check
   is now a direct-child check).
4. An unfinished mirror (label Zarr directories present but empty — the
   first-sync state) passed discovery and aborted patch finding for the
   whole scan. v2 checks that auto-discovered flat stores actually contain
   chunks, skips unfinished mirrors with a warning, and picks them up on a
   later run once their stores are filled.

Scope of the strictness (this is what keeps the existing tests green)
---------------------------------------------------------------------
The content checks (stores must contain chunks) apply ONLY to
auto-discovered flat segments — folders found by scanning segments_path
with no allowlist entry and no explicit surface volume. Segments the user
explicitly configured (allowlisted, or given an explicit surface/volume
path, local or remote) keep the current name-based acceptance: the user
vouched for them, and the existing test
test_segment_gathering_preserves_remote_and_explicit_volume_paths builds
exactly such segments out of empty Zarr directories and requires them to
be returned. For those, an unfinished store is caught later, per segment,
by patch finding instead of aborting the run (see the optional
dataset_resilience diff; it is NOT part of this file).

Failure policy: one bad folder never kills the run. Every per-directory
failure inside discovery is a warning + skip, except genuine configuration
errors about a segment the user named explicitly (native volume_path
missing, a surface_volume_paths entry missing for an allowlisted segment,
a single surface_volume_path with anything but one allowlisted segment),
which still raise, exactly as before.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import warnings

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Segment


_LABEL_SUFFIXES = (
    ("_inklabels", "inklabels"),
    ("_supervision_mask", "supervision_mask"),
    ("_validation_mask", "validation_mask"),
)
_LABEL_EXTENSIONS = {".tif", ".tiff", ".zarr"}
_REMOTE_VOLUME_PREFIXES = ("s3://", "http://", "https://")

# Zarr store metadata files: their presence alone does not make a store ready.
# Any other file inside the store is a chunk (v2: "0", "1", ...; v3: under
# "c/", ...). A store with only metadata files is an unfinished first-sync
# mirror, not a usable segment.
_ZARR_METADATA_NAMES = frozenset(
    {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}
)


def has_tifxyz_geometry(segment_dir: str | Path) -> bool:
    """True when the tifxyz reader can open coordinates in this directory.

    The reader opens ``x.tif`` by name, directly in the segment directory,
    so the check is a direct-child existence test — not a recursive glob.
    Only presence is checked (an empty x.tif still counts, as in main):
    coordinate completeness is the reader's job, unchanged from main.
    """
    return (Path(segment_dir) / "x.tif").is_file()


def zarr_store_has_chunks(store: str | Path) -> bool:
    """True when a Zarr store holds at least one chunk file.

    Metadata-only stores (``.zarray``/``.zattrs``/``zarr.json`` with no
    chunks) are the first-sync state of an unfinished mirror and do not
    count. Symlinked chunk directories count: mirrors built with symlinks
    (the tutorial's re-download/symlink setup) are followed. Anything that
    cannot be walked — missing path, not a directory, I/O error — is not
    ready and returns False instead of raising.
    """
    store = Path(store)
    if not store.is_dir():
        return False
    try:
        for _root, _dirs, files in os.walk(store, followlinks=True):
            for name in files:
                if name not in _ZARR_METADATA_NAMES:
                    return True
    except OSError:
        return False
    return False


def parse_label_asset_path(path: str | Path) -> dict[str, object] | None:
    """Parse the label basename vocabulary; unversioned means v1."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in _LABEL_EXTENSIONS:
        return None
    stem = path.stem
    version_num = 1
    stem_parts = stem.rsplit("_", 1)
    if (
        len(stem_parts) == 2
        and stem_parts[1].startswith("v")
        and stem_parts[1][1:].isdigit()
    ):
        version_num = int(stem_parts[1][1:])
        stem = stem_parts[0]
    for suffix, label_kind in _LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return {
                "path": path,
                "dir_prefix": path.parent,
                "name": path.name,
                "prefix": stem[: -len(suffix)],
                "label_kind": label_kind,
                "version_num": version_num,
                "extension": extension,
            }
    return None


def _warn_if_validation_unused(
    segment: Segment, selected: Path | None, extension: str
) -> None:
    """Warn when the directory holds validation masks but none was selected."""
    if selected is not None:
        return
    unused = sorted(
        path.name
        for path in segment.segment_dir.iterdir()
        if (parsed := parse_label_asset_path(path)) is not None
        and parsed["label_kind"] == "validation_mask"
    )
    if unused:
        warnings.warn(
            f"{segment.segment_dir}: no validation mask selected, but found "
            f"{unused}; label files must be named "
            f"{segment.segment_name}_<kind>[_vN]{extension} "
            f"(and match label_version if set). This segment has no validation.",
            stacklevel=3,
        )


def discover_segment_labels(
    segment: Segment, *, extension: str = ".zarr", required: bool = True
) -> Segment:
    """Resolve independently greatest versions, or one matching explicit version."""
    normalized_extension = str(extension).lower()
    requested = segment.data_config.label_version
    candidates_by_version: dict[int, dict[str, Path]] = {}
    candidates_by_kind: dict[str, dict[int, Path]] = {
        "inklabels": {},
        "supervision_mask": {},
        "validation_mask": {},
    }
    for path in segment.segment_dir.iterdir():
        parsed = parse_label_asset_path(path)
        if parsed is None:
            continue
        if parsed["prefix"] != segment.segment_name:
            continue
        if str(parsed["extension"]).lower() != normalized_extension:
            continue
        version_num = int(parsed["version_num"])
        kind = str(parsed["label_kind"])
        candidates_by_version.setdefault(version_num, {})[kind] = path
        candidates_by_kind[kind][version_num] = path

    if requested:
        selected = None
        for version_num, record in candidates_by_version.items():
            if (
                "inklabels" in record
                and "supervision_mask" in record
                and f"v{version_num}" == requested
            ):
                selected = record
                break
        if selected is None:
            if required:
                raise ValueError(
                    f"{segment.segment_dir} does not contain matching "
                    f"{normalized_extension} labels for version {requested}"
                )
            return replace(
                segment,
                inklabels=None, supervision_mask=None, validation_mask=None
            )
        _warn_if_validation_unused(
            segment, selected.get("validation_mask"), normalized_extension
        )
        return replace(
            segment,
            inklabels=selected["inklabels"],
            supervision_mask=selected["supervision_mask"],
            validation_mask=selected.get("validation_mask"),
        )

    if required:
        if not candidates_by_kind["inklabels"]:
            raise ValueError(
                f"{segment.segment_dir} must contain at least one inklabels "
                f"{normalized_extension} asset"
            )
        if not candidates_by_kind["supervision_mask"]:
            raise ValueError(
                f"{segment.segment_dir} must contain at least one supervision_mask "
                f"{normalized_extension} asset"
            )
    selected_paths: dict[str, Path | None] = {}
    for kind, candidates in candidates_by_kind.items():
        selected_paths[kind] = candidates[max(candidates)] if candidates else None
    if selected_paths["inklabels"] is not None:
        _warn_if_validation_unused(
            segment, selected_paths["validation_mask"], normalized_extension
        )
    return replace(
        segment,
        inklabels=selected_paths["inklabels"],
        supervision_mask=selected_paths["supervision_mask"],
        validation_mask=selected_paths["validation_mask"],
    )


def _looks_segment_like(segment_dir: Path) -> bool:
    """True when a folder looks like a segment attempt (warn-worthy skip)."""
    for path in segment_dir.iterdir():
        if path.suffix.lower() in _LABEL_EXTENSIONS:
            return True
        if path.is_dir() and path.suffix.lower() == ".zarr":
            return True
    return False


def _drop_empty_validation(segment: Segment) -> Segment:
    """Drop a selected validation mask whose store holds no chunks.

    Validation is optional: an unfinished validation store must not reject
    an otherwise usable segment. Returns the segment unchanged when there
    is no validation mask, or when its store is ready.
    """
    if segment.validation_mask is None:
        return segment
    if zarr_store_has_chunks(segment.validation_mask):
        return segment
    warnings.warn(
        f"{segment.segment_dir}: validation mask "
        f"{segment.validation_mask} holds no chunks yet (unfinished mirror); "
        "continuing without validation for this segment.",
        stacklevel=3,
    )
    return replace(segment, validation_mask=None)


def _resolve_labels_or_skip(segment: Segment, *, required: bool) -> Segment | None:
    """Resolve labels; a folder without usable labels is skipped, not fatal."""
    try:
        return discover_segment_labels(segment, extension=".zarr", required=required)
    except ValueError as exc:
        warnings.warn(
            f"{segment.segment_dir}: skipping segment: {exc}",
            stacklevel=3,
        )
        return None


def _gather_one(
    config: InkDataConfig,
    source,
    dataset_idx: int,
    segment_dir: Path,
    *,
    native_mode: bool,
    explicit_surface: bool,
) -> Segment | None:
    """Gather one directory, or None (with a warning) when it is not usable.

    Raises only for explicit-configuration errors about a segment the user
    named: a missing native volume_path, a surface_volume_paths entry
    missing for an allowlisted segment, or a single surface_volume_path
    with anything but one allowlisted segment.
    """
    allowlist = set(source.segment_names)
    is_allowlisted = segment_dir.name in allowlist
    relpath = segment_dir.relative_to(source.segments_path).as_posix()

    if native_mode:
        # Native discovery keeps its geometry requirement: without x.tif
        # the sample reader cannot open coordinates, so the folder is not
        # a native segment at all.
        if not has_tifxyz_geometry(segment_dir):
            if _looks_segment_like(segment_dir):
                warnings.warn(
                    f"{segment_dir}: skipping segment: no tifxyz geometry "
                    "(x.tif) for native mode; the sample reader opens "
                    "coordinate TIFFs by name.",
                    stacklevel=3,
                )
            return None
        if source.volume_path is None:
            raise ValueError(
                f"datasets[{dataset_idx}].volume_path is required for {config.mode}"
            )
        image_volume: str | Path = source.volume_path
    elif source.surface_volume_paths:
        explicit = source.surface_volume_paths.get(
            relpath, source.surface_volume_paths.get(segment_dir.name)
        )
        if explicit is None:
            if is_allowlisted:
                raise KeyError(
                    "surface_volume_paths has no entry for "
                    f"segment {relpath!r}"
                )
            # Unrelated folder next to an explicitly mapped source: skip it.
            return None
        image_volume = explicit
    elif source.surface_volume_path is not None:
        if len(allowlist) != 1:
            raise ValueError(
                "surface_volume_path is only unambiguous with exactly one "
                "allowlisted segment; use surface_volume_paths for multiple segments"
            )
        image_volume = source.surface_volume_path
    else:
        # Auto-discovered flat mirror: the image source is the conventional
        # <name>.zarr volume next to the labels. No TIFF is required.
        image_volume = segment_dir / f"{segment_dir.name}.zarr"

    segment = Segment(
        data_config=config,
        source=source,
        dataset_idx=dataset_idx,
        segment_relpath=relpath,
        segment_dir=segment_dir,
        segment_name=segment_dir.name,
        image_volume=image_volume,
    )

    explicitly_configured = bool(allowlist) or explicit_surface or native_mode
    required_labels = config.discovery_mode != "unlabeled"

    if not native_mode and not explicit_surface and not is_allowlisted:
        # ---- Auto-discovered flat candidate: strict, content-based gate. ----
        if not Path(image_volume).exists():
            if _looks_segment_like(segment_dir):
                warnings.warn(
                    f"{segment_dir}: skipping segment: no flat image volume "
                    f"{image_volume}; a flat mirror needs its <name>.zarr "
                    "volume next to the labels.",
                    stacklevel=3,
                )
            return None
        if not zarr_store_has_chunks(image_volume):
            warnings.warn(
                f"{segment_dir}: skipping segment: image volume "
                f"{image_volume} holds no chunks yet (unfinished mirror); "
                "it will be picked up once its store is filled.",
                stacklevel=3,
            )
            return None
        segment = _resolve_labels_or_skip(segment, required=required_labels)
        if segment is None:
            return None
        if required_labels:
            if segment.inklabels is None or segment.supervision_mask is None:
                warnings.warn(
                    f"{segment_dir}: skipping segment: incomplete labels "
                    "(inklabels + supervision_mask .zarr required).",
                    stacklevel=3,
                )
                return None
            if not zarr_store_has_chunks(segment.inklabels):
                warnings.warn(
                    f"{segment_dir}: skipping segment: inklabels store "
                    f"{segment.inklabels} holds no chunks yet (unfinished "
                    "mirror); it will be picked up once its store is filled.",
                    stacklevel=3,
                )
                return None
            if not zarr_store_has_chunks(segment.supervision_mask):
                warnings.warn(
                    f"{segment_dir}: skipping segment: supervision store "
                    f"{segment.supervision_mask} holds no chunks yet "
                    "(unfinished mirror); it will be picked up once its "
                    "store is filled.",
                    stacklevel=3,
                )
                return None
        else:
            # Unlabeled discovery: present-but-empty label stores are not
            # usable labels; drop them instead of rejecting the segment.
            if segment.inklabels is not None and not zarr_store_has_chunks(
                segment.inklabels
            ):
                segment = replace(segment, inklabels=None)
            if segment.supervision_mask is not None and not zarr_store_has_chunks(
                segment.supervision_mask
            ):
                segment = replace(segment, supervision_mask=None)
        segment = _drop_empty_validation(segment)
    else:
        # ---- Explicitly configured segment (allowlisted, explicit surface,
        # or native): name-based acceptance, unchanged from main. A segment
        # that carries the mode's own marker (tifxyz geometry in native
        # mode, an explicit volume path in flat mode) but cannot resolve
        # its labels is a broken segment, not an unrelated folder: fail
        # loudly, as main did.
        segment = discover_segment_labels(
            segment, extension=".zarr", required=required_labels
        )

    image_is_remote = str(image_volume).lower().startswith(
        _REMOTE_VOLUME_PREFIXES
    )
    if not image_is_remote and not Path(image_volume).exists():
        if config.discovery_mode == "unlabeled":
            return None
        if explicitly_configured:
            raise ValueError(
                f"{segment_dir.name} is missing its image volume {image_volume}"
            )
        warnings.warn(
            f"{segment_dir}: skipping segment: missing image volume "
            f"{image_volume}.",
            stacklevel=3,
        )
        return None
    if config.discovery_mode != "unlabeled":
        if not (
            segment.supervision_mask is not None
            and segment.inklabels is not None
            and segment.supervision_mask.exists()
            and segment.inklabels.exists()
        ):
            if explicitly_configured:
                raise ValueError(
                    f"{segment_dir.name} is missing required image, supervision, or labels"
                )
            warnings.warn(
                f"{segment_dir}: skipping segment: missing required image, "
                "supervision, or labels.",
                stacklevel=3,
            )
            return None
    return segment


def gather_segments(config: InkDataConfig) -> tuple[Segment, ...]:
    """Discover usable segment representations in stable directory-name order."""
    segments: list[Segment] = []
    native_mode = config.mode in {"full_3d", "full_3d_single_wrap"}
    skipped = 0
    for dataset_idx, source in enumerate(config.active_datasets):
        allowlist = set(source.segment_names)
        explicit_surface = bool(
            source.surface_volume_paths or source.surface_volume_path is not None
        )
        for segment_dir in sorted(source.segments_path.iterdir()):
            if not segment_dir.is_dir() or segment_dir.name == "unused":
                continue
            if allowlist and segment_dir.name not in allowlist:
                continue
            segment = _gather_one(
                config,
                source,
                dataset_idx,
                segment_dir,
                native_mode=native_mode,
                explicit_surface=explicit_surface,
            )
            if segment is None:
                skipped += 1
                continue
            segments.append(segment)
    if not segments and skipped:
        warnings.warn(
            "gather_segments found no usable segment: "
            f"{skipped} director{'y' if skipped == 1 else 'ies'} "
            "skipped (see warnings above); unfinished mirrors are picked up "
            "once their stores are filled.",
            stacklevel=2,
        )
    return tuple(segments)
