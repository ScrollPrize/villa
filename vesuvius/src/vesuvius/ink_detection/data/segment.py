"""Discover segment directories and select versioned label assets.

v3 — targeted flat-discovery fix for PR #2677 (review: villa-review, 24/11/2026,
second review of the v2 head at 8d50b31).

What v2 got wrong, and what v3 changes
----------------------------------------
1. An incomplete explicit map silently shrank the validation set. With
   surface_volume_paths naming only one of two valid segments and no
   allowlist, v2 omitted the unmapped segment and passed preflight with
   half the validation patches; main raises KeyError. v3 raises KeyError
   for an unmapped folder that carries segment markers (like main), and
   still skips silently only folders with no markers at all (a preview
   folder is not a segment). Same for label_version: a segment whose
   labels exist but match no requested version now fails loudly, like
   main, instead of warning and accepting the rest.
2. A recognized segment must be admitted whole or fail loudly. Only
   folders with no segment markers may be skipped silently. A held-out
   source (split == "validation") never skips a recognized segment: an
   unfinished store, missing labels or a version mismatch there raises,
   so preflight cannot report a partial validation set as complete.
3. Readable sparse labels were dropped. villa's label writer omits
   fill-only chunks, so a valid all-background ink store looks exactly
   like an unfinished mirror to a chunk-presence gate. v3 distinguishes
   the three states of a store — unreadable (no metadata), readable
   without chunks, ready — and uses the neighbouring store to decide:
   an ink store that is readable without chunks is sparse (admitted)
   when the supervision store has chunks; a supervision store without
   chunks can yield no supervised patches, so the segment waits.
4. A selected validation mask no longer gets dropped. v2 discarded an
   empty validation mask and trained over held-out pixels. v3 makes the
   segment wait (skip in training, fail in held-out) until its
   validation regions can be read.
5. The chunk-content gate is kept for the image and supervision stores
   only. The ink store is no longer gated on chunks.

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
by patch finding instead of aborting the run (see dataset.py).

Failure policy: a recognised segment that cannot be used fails loudly
when validation is at stake (held-out source, explicit-map
incompleteness, requested label version absent, missing labels) and is
warned about and skipped otherwise. A folder with no segment markers is
never a failure: it is skipped silently.
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


def zarr_store_has_metadata(store: str | Path) -> bool:
    """True when a Zarr store carries metadata (readable), even without chunks.

    A store with a metadata file (``.zarray``/``.zattrs``/``.zgroup``/
    ``.zmetadata``/``zarr.json``) at its root can be opened: reads return
    the fill value for missing chunks. A bare directory (no metadata at
    all) cannot be opened — it is unreadable, not merely empty. Together
    with :func:`zarr_store_has_chunks` this distinguishes the three
    states of a store: unreadable, readable-but-empty, ready.
    """
    store = Path(store)
    if not store.is_dir():
        return False
    try:
        return any(
            (store / name).is_file() for name in _ZARR_METADATA_NAMES
        )
    except OSError:
        return False


def _segment_markers(segment_dir: Path) -> dict[str, bool]:
    """Which segment markers a folder carries (tight definition).

    A preview TIFF is NOT a marker: only the assets a segment is made
    of count — native geometry (``x.tif``), the conventional flat image
    volume (``<name>.zarr``), or label assets whose parsed prefix equals
    the folder name. A folder with at least one marker is a recognised
    segment (or segment attempt); a folder with none is unrelated and
    may be skipped silently.
    """
    name = segment_dir.name
    has_geometry = (segment_dir / "x.tif").is_file()
    has_image = (segment_dir / f"{name}.zarr").is_dir()
    has_labels = False
    try:
        for path in segment_dir.iterdir():
            parsed = parse_label_asset_path(path)
            if parsed is not None and parsed["prefix"] == name:
                has_labels = True
                break
    except OSError:
        pass
    return {
        "geometry": has_geometry,
        "image": has_image,
        "labels": has_labels,
        "recognized": has_geometry or has_image or has_labels,
    }


def _label_assets_present(segment_dir: Path) -> bool:
    """True when the folder holds label assets under its own name.

    Any version, any supported extension: the question is only whether
    labels exist at all (a version mismatch is a different failure from
    missing labels, and both must fail loudly for a recognised segment).
    """
    return _segment_markers(segment_dir)["labels"]


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


def _fail_or_skip(
    segment_dir: Path, *, held_out: bool, message: str
) -> None:
    """Fail loudly for a held-out segment, warn-and-skip otherwise.

    A recognised held-out segment that cannot be used must fail
    preflight: silently dropping it would report a partial validation
    set as complete. A training segment in the same state is skipped
    with a warning and picked up on a later run once its stores are
    ready.
    """
    if held_out:
        raise ValueError(f"{segment_dir}: {message}")
    warnings.warn(f"{segment_dir}: {message}", stacklevel=3)


def _resolve_labels_or_fail(
    segment: Segment, *, required: bool, held_out: bool
) -> Segment | None:
    """Resolve labels; a recognised segment fails loudly on bad labels.

    A ValueError from :func:`discover_segment_labels` means either the
    requested label version matches nothing (the segment's labels exist
    but in another version) or the labels are missing entirely. Both
    are configuration errors for a recognised segment — main refuses
    them — so they raise here, except when the folder carries no segment
    markers at all (not a segment: skip silently).
    """
    try:
        return discover_segment_labels(segment, extension=".zarr", required=required)
    except ValueError as exc:
        if not _segment_markers(segment.segment_dir)["recognized"]:
            return None
        if held_out:
            raise
        raise ValueError(
            f"{segment.segment_dir}: {exc} (segment recognised but its "
            "labels cannot be used; fix the labels or the requested "
            "label_version)"
        ) from exc


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

    Raises for configuration errors and for recognised segments that
    cannot be used when validation is at stake: a missing native
    volume_path, a surface_volume_paths entry missing for an allowlisted
    segment or for any other recognised segment, a single
    surface_volume_path with anything but one allowlisted segment, a
    requested label version that matches nothing, missing labels, and
    any unfinished or unreadable store of a recognised segment in a
    held-out source. A training segment in the same state is warned
    about and skipped instead, and picked up on a later run.
    """
    allowlist = set(source.segment_names)
    is_allowlisted = segment_dir.name in allowlist
    relpath = segment_dir.relative_to(source.segments_path).as_posix()
    held_out = source.split == "validation"
    markers = _segment_markers(segment_dir)

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
            if markers["recognized"]:
                # An incomplete explicit map must fail preflight: the
                # folder is a segment the map forgot, and silently
                # dropping it would report a partial dataset as complete.
                # (Main raises KeyError for any unmapped folder; v3
                # narrows the raise to folders carrying segment markers
                # so an unrelated folder, e.g. a preview-TIFF folder,
                # still does not abort the run.)
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
            if markers["labels"] or markers["geometry"]:
                _fail_or_skip(
                    segment_dir,
                    held_out=held_out,
                    message=(
                        "skipping segment: no flat image volume "
                        f"{image_volume}; a flat mirror needs its <name>.zarr "
                        "volume next to the labels."
                    ),
                )
                return None
            if _looks_segment_like(segment_dir):
                warnings.warn(
                    f"{segment_dir}: skipping segment: no flat image volume "
                    f"{image_volume}; a flat mirror needs its <name>.zarr "
                    "volume next to the labels.",
                    stacklevel=3,
                )
            return None
        if not zarr_store_has_chunks(image_volume):
            _fail_or_skip(
                segment_dir,
                held_out=held_out,
                message=(
                    "skipping segment: image volume "
                    f"{image_volume} holds no chunks yet (unfinished mirror); "
                    "it will be picked up once its store is filled."
                ),
            )
            return None
        segment = _resolve_labels_or_fail(
            segment, required=required_labels, held_out=held_out
        )
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
            if not zarr_store_has_chunks(segment.supervision_mask):
                # A supervision store without chunks holds no supervised
                # voxels, so the segment could yield no patches anyway.
                _fail_or_skip(
                    segment_dir,
                    held_out=held_out,
                    message=(
                        "skipping segment: supervision store "
                        f"{segment.supervision_mask} holds no chunks yet "
                        "(unfinished mirror); it will be picked up once "
                        "its store is filled."
                    ),
                )
                return None
            if not zarr_store_has_metadata(segment.inklabels):
                # The ink store cannot be opened at all: the segment is
                # unfinished, not sparse. (A readable ink store without
                # chunks is admitted below as sparse labels.)
                _fail_or_skip(
                    segment_dir,
                    held_out=held_out,
                    message=(
                        "skipping segment: inklabels store "
                        f"{segment.inklabels} is unreadable (unfinished "
                        "mirror); it will be picked up once its store is "
                        "filled."
                    ),
                )
                return None
            # No chunk gate on the ink store (point 3): a readable ink
            # store without chunks is an all-background store — the
            # label writer omits fill-only chunks — and the supervision
            # store above proves the segment is synchronised. It stays
            # eligible whenever the selected filter allows background
            # examples.
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
        if segment.validation_mask is not None and not zarr_store_has_chunks(
            segment.validation_mask
        ):
            # A selected validation mask must govern training supervision,
            # or the segment must wait until its validation regions can
            # be read. Dropping the mask (v2) would train over held-out
            # pixels, so the segment waits instead.
            _fail_or_skip(
                segment_dir,
                held_out=held_out,
                message=(
                    "skipping segment: validation mask "
                    f"{segment.validation_mask} holds no chunks yet "
                    "(unfinished mirror); the segment waits until its "
                    "validation regions can be read."
                ),
            )
            return None
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
        # ---- Configuration completeness (v3): every explicitly named
        # segment must exist. A name in the allowlist or in
        # surface_volume_paths that matches no folder under
        # segments_path is a configuration error (a typo, a folder that
        # has not synced yet) and fails loudly: preflight must not
        # report a dataset as complete when an explicitly requested
        # segment is absent. A map key naming a folder that exists but
        # is excluded by the allowlist is ignored: the allowlist wins,
        # it is the explicit selection, and the map is never consulted
        # for an excluded folder (same as main).
        present_names = {
            d.name
            for d in Path(source.segments_path).iterdir()
            if d.is_dir()
        }
        for name in sorted(allowlist):
            if name not in present_names:
                raise ValueError(
                    f"{source.segments_path}: allowlist names segment "
                    f"{name!r}, but no such folder exists under segments_path"
                )
        for key in sorted(source.surface_volume_paths):
            folder_name = Path(key).name
            if folder_name not in present_names:
                raise KeyError(
                    f"surface_volume_paths names segment {key!r}, but no "
                    f"such folder exists under {source.segments_path}"
                )
            # A key naming a folder that exists but is excluded by the
            # allowlist is inert: the allowlist wins, the map is never
            # consulted for an excluded folder (same as main).
    if not segments and skipped:
        warnings.warn(
            "gather_segments found no usable segment: "
            f"{skipped} director{'y' if skipped == 1 else 'ies'} "
            "skipped (see warnings above); unfinished mirrors are picked up "
            "once their stores are filled.",
            stacklevel=2,
        )
    return tuple(segments)
