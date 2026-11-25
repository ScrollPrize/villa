"""Discover segment directories and select versioned label assets.

v5 -- focused labeled discovery for PR #2677 (review: villa-review,
24/11/2026, fourth review of the v4 head at 1c2f491). The reviewer
recommends merging a FOCUSED discovery change after repair, and says
the wider changes (downloader, cache) add maintenance beyond the
Tutorial 5 workflow. v5 is that narrowing:

- ``download_required_zarr_chunks.py`` is restored to main EXACTLY
  (review 4, point 1, disappears by construction: main already
  refuses an overwrite from an incomplete plan and resumes after
  restoration).
- ``dataset.py`` keeps only the warn-and-skip resilience and the
  failed-segment exclusion with the held-out re-raise (needed by
  reason (b) below, and by B5/X3). The unlabeled cache-coverage check
  (v3) and the probe (v4) are removed: cache reuse is main's again
  (review 4, point 6, disappears by construction).
- ``segment.py`` keeps ONLY the automatic discovery of TIFF-free flat
  mirrors, plus the scale-aware and selection-aware gates review 4
  demands. Everything else is main's behaviour.

What review 4 demanded, and where it lives in v5
--------------------------------------------------
1. Incomplete supervision destroying a valid export (downloader).
   Gone: the downloader is main's.
2. Partly synced sibling stopping training. An auto-discovered flat
   sibling whose labels are only partly there (ink without
   supervision, or the reverse) is skipped with a warning; ready
   segments proceed. Selected inputs, held-out sources and requested
   label versions keep their completeness checks (they raise).
3. Unreadable mask losing its exclusions. A mask store is judged AT
   THE REQUESTED RESOLUTION LEVEL, not at the store root: a level
   whose subgroup is absent is unreadable, never silently cleared.
   And a validation mask with metadata but no chunks at the required
   level is an unfinished sync, not a legitimate all-zero mask: a
   newly admitted segment must not contribute training patches on
   the strength of it. A validation mask WITH chunks (even all-zero
   content) stays a legitimate sparse mask. The same ambiguity exists
   for inputs main already discovers; those keep main's behaviour.
4. Held-out preflight counting unscorable patches. A newly admitted
   held-out segment must supply the image AND the labels with chunks
   (image, validation) or readable metadata (ink, supervision) AT
   THE SCALE THEY WILL BE READ (volume_scale for sampling,
   patch_finding_scale for patch finding). Labels available only at
   the scan scale no longer pass preflight (the bot's discarded
   reason (a)).
5. Allowlist losing control. Map keys and allowlist names that match
   no folder are INERT (main never consults them); the v3/v4
   completeness check that raised for them is removed. A read
   failure on a SELECTED segment (allowlisted, explicitly mapped, or
   held-out) stops the run before it trains or scores a subset --
   enforced in dataset.py, at patch finding, as main does it: main
   gathers such a segment (the repo test
   test_segment_gathering_preserves_remote_and_explicit_volume_paths
   requires it) and stops when its masks cannot be read, so v5
   gathers it too and the dataset re-raises a patch-finding failure
   on a selected segment instead of skipping it.
6. Unlabeled cache rescanning. Gone: the cache is main's.

Reason (b) for the bot discarding its own narrowing -- a complete
sibling below Tutorial 5's ink-coverage threshold aborting a scan
that otherwise yields usable patches -- is fixed by the dataset.py
resilience kept in v5: a segment that yields no patches at patch
finding is warned about and skipped, and the run proceeds on the
ready segments (it still fails loudly if NO segment yields patches).
Discovery cannot know the coverage without re-implementing patch
finding, so the fix belongs there, not in discovery.

Scope of the new gates (this is what keeps the change focused)
---------------------------------------------------------------
The scale-aware gates (image chunks, label readability, validation
chunks at the required levels) apply ONLY to segments the PR admits
that main would not: flat mirrors with no tifxyz geometry and no
explicit surface/volume path, whether auto-discovered or allowlisted.
Segments main already discovers -- native geometry (x.tif), an
explicit surface_volume_paths entry, or a single surface_volume_path
-- keep main's name-based acceptance, with no addition at discovery:
a mask that cannot be read there is main's business at patch
finding, and the dataset (point 5) stops the run then for a
selected segment.

A folder is a segment candidate when it carries an image volume,
native geometry, or Zarr label assets. A folder with ONLY its
original label TIFFs (no image, no geometry, no Zarr labels) is not
a candidate: it is skipped silently, even in a held-out source
(the v4 H2 rule is reversed -- Y6 -- because main skips it too and
the outcome, "not counted as a validation segment", is the same).
A preview TIFF is still not a marker.

Failure policy: a recognised segment that cannot be used fails
loudly when it was selected (allowlisted, explicitly mapped, or in a
held-out source) or when validation is at stake; an auto-discovered
training segment in the same state is warned about and skipped, and
picked up on a later run once its stores are ready. A folder with no
segment markers is never a failure: it is skipped silently.
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


def _zarr_level_dir(store: str | Path, level: int | str) -> Path | None:
    """Directory holding one resolution level of a Zarr store, or None.

    A Zarr store is either an ARRAY (``.zarray`` / ``zarr.json`` with
    node_type "array" at its root: the level must be 0 and lives at the
    root) or a GROUP (``.zgroup`` / ``zarr.json`` group: each level is
    a subgroup named after its resolution). ``open_volume`` selects the
    level exactly this way (``root[str(resolution)]`` for a group), so
    a level whose directory is absent is unreadable at that level --
    even when the store root carries metadata (the review-4 point-3
    case: ``.zgroup`` present but the requested level absent).
    """
    import json

    store = Path(store)
    if not store.is_dir():
        return None
    level_text = str(level)
    if (store / ".zarray").is_file():
        return store if level_text in {"0", ""} else None
    if (store / ".zgroup").is_file():
        candidate = store / level_text
        return candidate if candidate.is_dir() else None
    zarr_json = store / "zarr.json"
    if zarr_json.is_file():
        try:
            node_type = json.loads(zarr_json.read_text()).get("node_type")
        except (OSError, ValueError):
            return None
        if node_type == "array":
            return store if level_text in {"0", ""} else None
        if node_type == "group":
            candidate = store / level_text
            return candidate if candidate.is_dir() else None
    return None


def zarr_store_level_has_metadata(store: str | Path, level: int | str) -> bool:
    """True when a Zarr store carries metadata at the requested level.

    This is the level-aware form of :func:`zarr_store_has_metadata`:
    a store whose root has metadata but whose requested level subgroup
    is absent is NOT readable at that level (``open_volume`` raises
    FileNotFoundError there), so it must never be treated as readable.
    """
    level_dir = _zarr_level_dir(store, level)
    if level_dir is None:
        return False
    try:
        return any(
            (level_dir / name).is_file() for name in _ZARR_METADATA_NAMES
        )
    except OSError:
        return False


def zarr_store_level_has_chunks(store: str | Path, level: int | str) -> bool:
    """True when a Zarr store holds at least one chunk file at the level.

    The level-aware form of :func:`zarr_store_has_chunks`. A store
    with metadata at level 0 and chunks only at level 1 does NOT have
    chunks at level 0 (the review-4 point-4 case: the first image read
    at volume_scale=0 would be all zeros). Symlinked chunk
    directories count, as in the whole-store check.
    """
    level_dir = _zarr_level_dir(store, level)
    if level_dir is None:
        return False
    try:
        for _root, _dirs, files in os.walk(level_dir, followlinks=True):
            for name in files:
                if name not in _ZARR_METADATA_NAMES:
                    return True
    except OSError:
        return False
    return False


def _required_levels(segment: Segment) -> tuple[int, ...]:
    """Resolution levels a segment's stores must serve.

    The sample scale (``segment.scale`` == the source's volume_scale)
    is always required: samples read image and labels there. The
    patch-finding scale (``patch_finding.scan_scale``, from the
    ``patch_finding_scale`` config key) is additionally required when
    it differs: patch finding reads image and labels there. A store
    that serves only the scan scale passes patch finding and then
    fails on the first sample (FileNotFoundError) -- the bot's
    discarded reason (a) -- so both levels are gated.
    """
    scale = int(segment.scale)
    scan = segment.data_config.patch_finding.scan_scale
    if scan is None or int(scan) == scale:
        return (scale,)
    return (scale, int(scan))


def _has_original_label_tiffs(segment_dir: Path) -> bool:
    """True when the folder holds original label TIFFs (the TIFF files
    the Zarr labels were converted from).

    A ``.tif``/``.tiff`` whose stem, ignoring a trailing ``_vN`` version,
    ends with a label kind name (``inklabels``, ``supervision_mask``,
    ``validation_mask``) — with or without a segment-name prefix. These
    are segment markers even when no Zarr label asset exists yet: a
    sibling with only its label TIFFs is a segment attempt, not an
    unrelated folder, so it must not disappear silently from a scan.
    A preview TIFF (any other name) is still not a marker.
    """
    kinds = tuple(kind for _suffix, kind in _LABEL_SUFFIXES)
    try:
        for path in segment_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            stem = path.stem
            parts = stem.rsplit("_", 1)
            if (
                len(parts) == 2
                and parts[1].startswith("v")
                and parts[1][1:].isdigit()
            ):
                stem = parts[0]
            if stem.endswith(kinds):
                return True
    except OSError:
        return False
    return False


def _segment_markers(segment_dir: Path) -> dict[str, bool]:
    """Which segment markers a folder carries (tight definition).

    A preview TIFF is NOT a marker, and neither are original label
    TIFFs on their own: only the assets a segment is made of count --
    native geometry (``x.tif``), the conventional flat image volume
    (``<name>.zarr``), or Zarr label assets whose parsed prefix equals
    the folder name. A folder with only its original label TIFFs (no
    image, no geometry, no Zarr labels) is not a segment yet -- it is
    skipped silently, even in a held-out source (the v4 H2 rule is
    reversed in v5: main skips it too, and the outcome -- not counted
    as a validation segment -- is the same). The ``label_tifs`` entry
    is kept only for the warn-when-skipping decision. A folder with
    at least one of geometry / image / labels is a recognised segment
    (or segment attempt); a folder with none is unrelated and may be
    skipped silently.
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
    has_label_tifs = _has_original_label_tiffs(segment_dir)
    return {
        "geometry": has_geometry,
        "image": has_image,
        "labels": has_labels,
        "label_tifs": has_label_tifs,
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
    segment_dir: Path, *, strict: bool, message: str
) -> None:
    """Fail loudly for a selected segment, warn-and-skip otherwise.

    A segment that was selected (allowlisted, explicitly mapped, or in
    a held-out source) and cannot be used must fail loudly: silently
    dropping it would train or score a subset while reporting the
    selected set as complete (review 4, point 5). An auto-discovered
    training segment in the same state is skipped with a warning and
    picked up on a later run once its stores are ready.
    """
    if strict:
        raise ValueError(f"{segment_dir}: {message}")
    warnings.warn(f"{segment_dir}: {message}", stacklevel=3)


def _label_assets_by_kind(segment_dir: Path) -> dict[str, list[Path]]:
    """Zarr label assets per kind, by the segment's own name prefix."""
    found: dict[str, list[Path]] = {
        "inklabels": [],
        "supervision_mask": [],
        "validation_mask": [],
    }
    try:
        for path in segment_dir.iterdir():
            parsed = parse_label_asset_path(path)
            if parsed is None or parsed["prefix"] != segment_dir.name:
                continue
            if str(parsed["extension"]).lower() != ".zarr":
                continue
            found[str(parsed["label_kind"])].append(path)
    except OSError:
        pass
    return found


def _resolve_labels_or_fail(
    segment: Segment, *, required: bool, strict: bool
) -> Segment | None:
    """Resolve labels for a PR-admitted flat segment.

    Returns the segment with its labels, or None (with a warning) when
    the labels have not arrived yet. A ValueError from
    :func:`discover_segment_labels` has two distinct causes, and they
    are treated differently:

    - Version mismatch: ink and supervision assets both exist, but no
      version carries both at the requested version. That is a
      configuration error, never an unfinished sync: it raises,
      always, even for an auto-discovered segment.
    - Missing or partly synced labels: no label asset at all (C1: the
      image synced first), or only one of ink/supervision has arrived
      (review 4, point 2). The sibling is unfinished: it is skipped
      with a warning and training proceeds on the ready segments, and
      it is picked up on a later run once its stores are filled. A
      selected segment (``strict``: allowlisted, explicitly mapped, or
      held-out) raises instead -- its labels were requested, so their
      absence is a failure, not a sync in progress.

    This function only runs for segments the PR admits (flat, no
    tifxyz geometry, no explicit surface). Segments main already
    discovers call :func:`discover_segment_labels` directly, so their
    completeness checks are unaffected. In unlabeled mode
    (``required`` False) ``discover_segment_labels`` never raises for
    missing labels, so this function simply returns the segment,
    possibly with ``None`` labels.
    """
    try:
        return discover_segment_labels(segment, extension=".zarr", required=required)
    except ValueError as exc:
        assets = _label_assets_by_kind(segment.segment_dir)
        if assets["inklabels"] and assets["supervision_mask"]:
            # Both label kinds exist: the failure is a version
            # mismatch, not an unfinished sync.
            raise ValueError(
                f"{segment.segment_dir}: {exc} (segment recognised but its "
                "labels cannot be used; fix the labels or the requested "
                "label_version)"
            ) from exc
        if strict:
            raise
        warnings.warn(
            f"{segment.segment_dir}: skipping segment: labels have not "
            "arrived yet (unfinished mirror); training proceeds on the "
            "ready segments, and it will be picked up once its labels "
            "arrive.",
            stacklevel=3,
        )
        return None


def _gather_pr_admitted(
    config: InkDataConfig,
    source,
    dataset_idx: int,
    segment_dir: Path,
    *,
    relpath: str,
    markers: dict[str, bool],
    held_out: bool,
    is_allowlisted: bool,
) -> Segment | None:
    """Gather a flat mirror main would not discover (no x.tif, no explicit surface).

    This is the PR's focused discovery: a TIFF-free flat mirror is
    admitted on the strength of its Zarr stores, but only when those
    stores serve the resolutions the run will read (``_required_levels``).
    A folder with no segment markers at all is not a segment -- it is
    skipped silently, even when allowlisted or held-out (Y5/Y6: main
    skips it too). A folder with markers but unfinished stores fails
    loudly when it was selected (allowlisted or held-out) and is warned
    about and skipped otherwise.
    """
    strict = held_out or is_allowlisted
    required_labels = config.discovery_mode != "unlabeled"
    image_volume: str | Path = segment_dir / f"{segment_dir.name}.zarr"

    if not markers["recognized"]:
        # Not a segment at all (no image, no geometry, no Zarr labels).
        # A folder with only its original label TIFFs is not a segment
        # yet either.
        return None
    if not Path(image_volume).exists():
        _fail_or_skip(
            segment_dir,
            strict=strict,
            message=(
                "no flat image volume "
                f"{image_volume}; a flat mirror needs its <name>.zarr "
                "volume next to the labels."
            ),
        )
        return None

    segment = Segment(
        data_config=config,
        source=source,
        dataset_idx=dataset_idx,
        segment_relpath=relpath,
        segment_dir=segment_dir,
        segment_name=segment_dir.name,
        image_volume=image_volume,
    )
    levels = _required_levels(segment)

    # (a) The image must hold chunks at every required level. A store
    # with metadata at level 0 and chunks only at level 1 reads as all
    # zeros at volume_scale=0 (review 4, point 4).
    for level in levels:
        if not zarr_store_level_has_chunks(image_volume, level):
            _fail_or_skip(
                segment_dir,
                strict=strict,
                message=(
                    "image volume "
                    f"{image_volume} holds no chunks at resolution {level} "
                    "(unfinished mirror); it will be picked up once its "
                    "store is filled."
                ),
            )
            return None

    # (b) Labels: a partly synced sibling is skipped (point 2), a
    # version mismatch always raises.
    segment = _resolve_labels_or_fail(
        segment, required=required_labels, strict=strict
    )
    if segment is None:
        return None

    if required_labels:
        if segment.inklabels is None or segment.supervision_mask is None:
            # discover_segment_labels with required=True raises for
            # these, so this is belt-and-braces.
            _fail_or_skip(
                segment_dir,
                strict=strict,
                message=(
                    "incomplete labels (inklabels + supervision_mask .zarr "
                    "required)."
                ),
            )
            return None
        # (c) Supervision and ink must be READABLE at every required
        # level. A readable store without chunks is an all-zero mask --
        # the label writer omits fill-only chunks -- so the segment is
        # ready: it yields no training patches but may yield validation
        # patches (C3). No chunk gate here.
        for kind in ("supervision_mask", "inklabels"):
            store = getattr(segment, kind)
            for level in levels:
                if not zarr_store_level_has_metadata(store, level):
                    _fail_or_skip(
                        segment_dir,
                        strict=strict,
                        message=(
                            f"{kind} store {store} is unreadable at "
                            f"resolution {level} (unfinished mirror); it "
                            "will be picked up once its store is filled."
                        ),
                    )
                    return None
        # (d) A validation mask, when present, must hold CHUNKS at every
        # required level. Metadata without chunks is an unfinished sync,
        # not a legitimate all-zero mask: treating it as all-zero would
        # let the segment contribute training patches that its real
        # validation regions (once synced) would have held out
        # (review 4, point 3). A mask WITH chunks -- even all-zero
        # content -- stays a legitimate sparse mask.
        if segment.validation_mask is not None:
            for level in levels:
                if not zarr_store_level_has_chunks(
                    segment.validation_mask, level
                ):
                    _fail_or_skip(
                        segment_dir,
                        strict=strict,
                        message=(
                            "validation mask "
                            f"{segment.validation_mask} holds no chunks at "
                            f"resolution {level} (unfinished sync); the "
                            "segment waits until its validation regions "
                            "can be read."
                        ),
                    )
                    return None
    else:
        # Unlabeled discovery: present-but-empty label stores are not
        # usable labels; drop them instead of rejecting the segment.
        # A mask that cannot be READ at a required level raises (C4 /
        # review 4, point 3): dropping it would silently bypass the
        # supervision exclusion. A readable but empty mask is dropped:
        # it excludes nothing.
        if segment.inklabels is not None and not any(
            zarr_store_level_has_chunks(segment.inklabels, level)
            for level in levels
        ):
            segment = replace(segment, inklabels=None)
        if segment.supervision_mask is not None:
            if not all(
                zarr_store_level_has_metadata(segment.supervision_mask, level)
                for level in levels
            ):
                raise ValueError(
                    f"{segment.supervision_mask} is unreadable "
                    "(unfinished mirror); a selected supervision mask "
                    "must be readable before its segment supplies "
                    "unlabeled patches"
                )
            if not any(
                zarr_store_level_has_chunks(segment.supervision_mask, level)
                for level in levels
            ):
                segment = replace(segment, supervision_mask=None)
        if segment.validation_mask is not None:
            if not all(
                zarr_store_level_has_metadata(segment.validation_mask, level)
                for level in levels
            ):
                raise ValueError(
                    f"{segment.validation_mask} is unreadable "
                    "(unfinished mirror); a selected validation mask "
                    "must be readable before its segment supplies "
                    "unlabeled patches"
                )
            if not any(
                zarr_store_level_has_chunks(segment.validation_mask, level)
                for level in levels
            ):
                segment = replace(segment, validation_mask=None)

    if required_labels:
        if not (
            segment.supervision_mask is not None
            and segment.inklabels is not None
            and segment.supervision_mask.exists()
            and segment.inklabels.exists()
        ):
            _fail_or_skip(
                segment_dir,
                strict=strict,
                message=(
                    "missing required image, supervision, or labels."
                ),
            )
            return None
    return segment


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

    Two populations, two rules. Segments main already discovers --
    native geometry (``x.tif``), an explicit ``surface_volume_paths``
    entry, or a single ``surface_volume_path`` -- keep main's
    name-based acceptance, with no addition at discovery: a mask
    that cannot be read there is main's business at patch finding
    (the dataset stops the run then for a selected segment, review
    4, point 5). Segments only the PR admits -- flat mirrors with no
    tifxyz geometry and no explicit surface -- go through the
    scale-aware gates of :func:`_gather_pr_admitted`.

    A folder with no segment markers at all is never a failure: it is
    skipped silently, even when allowlisted or in a held-out source
    (Y5/Y6). A recognised segment that cannot be used fails loudly
    when it was selected (allowlisted, explicitly mapped, or held-out)
    and is warned about and skipped otherwise, to be picked up on a
    later run once its stores are ready.
    """
    allowlist = set(source.segment_names)
    is_allowlisted = segment_dir.name in allowlist
    relpath = segment_dir.relative_to(source.segments_path).as_posix()
    held_out = source.split == "validation"
    markers = _segment_markers(segment_dir)
    is_native = has_tifxyz_geometry(segment_dir)
    # A segment is "selected" when the user asked for it by name, gave
    # it an explicit volume, or when it belongs to the validation set.
    strict = held_out or is_allowlisted or explicit_surface
    required_labels = config.discovery_mode != "unlabeled"

    if native_mode:
        # Native discovery keeps its geometry requirement: without x.tif
        # the sample reader cannot open coordinates, so the folder is not
        # a native segment at all. An allowlisted folder without x.tif is
        # skipped like any other (v5 reverses the v4 H1 raise: main
        # skips it too -- Y7).
        if not is_native:
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
            if markers["recognized"]:
                # An incomplete explicit map must fail preflight: the
                # folder is a segment the map forgot, and silently
                # dropping it would report a partial dataset as complete.
                # (Main raises KeyError for any unmapped folder; v5
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
        # Flat, no explicit surface. A mirror with native geometry is a
        # main-discovered segment; a TIFF-free mirror is the PR's
        # focused discovery.
        if not is_native:
            return _gather_pr_admitted(
                config,
                source,
                dataset_idx,
                segment_dir,
                relpath=relpath,
                markers=markers,
                held_out=held_out,
                is_allowlisted=is_allowlisted,
            )
        image_volume = segment_dir / f"{segment_dir.name}.zarr"

    # ---- Segments main already discovers: name-based acceptance. ----
    segment = Segment(
        data_config=config,
        source=source,
        dataset_idx=dataset_idx,
        segment_relpath=relpath,
        segment_dir=segment_dir,
        segment_name=segment_dir.name,
        image_volume=image_volume,
    )
    segment = discover_segment_labels(
        segment, extension=".zarr", required=required_labels
    )
    image_is_remote = str(image_volume).lower().startswith(
        _REMOTE_VOLUME_PREFIXES
    )
    if not image_is_remote and not Path(image_volume).exists():
        if config.discovery_mode == "unlabeled":
            if strict and markers["recognized"]:
                # A selected segment must not disappear silently (H1):
                # its folder exists and carries segment markers, but
                # its image volume is missing.
                raise ValueError(
                    f"{segment_dir.name} is missing its image volume {image_volume}"
                )
            return None
        raise ValueError(
            f"{segment_dir.name} is missing its image volume {image_volume}"
        )
    if config.discovery_mode != "unlabeled":
        if not (
            segment.supervision_mask is not None
            and segment.inklabels is not None
            and segment.supervision_mask.exists()
            and segment.inklabels.exists()
        ):
            raise ValueError(
                f"{segment_dir.name} is missing required image, supervision, or labels"
            )
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
        # No configuration-completeness check here (v5 removes the v3/v4
        # one): map keys and allowlist names that match no folder are
        # INERT, as in main -- gather iterates folders, never map keys,
        # so an unselected key is never consulted (review 4, point 5).
        # A typo'd allowlist or map key therefore yields no segment for
        # that name, and the warning below fires when nothing at all
        # was gathered.
    if not segments and skipped:
        warnings.warn(
            "gather_segments found no usable segment: "
            f"{skipped} director{'y' if skipped == 1 else 'ies'} "
            "skipped (see warnings above); unfinished mirrors are picked up "
            "once their stores are filled.",
            stacklevel=2,
        )
    return tuple(segments)
