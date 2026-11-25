"""Discover segment directories and select versioned label assets.

v8 -- focused LABELED discovery for PR #2677 (review: villa-review,
25/11/2026, seventh review of the v7 head at 8f34af3). The reviewer
still recommends merging a FOCUSED discovery change after repair.
v8 repairs the six points of review 7 on top of the v7 base:

1. Mirror completeness (point 1, both directions). The v7 cross-level
   fill proof's UPWARD direction (a present coarser chunk whose
   footprint is all fill proves the missing fine chunk is fill) is
   UNSOUND for thin features: nearest-neighbour downsampling erases a
   one-voxel stroke, so a deleted chunk holding one reads as fill at
   the coarser level. The upward direction is dropped. What replaces
   it: (a) the downward proof is kept (finer chunks present and fill
   prove the coarser chunk is fill -- sound for nearest and mean);
   (b) the upward NON-FILL proof is kept (a coarser chunk present and
   non-fill in the footprint proves the missing chunk is NOT fill --
   sound); (c) NEW: a missing chunk whose covering chunks at the
   immediately coarser level are ALL ABSENT, in a store that holds at
   least one chunk somewhere, is fill -- the writer omitted a
   chunk-aligned empty region at every resolution (review 7, point 2).
   Deliberate, disclosed trade-off: the review-6 F1 writer mask (1
   populated + 1 omitted fill chunk, admitted by v7 via the upward
   proof) is now REJECTED -- it is observationally identical to the
   review-7 1b tamper (one fine chunk deleted, thin stroke erased at
   the coarser level), and no signal in the store distinguishes them.
   The fully-stripped 6-level mask (review 7, point 1a) is
   byte-identical to a genuine uniform mask and stays admitted by the
   uniform exception: provably undecidable, documented, not fixed.
2. Sparse ink pyramids (point 2). A complete writer pyramid whose
   empty area spans chunks at every level is now admitted by the new
   coarser-absent rule (1c above): the writer omitted those fill
   chunks, and no coarser chunk exists to prove otherwise. The rule
   requires at least one present chunk in the store, so the review-6
   V2 metadata-only 2-level store stays rejected.
3. Chunk-key validation (point 3). The key parser now validates the
   EXACT storage key against the store's declared format and
   separator: v2 dot-separated canonical integers only (no leading
   zeros, no slashes, no backslashes); v2 slash-separated nested
   paths only; v3 ``c/<ints>`` only, with no backslash anywhere. A
   file named ``0\\0\\0`` is ONE filename, never a nested path.
4. Reader agreement (point 4). Readiness content reads go through
   ``open_volume`` (vc_reader) -- the same reader the training path
   uses -- instead of Python zarr. Discovery therefore works in the
   documented ``models`` install (which declares ``vc-reader`` but no
   Python ``zarr``) and agrees with the training reader on v3 stores.
   segment.py no longer imports zarr at all.
5. Unlabeled missing image (point 5). In unlabeled mode a missing
   local image is ALWAYS skipped, whether flat or explicitly mapped:
   that is main's rule (``continue``), and v7's H1a raise for an
   explicitly configured image contradicted it. The V8 control now
   expects main's KeyError (the ready segment has no map entry).
6. Cache-first discovery (point 6). ``gather_segments`` takes a
   ``light`` mode (filesystem checks only, no chunk decoding); the
   dataset gathers lightly, tries the cache, and only on a miss
   re-gathers with full content validation. A store fingerprint
   (exact chunk inventory -- relative paths, sizes, mtimes -- hashed
   per store) is recorded at cache-write time and verified on a hit
   by listing only, so a chunk removed after caching invalidates the
   cache and the fresh run detects it. Selection helpers
   (``_is_selected_segment``, ``_is_main_discovered_segment``) live
   here now instead of being mirrored in dataset.py and
   patch_cache.py.

v6 -- focused LABELED discovery for PR #2677 (review: villa-review,
25/11/2026, fifth review of the v5 head at 3a46e88). The reviewer
still recommends merging a FOCUSED discovery change after repair, and
now says explicitly to keep the new discovery within labeled mode.
v6 repairs the five points of review 5 on top of the v5 base:

1. Chunk completeness (point 1). A surviving chunk no longer passes
   the readiness check: the image and the validation mask must hold
   EVERY expected chunk at every required level, not just one. A
   complete fill-only (uniform) validation mask is preserved: the
   writer omits fill-only chunks (skip_empty), so a writer-style
   pyramid whose every level holds zero chunks is complete, while a
   single-level zero-chunk store is the first-sync state (review 4,
   R3b). Ink and supervision keep the v5 rule (readable metadata is
   an all-zero mask): a missing chunk there reads as fill, which
   costs patches instead of leaking held-out pixels or blank samples.
2. Scale checks (point 2). A level counts as readable only when it
   is an ARRAY (.zarray, or zarr.json node_type "array"): a directory
   holding only .zattrs, or a Zarr group, is not an array -- open_volume
   either raises FileNotFoundError there or returns a group with no
   .shape. And the required levels follow the finder that will run:
   subtiling reads everything at volume_scale, so a complete level-0
   mirror with patch_finding_scale=1 is admitted; the default finder
   reads labels at the scan scale and samples at volume_scale, so
   both levels stay required there.
3. Read failures in existing layouts (point 3, dataset.py). The
   warn-and-skip resilience applies ONLY to segments the PR admits;
   a failure on a segment main discovers (native tifxyz geometry or
   an explicit surface) re-raises, as main does. A failure that only
   says the segment yielded no patches is a legitimate zero-patch
   outcome (complete but below the ink-coverage threshold): the
   segment is kept and covered by the patch-cache sidecar, so an
   unchanged rerun reuses the cache instead of re-discovering it.
4. Allowlist / map validation (point 4). An allowlisted folder that
   exists on disk takes part in map validation: no map entry for it
   raises KeyError, even when the folder carries no segment markers
   (an empty allowlisted folder must not disappear from preflight).
   Names with no on-disk folder stay inert, as in main.
5. Unlabeled path (point 5). The new TIFF-free discovery is
   labeled-only, as the reviewer explicitly offers: in unlabeled
   mode these mirrors are skipped exactly as main skips them, so the
   unlabeled extension -- which lost mask exclusions and missed
   repaired mirrors on cached reruns -- is gone by construction.

What the reviewer tried and discarded (its own narrowing failed
because partial stores and a Zarr group at the sampling level still
passed preflight, and a complete sibling below the ink threshold
stopped an otherwise usable scan): v6 closes exactly those two
holes -- full chunk enumeration instead of a "some chunk" probe, and
the array check instead of a metadata check, with the below-threshold
sibling handled in the dataset (point 3), not at discovery.

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
   on a selected segment instead of skipping it. (v6 extends that
   re-raise to ANY segment main discovers, review 5 point 3.)
6. Unlabeled cache rescanning. Gone: the cache is main's.

Reason (b) for the bot discarding its own narrowing -- a complete
sibling below Tutorial 5's ink-coverage threshold aborting a scan
that otherwise yields usable patches -- is fixed by the dataset.py
resilience kept in v5: a segment that yields no patches at patch
finding is warned about and skipped, and the run proceeds on the
ready segments (it still fails loudly if NO segment yields patches).
Discovery cannot know the coverage without re-implementing patch
finding, so the fix belongs there, not in discovery. (v6 keeps such
a segment in the dataset and covers it with the patch-cache sidecar,
review 5 point 3, so an unchanged rerun reuses the cache instead of
re-discovering it.)

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
import itertools
import json
import math
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


def _zarr_level_array_meta(store: str | Path, level: int | str) -> tuple | None:
    """(shape, chunks, dimension_separator) of the array at the level, or None.

    The level node must be an ARRAY: a v2 ``.zarray`` file, or a v3
    ``zarr.json`` with node_type "array". A directory holding only
    ``.zattrs`` (review 5, point 2a) or a ``.zgroup`` (point 2b) is a
    group, not an array: ``open_volume`` either raises FileNotFoundError
    there or returns a group with no ``.shape`` (AttributeError at the
    first sample), so it must never count as readable. A level whose
    metadata cannot be parsed is unreadable too, never assumed.
    """
    level_dir = _zarr_level_dir(store, level)
    if level_dir is None:
        return None
    zarray = level_dir / ".zarray"
    if zarray.is_file():
        try:
            meta = json.loads(zarray.read_text())
        except (OSError, ValueError):
            return None
        shape = meta.get("shape")
        chunks = meta.get("chunks")
        if not isinstance(shape, list) or not isinstance(chunks, list):
            return None
        return (
            tuple(int(value) for value in shape),
            tuple(int(value) for value in chunks),
            str(meta.get("dimension_separator", ".")),
        )
    zarr_json = level_dir / "zarr.json"
    if zarr_json.is_file():
        try:
            meta = json.loads(zarr_json.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("node_type") != "array":
            return None
        shape = meta.get("shape")
        grid = (meta.get("chunk_grid") or {}).get("configuration", {}).get(
            "chunk_shape"
        )
        if not isinstance(shape, list) or not isinstance(grid, list):
            return None
        return (
            tuple(int(value) for value in shape),
            tuple(int(value) for value in grid),
            "/",
        )
    return None


# Villa's label writer writes this many pyramid levels by default
# (``label_zarr.DEFAULT_LEVELS``); ``create_label_zarrs --levels``
# can override it, but the tutorial path uses the default.
_WRITER_DEFAULT_LEVELS = 6


def _zarr_level_format(store: str | Path, level: int | str) -> str | None:
    """\"v2\" or \"v3\" for the array at the level, or None when not an array.

    The format decides which storage keys the real reader looks up:
    v2 joins chunk indices with the declared dimension separator
    ("." or "/"), v3 prefixes them with ``c`` and joins with the
    chunk-key-encoding separator. A store whose format cannot be
    determined is not an array (see :func:`_zarr_level_array_meta`).
    """
    level_dir = _zarr_level_dir(store, level)
    if level_dir is None:
        return None
    if (level_dir / ".zarray").is_file():
        return "v2"
    zarr_json = level_dir / "zarr.json"
    if zarr_json.is_file():
        try:
            meta = json.loads(zarr_json.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("node_type") == "array":
            return "v3"
    return None


def _zarr_v3_chunk_separator(store: str | Path, level: int | str) -> str:
    """Declared v3 chunk-key separator ("/" when absent or unreadable).

    Zarr v3's ``chunk_key_encoding`` names the separator explicitly;
    the default encoding uses "/". The reader joins chunk indices
    with exactly this separator, so a key written with any other
    separator is not a chunk.
    """
    level_dir = _zarr_level_dir(store, level)
    if level_dir is None:
        return "/"
    zarr_json = level_dir / "zarr.json"
    if not zarr_json.is_file():
        return "/"
    try:
        meta = json.loads(zarr_json.read_text())
    except (OSError, ValueError):
        return "/"
    encoding = meta.get("chunk_key_encoding") or {}
    separator = encoding.get("separator", "/")
    return separator if isinstance(separator, str) and separator else "/"


def _grid_index_for_chunk_file(
    relpath: str,
    ndim: int,
    *,
    format: str,
    separator: str,
    grid_counts: tuple[int, ...] | None = None,
) -> tuple[int, ...] | None:
    """Chunk-grid index for a file under a level directory, or None.

    The file must sit at EXACTLY the storage key the real reader
    looks up for a chunk (review 7, point 3): v2 joins the indices
    with the declared dimension separator, v3 prefixes them with
    ``c`` and joins with the chunk-key-encoding separator, and every
    component is a canonical integer (``str(int(part)) == part``: no
    leading zeros, no signs, no whitespace). A file at any other key
    is not a chunk -- it is ignored, never counted -- even when the
    sampler would read fill for the tile it pretends to cover.

    A backslash anywhere rejects the file outright: on Linux a
    filename may literally contain backslashes, so ``0\\0\\0`` is ONE
    filename, not a nested path, and must never be read as the chunk
    ``(0, 0, 0)``. The old code replaced backslashes with slashes
    before splitting, which is exactly the bug.
    """
    if "\\" in relpath:
        return None
    if format == "v3":
        parts = relpath.split(separator) if separator else [relpath]
        if not parts or parts[0] != "c":
            return None
        parts = parts[1:]
    elif separator == "/":
        parts = relpath.split("/")
    elif separator == ".":
        if "/" in relpath:
            return None
        parts = relpath.split(".")
    else:
        return None
    if len(parts) != ndim:
        return None
    index: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            return None
        if str(value) != part:
            return None
        index.append(value)
    if grid_counts is not None:
        if len(grid_counts) != ndim:
            return None
        if any(value < 0 or value >= count for value, count in zip(index, grid_counts)):
            return None
    elif any(value < 0 for value in index):
        return None
    return tuple(index)


def _zarr_level_chunk_presence(
    store: str | Path, level: int | str
) -> tuple | None:
    """(meta, present, expected) chunk-grid sets at a level, or None.

    Cheap: enumerates the files under the level directory and maps
    them to grid positions (no content reads). ``present`` holds the
    grid indices with a file at that position; ``expected`` holds
    every grid index from the array metadata. Returns None when the
    level node is not an array.
    """
    meta = _zarr_level_array_meta(store, level)
    if meta is None:
        return None
    shape, chunks, separator = meta
    ndim = len(shape)
    grid_counts = [
        -(-int(size) // int(chunk)) for size, chunk in zip(shape, chunks)
    ]
    expected = set(
        itertools.product(*[range(count) for count in grid_counts])
    )
    level_dir = _zarr_level_dir(store, level)
    if level_dir is None:
        return None
    # The storage key depends on the store's declared format: v2 joins
    # indices with the dimension separator, v3 prefixes them with "c"
    # and joins with the chunk-key-encoding separator (review 7,
    # point 3). A file at any other key is not a chunk.
    format = _zarr_level_format(store, level)
    if format is None:
        return None
    if format == "v3":
        separator = _zarr_v3_chunk_separator(store, level)
    present: set[tuple[int, ...]] = set()
    try:
        for root, _dirs, files in os.walk(level_dir, followlinks=True):
            for name in files:
                if name in _ZARR_METADATA_NAMES:
                    continue
                rel = os.path.relpath(os.path.join(root, name), level_dir)
                index = _grid_index_for_chunk_file(
                    rel,
                    ndim,
                    format=format,
                    separator=separator,
                    grid_counts=tuple(grid_counts),
                )
                if index is not None and index in expected:
                    present.add(index)
    except OSError:
        return None
    return (meta, present, expected)


def _open_level_volume(store: str | Path, level: int | str):
    """Open one resolution level through the training reader, or None.

    The content reads go through ``open_volume`` (vc_reader) -- the
    same reader the training path uses (dataset ``_open``) -- instead
    of Python zarr (review 7, point 4). The documented ``models``
    extra declares ``vc-reader`` but no Python ``zarr``, so a
    zarr-based readiness rejected stores the training reader opens
    (a complete published v2 crop with zarr unimportable, and a v3
    re-encoded crop even with zarr 2.18.7 installed). Reading through
    the training reader makes readiness agree with training by
    construction, in the documented install and on v3 stores.

    Lazy import (segment.py otherwise reads only JSON metadata).
    None when the reader is unavailable or the level cannot be opened
    (a corrupt store): callers treat present chunks as corrupt --
    fail closed, since the run could not read the level anyway.
    """
    if _zarr_level_dir(store, level) is None:
        return None
    try:
        from vesuvius.ink_detection.volume_io import open_volume
    except ImportError:
        return None
    try:
        return open_volume(store, level)
    except Exception:
        return None


def _zarr_level_chunk_states(
    store: str | Path, level: int | str
) -> tuple | None:
    """(meta, states): per-chunk states at a level, or None if not an array.

    ``states`` maps every expected chunk-grid index to one of:

    - ``"valid"``: a file sits at the grid position and reads as a
      chunk of this array (one voxel decompresses).
    - ``"missing"``: no file at the grid position.
    - ``"corrupt"``: a file sits at the grid position but does not
      read as a chunk (not a chunk file, truncated, wrong codec), or
      the array cannot be opened at all.

    Only files at valid grid positions count (review 6, point 2): v6
    counted every non-metadata file, so an unrelated file kept the
    count unchanged and an accepted patch loaded as zeros. A file at a
    grid position still has to prove it is a chunk: the voxel read is
    the content check, codec-agnostic (it goes through the training
    reader, so any compressor the array uses is honoured). Residual hole, documented:
    a chunk file that is a valid compressed chunk for the same
    shape/dtype but from another store is indistinguishable -- zarr v2
    carries no checksum or manifest.
    """
    result = _zarr_level_chunk_presence(store, level)
    if result is None:
        return None
    (meta, present, expected) = result
    _shape, chunks, _separator = meta
    states: dict[tuple[int, ...], str] = {
        index: "missing" for index in expected - present
    }
    if not present:
        return (meta, states)
    array = _open_level_volume(store, level)
    if array is None:
        for index in present:
            states[index] = "corrupt"
        return (meta, states)
    for index in sorted(present):
        coord = tuple(i * c for i, c in zip(index, chunks))
        try:
            array[coord]
        except Exception:
            states[index] = "corrupt"
        else:
            states[index] = "valid"
    return (meta, states)


def _zarr_level_chunk_counts(store: str | Path, level: int | str) -> tuple | None:
    """(present, expected) chunk files at the level, or None if not an array.

    Present counts only files at valid grid positions whose content
    reads as a chunk (see :func:`_zarr_level_chunk_states`); kept for
    callers that only need the counts.
    """
    result = _zarr_level_chunk_states(store, level)
    if result is None:
        return None
    _meta, states = result
    present = sum(1 for state in states.values() if state == "valid")
    return (present, len(states))


def _store_array_levels(store: str | Path) -> list[int]:
    """Sorted array-level numbers of a store (empty when not a store).

    A single-level ARRAY store (``.zarray`` at the root) has level
    [0]; a GROUP store has one entry per numeric child directory that
    holds an array. Non-numeric children are not levels.
    """
    store = Path(store)
    if not store.is_dir():
        return []
    if _zarr_level_dir(store, 0) == store:
        return [0]
    try:
        children = sorted(store.iterdir())
    except OSError:
        return []
    levels = []
    for child in children:
        if not child.is_dir() or not child.name.isdigit():
            continue
        if _zarr_level_array_meta(store, int(child.name)) is not None:
            levels.append(int(child.name))
    return sorted(levels)


def _zarr_store_is_writer_default_uniform_pyramid(store: str | Path) -> bool:
    """True when a zero-chunk store is a writer-default pyramid.

    Villa's label writer (``create_label_zarrs``) writes a pyramid
    group whose every level is an array (``DEFAULT_LEVELS`` = 6, unless
    ``--levels`` overrides it) and omits fill-only chunks
    (``skip_empty=True``): a complete uniform (fill-only) mask
    therefore holds ZERO chunks at every level of a 6-level pyramid,
    and is still complete (review 5, point 1). The level count is the
    only structural difference between that mask and an interrupted
    transfer (review 6, point 1): a 2-level metadata-only store is not
    a writer product, and a single-level zero-chunk store is the
    first-sync state (review 4, R3b). So the uniform exception now
    requires exactly ``DEFAULT_LEVELS`` (6) array levels, each holding
    zero chunks. Residual holes, documented: a 6-level stripped
    populated mask is byte-identical to a genuine uniform mask and
    cannot be distinguished; a writer mask made with a non-default
    ``--levels`` (2-5) that is all-fill is wrongly rejected.
    """
    levels = _store_array_levels(store)
    if len(levels) != _WRITER_DEFAULT_LEVELS:
        return False
    for level in levels:
        result = _zarr_level_chunk_presence(store, level)
        if result is None:
            return False
        _meta, present, _expected = result
        if present:
            return False
    return True


def _footprint_at_level(
    shape_from: tuple[int, ...],
    extent_from: tuple[tuple[int, int], ...],
    shape_to: tuple[int, ...],
) -> tuple[tuple[int, int], ...] | None:
    """Map a voxel extent from one level's shape to another's.

    The extent is scaled by the shape ratio (a real writer pyramid
    halves Y/X each level; a same-shape pyramid maps identically), so
    no downsample factor is assumed. Bounds are widened (floor start,
    ceil stop) so the footprint covers every voxel that could
    contribute to the extent. None when a shape is degenerate or the
    footprint is empty.
    """
    footprint = []
    for (start, stop), size_from, size_to in zip(
        extent_from, shape_from, shape_to
    ):
        if size_from <= 0 or size_to <= 0:
            return None
        ratio = size_from / size_to
        lo = max(0, min(int(start / ratio), size_to))
        hi = max(0, min(math.ceil(stop / ratio), size_to))
        if lo >= hi:
            return None
        footprint.append((lo, hi))
    return tuple(footprint)


def _chunk_indices_intersecting(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    footprint: tuple[tuple[int, int], ...],
) -> set[tuple[int, ...]]:
    """Grid indices of the chunks intersecting a voxel footprint."""
    ranges = []
    for (lo, hi), size, chunk in zip(footprint, shape, chunks):
        first = lo // chunk
        last = (hi - 1) // chunk
        ranges.append(range(first, last + 1))
    return set(itertools.product(*ranges))


def _region_fill_state(
    store: str | Path, level: int | str, footprint: tuple[tuple[int, int], ...]
) -> bool | None:
    """True/False/None: is the footprint region all-fill at the level?

    True when every voxel in the region is the fill value (0); False
    when any voxel is non-zero; None when the region cannot be read
    (a chunk covering it is missing or corrupt, or the level is not an
    array). The presence check matters: the reader reads a missing chunk as
    fill, so a region read alone cannot tell a genuinely all-fill
    region from a missing one -- the chunks covering the footprint
    must be present first. Only the covering chunks are validated
    (one region read, which decompresses them), not the whole level.
    """
    result = _zarr_level_chunk_presence(store, level)
    if result is None:
        return None
    (shape, chunks, _sep), present, _expected = result
    covering = _chunk_indices_intersecting(shape, chunks, footprint)
    if not covering or not covering.issubset(present):
        return None
    array = _open_level_volume(store, level)
    if array is None:
        return None
    try:
        region = array[tuple(slice(lo, hi) for lo, hi in footprint)]
    except Exception:
        return None
    return not bool(region.any())


def _level_has_any_chunk(store: str | Path, level: int | str) -> bool:
    """True when the level holds at least one chunk.

    Listing only, no content reads. Used by the coarser-absent rule
    below: the rule requires a chunk at the IMMEDIATELY COARSER level
    itself, not merely somewhere in the store -- a store whose L0
    chunks arrived but whose coarser levels are all empty is an
    interrupted transfer mid-L0 (the writer writes L0 first), not a
    sparse pyramid (red-team R2).
    """
    result = _zarr_level_chunk_presence(store, level)
    return result is not None and bool(result[1])


def _coarser_level_covering_absent(
    store: str | Path, level: int | str, chunk_index: tuple[int, ...]
) -> bool:
    """True when every chunk covering the missing chunk at the
    immediately coarser level is absent.

    The covering chunks are the ones intersecting the missing chunk's
    footprint at level ``level + 1``. When they are all absent, the
    region is chunk-aligned empty at the coarser resolution: the
    writer (``skip_empty``) omitted them because the region was empty
    there, and a single deleted chunk in a populated area cannot
    satisfy this -- its coarser parents are present via its
    neighbours. False when there is no coarser level, when the
    footprint cannot be mapped, or when any covering chunk is present.
    """
    result = _zarr_level_chunk_presence(store, level)
    if result is None:
        return False
    (shape, chunks, _sep), _present, _expected = result
    extent = tuple(
        (i * c, min(i * c + c, size))
        for i, c, size in zip(chunk_index, chunks, shape)
    )
    levels = _store_array_levels(store)
    coarser = [other for other in levels if other > level]
    if not coarser:
        return False
    other = min(coarser)
    other_result = _zarr_level_chunk_presence(store, other)
    if other_result is None:
        return False
    (other_shape, other_chunks, _other_sep), other_present, _other_expected = other_result
    footprint = _footprint_at_level(shape, extent, other_shape)
    if footprint is None:
        return False
    covering = _chunk_indices_intersecting(other_shape, other_chunks, footprint)
    if not covering:
        return False
    return covering.isdisjoint(other_present)


def _missing_chunk_is_proven_fill(
    store: str | Path, level: int | str, chunk_index: tuple[int, ...]
) -> bool:
    """True when cross-level evidence proves a missing chunk is fill.

    Three directions (see DESIGN_NOTE_V8.md for the proofs and the
    residual holes):

    - Downward (sound): the finer level's chunks over the footprint
      are all present and all-fill, so the coarser chunk is fill
      (nearest and mean both map all-zero finer voxels to zero).
    - Upward non-fill (sound): a coarser chunk over the footprint is
      present and non-fill, so the finer chunk is NOT fill -- the
      proof returns False immediately (proven non-fill).
    - Coarser-absent (review 7, point 2): every chunk covering the
      missing chunk at the immediately coarser level is ABSENT, and
      that immediately coarser level holds at least one chunk
      somewhere else. The writer omitted a chunk-aligned empty region
      at every resolution, so the missing chunk is fill. A single
      deleted chunk in a populated area cannot satisfy this (its
      coarser parents are present via its neighbours), and a store
      whose coarser levels are ALL empty cannot satisfy it either:
      that is an interrupted transfer mid-L0, since the writer writes
      L0 first (red-team R2: rule (c) used to require only "some
      chunk somewhere in the store", which an interrupted transfer
      satisfies).

    The v7 UPWARD-FILL direction is GONE: a present coarser chunk
    whose footprint is all fill proves nothing about the missing
    chunk, because nearest-neighbour downsampling erases a
    one-voxel stroke (review 7, point 1b: deleting the fine chunk
    holding a thin validation stroke left the coarser footprint all
    fill, so v7 admitted the tampered mask). A present all-fill
    coarser footprint is now NO evidence, and the chunk is unproven
    unless a sound direction or the coarser-absent rule applies.

    Deliberate trade-off: the review-6 F1 writer mask (1 populated +
    1 omitted fill chunk, admitted by v7 via the upward proof) is
    now REJECTED -- it is observationally identical to the review-7
    1b tamper (same missing chunk, same present coarser chunk, same
    all-fill footprint), and no signal in the store distinguishes
    them. The fully-stripped 6-level mask (review 7, point 1a) is
    byte-identical to a genuine uniform mask and stays admitted by
    the uniform exception: provably undecidable.

    The proof walks outward from the missing chunk's level (coarser
    levels first, checking non-fill only, then the coarser-absent
    rule, then finer) and uses the first level that gives evidence.
    A level whose covering chunks are missing or corrupt gives no
    evidence and is skipped. When no level gives evidence, the chunk
    is unproven: an interrupted transfer and a complete uniform mask
    are indistinguishable there (the residual hole).
    """
    result = _zarr_level_chunk_presence(store, level)
    if result is None:
        return False
    (shape, chunks, _sep), _present, _expected = result
    extent = tuple(
        (i * c, min(i * c + c, size))
        for i, c, size in zip(chunk_index, chunks, shape)
    )
    levels = _store_array_levels(store)
    for other in sorted(l for l in levels if l > level):
        other_result = _zarr_level_chunk_presence(store, other)
        if other_result is None:
            continue
        footprint = _footprint_at_level(shape, extent, other_result[0][0])
        if footprint is None:
            continue
        fill = _region_fill_state(store, other, footprint)
        # Only NON-fill is evidence here: a present coarser chunk
        # whose footprint is non-fill proves the missing chunk is NOT
        # fill. An all-fill footprint proves nothing (a thin signal
        # can be erased by downsampling) and is skipped.
        if fill is False:
            return False
    # Coarser-absent (review 7, point 2): the writer omitted the
    # whole coarser neighbourhood of this chunk, so the region was
    # empty at the coarser resolution too. Tightened after the
    # red-team R2 attack: the immediately coarser level must itself
    # hold at least one chunk -- requiring only "some chunk somewhere
    # in the store" admits an interrupted transfer mid-L0 (the writer
    # writes L0 first, so L0 chunks present + all coarser levels
    # empty is exactly a transfer that died after L0).
    coarser_levels = sorted(l for l in levels if l > level)
    if (
        coarser_levels
        and _coarser_level_covering_absent(store, level, chunk_index)
        and _level_has_any_chunk(store, coarser_levels[0])
    ):
        return True
    for other in sorted((l for l in levels if l < level), reverse=True):
        other_result = _zarr_level_chunk_presence(store, other)
        if other_result is None:
            continue
        footprint = _footprint_at_level(shape, extent, other_result[0][0])
        if footprint is None:
            continue
        fill = _region_fill_state(store, other, footprint)
        if fill is True:
            return True
        if fill is False:
            return False
    return False


def _zarr_level_store_state(store: str | Path, level: int | str) -> str:
    """One of "complete", "fill", "uniform", "empty", "partial",
    "corrupt", "unreadable" for a store level.

    - "unreadable": the level node is not an array (missing, a group,
      or metadata-only without ``.zarray``) -- review 5, point 2.
    - "complete": every expected chunk is present and reads as a chunk.
    - "fill": some chunks are missing, but every missing chunk is
      proven fill by cross-level evidence (review 6, point 1) -- the
      store is effectively complete.
    - "uniform": zero chunks at the level, and the store is a
      writer-default pyramid (exactly 6 array levels, zero chunks at
      every level) -- a complete fill-only mask (review 5, point 1;
      review 6, point 1 narrows the level count to the writer
      default).
    - "empty": zero chunks at the level, and the store is a
      single-level flat store -- a readable all-zero mask (review 3,
      C3/Z8). Admitted for ink/supervision, never for validation
      (review 4, R3b: a flat metadata-only validation mask is the
      first-sync state, not a uniform mask).
    - "partial": some chunks are missing and not all are proven fill
      -- an unfinished sync (review 5, point 1; review 6, point 1
      F2 and point 2).
    - "corrupt": a chunk file at a grid position does not read as a
      chunk (review 6, point 2) -- an unrelated file, a truncated
      chunk, a wrong codec.
    """
    result = _zarr_level_chunk_states(store, level)
    if result is None:
        return "unreadable"
    _meta, states = result
    present = sum(1 for state in states.values() if state == "valid")
    missing = [index for index, state in states.items() if state == "missing"]
    if any(state == "corrupt" for state in states.values()):
        return "corrupt"
    if not missing:
        return "complete"
    if present == 0:
        if _zarr_store_is_writer_default_uniform_pyramid(store):
            return "uniform"
        if len(_store_array_levels(store)) == 1:
            return "empty"
        # 2-5 levels, zero chunks here: not a writer product and not a
        # flat store. Fall through to the cross-level proof (another
        # level may hold the evidence); otherwise partial.
    if all(
        _missing_chunk_is_proven_fill(store, level, index) for index in missing
    ):
        return "fill"
    return "partial"


def _required_levels(segment: Segment) -> tuple[int, ...]:
    """Resolution levels a segment's stores must serve.

    The sample scale (``segment.scale`` == the source's volume_scale)
    is always required: samples read image and labels there. Which
    other level patch finding reads depends on the finder that will
    run (review 5, point 2c): the DEFAULT finder reads labels at the
    patch-finding scan scale (``patch_finding.scan_scale``, from the
    ``patch_finding_scale`` config key) when it differs, so both
    levels are required there -- a store that serves only the scan
    scale passes patch finding and then fails on the first sample
    (FileNotFoundError), the bot's discarded reason (a). The SUBTILING
    finder reads everything at ``segment.scale`` (it never touches
    ``scan_scale``), so only the sample scale is required there: a
    complete level-0 mirror with ``patch_finding_scale=1`` must be
    admitted, not skipped for a level it will never read.
    """
    scale = int(segment.scale)
    if segment.data_config.patch_finding.kind == "subtiling":
        return (scale,)
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


def _resolve_labels_or_fail(segment: Segment, *, strict: bool) -> Segment | None:
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
    tifxyz geometry, no explicit surface, labeled mode). Segments main
    already discovers call :func:`discover_segment_labels` directly,
    so their completeness checks are unaffected.
    """
    try:
        return discover_segment_labels(segment, extension=".zarr", required=True)
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
    light: bool = False,
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

    Labeled mode only (review 5, point 5): in unlabeled mode the PR
    admits nothing -- main skips these mirrors, and the unlabeled
    extension lost mask exclusions and missed repaired mirrors on
    cached reruns. The reviewer explicitly offers keeping the new
    discovery within labeled mode.
    """
    if config.discovery_mode == "unlabeled":
        return None
    strict = held_out or is_allowlisted
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

    # (a) The image must hold EVERY expected chunk at every required
    # level (review 5, point 1). A store with some chunks missing
    # reads as all zeros where the chunks are absent: patch finding
    # still announces patches, and the first sample loads blank
    # images. A store with metadata at level 0 and chunks only at
    # level 1 reads as all zeros at volume_scale=0 (review 4,
    # point 4). No uniform exception: an image is never legitimately
    # chunkless.
    #
    # Light gather (review 7, point 6): skip the content gates --
    # they decode chunks. A light segment is only used to look up the
    # patch cache; the fingerprint check there (patch_cache) and the
    # full re-gather on a miss are what catch an unfinished store.
    if not light:
        for level in levels:
            if _zarr_level_store_state(image_volume, level) != "complete":
                _fail_or_skip(
                    segment_dir,
                    strict=strict,
                    message=(
                        "image volume "
                        f"{image_volume} is incomplete at resolution {level} "
                        "(unfinished mirror: a chunk is missing, or the level "
                        "is not an array); it will be picked up once its "
                        "store is filled."
                    ),
                )
                return None

    # (b) Labels: a partly synced sibling is skipped (point 2), a
    # version mismatch always raises.
    segment = _resolve_labels_or_fail(segment, strict=strict)
    if segment is None:
        return None

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
    # (c) Supervision and ink must be COMPLETE at every required
    # level -- every expected chunk present and reading as a chunk --
    # or provably fill ("fill": every missing chunk is proven fill by
    # cross-level evidence), or a uniform/empty all-zero mask.
    # Review 6, point 2 refutes the v5/v6 rule ("readable metadata =
    # all-zero mask"): a missing chunk reads as fill, so a partly
    # synced ink store silently turns positives into supervised
    # background, and a partly synced supervision store silently drops
    # validation patches. A readable store without chunks is still an
    # all-zero mask -- the label writer omits fill-only chunks -- so
    # the segment is ready: it yields no training patches but may
    # yield validation patches (C3, Z8). A corrupt chunk (an unrelated
    # file at a grid position, a truncated chunk) fails here, not at
    # the first sample.
    #
    # Light gather (review 7, point 6): skip the content gates (see
    # gate (a) above).
    if not light:
        for kind in ("supervision_mask", "inklabels"):
            store = getattr(segment, kind)
            for level in levels:
                state = _zarr_level_store_state(store, level)
                if state not in ("complete", "fill", "uniform", "empty"):
                    _fail_or_skip(
                        segment_dir,
                        strict=strict,
                        message=(
                            f"{kind} store {store} is {state} at "
                            f"resolution {level} (unfinished mirror: a chunk "
                            "is missing or does not read as a chunk); it will "
                            "be picked up once its store is filled."
                        ),
                    )
                    return None
    # (d) A validation mask, when present, must be COMPLETE at every
    # required level -- every expected chunk present -- or FILL (every
    # missing chunk proven fill by cross-level evidence, review 6
    # point 1: a complete writer mask with an omitted fill chunk) or
    # UNIFORM: a writer-default pyramid (exactly 6 levels) whose every
    # level holds zero chunks, which the label writer produces for a
    # fill-only mask (skip_empty). A PARTIAL mask (some chunks missing)
    # is an unfinished sync: treating its missing chunks as fill would
    # let the segment contribute training patches that its real
    # validation regions (once synced) would have held out (review 5,
    # point 1). A single-level zero-chunk store ("empty") is the
    # first-sync state, not a uniform mask (review 4, point 3 / R3b),
    # and a 2-5-level zero-chunk store is not a writer product either
    # (review 6, point 1, F2): both wait until their validation
    # regions can be read.
    if not light and segment.validation_mask is not None:
        for level in levels:
            if _zarr_level_store_state(segment.validation_mask, level) not in (
                "complete",
                "fill",
                "uniform",
            ):
                _fail_or_skip(
                    segment_dir,
                    strict=strict,
                    message=(
                        "validation mask "
                        f"{segment.validation_mask} is incomplete at "
                        f"resolution {level} (unfinished sync); the "
                        "segment waits until its validation regions "
                        "can be read."
                    ),
                )
                return None

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
    light: bool = False,
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
            if is_allowlisted or markers["recognized"]:
                # An incomplete explicit map must fail preflight: the
                # folder is a segment the map forgot, and silently
                # dropping it would report a partial dataset as complete.
                # (Main raises KeyError for any unmapped folder; v5
                # narrowed the raise to folders carrying segment markers
                # so an unrelated folder, e.g. a preview-TIFF folder,
                # still does not abort the run. v6 restores the raise
                # for an allowlisted folder that exists on disk even
                # when it carries no markers -- review 5, point 4: an
                # empty allowlisted folder must take part in map
                # validation, or it disappears from preflight. A name
                # with no on-disk folder never reaches here, so it
                # stays inert, as in main.)
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
                light=light,
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
            # Main's rule (review 7, point 5): in unlabeled mode a
            # missing local image is ALWAYS skipped, whether flat or
            # explicitly mapped -- main does ``continue`` here, with
            # no exception for an explicitly configured volume. v7's
            # H1a raise (ValueError for an explicitly configured
            # image) contradicted main: with two mapped segments and
            # one missing image, main yields the ready segment's
            # patches, and a ready segment with no map entry then
            # raises KeyError at map validation. Skipping here
            # reproduces exactly that.
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


def _store_chunk_inventory_digest(store: str | Path | None) -> str:
    """Short digest of one store's exact chunk inventory (listing only).

    Hashes every file under the store (relative path, size, mtime_ns),
    sorted -- the same philosophy as ``patch_cache.label_asset_fingerprint``:
    it reads no chunk contents, so it costs one directory walk, and any
    change (a chunk removed, added, renamed, or rewritten) changes the
    digest and invalidates the cache. A missing store digests as
    ``missing``; a remote store (``s3://``, ``http://``, ``https://``)
    digests its URL, since its chunks cannot be listed locally.

    Residual limitation, documented: a chunk replaced by a different
    file with the SAME name, size and mtime is undetectable without
    reading contents -- that is deliberate forgery, not a sync issue,
    and the fresh run's content validation is what catches it.
    """
    import hashlib

    digest = hashlib.sha256()
    if store is None:
        digest.update(b"\0none")
        return digest.hexdigest()[:16]
    text = str(store)
    digest.update(text.encode())
    if text.lower().startswith(_REMOTE_VOLUME_PREFIXES):
        return digest.hexdigest()[:16]
    root = Path(store)
    if not root.exists():
        digest.update(b"\0missing")
        return digest.hexdigest()[:16]
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


def _segment_store_fingerprint(segment: Segment) -> str:
    """Short digest of a segment's stores' exact chunk inventories.

    Covers the image volume AND the ink, supervision and validation
    stores (review 7, point 6): the patch cache's label fingerprint
    covers labels only, so an image chunk removed after caching would
    not invalidate it. Recorded at cache-write time and verified on a
    cache hit by listing only (no chunk decoding), so a chunk removed
    after caching invalidates the cache and the fresh run detects it.
    """
    import hashlib

    digest = hashlib.sha256()
    for label, store in (
        ("image", segment.image_volume),
        ("inklabels", segment.inklabels),
        ("supervision_mask", segment.supervision_mask),
        ("validation_mask", segment.validation_mask),
    ):
        digest.update(label.encode())
        digest.update(_store_chunk_inventory_digest(store).encode())
    return digest.hexdigest()[:16]


def _is_selected_segment(segment) -> bool:
    """True when the user explicitly selected this segment.

    A segment is selected when its name is in the source's allowlist,
    when it has an explicit surface volume (a single
    ``surface_volume_path`` or a ``surface_volume_paths`` entry for
    its relpath or name), or when its source is held out
    (``split == "validation"``). A read failure on a selected segment
    must stop the run before it trains or scores a subset (review 4,
    point 5): main stops there too, when patch finding opens its
    masks. An auto-discovered segment in the same state is warned
    about and skipped instead, and picked up on a later run.

    Single definition (review 7, point 6): dataset.py and
    patch_cache.py import it from here instead of mirroring it.
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

    Main discovers a flat segment when it carries native tifxyz
    geometry (``x.tif`` in its directory) or when it has an explicit
    surface volume (a single ``surface_volume_path`` or a
    ``surface_volume_paths`` entry for its relpath or name). A
    TIFF-free flat mirror with no explicit surface exists only
    because of the PR's discovery. A patch-finding failure on a
    main-discovered segment must stop the run (review 5, point 3):
    main stops there too, and silently dropping one would train or
    score a subset while reporting the full set -- and the broad
    catch also reaches native discovery, where removing a segment
    changes which neighbours contribute labels.

    Single definition (review 7, point 6): dataset.py and
    patch_cache.py import it from here instead of mirroring it.
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


def gather_segments(
    config: InkDataConfig, *, light: bool = False
) -> tuple[Segment, ...]:
    """Discover usable segment representations in stable directory-name order.

    ``light=True`` (review 7, point 6) gathers with filesystem checks
    only -- segment markers, label resolution by name, image
    existence -- and skips the content gates (chunk completeness),
    which decode chunks. A light gather is only ever used to look up
    the patch cache; on a cache miss the caller re-gathers with full
    validation, and the cache's store fingerprint (patch_cache) is
    what catches a chunk removed after caching. The Segment objects
    a light gather returns are identical to a full gather's (same
    fields, same paths); only the admission decision differs.
    """
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
                light=light,
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
