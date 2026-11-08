"""Ink label tools: open, paint, fill, suggest, clean up, import, read, undo
and save the ink, background and validation labels drawn on a segment's
flattened view (the trainer's <seg>_inklabels.tif, <seg>_supervision_mask.tif
and <seg>_validation_mask.tif)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from ..core import mcp, _call, _strip_none

# The bridge waits up to 300 s for a save or an import; give the client a
# margin over that.
_SAVE_TIMEOUT_S = 330.0
_IMPORT_TIMEOUT_S = 330.0


class _LabelPoint(TypedDict):
    x: float
    y: float
    z: NotRequired[float]


class _Stroke(TypedDict):
    points: list[_LabelPoint]
    radius: float


class _Polygon(TypedDict):
    points: list[_LabelPoint]


class _Origin(TypedDict):
    x: int
    y: int


class _Mask(TypedDict):
    path: str
    origin: _Origin
    format: NotRequired[Literal["paint", "values"]]


class _Expect(TypedDict, total=False):
    segment: str
    serial: int


class _Rect(TypedDict):
    x: int
    y: int
    width: int
    height: int


class _FillPoint(TypedDict):
    x: float
    y: float
    z: NotRequired[float]


class _Display(TypedDict, total=False):
    show: bool
    ink: bool
    supervision: bool
    validation: bool
    opacity: float


class _AssistSettings(TypedDict, total=False):
    guide: Literal["prediction", "labels"]
    threshold: int
    above: bool
    connectivity: int
    unlabeledOnly: bool
    median: int
    minArea: int
    cleanupSize: int


@mcp.tool()
async def vc3d_ink_labels_state() -> dict[str, Any]:
    """Report the ink label session of the active segment.

    Returns {"open", "readOnly", "locked", "lockReason", "segment", "store"
    (the label folder <project>/ink_labels/<segment>), "canvas" {"width",
    "height", "gridCols", "gridRows", "scaleX", "scaleY", "renderScale",
    "centerX", "centerY"}, "labelMode", "class", "erase", "tool", "radius",
    "dirty", "stateSerial", "version" (the open label version; 1 is the
    unnumbered files), "versions" (all versions in the store), "undo" (newest
    first, at most 10, each {"serial", "label", "actor"}), "undoDepth",
    "redoDepth", "historyBytes", "counts" {"ink", "background", "validation",
    "unlabeled"} in label pixels, "lastSave" {"atUtc", "files"} or null,
    "autosave" {"atUtc"} or null, "assist" {"running", "suggestions",
    "picked", "settings"}, "prediction" (the ink prediction selected in the
    Ink Detection dock: {"path", "singleChannel", "hflip", "vflip", "width",
    "height", "labelPxPerPixelX", "labelPxPerPixelY"}, or null)}. Fill and
    suggestions need a single-channel prediction.

    Label pixels are the pixels of the vc_render_tifxyz canvas of the segment
    at full resolution (one per voxel). A grid position g maps to label pixel
    centerX + (renderScale / scaleX) * (g.x - gridCols / 2), and likewise in y.
    Call this first; "open": false means vc3d_ink_labels_open has not been
    called for the active segment.
    """
    return await _call("ink_labels.state", {})


@mcp.tool()
async def vc3d_ink_labels_open(
    recover: Literal["restore", "discard", "fail"] = "fail",
    read_only_if_locked: bool = True,
    version: Optional[int] = None,
    start_new_version: bool = False,
) -> dict[str, Any]:
    """Open (or reuse) the ink label session of the active segment.

    recover: what to do with an autosave newer than the last save (unsaved
    labels from a crash or an earlier session). "fail" (default) refuses with
    -32602 data.kind "autosave_pending" and data.autosaveAtUtc, so you can
    decide; "restore" loads the autosave; "discard" deletes it.
    read_only_if_locked: when another VC3D holds the labels, open read-only
    (true, default) or fail with -32009 data.kind "locked_by_other".
    version: label version to open (2 opens <seg>_inklabels_v2.tif and so on);
    omit to open the newest version in the store. A version that does not
    exist is -32602. start_new_version=true starts empty labels as the next
    version instead (also the way past a label_layer_locked store); Save
    writes it.

    Errors: -32007 data.kind "segment" with no active segment; -32009
    data.kind "label_layer_locked" (with data.expected / data.got) when the
    labels were saved for a different grid or segment, which nothing can paint
    over. Returns the state (as vc3d_ink_labels_state) plus "restoredAutosave"
    and "warnings" (such as files changed outside VC3D).
    """
    return await _call(
        "ink_labels.open",
        _strip_none(
            {
                "recover": recover,
                "readOnlyIfLocked": read_only_if_locked,
                "version": version,
                "startNewVersion": start_new_version,
            }
        ),
    )


@mcp.tool()
async def vc3d_ink_labels_paint(
    label_class: Optional[Literal["ink", "background", "validation"]] = None,
    erase: bool = False,
    background_keeps_ink: bool = False,
    space: Literal["label", "volume"] = "label",
    stroke: Optional[_Stroke] = None,
    polygon: Optional[_Polygon] = None,
    mask: Optional[_Mask] = None,
    expect: Optional[_Expect] = None,
) -> dict[str, Any]:
    """Paint one operation into the open ink label session (one undo entry,
    recorded with actor "agent").

    label_class: "ink" (ink and supervised), "background" (supervised, not
    ink; with background_keeps_ink=true existing ink is kept) or "validation"
    (the validation area); required except for a values mask. erase=true
    removes the class instead: ink or background go back to unknown
    (unsupervised), validation is cleared.

    Give exactly one shape:
    - stroke {"points": [{x, y}, ...], "radius": r}: a disc for one point, else
      round-capped segments through the points; r is 0.5 to 512 label px.
    - polygon {"points": [...]}: at least 3 vertices, even-odd fill.
    - mask {"path", "origin": {x, y}, "format"?}: an absolute single-channel
      8-bit PNG or TIFF of at most 8192 x 8192 px, with mask pixel (0, 0) at
      label pixel origin. Label space only. format "paint" (default): nonzero
      pixels are painted with label_class. format "values": the image holds
      label values 0 to 7 (bit 1 ink, 2 supervised, 4 validation), exactly as
      vc3d_ink_labels_read writes them, and every pixel of the rectangle is
      set to its value (label_class is ignored), so labels read with that
      tool can be edited and written back.
    space: "label" (default; continuous label px, pixel (c, r) covers
    [c, c+1) x [r, r+1)) or "volume" ({x, y, z} voxels on the segment, mapped
    through the segmentation viewer; off-surface points are -32003).
    expect: {"segment", "serial"} guard; if the session's segment or
    stateSerial differs the call fails -32602 data.kind "expect_mismatch" with
    data.current, and nothing is painted.

    Returns {"changedPixels", "stateSerial", "undoDepth", "clipped"}; clipped is
    true when part of the shape fell outside the canvas. Read-only or locked
    sessions, and a stroke the user is drawing (data.kind "stroke_open"), are
    -32009. Volume space needs the segmentation viewer to show the labeled
    segment flattened (data.kind "viewer" otherwise): call
    vc3d_activate_segment for it, or paint in label space.
    """
    return await _call(
        "ink_labels.paint",
        _strip_none(
            {
                "class": label_class,
                "erase": erase,
                "backgroundKeepsInk": background_keeps_ink,
                "space": space,
                "stroke": stroke,
                "polygon": polygon,
                "mask": mask,
                "expect": expect,
            }
        ),
    )


@mcp.tool()
async def vc3d_ink_labels_read(
    rect: _Rect,
    path: str,
    format: Literal["values", "ink", "supervision", "validation"] = "values",
) -> dict[str, Any]:
    """Write the current (also unsaved) label pixels of a rectangle to an
    image file, without saving the labels.

    rect: {"x", "y", "width", "height"} in label px, at most 4096 x 4096; it is
    clipped to the canvas. path: absolute .png or .tif whose folder exists.
    format: "values" (default; one byte per pixel, bit 1 ink, bit 2
    supervised, bit 4 validation), or "ink" / "supervision" / "validation"
    (0 or 255). Returns {"path", "rect" (clipped), "format", "counts" {"ink",
    "background", "validation", "unlabeled"} inside the rect, "stateSerial"}.
    """
    return await _call(
        "ink_labels.read",
        {"rect": rect, "format": format, "path": path},
    )


@mcp.tool()
async def vc3d_ink_labels_history(
    action: Literal["undo", "redo"],
    steps: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """Undo or redo ink label edits (separate from segmentation undo).

    steps: 1 to 1000. Undo stops at an entry the user made unless force=true;
    if the very first entry is the user's the call fails -32009 data.kind
    "other_actor_entry". Returns {"applied", "stoppedAtOtherActor",
    "stateSerial", "undo" (newest first), "redoDepth"}.
    """
    return await _call(
        "ink_labels.history",
        {"action": action, "steps": steps, "force": force},
    )


@mcp.tool()
async def vc3d_ink_labels_save(
    overwrite_external_edits: bool = False,
    as_new_version: bool = False,
) -> dict[str, Any]:
    """Save the open ink labels as the trainer's TIFFs and wait for the write.

    Writes <seg>_inklabels.tif and <seg>_supervision_mask.tif (and
    <seg>_validation_mask.tif when validation is painted or the file exists):
    uint8 0/255, tiled, LZW, at the vc_render_tifxyz canvas size, plus a JSON
    sidecar; the replaced files are kept as .prev. Returns {"saved": true,
    "files": [{"kind", "path", "sha256"}], "stateSerial", "ms", "version"},
    or {"saved": false} at once when nothing changed since the last save.
    as_new_version=true writes all three files as the next version
    (<seg>_inklabels_vN.tif, ...) and the session continues on it.

    A file changed outside VC3D since it was loaded is -32602 data.kind
    "external_edit" with data.files; call again with
    overwrite_external_edits=true to replace it. A save already running is
    -32004; a failed write is -32005.
    """
    return await _call(
        "ink_labels.save",
        {"overwriteExternalEdits": overwrite_external_edits, "asNewVersion": as_new_version},
        timeout=_SAVE_TIMEOUT_S,
    )


@mcp.tool()
async def vc3d_ink_labels_versions() -> dict[str, Any]:
    """List the label versions in the open session's store.

    Returns {"versions": [1, 2, ...], "open": N}; 1 is the unnumbered
    <seg>_inklabels.tif, N >= 2 is <seg>_inklabels_vN.tif. Open another with
    vc3d_ink_labels_open(version=N).
    """
    return await _call("ink_labels.versions", {})


@mcp.tool()
async def vc3d_ink_labels_dataset_entry() -> dict[str, Any]:
    """The ink detection trainer's dataset entry for the open labels.

    Returns {"entry": {"segments_path", "segment_names",
    "surface_volume_paths", "label_version"?}, "store", "version", "saved"}.
    Add volume_scale as in the training config. label_version appears when the
    store has several versions. Save first ("saved" says whether a save
    exists); docs/ink_labels.md has the label zarr and surface volume commands.
    """
    return await _call("ink_labels.dataset_entry", {})


@mcp.tool()
async def vc3d_ink_labels_set_tool_state(
    label_mode: Optional[bool] = None,
    label_class: Optional[Literal["ink", "background", "validation"]] = None,
    erase: Optional[bool] = None,
    background_keeps_ink: Optional[bool] = None,
    tool: Optional[Literal["brush", "polygon", "fill"]] = None,
    radius: Optional[float] = None,
    display: Optional[_Display] = None,
    assist: Optional[_AssistSettings] = None,
) -> dict[str, Any]:
    """Set the label tools the user sees in the Ink Detection dock: label mode,
    class, eraser, tool, brush radius (0.5 to 512 label px), display
    {"show", "ink", "supervision", "validation", "opacity" 0..1} and assist
    settings {"guide" "prediction"|"labels", "threshold" 0..255, "above",
    "connectivity" 4|8, "unlabeledOnly", "median" 1|3|5, "minArea",
    "cleanupSize" 1..256}. Omitted fields stay as they are.

    label_mode=true needs an open session (vc3d_ink_labels_open) and is
    refused with -32009 data.kind "label_mode_refused" while segmentation
    editing is on. Returns the state, as vc3d_ink_labels_state.
    """
    return await _call(
        "ink_labels.set_tool_state",
        _strip_none(
            {
                "labelMode": label_mode,
                "class": label_class,
                "erase": erase,
                "backgroundKeepsInk": background_keeps_ink,
                "tool": tool,
                "radius": radius,
                "display": display,
                "assist": assist,
            }
        ),
    )


@mcp.tool()
async def vc3d_ink_labels_close() -> dict[str, Any]:
    """Close the ink label session. Unsaved labels are kept as an autosave
    checkpoint (restored by vc3d_ink_labels_open(recover="restore")); the
    undo history ends. Returns {"closed", "keptAsAutosave"}.
    """
    return await _call("ink_labels.close", {})


@mcp.tool()
async def vc3d_ink_labels_fill(
    point: _FillPoint,
    space: Literal["label", "volume"] = "label",
    label_class: Literal["ink", "background", "validation"] = "ink",
    erase: bool = False,
    background_keeps_ink: bool = False,
    guide: Optional[Literal["prediction", "labels"]] = None,
    threshold: Optional[int] = None,
    above: Optional[bool] = None,
    connectivity: Optional[int] = None,
    unlabeled_only: Optional[bool] = None,
    bound: Optional[_Rect] = None,
) -> dict[str, Any]:
    """Flood-fill the connected region around a point with a class, as one
    undo entry (actor "agent").

    guide "prediction": the region where the selected ink prediction is at or
    above (above=true) or at or below the threshold (0..255, the prediction's
    display units); "labels": where the label values equal the point's.
    connectivity 4 or 8; unlabeled_only stops at pixels already ink or
    background. Omitted settings keep the dock's. bound limits the region (by
    default the visible part of the segment view); a fill covers at most
    4096 x 4096 label px. Returns {"changedPixels", "stateSerial",
    "undoDepth", "reachedBound"}; reachedBound means the region was cut by
    the bound. No single-channel prediction: -32009 data.kind "no_prediction".
    """
    return await _call(
        "ink_labels.fill",
        _strip_none(
            {
                "point": point,
                "space": space,
                "class": label_class,
                "erase": erase,
                "backgroundKeepsInk": background_keeps_ink,
                "guide": guide,
                "threshold": threshold,
                "above": above,
                "connectivity": connectivity,
                "unlabeledOnly": unlabeled_only,
                "bound": bound,
            }
        ),
    )


@mcp.tool()
async def vc3d_ink_labels_suggest(
    threshold: Optional[int] = None,
    above: Optional[bool] = None,
    median: Optional[int] = None,
    min_area: Optional[int] = None,
    bound: Optional[_Rect] = None,
) -> dict[str, Any]:
    """Find ink suggestions: connected components of the thresholded ink
    prediction (median 1|3|5 smoothing, a 3 x 3 opening, components of at least
    min_area label px), leaving out pixels already labeled ink or background.
    They are shown hatched in the view until accepted or discarded.

    Returns {"count", "suggestions": [{"id", "area", "bbox": {x, y, width,
    height}}] (at most 2000 listed), "truncated", "window"}. Then call
    vc3d_ink_labels_accept_suggestions or vc3d_ink_labels_discard_suggestions.
    """
    return await _call(
        "ink_labels.suggest",
        _strip_none(
            {
                "threshold": threshold,
                "above": above,
                "median": median,
                "minArea": min_area,
                "bound": bound,
            }
        ),
    )


@mcp.tool()
async def vc3d_ink_labels_accept_suggestions(
    ids: Optional[list[int]] = None,
    all: bool = False,
) -> dict[str, Any]:
    """Paint suggestions as ink, as one undo entry (actor "agent"): the given
    ids, or all=true for every one, or the ones the user picked when neither is
    given. Pixels already ink or background are never changed. Returns
    {"changedPixels", "stateSerial", "undoDepth", "accepted"}. Without
    suggestions: -32007 data.kind "suggestions".
    """
    return await _call(
        "ink_labels.accept_suggestions",
        _strip_none({"ids": ids, "all": all}),
    )


@mcp.tool()
async def vc3d_ink_labels_discard_suggestions() -> dict[str, Any]:
    """Drop the shown suggestions (and a fill, suggestion or cleanup still
    running from the dock). Returns {"discarded"}."""
    return await _call("ink_labels.discard_suggestions", {})


@mcp.tool()
async def vc3d_ink_labels_cleanup(
    kind: Literal["remove_specks", "fill_holes", "grow", "shrink", "background_ring"],
    size: Optional[int] = None,
    bound: Optional[_Rect] = None,
) -> dict[str, Any]:
    """Clean up the ink labels in a region, as one undo entry (actor "agent").

    size n (1..256 px): remove_specks clears ink components under n px (they
    stay supervised), fill_holes fills holes under n px inside ink,
    grow and shrink move the ink edge by n px (shrunk pixels stay
    supervised), background_ring marks unlabeled pixels within n px of ink as
    background. bound defaults to the visible part of the view (at most
    8192 x 8192 px). Returns {"changedPixels", "stateSerial", "undoDepth"}.
    """
    return await _call(
        "ink_labels.cleanup",
        _strip_none({"kind": kind, "size": size, "bound": bound}),
    )


@mcp.tool()
async def vc3d_ink_labels_import(
    path: str,
    label: Optional[Literal["ink", "supervision", "validation"]] = None,
    allow_resample: bool = False,
    legacy_mask_as: Optional[Literal["supervision", "ignore"]] = None,
) -> dict[str, Any]:
    """Import a label image into one label bit, as one undo entry, and wait.

    path: absolute label TIFF or PNG, a create_label_zarrs label zarr, or a 2D
    zarr. label: which bit it sets (by default from the file name:
    *_supervision_mask*, *_validation_mask*, else ink). The image must be the
    canvas size; a render at scale 0.5 or 0.25 needs allow_resample=true
    (else -32602 data.kind "resample_needed"). A legacy <seg>_inklabels.png
    with <seg>_mask.png needs legacy_mask_as (else -32602 data.kind
    "legacy_mask"). Returns {"changedPixels", "stateSerial", "undoDepth",
    "label", "format", "setPixels", "resampledFrom", "warnings"}.
    """
    return await _call(
        "ink_labels.import",
        _strip_none(
            {
                "path": path,
                "label": label,
                "allowResample": allow_resample,
                "legacyMaskAs": legacy_mask_as,
            }
        ),
        timeout=_IMPORT_TIMEOUT_S,
    )
