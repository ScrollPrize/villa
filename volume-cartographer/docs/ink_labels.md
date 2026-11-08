# Ink labels

VC3D can draw ink detection training labels directly on a segment's flattened
view and save them as the files the ink detection trainer reads. Labels are
drawn at the exact pixel grid `vc_render_tifxyz` writes for the segment, so
they line up with its renders without any manual alignment.

The label tools are the **Ink labels** section at the bottom of the
**Ink Detection** dock (**View → Ink Detection**). **View → Ink label mode**
turns label mode on and off too.

## Label mode

Select a segment, then press **I** (or **Label mode** in the dock). The
segment view's left mouse button now paints labels; the right button still
pans and the wheel still zooms. Label mode is refused while segmentation
editing is on, because editing moves the grid the labels are attached to.
Turning label mode on turns approval painting off, and turning editing or
approval painting on turns label mode off at the next click. While label mode
is on, the status bar shows the class, tool and radius, for example
"Ink labels: ink, brush r 12".

Every label pixel has three independent bits, as the trainer sees them:

* **Ink (W)** marks the pixel as ink and as supervised.
* **Background (F)** marks it as supervised and not ink. With
  **Background keeps ink** checked, painting background over ink keeps the ink.
* **Validation (V)** marks the pixel as part of the validation area.
* **Eraser (Z)** returns ink and background to unknown (not supervised, so
  the trainer ignores the pixel) and clears validation. To turn ink into
  background, paint background instead.

## Tools

* **Brush.** Click to stamp a disc, drag to paint, **Shift+Click** for a
  straight stroke from the last point. **Ctrl+Mouse Wheel** changes the radius
  (0.5 to 512 label pixels, which are voxels at full resolution).
* **Polygon.** Click vertices and press **Enter** or double-click to fill, or
  drag a lasso and release. **Backspace** removes the last vertex, **Esc**
  cancels. Self-intersecting outlines fill even-odd.

**L** switches tools (brush, polygon and the fill tool of the Assist rows).
Each stroke, polygon or Shift+Click segment is one undo step: **Ctrl+Z**
undoes and **Ctrl+Shift+Z** (or **Ctrl+Y**) redoes, also when a dock control
has focus. Label undo is separate from segmentation undo. The
history keeps whole operations up to 256 MiB. Switching to another segment
keeps unsaved labels (as an autosave) but ends their undo history; switching
back restores the labels with a fresh history.

## Display

**H** shows or hides the labels; **Space** still toggles the ink prediction
below them. Ink is drawn red, validation yellow and supervised background
blue; the dock sets which of them are shown and their opacity. When zoomed out,
blocks of label pixels are pooled: a block is red if any pixel is ink, yellow
if any is validation, solid blue if all are supervised and faint blue if some
are. Labels are not drawn on the volumetric composite camera.

## Assist

The **Assist** group (folded by default; click its title to open it) uses the
ink prediction selected in the Ink Detection dock
(single-channel predictions only; a color prediction has no values to
threshold). Each tool works on the part of the segment visible in the view,
runs in the background, and lands as one undo step. **Esc** drops a result
that is still being computed.

* **Fill** (the third tool; **L** cycles brush, polygon and fill). A click
  fills the connected region where the prediction is at or above (or at or
  below) the **Threshold**, or, with the **Labels** guide, where the labels
  equal the clicked pixel. It paints the current class, 4- or 8-connected,
  and with **Unlabeled only** stops at pixels that are already ink or
  background. A fill covers at most 4096 x 4096 label pixels around the
  click; when it stops at the edge of the view, the status bar says so (zoom
  out to fill further).
* **Suggest (U)** smooths the prediction with an optional median, thresholds
  it, removes one-pixel speckle and thin lines with a small opening, and shows
  the connected components of at least the minimum area, hatched in cyan.
  Pixels already labeled ink or background are left out, so accepting a
  suggestion never overwrites the annotator's background. Click components to
  pick them (they turn red) or press **Shift+U** (**Pick all**) to pick them
  all, then press **Enter** to accept them as ink, or **Esc** to discard them.
  **Enter** with nothing picked keeps the suggestions. Accepted suggestions
  are recorded in the sidecar's `provenance` with the threshold and filter
  settings.
* **Cleanup** works on the visible labels with size n: **Remove specks**
  (ink components under n px become background), **Fill holes** (holes under
  n px inside ink become ink), **Grow** and **Shrink** (ink by n px; shrunk
  pixels stay supervised as background) and **Background ring** (unlabeled
  pixels within n px of ink become background).

The threshold is in the prediction's display units: 8-bit images as they
are, other depths scaled to 0 to 255. The brush ring takes the class color:
red for ink, blue for background, yellow for validation and gray for the
eraser.

## Files

Labels live outside the segment folder, in

```text
<project folder>/ink_labels/<segment folder name>/
  <seg>_inklabels.tif              ink           (written by Save)
  <seg>_supervision_mask.tif       supervised    (written by Save)
  <seg>_validation_mask.tif        validation    (only when used)
  <seg>_inklabels.json             sidecar: canvas, hashes, edit counts
  *.tif.prev                       the previous saved generation
  *.autosave                       autosave checkpoints
  <seg>_inklabels.lock             held while a VC3D has the labels open
```

A project opened from the open-data catalog keeps its project file in the
remote cache, which can be wiped and is shared by every sample, so its labels
go to `<application data folder>/open_data_labels/<sample>/ink_labels/` instead
(on Linux `~/.local/share/Vesuvius Challenge/VC3D/...`). The dock shows the folder in use. The
folder and its lock file are created at the first edit, so opening a segment
writes nothing.

**Save (Ctrl+S)** writes the three TIFFs as uint8 0/255, one sample, tiled
1024, LZW, at the canvas size `vc_render_tifxyz --scale 1` produces for the
segment, then the sidecar. The trainer and `create_label_zarrs` read the TIFFs
and ignore the other files. The sidecar records the canvas (grid size and the
scale's exact float bits), the segment uuid, a hash of the segment's geometry
and a SHA-256 of each file.

* **Autosave.** Two minutes after the last checkpoint, once painting has paused
  for two seconds, unsaved labels are written to the `.autosave` files. They
  never replace the trainer files. Switching segment or closing the project
  checkpoints too. After a crash, opening the labels offers to restore the
  checkpoint. A save deletes the checkpoint, so an undo followed by a save
  can never bring the undone strokes back. A checkpoint that cannot be
  restored (written for another canvas, or with an unreadable sidecar) is
  renamed to `*.unusable`, never deleted.
* **Outside edits.** If a trainer file changed since VC3D loaded or saved it
  (an edit in another program, or an interrupted save), Save asks before
  replacing it; the replaced file stays as `.prev`.
* **Another VC3D.** A second VC3D opening the same labels gets them read-only,
  with the holder's process and host in the dock. If both opened a store that
  did not exist yet, the first to paint takes it and the other turns read-only
  at its first edit.
* **A changed segment.** If the segment's grid or scale changed since the
  labels were saved (a re-render or a regrown segment), the labels lock and are
  neither shown nor saved; the dock names both canvases. To carry labels
  across a re-render use `vesuvius.tifxyz_label_transfer`. A geometry edit
  that keeps the grid only warns: labels stay on the same grid positions.
* **Closing VC3D** with unsaved labels asks to save, discard or cancel.

## Importing labels

**Import...** reads an existing label image into one label bit, chosen by
its file name (`*_supervision_mask*` and `*_validation_mask*`, otherwise
ink): pixels above 0 get the bit and pixels at 0 lose it, as the trainer
reads them. It accepts:

* TIFF (read block by block, so large files are fine) and PNG (up to 1 GiB);
  extra channels are ignored, as `create_label_zarrs` ignores them;
* a label zarr written by `create_label_zarrs` (plane 32 of `<x>.zarr/0`),
  or a 2D uint8 or uint16 zarr (v2 or v3) at `<x>.zarr/0`;
* a legacy `<seg>_inklabels.png` with `<seg>_mask.png` beside it, where
  VC3D asks whether the mask (the valid surface) becomes the supervision mask.

The image must have the canvas size. A render at scale 0.5 or 0.25 is
offered nearest-neighbor resampling; a rotated or unrelated size is refused
with both sizes. Imports are one undo step and are recorded in the sidecar's
`provenance`.

## Versions and training

The trainer picks the newest `_vN` of each label file. **Save as new
version** writes all three files as the next version (an all-zero
validation mask when none is painted), so the newest ink, supervision and
validation always belong together; the dock then edits that version. The
version list opens any other version, and the newest opens by default. When
labels are locked to an old canvas, **Start empty labels in a new version**
labels the current canvas without touching them.

**Copy dataset entry** (in the folded **Versions and import** group, with the
version list and **Import...**) puts the trainer's dataset entry for these
labels on the clipboard as JSON:

```json
{
    "label_version": "v2",
    "segment_names": ["w045"],
    "segments_path": "/data/proj/ink_labels",
    "surface_volume_paths": {"w045": "/data/proj/ink_labels/w045/w045.zarr"}
}
```

`label_version` appears only when the store has more than one version, so
the trainer reads the version you saved instead of the newest. Add
`volume_scale` as in your config. Setting `surface_volume_paths` means
`x.tif` is not needed in non-native modes; native 3D modes need `x.tif` in
the segment folder, which the label folder does not have.

To turn saved labels into trainer input (POSIX shell; adjust paths and
quoting on Windows):

```sh
# 1. label zarrs for this segment only
cd villa/vesuvius && uv run --extra models python -m vesuvius.ink_detection.preprocessing.create_label_zarrs \
    "<store>/<seg>" --workers 4 --overwrite
# 2. the surface volume next to the labels (time a crop first)
vc_render_tifxyz -v <volume> -s "<segment folder>" --scale 1 -g 0 -n 65 --zarr-output "<store>/<seg>/<seg>.zarr"
# 3. validation, on a <root>/<scroll>/<segment> layout
mkdir -p /tmp/vc3d-labels/<scroll> && ln -sfn "<store>/<seg>" /tmp/vc3d-labels/<scroll>/<seg>
uv run --extra models python -m vesuvius.ink_detection.preprocessing.validate_segments /tmp/vc3d-labels
```

Here `<store>/<seg>` is the label folder the dock shows.

## Agents

The VC3D agent bridge exposes the same session, with a method for every
control of the label section (MCP tools `vc3d_ink_labels_*`): `state`,
`open`, `paint` (strokes, polygons, masks, or exact label values),
`read`, `history`, `save` (also as a new version), `fill`, `suggest`,
`accept_suggestions`, `discard_suggestions`, `cleanup`, `import`,
`versions`, `dataset_entry`, `set_tool_state` and `close`, all under
`ink_labels.`. No method opens a dialog. Agent edits show live in the view,
are recorded with actor `agent`, and an agent undo stops at the user's edits
unless forced. See the "Ink labels" section of
`apps/VC3D/agent_bridge/SPEC.md`.

## Pixel mapping

Label pixel `(c, r)` covers `[c, c+1) x [r, r+1)` of the canvas of size
`W = max(1, round(cols / scale_x))` by `H = max(1, round(rows / scale_y))`.
Its grid position is

```text
grid_x = cols/2 + scale_x * (x - W/2)
grid_y = rows/2 + scale_y * (y - H/2)
```

which at pixel centers is exactly the position `vc_render_tifxyz` samples for
output pixel `(c, r)`. The formula lives in `vc/core/util/RenderCanvas.hpp`,
which the renderer uses too.

## Keys

| Key | Action |
| --- | --- |
| I | Toggle ink label mode |
| W, F, V | Paint ink, background, validation |
| Z | Toggle the eraser |
| L | Switch tools: brush, polygon, fill |
| U | Suggest ink from the prediction |
| H | Show or hide the labels |
| Ctrl+Z, Ctrl+Shift+Z | Undo, redo (in label mode) |
| Ctrl+S | Save (in label mode) |
| Ctrl+Mouse Wheel | Brush radius |
| Shift+Click | Straight stroke from the last point |
| Enter, Backspace, Esc | Close, shorten or cancel a polygon |
| Shift+U | Pick every suggestion |
| Enter, Esc | Accept the picked suggestions; drop suggestions or a running assist |

The keys work wherever focus is in the main window, including the docks and the
dock bar's pop-up, while the **main** (segment) workspace is
the current tab; other workspaces keep their own bindings of these letters.
Enter, Backspace, Esc and Shift+U are taken only while a polygon is open or
suggestions are shown.

## Object-store projects

On current Villa, a surface in a local or hosted object store is identified by its stable object UUID. Label exports use that UUID under the local project's `ink_labels` folder, rather than a temporary downloaded geometry folder. The label folder is shown in the Ink labels panel and can be selected and copied. Labels remain local trainer exports; Save does not submit them to a hosted store. Geometry and canvas checks still guard reopening an export after the surface changes. Export matching surface geometry and a surface volume separately before training.
