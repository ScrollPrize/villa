---
name: vc3d-visual-evidence
description: Capture VC3D screenshots that support the claimed result: choose the correct pane, frame real content, distinguish base and overlay volumes, hold the camera fixed, inspect pixels, and checksum artifacts. Load before vc3d_screenshot, vc3d_set_overlay, or before/after image work.
---

# Visual evidence

Assume `vc3d-bridge-session` and load `vc3d-reading-the-image` when choosing or
interpreting a point.

## Choose a meaningful pane

- The main `segmentation` / L0 Surface pane needs an active materialized
  segment. Without one, a gray pane proves nothing.
- XY/XZ/YZ panes show the selected volume.
- Fiber traces live in Line Annotation panes.
- Re-read `vc3d_get_state` after opening data or switching workspaces; viewer
  ids are process-local and panes may be recreated.

## Frame and render

1. Center the intended L0 point with `vc3d_center_viewer`.
2. Zoom with `vc3d_zoom_viewer` until the subject fills the pane. The RPC
   returns scale, not pyramid level; record the scale and the pane's visible
   L-level label when level matters.
3. Set an explicit volume window with `vc3d_set_render_settings`; persistent
   settings can otherwise leave a valid volume clipped white or black.
4. Capture the pane by `target`, not `viewer`. Use `file_path` for a durable
   artifact or omit it to inspect inline.
5. The screenshot RPC does not wait for remote fetch quiescence. For remote
   data, capture until two consecutive fixed-state images stabilize or a
   bounded timeout expires; preserve a timeout rather than accepting a blank
   first frame.

A hidden/background pane returns `-32009`. Switch to its workspace or choose a
visible pane.

## Overlay proof

`vc3d_list_overlay_volumes` reports ids and which is current. Do not overlay the
current/base volume onto itself. Choose a derived volume using the catalog
representation metadata and attached-volume identities, then call
`vc3d_set_overlay`.

Hold selected base volume, center, scale, rotation, volume window, colormap,
opacity, threshold, and resolution cap fixed. Capture:

1. overlay off;
2. overlay on;
3. overlay off again.

The first and third checksums must match; the middle must differ. Echoed overlay
state alone is not pixel evidence. If a derived representation uses a different
coordinate level, select its matching virtual source instead of forcing a
mismatched base.

## Validate artifacts

Inspect every image. Record dimensions, checksum, pane, center, scale, selected
base, overlay id, window, and build SHA. For output images or rendered TIFFs,
also record decoded dtype, min/max, nonzero count, and unique/range evidence.
Job completion or file existence is not enough; all-zero or unreadable pixels
are a failure.

Do not commit generated screenshots, WIP evidence directories, or local logs
unless the user explicitly asks for those artifacts in the PR.
