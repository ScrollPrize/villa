# Fiber drift and metric notes

Use this reference only after the core fiber workflow.

## Visual drift

- Inspect both orthogonal cut panes and the strip panes. A single view can make
  an oblique trace look centered when it is not.
- Keep base CT and prediction overlays distinct. An overlay of the base volume
  onto itself is not evidence.
- Record the seed, selected volume, Lasagna manifest identity, viewer names,
  and screenshots before changing controls.
- Re-read `vc3d_get_state` after each edit because pane ids can be recreated.
- The MCP contract cannot set an arbitrary along-line position or directly
  add, move, or remove a control point. Use ordinary pane clicks and describe
  that limitation rather than naming a nonexistent tool.
- Save and list the fiber before quoting control-point counts or spans.
  Unsaved workspace state has no structured MCP getter.

## Coordinate levels

Open Data predictions and normals can live at different pyramid levels. Derive
an L0 seed from the representation metadata and selected base volume; do not
reuse a displayed L1/L2 coordinate as L0. Preserve sample id, volume id,
representation ref, source level, and coordinate-space tags in evidence.

## Native metric

`vc_fiber_trace_metric` is a repository CLI, outside the MCP job registry.
Inspect its current `--help` before running it and use explicit fiber,
prediction-manifest, normal-manifest, cache, and output paths.

A useful metric needs real endpoint controls. A seed-only fiber can produce
self-consistent numbers against extrapolated endpoints that do not measure
agreement with placed anchors. Record:

- fiber id/path and control-point count;
- prediction and normal manifest identities and levels;
- voxel-size and scale conversions;
- successful and failed spans;
- distance/error threshold and restart count;
- warnings and output path.

Do not convert CLI completion into an MCP success claim, and do not treat a
local metric as a substitute for a failed Open Data workflow.
