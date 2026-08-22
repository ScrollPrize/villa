---
name: vc3d-rendering
description: Render a tifxyz segment against a VC3D volume through MCP, choose TIFF-stack or OME-Zarr output, control slice count and sampling, monitor the tool job, and validate actual output pixels. Load for render.tifxyz or flattened-surface image generation.
---

# Render a tifxyz surface

Assume `vc3d-bridge-session`; use `vc3d-open-data` for remote volumes and
`vc3d-flattening` when geometry must be flattened first.

## Prepare

1. Call `vc3d_get_state`, `vc3d_list_attached_volumes`, and
   `vc3d_list_segments`.
2. Confirm the volume is data-bearing. For a local OME-Zarr, distinguish array
   metadata from chunk payloads: a store containing only `.zarray`/`.zattrs`
   with `fill_value: 0` renders zeros by construction. For a remote volume,
   confirm its locator is readable with a bounded view or render probe.
3. Materialize a remote placeholder with `vc3d_fetch_segment(..., wait=true)`.
4. Choose the segment and volume by identity. Pass `volume_id` explicitly when
   more than one volume is attached or provenance matters.
5. Read the actual tifxyz grid dimensions and choose `group_idx` and `scale`
   that keep the probe bounded while retaining enough output pixels to vary.
   Catalog metadata may describe the original grid rather than the materialized
   grid, so inspect `x.tif`, `y.tif`, or `z.tif` when available. Derive and
   record the expected output dimensions from that grid and the selected
   pyramid transform; in a standard 2x pyramid they are approximately
   `grid_dimensions * scale / 2**group_idx`. Set a numeric output-pixel budget
   before launch. If the prediction exceeds it, choose a smaller materialized
   segment, a coarser group, or a lower scale; do not launch an accidentally
   unbounded probe.
6. Choose a new output directory visible to VC3D. Do not reuse an existing
   output unless overwrite behavior is documented and requested.

Viewer render settings and overlay settings do not configure
`vc3d_render_tifxyz`; the tool's own arguments are authoritative.

## Render

Call `vc3d_render_tifxyz` with:

- `output_format="tif_stack"` for per-slice TIFFs, or `"zarr"` for OME-Zarr;
- `scale > 0` for pixels per source voxel;
- `group_idx >= 0` for the source OME-Zarr group;
- `num_slices >= 1` for the normal-direction slice count;
- `voxel_size` only when overriding usable volume metadata intentionally;
- `wait=true` for a bounded foreground run.

The job source is `tool`, not `flatten`. On timeout, keep the returned job id
and continue with `vc3d_wait_job` or `vc3d_job_status`.

For a bounded validation run, choose and record a wall-clock budget before
launch. If the budget expires while the `tool` job is still running, call
`vc3d_cancel_job(job_id=...)` and poll for a terminal record. If cancellation
or the bridge itself is unresponsive, preserve partial files/status as timeout
evidence and terminate only an explicitly owned disposable VC3D session. Do
not rewrite a non-terminal timeout as a job failure or success.

Advanced crop, affine, rotation, flip, composite, alpha, and in-render
flattening controls are not exposed by this MCP tool. Do not imply that they
were applied.

## Validate the artifact

A succeeded job is necessary but insufficient.

A tiny coarse-group render is only a health probe. Label it as such and do not
present it as a substantive deliverable; after the probe, choose the finest
group that meets the agreed wall-clock and output-size budget.

1. Record the terminal job, output directory, format, selected volume, scale,
   group, and requested slice count.
2. Confirm the expected TIFF files or Zarr metadata/chunks exist.
3. Read at least one emitted plane with an available image/Zarr reader and
   report dimensions, dtype, minimum, maximum, nonzero count or fraction, and
   whether values vary.
4. For multiple slices, verify the emitted slice count and inspect a boundary
   and interior slice when available.
5. Reject an all-empty, unreadable, truncated, or dimensionally inconsistent
   artifact even if the job reports success.
6. For remote input, also record sample id, representation, source level,
   bounded access region where observable, and cache/download measurements.

When pixels are all zero, first distinguish a metadata-only/fill-only volume
from a segment/volume coordinate mismatch. Then compare an unflattened source
render with the newly attached flattened id using the same explicit volume. Do
not report either output as successful until one contains real sampled values.

Validate dimensions against the current segment's recorded prediction, not a
size observed for another segment or sample. A different but correctly
predicted size is not a failure.

Use a disposable output directory for validation and preserve it with the log
when a failure needs diagnosis.
