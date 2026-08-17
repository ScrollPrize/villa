---
name: vc3d-spiral-checkpoint-flattening
description: Export a fitted Spiral `.ckpt` into a combined surface and flatten it to a final TIFXYZ with `scripts/spiral/flatten_spiral_checkpoint.py` and a private Lasagna service. Load for standalone Spiral checkpoint flattening, checkpoint-to-TIFXYZ conversion, or diagnosing that exporter; do not use for MCP `flatten.*` jobs or ordinary Spiral preview export.
---

# Flatten a fitted Spiral checkpoint

This is a standalone host workflow, not an MCP RPC. The script reconstructs a
combined Spiral surface, starts a private Lasagna service on an ephemeral
loopback port, runs `flatten_fast_nofilter.json`, publishes the final TIFXYZ
atomically, and stops the service.

Use `vc3d-spiral` to obtain or download a fitted checkpoint. Use
`vc3d-flattening` for an already-materialized VC3D segment. A normal
`vc3d_spiral_export_preview` is flattened and published by the connected
Spiral host; do not run this standalone exporter as an extra preview step.

## Preflight

1. Resolve `scripts/spiral/flatten_spiral_checkpoint.py` and a fitted
   checkpoint containing `cfg`, `z_begin`, `z_end`, and
   `spiral_and_transform`.
2. Use an existing Spiral Python environment. The project requires Python
   3.14+ and imports PyTorch plus the Spiral modules. Do not install or run
   `uv sync` without authorization.
3. Resolve `umbilicus.json`. It is not embedded in the checkpoint. Prefer
   `--umbilicus`; otherwise the script searches `SPIRAL_DATASET`, checkpoint
   ancestors, and legacy local locations.
4. Resolve a Lasagna checkout containing both `fit_service.py` and
   `configs/flatten_fast_nofilter.json`. Prefer `--lasagna-dir`; the fallback
   uses `LASAGNA_SERVICE_PATH`, the repository sibling, then `~/villa/lasagna`.
5. Verify the output does not exist. The script deliberately refuses to
   overwrite it. Confirm its parent is writable and has capacity for the
   reconstructed temporary surface, model output, and final TIFXYZ.
6. Verify source voxel size. The CLI default is `9.6` micrometers; pass
   `--voxel-size-um` when that is not authoritative for the checkpoint.

CUDA is the default reconstruction device. Use `--device cpu` only as an
intentional, potentially slow fallback. Leave `--chunk-size 65536` unchanged
unless memory measurement justifies a different transform batch size.

## Run

From the repository root, using the already-provisioned interpreter:

```sh
AGENTS_AGENT_MODE=1 /path/to/spiral-python \
  scripts/spiral/flatten_spiral_checkpoint.py \
  /path/to/checkpoint_fitted.ckpt /path/to/output.tifxyz \
  --umbilicus /path/to/umbilicus.json \
  --lasagna-dir /path/to/lasagna \
  --voxel-size-um 9.6
```

Record the interpreter, checkpoint, checkpoint hash when practical,
umbilicus, Lasagna directory/config, device, voxel size, chunk size, output,
and start/end times. Do not substitute `vc3d_lasagna_start_optimization`: the
private service and its job are not visible to `vc3d_lasagna_jobs` or
`vc3d_job_status`.

The log should show reconstruction progress, the selected Lasagna config,
service stages, and finally `wrote <output>`. Interrupt the top-level script
rather than targeting an inferred child PID; its cleanup stops the private
service and removes the temporary sibling work directory.

## Validate

A zero exit status is necessary but insufficient:

1. Confirm the output is a directory with non-empty `meta.json`, `x.tif`,
   `y.tif`, and `z.tif`.
2. Read all three coordinate grids and verify equal dimensions, readable
   float data, finite valid coordinates, and nontrivial bounds.
3. Check metadata for the expected TIFXYZ identity, grid scale, Lasagna fit
   configuration/job, and linked-surface object references. Confirm voxel size
   through the requested value and area metadata where present; the final
   metadata need not contain the exporter script name as a dedicated field.
4. Confirm no partial output was published after a failure and no
   `.<output-name>.flatten-*` sibling remains.
5. Attach the result to VC3D only after validation, then use `vc3d-rendering`
   with a data-bearing, coordinate-compatible volume for pixel evidence.

Distinguish checkpoint/config errors, missing umbilicus or Lasagna files,
CUDA unavailability, service startup failure, Lasagna job error/cancellation,
and a reported success with no TIFXYZ. Preserve the log and exact exception;
never present temporary source geometry as the final flattened result.
