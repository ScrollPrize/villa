---
name: vc3d-flattening
description: Flatten or straighten a VC3D tifxyz segment through MCP using SLIM, ABF++, or vc_straighten, manage the shared flatten job, verify the generated geometry, attach it when needed, and optionally render it. Load for flatten.slim, flatten.abf, or flatten.straighten workflows.
---

# Flatten or straighten a segment

Assume `vc3d-bridge-session` and `vc3d-segment-lifecycle`.

For a fitted Spiral `.ckpt`, use `vc3d-spiral-checkpoint-flattening` instead.
That standalone exporter reconstructs the surface and uses a private Lasagna
service; it is not one of the MCP `flatten.*` jobs below.

## Choose the operation

- Prefer `vc3d_flatten_slim` for production flattening. Its default
  full-resolution symmetric-Dirichlet path is the baseline; lowering
  `keep_percent` introduces decimation and an additional UV-lift executable.
- Use `vc3d_flatten_abf` when ABF++ is specifically requested. It writes
  `<segment>_abf`, has no output-directory argument, and does not require a
  current volume.
- Use `vc3d_flatten_straighten` for spine unbending, overlap passes,
  orthogonalization, and trimming rather than UV flattening.

Do not change energy, decimation, inpainting, trimming, or iteration controls
without recording the choice and rationale.

## Run

1. List segments, choose by id, and materialize an Open Data placeholder.
2. Use a disposable output path where the tool permits one. Avoid the source
   directory: SLIM may otherwise rebuild in place.
3. Launch exactly one flatten operation with `wait=true`. All three share the
   non-cancellable `flatten` job source.
4. On failure, preserve `consoleTail` and distinguish invalid inputs, missing
   executables (`-32006`), pre-existing output, and geometry failure.

SLIM needs `flatboi`, `vc_tifxyz2obj`, and `vc_obj2tifxyz`; decimated runs also
need `vc_obj_uv_lift`. Straighten needs `vc_straighten` and refuses to overwrite
an existing output directory.

## Verify and consume the result

1. Record the terminal job and `outputDir`; inspect the directory even when the
   surface list does not refresh.
2. Confirm tifxyz metadata/grid files exist and are non-empty. Compare source
   and output bounds, grid dimensions, valid-point count, and finite-coordinate
   count where readers are available.
3. Confirm the output is distinct from the source unless in-place behavior was
   explicitly requested.
4. If the result is absent from `vc3d_list_segments`, attach its directory with
   `vc3d_attach_segments`, then list/activate it by returned identity. An
   `attached:true` response is not enough: if no new selectable id appears,
   classify the output as filesystem-only. The MCP render tool cannot target a
   directory, so do not render the source segment and present it as flattened
   output.
5. For end-to-end proof, use `vc3d-rendering` to render at least one output
   slice against the intended volume and inspect real pixels.

Flattening changes surface parameterization, not the source volume. Do not
claim geometric quality from file existence alone; record available distortion
or coverage evidence and any limitation.
