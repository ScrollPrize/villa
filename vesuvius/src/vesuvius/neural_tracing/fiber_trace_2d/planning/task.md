# Shared 3D Tiled Inference For Lasagna Cos And Fiber Models

Update the stale backed-up predict3d planning task after the main merge.

Current baseline:

- `lasagna/preprocess_cos_omezarr.py predict3d` already contains the rolling
  z-band accumulator, per-channel mmap scratch files, stale temp cleanup,
  atomic chunk writes, per-output-channel completeness checks, canonical global
  tile/output chunk helpers, chunk-level resume behavior, OME-Zarr pyramid
  generation, and independent `pred_dt` handling.
- Those features are current behavior and must be preserved.
- The remaining task is not to re-plan the rolling accumulator itself. The
  remaining task is to factor the shared inference mechanics out of the
  monolithic cos/Lasagna script and use them for 3D fiber inference as well.

Requirements:

- Keep all current Lasagna `predict3d` features and command-line compatibility.
- Extract the reusable tiled 3D inference/resume/output mechanics into a shared
  module instead of duplicating them in a fiber-specific script.
- Keep `preprocess_cos_omezarr.py predict3d` as the Lasagna cos/normal wrapper
  around that shared module.
- Add a separate 3D fiber inference entry point using the same shared runner
  and the same common arguments where applicable:
  `--input`, `--output`, `--checkpoint`, `--tile-size`, `--overlap`,
  `--border`, `--scaledown`, `--crop`, `--device`, `--no-download`,
  `--levels`, and `--ome-chunk`.
- Preserve crop-composable resume semantics:
  output chunks and inference tiles use a canonical global lattice; overlapping
  or separate bbox/crop runs must produce the same bytes for the same global
  output chunk.
- Preserve resume/completeness semantics:
  no done markers; output channel chunks are the durable state; missing
  auxiliary `pred_dt` chunks must not trigger Lasagna model inference.
- Preserve atomic write semantics:
  write every output chunk to a unique temp path on the same filesystem, then
  install it with `os.replace`.
- Fiber inference must use the existing 3D fiber model APIs in
  `vesuvius.neural_tracing.fiber_trace_3d`, including checkpoint/config
  loading, input normalization, BF16/FP16 autocast compatibility where
  configured, and Lasagna 3x2 direction encoding helpers.
- Fiber inference outputs are not Lasagna `grad_mag/nx/ny`.
  Each fiber output option is a coherent seven-channel bundle:
  six Lasagna 3x2 direction channels plus one presence channel.
- Multi-branch or recurrent/conditioned fiber outputs must be preserved as
  separate coherent options. Inference must not collapse them to branch 0 or to
  a min/max/average summary unless an explicit postprocessing mode is added.
- Product completeness for fiber inference is per option/bundle:
  an option chunk is complete only when all seven required output channel
  chunks exist.
- The shared runner must keep product-specific behavior behind adapters:
  model loading, output channel schema, tile postprocessing, manifest/group
  metadata, product completeness, and pyramid behavior.
- Do not change model output semantics, normalization, selected scale handling,
  Lasagna normal encoding, 3D fiber training, or Trace2CP tracing behavior as
  part of this task.
