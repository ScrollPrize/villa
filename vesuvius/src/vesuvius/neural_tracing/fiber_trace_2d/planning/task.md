# Task: Native 3D Trace2CP GPU Sparse Sampling

Optimize native 3D Trace2CP tracing by removing per-candidate CPU callbacks
from the beam/candidate loop.

Current profiling shows the tracer is effectively single-core CPU-bound:

- `lasagna_normal_sample` / `trace_candidate_normals`: about 61% wall time.
- `trace_candidate_score`: about 31% wall time.
- `inference_forward`: about 8% wall time.
- Source block reads are below 1% wall time.

The implementation must reuse and, where necessary, extend Lasagna's existing
GPU sparse chunk/cache machinery. It must not introduce a parallel duplicate
block table/cache design in the 3D tracer.

Requirements:

- Candidate evaluation must operate on batched tensors over beams, candidates,
  lookahead states, and substeps. No per-candidate Python callback path.
- Lasagna normal sampling must use the existing sparse GPU chunk cache path for
  `grad_mag`, `nx`, and `ny`, via streaming `FitData3D` where possible.
  Compact `nx`/`ny` samples must be decoded only at the requested-channel voxel
  corners, converted to Lasagna's sign-ambiguous second-moment tensor
  convention before interpolation, and then reconstructed with Lasagna's
  existing closed-form tensor-to-encoding plus `estimate_normal` path to get
  one local ambiguous normal axis for candidate smoothness. Raw `nx`/`ny`
  interpolation, `FitData3D.normal_3d` on interpolated `nx`/`ny`, grid search,
  and power/eigen fallback decoding are not allowed in the tracer.
- Inferred fiber prediction fields should remain GPU-resident for sampling.
  Reuse/extend Lasagna sparse-cache/sparse-sampling code for this rather than
  maintaining tracer-local CPU block lookup and per-block GPU copies.
- Fast cached zarr/chunk loading is part of the sparse chunk tensor design and
  must remain in that shared Lasagna code path.
- Keep native 3D Trace2CP semantics, determinism, multi-branch handling,
  scaledown/blur behavior, and current metric outputs unchanged except for
  performance and added diagnostics.
- Full Python-free beam reconstruction can be delayed until after the normal
  and field sampling bottlenecks are removed.
