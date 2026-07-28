# Plan: Native 3D Trace2CP GPU Sparse Sampling

## Scope

Remove the current per-candidate CPU normal callback and CPU-routed inferred
field lookup from native 3D Trace2CP tracing. Reuse Lasagna's sparse GPU chunk
cache and sampling infrastructure; do not create an independent tracer-specific
block table implementation.

## Implementation

1. Preserve and rerun the current profile baseline.
   - Use the same metric command shape the user has been running.
   - Capture `native_trace2cp_profile` before and after.
   - Track at least `lasagna_normal_sample`, `trace_candidate_normals`,
     `trace_candidate_score`, `field_sample_lookup`, `inference_forward`, total
     wall/cpu time, and final metric score.

2. Add a native Trace2CP normal sampler backed by Lasagna streaming data.
   - Load Lasagna normals through `lasagna.fit_data.load_3d_streaming(...)` or
     the equivalent existing CLI/data helper, restricted to
     `grad_mag`, `nx`, and `ny`.
   - Use the returned `FitData3D.grid_sample_fullres(...)` path so sampling
     routes through `SparseChunkGroupCache` or
     `TensorStoreSparseChunkGroupCache` and the existing CUDA sparse sampler.
   - Convert selected-level tracer points to base/fullres XYZ tensor
     coordinates once per batched candidate call, on GPU.
   - Prefetch/sync the sparse caches once per whole candidate batch, not once
     per candidate.
   - Sample `nx`/`ny` at the eight requested-channel voxel corners, decode only
     those corner compact normals, convert each one to Lasagna's sign-invariant
     six-component second-moment tensor (`nx^2`, `ny^2`, `nz^2`, `nx*ny`,
     `nx*nz`, `ny*nz`), and trilinearly blend those tensors.
   - Re-encode the blended tensor with Lasagna's existing
     `tifxyz_labels.encode_from_tensor(...)` helper, then decode it with
     Lasagna's closed-form `estimate_normal(...)` path to get one local
     ambiguous normal axis for smoothness.
   - Validate with `grad_mag > 0`; do not call `FitData3D.normal_3d` on
     interpolated compact `nx`/`ny`, and do not use grid-search, power, or
     eigen fallback decoding in the tracer.
   - Replace `_NativeLasagnaNormalSampler` in the 3D tracer with this sparse
     sampler when a Lasagna manifest is available.
   - Keep a loud error if normal-aware smoothness is requested but no Lasagna
     normal source is available.

3. Make candidate normal sampling shape-preserving and tensor-native.
   - Accept `B x K x 3` and `B x K x S x 3` candidate point tensors.
   - Flatten only as a tensor view for sparse sampling.
   - Return normal tensors and valid masks reshaped to the original candidate
     tensor shape.
   - Remove the current tensor-to-NumPy conversion from the candidate normal
     path.

4. Reuse/extend Lasagna sparse sampling for inferred prediction fields.
   - Do not add a second independent tracer block-table design.
   - Factor a minimal shared sparse field/cache interface out of Lasagna sparse
     cache code if needed, preserving the existing zarr-backed cache behavior.
   - Add a float32 inferred-field cache implementation only as an extension of
     that shared interface:
     - same chunk-table ownership pattern,
     - same vectorized point-to-chunk coverage checks,
     - same GPU-resident chunk storage,
     - same CUDA sparse point sampling concept.
   - If the existing CUDA sparse sampler only supports uint8 chunks, add a
     shared float32 variant or typed path in Lasagna next to the existing
     `sparse_grid_sample_3d_u8` kernels. Keep indexing/chunk-table semantics
     shared.
   - Store decoded inference products and valid masks on GPU in that shared
     sparse field cache. Do not store CPU tensors and copy blocks back to GPU
     for every lookup.

5. Replace `NativeTraceFieldCache.sample_point_choices_torch(...)` routing.
   - Keep inference block generation and model forward behavior unchanged.
   - After model output decode, insert GPU-resident inferred blocks into the
     shared sparse field cache.
   - Sample all candidate/current/start points directly from the shared sparse
     field sampler.
   - Remove `np.unique` block grouping, Python per-block loops, and per-block
     `block.output_czyx.to(device=...)` copies from lookup.

6. Keep beam scoring vectorized; defer full beam object removal.
   - `_score_candidate_loss_tensors_batched(...)` already does most scoring as
     tensors. Keep using it.
   - Keep the current Python node reconstruction at prune/commit boundaries for
     this task, because the profile shows it is not the main bottleneck.
   - Ensure the lookahead expansion still evaluates all active beams and all
     candidate/substep samples in batched tensor calls.

7. Diagnostics and safety checks.
   - Add profile stages separating:
     - normal sparse prefetch/sync,
     - normal sparse grid sampling,
     - inferred-field cache insertion,
     - inferred-field sparse lookup.
   - Keep final metric determinism checks by comparing restart count/error on
     the same command before and after.
   - Add an opt-in debug guard using existing sparse cache coverage checks
     rather than tracer-specific partial-cache checks.

## Spec Update

Add native 3D Trace2CP tracer specs:

- Candidate normal sampling must use Lasagna streaming `FitData3D` sparse GPU
  chunk sampling for `grad_mag`, `nx`, and `ny`. It must follow Lasagna's
  sign-ambiguous second-moment tensor convention before interpolation, then use
  Lasagna's closed-form tensor-to-encoding and `estimate_normal` reconstruction
  path to get one local ambiguous normal axis for projection/smoothness.
  Calling `FitData3D.normal_3d` on interpolated compact normals, doing a
  grid-search decode, or using power/eigen fallback decoding in the hot path is
  not allowed. Per-candidate CPU geometry callbacks are not allowed in the hot
  path.
- Native 3D Trace2CP inferred prediction fields must be sampled through shared
  Lasagna sparse GPU field/cache infrastructure. The tracer must not own a
  duplicate sparse block-table implementation.
- Candidate/beam/lookahead sampling must be batched over all active samples;
  per-candidate Python callbacks are forbidden in the candidate scoring path.
- Python beam object reconstruction may remain at prune/commit boundaries until
  it becomes a measured bottleneck.

## Docs Updates

Update the relevant fiber 3D tracing documentation to describe:

- the sparse normal sampler path,
- the shared inferred-field sparse cache,
- which parts remain CPU-side after this task,
- how to interpret the new profile stages.

## Tests

Add focused tests for:

1. Sparse normal sampler uses `FitData3D.grid_sample_fullres(...)` once per
   batched candidate tensor, preserves candidate shape, and returns
   Lasagna-closed-form reconstructed local normal axes from sign-invariant
   blended tensors rather than interpolated compact-normal vectors.
2. Candidate normal sampling no longer converts each candidate through a Python
   callback.
3. Inferred-field sparse cache lookup returns the same directions/presence as
   the old block lookup on a synthetic block arrangement.
4. Multi-branch candidate scoring is unchanged for single-branch and
   multi-branch outputs.
5. Missing normal chunks/invalid `grad_mag` preserve current invalid-candidate
   behavior.

Validation commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d or whole_fiber_trace"
```

Also run the user metric command with `--profile` before/after and compare
final metric plus profile table.

## Changelog

Add one changelog entry noting that native 3D Trace2CP candidate normal and
prediction-field sampling were moved onto shared Lasagna sparse GPU sampling.

## Deferred Items

- Full Python-free beam storage/reconstruction is intentionally deferred. The
  profile shows beam rebuild/prune is currently about 1-2% of wall time, while
  normal sampling and candidate field scoring dominate.
- The shared float32 inferred-field sparse cache remains pending after the
  current implementation pass. Lasagna's existing sparse cache is uint8
  zarr-chunk specific and directly fits `grad_mag`/`nx`/`ny` normal sampling;
  live checkpoint outputs need a shared typed float32 sparse field extension
  before the tracer-local trusted-core block routing can be fully removed
  without replacing it by another duplicate implementation.
