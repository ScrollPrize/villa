# Mixed-model trace inference

The live evaluator can take presence and fiber-direction nx/ny from the primary
checkpoint, and signed float32 embeddings from a second checkpoint. External
Lasagna sheet normals (used for the reference ellipse and strips) are unchanged.
The tracer, embedding accumulator policy and sampling math are unchanged.

Add this object to the evaluation config (under `evaluation` in a workload
config used by the approved runner):

```json
{
  "embedding_inference": {
    "checkpoint": "/absolute/path/to/frozen-embedding-snapshot.pt",
    "checkpoint_sha256": "optional-expected-sha256",
    "inference_cache_namespace": "existing-embedding-namespace",
    "inference": {
      "emit_embeddings": true,
      "max_batch_size": 8
    }
  }
}
```

The primary checkpoint/inference/namespace fields keep their usual meaning.
Alternate inference settings inherit the primary settings, then apply overrides;
embedding output is always enabled for the alternate. Optional `model_config`
is for legacy checkpoints only. Use absolute checkpoint/config paths with the
prepared workload runner. Keep each model's original numerical parameters and
namespace to reuse its existing chunks. Do not enable embeddings on a primary
model that lacks an embedding head. The tracer's embedding weight must still be
positive to use the alternate embeddings.

## Ownership and readiness

Both roles use ordinary inference caches with their existing source/model keys.
A small manifest under `live_inference/_composites/<identity>/` points directly
to the selected scalar arrays. It has no voxel payloads and does not replace or
rewrite either original cache. Both complete cache identities and manifest
paths are recorded. Equal physical source identity, base coordinates and output
voxel lattice are required; there is no implicit resampling. Output chunk sizes
may differ because readiness routes each array to its owning store.

The native sampler requests both roles through the existing authenticated
control channel. A request is acknowledged only after all requested underlying
chunks are published. Reference-intersection prefetch covers both models with
its usual lower priority. Cache hits are reused; missing chunks trigger only
their owning model. Identical cache roots are deduplicated before writer locks.
Alternate inference still publishes its ordinary complete products, including
its unused presence/nx/ny, to preserve its existing cache/completeness contract.

## Resource use

One manager grant owns the entire run. Existing numerical worker pools remain
model-specific: with eight GPUs, four serve each model. CPU read/publication
and explicit shared-buffer slot counts are divided between the pools, with a
minimum of one each. Base settings control shared CPU resource budgets; model
tile geometry, precision, accumulator parameters and batch cap remain independent.
With one GPU, both model workers reside there; each receives 45% of the inherited
manager memory allowance (or physical VRAM for explicitly unmanaged operation).
Each worker retains the existing 2GiB non-PyTorch reserve; too-small allowances
or oversized batches fail, not silently shrink. This is a supported resource
policy, not a measured throughput optimization. Work stealing between models
is not implemented.

## Ready-to-run comparison

`fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json`
selects non-embedding EMA999 u11800 presence/nx/ny and paired best EMA999 u10800
embeddings, both using their existing namespaces, with EMA-only scoring:
weight0.25, half-life512basevoxels, no anchors. The companion
`eval_mixed_current_ema_only_20260915_v1.json` selects paired-current u42000
embeddings. The requested full50fiber matrix is tracked in
`planning/task_log.md`; the earlier Anchor-LRU prototype remains an archived config.

From the repository root:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json
```

Rebuild the native evaluator once: readiness now includes embedding arrays
explicitly rather than assuming they are siblings of the prediction channels.
Embedding arrays may use a different chunk grid on the same voxel lattice.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Regression command:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
```

Tests include real native sampling/replay from conflicting synthetic model
heads, delayed readiness of one role, channel ownership, incompatible-source
rejection, device partitioning and inherited manager-budget division.
Validated 2026-09-15:102 native cases and215 Python tests passed;8 optional
tests skipped. Python suite72.78s. Workspace RelWithDebInfo build succeeded.

## Current limitations

The single-model `cached_trace_profile` profiling/extension commands explicitly
reject composite evaluations; their containment checks are not weakened.
Resume/replay mixed evaluations through the normal live evaluator with the same
two model configurations and a new evaluation output directory; it reuses both
caches automatically. Direct native rendering/sampling can consume the composite
manifest, but (as with any partial cache) cannot generate missing chunks alone.
Both full mixed EMA-only evaluations completed on all50fibers/180directions:
see [the full matrix](benchmarks/ema_only_matrix_20260915.md).
There is no controlled mixed-versus-single-model throughput comparison.
