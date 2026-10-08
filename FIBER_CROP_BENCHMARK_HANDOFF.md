# PHercParis4 fiber crop benchmark handoff

## Scope

- Crop, half-open base-volume XYZ: `[10240,22016,6144) .. [11264,23040,7168)` (1024 cubed).
- References: `/home/hendrik/test_datasets/2026-09-01_fiber_stack2/`.
- All 26 JSON files carry tag `hendrik_crop1`; the established benchmark selects 24 in-crop fibers / 48 directed cases and about 101.036 mm total directed length.
- Current checkout containing the evaluation implementation: `/home/hendrik/villa`, branch `fiber-exps2`. Commit `b660837d9` (`origin/fiberlets-development-integration`) is in its history. This `villa2` checkout is currently branch `fiber-exps` at `fca856861`.

## Local inputs

```text
new Fiber prediction:
/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-las-sd1-7ff0ce6c/artifacts/PHercParis4-20260411134726-las-sd1-7ff0ce6c.lasagna.json

classic Fiber prediction:
/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-las-sd1-d74cca79/artifacts/PHercParis4-20260411134726-las-sd1-d74cca79.lasagna.json

Lasagna normals:
/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-las-sd2-5b17ff6c/artifacts/PHercParis4-20260411134726-las-sd2-5b17ff6c.lasagna.json

existing combined Fiberlets (classic model only):
/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-fiberlets-e7aa1a99/artifacts/fiberlets.zarr
```

The `7ff0ce6c` prediction is the current best-so-far Fiber prediction and is the
primary benchmark target. It was inferred from:

```text
training run: 5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757
checkpoint: best_ema_0p999.pt, step 22000
checkpoint SHA-256: 67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742
Hugging Face model: scrollprize/lasagna-fiber
```

Do not compare the new `7ff0ce6c` direct result against the existing combined Fiberlets as if they used the same prediction: `e7aa1a99` was generated from the older `d74cca79`/classic snapshot.

## Build

`vc_fiber_trace_chunk` is not currently present in the `villa2` build tree. Configure/build the source revision being evaluated and record the revision/build type. The target is:

```bash
cmake --build volume-cartographer/build --target vc_fiber_trace_chunk -j 8
```

## Direct dense-prediction benchmark

This path is ready once the binary is built. It is the historical greedy tracer (beam width 1, lookahead 1), evaluated with the shared reference-replay metric.

```bash
BIN=volume-cartographer/build/bin/vc_fiber_trace_chunk
REF=/home/hendrik/test_datasets/2026-09-01_fiber_stack2
FIBER=/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-las-sd1-7ff0ce6c/artifacts/PHercParis4-20260411134726-las-sd1-7ff0ce6c.lasagna.json
NORMAL=/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-las-sd2-5b17ff6c/artifacts/PHercParis4-20260411134726-las-sd2-5b17ff6c.lasagna.json
OUT=/ephemeral/hendrik/fiber_crop_bench/direct-7ff0ce6c.json
mkdir -p "$(dirname "$OUT")"
"$BIN" reference-replay-benchmark - --replay-tracer greedy --fiber-manifest "$FIBER" --normal-manifest "$NORMAL" --bbox 10240 22016 6144 11264 23040 7168 --reference-fiber-dir "$REF" --reference-fiber-tag hendrik_crop1 --base-voxel-size-um 2.4 --threads 128 --output "$OUT"
```

The historical classic run completed in about 0.49 seconds on a 32-thread host, with 13 failures and 7.217 mm mean segment length. That is only a sanity reference, not the expected new-model result.

## Fiberlet benchmark

The existing classic combined Fiberlets can be benchmarked immediately:

```bash
BIN=volume-cartographer/build/bin/vc_fiber_trace_chunk
REF=/home/hendrik/test_datasets/2026-09-01_fiber_stack2
FIBERLETS=/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-fiberlets-e7aa1a99/artifacts/fiberlets.zarr
NORMAL=/ephemeral/hendrik/las_manager_inference/PHercParis4-20260411134726-las-sd2-5b17ff6c/artifacts/PHercParis4-20260411134726-las-sd2-5b17ff6c.lasagna.json
OUT=/ephemeral/hendrik/fiber_crop_bench/fiberlet-classic.json
mkdir -p "$(dirname "$OUT")"
"$BIN" reference-replay-benchmark "$FIBERLETS" --normal-manifest "$NORMAL" --bbox 10240 22016 6144 11264 23040 7168 --reference-fiber-dir "$REF" --reference-fiber-tag hendrik_crop1 --base-voxel-size-um 2.4 --threads 128 --output "$OUT"
```

For a like-for-like `7ff0ce6c` comparison, generate Fiberlets from that prediction first. The core already provides `FiberletOnDemandPreprocessor`, persistent anchor/Fiberlet Zarr caches, dependency generation and reference-based prefetching. `vc_fiberlets fiberlet-replay` uses it. The remaining integration is to let `vc_fiber_trace_chunk` create/read that on-demand cache for the crop benchmark. Cache identity must depend on prediction, normals, algorithm version and Fiberlet parameters, not the bbox, so overlapping crops reuse chunks.

## Authoritative documentation

- `volume-cartographer/docs/fiber_chunk_tracing.md`
- `volume-cartographer/docs/fiber_benchmark_runs/2026-09-03-6c006d9b0-greedy-reference-replay.md`
- `volume-cartographer/docs/fiber_benchmark_runs/2026-09-03-3046918b5-staged-reference-replay.md`

Always preserve the exact source revision, build type, commands, input identities, wall/user/system time, peak RSS, total directed length, failure count and mean segment length in the result record.
