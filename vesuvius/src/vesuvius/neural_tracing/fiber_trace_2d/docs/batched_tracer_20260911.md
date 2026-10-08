# Canonical batched tracer and readiness — 2026-09-11

## Result

Candidate evaluation now always takes the existing efficient batch route, independent of worker count.
The dedicated single-worker scalar candidate loop and candidateLoss implementation are removed.
Fused corners are used when prediction and normal grids match; otherwise the same path uses separate prediction/normal batches.
This preserves generic prediction sources and their concurrency capability limits.

Final cache-only test: **0.981679 / 1.034555s native wall**,
mean/median **1.008117s** for four directions.
Coordinator walls 1.081664 / 1.130754s.
One candidate worker per fiber, no adaptive worker scheduling and no new inference.
Two repetitions, GCC13 RelWithDebInfo, existing beam8/lookahead2/step4 and201.54GiB shared decoded cache.
Same classic checkpoint cached outputs and original annotation files.

| Direction | Previous mean case time | Batched mean case time | Previous candidates | Batched candidates |
|---|---:|---:|---:|---:|
| PHerc0139 forward | 30.716s | 0.599s | 660393 | 660393 |
| PHerc0139 reverse | 29.903s | 0.596s | 643059 | 643059 |
| Paris4 forward | 46.541s | 0.949s | 1043118 | 1039878 |
| Paris4 reverse | 51.859s | 1.007s | 1168182 | 1168182 |

**Comparison caveat:** previous measurements ran eight directions concurrently; the completed new pilot runs four.
These are matched per-direction timings, not a controlled identical-cohort wall-speedup ratio.
The original eight-case attempt failed closed because changed PHerc0332 trajectories required uncached chunk[81,25,42].
No inference was launched to fill it. Two shortest whole fibers (0139 and Paris4) were retained for both final repeats.
No whole100-case or adaptive-multicore claim is made.

Both final repeats are exactly identical to each other. Serialized replays differ from the old scalar path, as explicitly authorized.
Failure counts per direction remain0/2/5/13 (total20), unchanged from those old cases.
Candidate counts are unchanged in three cases and slightly lower in Paris4 forward; the gain is not from eliminating most search.
The prior intermediate batched v2 has the same serialized replay hash as final v3; the final readiness-only cleanup changes no trace results.

## Readiness fixes

- Precompute immutable exact-lattice groups and spacing1 bindings once when the gate is configured.
- Plan shared channel lattices once per batch, deduplicate integer keys, then expand channel paths.
- Reuse the existing valid cube-key helper: a single-chunk request has only one initialized key. The old append routine emitted unused slots and fictitious chunk0 dependencies.
- Expose the existing exact corner-grid predicate through sharesSamplingGrid and reuse it in validation and fused eligibility. Incompatible fused attempts now do no readiness work; fallback batch gates once.
- Gate before internal workers/read access; retain successful acknowledgement, cancellation and error propagation. Call-local scratch remains safe for concurrent public reads.
- Preserve both float and double public sampling footprints. No file-existence assumption replaces acknowledgement.

For these four directions, readiness callbacks fall from3514776 in the old counterpart profiles to2200.
Final profiled candidate points3511512; readiness planned points3511512 (previous intermediate batch version7023024).
Final planned path/chunk dependencies8034, bridge entries89, submitted missing dependencies303 per pass.
Readiness planning totals0.360886 / 0.360371s summed across cases.
Planning counters cover profiled candidate batches, not scalar seed samples.
Timers are inclusive: prediction-batch time contains readiness. Never sum nested metrics.

The completed real-data pilot uses separate prediction/normal batches (different grids); it does not use fused corner scoring.
The compatible-grid fused route, including threads1, is covered by regression tests.

## Replay versus GUI reoptimization

Both execute traceOneWayCore and the same beam candidate/scoring machinery.
Replay adds incremental reference matching, an ellipsoid failure check and reference restart handling.
GUI control-point spans instead use target-plane stopping and forward/reverse segment fusion; some GUI spans may be reused or use other interpolation modes.
Therefore the amount and stopping of search can differ, but reference checks are not a separate expensive tracing algorithm.
Before this change99.74–99.80% of the short trace's measured case time was inside candidate evaluation; reference checks were outside that hot stage.

## Validation and commands

318 native cases plus77 Python tests passed (34.01s Python).
Coverage: threads1 and2 generic batches; real fused-sampler threads1/4 exact replay equality;
mismatched-grid delayed publication/live-offline equality; one readiness planning pass;
float boundary dependencies; no phantom chunk0 and1000 duplicate points; cache cancellation/errors.

```bash
cmake --build /tmp/vc-live-native-cmake --target vc_fiber_trace_metric test_fiber_trace3d test_fiber_anchors test_fiber_replay test_fiber_trace_review test_fiber_reference_replay_benchmark test_lasagna_normal_sampler test_chunked_plane_sampler test_chunk_cache test_global_cache_root -j 8
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_cached_batched_profile.json
```

Approved runner exits0 under exclusive manager taskf18f2d31-2ad5-4eb0-ae33-19e29432f965.
Evidence: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_cached_batched_20260911_v3`.
Use a fresh profile.output_root on rerun; do not overwrite evidence.
Historical scalar baseline: classic_cached_readiness_20260911_v3.
Explicit profile.allow_original_replay_difference permits the authorized canonical-path change;
default strict historical validation remains unchanged, as does exact within-run-matrix repetition validation.
Every inference_summary.json verifies worker_pool=null and zero inferred/published chunks.

Independent plan and final implementation reviews approved.
Linux amd64 only tested; no macOS/arm64 runtime validation. No installs, no compiler fast-math changes.
Adaptive per-fiber worker allocation is not implemented in this task.
