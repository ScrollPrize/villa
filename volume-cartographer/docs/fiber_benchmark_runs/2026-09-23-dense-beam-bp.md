# VC3D dense beam candidates → BP (2026-09-23)

## Protocol

Classic `d74cca79` and new `7ff0ce6c` prediction manifests, same normals
`5b17ff6c` and Paris4 crop `[10240,22016,6144] .. [11264,23040,7168)`.
References are the 26 `hendrik_crop1` JSONs in
`/home/hendrik/test_datasets/2026-09-01_fiber_stack2`.

Candidate generation uses canonical prediction-derived anchors, the existing
crop-wide coverage/scheduling, and production VC3D beam tracing in both directions:
beam8/lookahead2/step4/cone25, no embeddings and no full-resolution tuning.
Ground-truth fibers are not inputs to candidate generation. In particular, the
previous reference-replay traces (which reset onto GT after failures) are not
used as BP candidates.

The artifact stores production beam loss, not Fiberlet graph loss. The graph-only
ambiguity margin0.40 and quality cutoff0.21 therefore do not apply to these beam
traces. All generated candidate traces enter BP. This difference from the earlier
tuned Fiberlet graph preprocessing is explicit; it is not a pure backend ablation
with identical quality filtering.

BP retains the historical piece length512, dominance0.9, fixed winding orientation,
sum-product-mixed, oracle-inliers/reference-assisted setup with one diagnostic
round. The two requested metrics are the existing plot metrics:

- Constraint score: `-100 * problematic_unique_constraints / retained_fulfilled_unique_constraints`.
  Higher (closer to zero) is better.
- Reference score: `100 * exact / (exact + wrong + missing)` at oracle round zero,
with all26 tagged references in the denominator. Higher is better.

Oracle round zero follows the historical conditioned-inlier closure (which can
already remove pieces); it does not mean no pruning has happened.

These are reference-assisted, reference-tuned crop diagnostics, not held-out BP
accuracy. They are distinct from the replay score `100/(failures+1)`.

## Results

Historical comparison and preceding crop results:
[previous report](2026-09-23-crop-new-predictions.md).

| Prediction / candidate pipeline | Constraint score ↑ | Exact references ↑ |
|---|---:|---:|
| Classic / Fiberlet q0.21 | -84.42% | 84.62% |
| New / Fiberlet q0.21 | -91.79% | 88.46% |
| Classic / VC3D beam, unfiltered | -240.90% | 88.46% |
| New / VC3D beam, unfiltered | -178.90% | 88.46% |

Classic beam raw counts: 997558 problematic / 414091 retained fulfilled
constraints; 23 exact, 3 wrong, 0 missing references.
New beam raw counts: 27187 problematic / 15197 retained fulfilled constraints;
23 exact, 3 wrong, 0 missing references. New Fiberlet: 5543/6039 constraints;
23 exact, 2 wrong, 1 missing. Classic Fiberlet: 14598/17293 constraints;
22 exact, 3 wrong, 1 missing. Percentages are calculated from integer counts,
not the rounded native display.

Candidate generation: Classic 2646 traces in 13.40 s; new 485 in 5.93 s.
New BP took 45.73 s. Classic requires 16,823,829 initial joint-grid states,
exceeding the default 4M memory guard. Its retry uses a 32M safety limit;
this changes neither candidates nor objective. The failed attempt is archived,
and the retry reuses the saved traces rather than regenerating them.

Classic BP completed successfully in 1145.24 s (19m05s), about 25× the new
runtime, with peak RSS 5,353,188 KiB. It had 6,833 pieces and 1,411,649
constraints versus new 1,204 pieces and 42,384 constraints. The initial Classic
solve hit the unchanged 500-message iteration limit; its reported conditioned
oracle round zero converged. The new fresh solve also reached its message
limit, with conditioned oracle round zero converged. These are the historical
accepted-message-limit protocol, not claims that all solves converged.

Conclusion: superior reference-replay tracing does not imply superior BP
candidate graphs. Both beam runs reach 88.46% exact references, but both have
worse constraint scores than the quality-filtered Fiberlet pipelines. Candidate
redundancy/filtering merits separate study; no new filtering was tuned here.

Updated [constraint plot](../fiber_benchmarks/imgs/fiber_crop_pruning_progress.svg)
and [reference plot](../fiber_benchmarks/imgs/fiber_crop_reference_accuracy.svg).

## Reproduction

Use the existing approved native build runner and evaluation runner's
`crop-benchmark --config ...` mode. Commands, source patches, input identities,
binary/shared-library hashes and wall/user/system/RSS are captured in
`/ephemeral/hendrik/fiber_crop_bench/20260923_v1/`.

No neural inference or GPU allocation is required; predictions already exist.

Configuration files under
`vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/`:

- `crop_benchmark_beam_bp_classic_20260923.json`: full Classic pipeline.
- `crop_benchmark_beam_bp_classic_retry_20260923.json`: BP-only reuse.
- `crop_benchmark_beam_bp_new_20260923.json`: full new pipeline.
- `crop_benchmark_beam_bp_publish_20260923.json`: regenerate historical BP plots.
- `crop_benchmark_dense_tests_20260923.json`: native regressions.

Example invocation (substitute the other config basenames above):

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_beam_bp_new_20260923.json
```

Validation: native crop tracing 82 tests (including trace2/prediction8 scale
regression), production tracing 86, replay 23 passed; Python runner/parser 5
passed. Independent native review found no blocking issue. Only one timed run
per pipeline: these timings are workload context, not a throughput study.
Build type: RelWithDebInfo, 128 requested CPU workers. Default solver iteration
limits remain unchanged.
