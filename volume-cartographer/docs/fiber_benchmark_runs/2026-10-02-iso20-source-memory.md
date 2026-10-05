# Coarse/fine source-memory BP

Paris4 1024³ crop, baseXYZ half-open [10240,22016,6144]–[11264,23040,7168]; hendrik_crop1 references from /home/hendrik/test_datasets/2026-09-01_fiber_stack2/. Frozen source: /ephemeral/hendrik/fiber_crop_bench/20261002_strict_reverse_crop_v1/beam-isotropic20.zarr. Raw40500 model SHA256 9d3bb1c0cd5b0ee58d16d9b2bdcec931faec675421e6e2c707f7fd10ce059f7e.

Frozen raw40500 iso20, fine orthogonal-event minimum64; RelWithDebInfo,128 CPU workers, 500 iterations. No inference/retracing, no interval repair. All rows inherit100% replay (0 ordinary failures,6 filtered stops,93.7463% retained). Accuracy/26, shift observed-only; ties use deterministic rank1. Local-event constraint-removal and oracle metrics remain unavailable due the previously documented evaluator calibration issue.

| Method | Replay % | Constraint score % | Oracle % | Non-oracle % | Shift % (N) | Disabled length % | Gaps | Boundaries | Memory-projected pieces | Seconds |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Historical overlap512/128 | 100.00% | −25.24% | 80.77% | 50.00% | 83.33% (24) | — | — | — | — | — |
| Prior fine64 tuned | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 16.054% | 561 | 61 | 0 | — |
| chain-break2000-open0 | 100.00% | Unavailable | Unavailable | 23.08% | 30.43% (23) | 21.230% | 624 | 61 | 0 | 35.93 |
| chain-break2000-open1000 | 100.00% | Unavailable | Unavailable | 15.38% | 17.39% (23) | 27.108% | 674 | 78 | 0 | 35.08 |
| chain-break500-open0 | 100.00% | Unavailable | Unavailable | 23.08% | 30.43% (23) | 21.243% | 624 | 61 | 0 | 36.26 |
| chain-break500-open1000 | 100.00% | Unavailable | Unavailable | 15.38% | 30.43% (23) | 25.021% | 654 | 82 | 0 | 36.54 |
| control | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 16.054% | 561 | 61 | 0 | 32.67 |
| overlap512-128 | 100.00% | Unavailable | Unavailable | 23.08% | 21.74% (23) | 20.993% | 621 | 61 | 0 | 36.61 |
| window512 | 100.00% | Unavailable | Unavailable | 26.92% | 30.43% (23) | 20.940% | 620 | 59 | 0 | 36.61 |

## Formulation and limitations

- Each fine piece owns its evidence and physical-length disabling cost once. Windows repeat only prior coupling.
- Window membership uses piece arc midpoint in half-open windows, anchored at each contiguous compatible source-run start; window512, stride512 or384. Long fine pieces are not forcibly split.
- All retained labels within a hard window must agree even across disabled pieces. Overlapping windows can transitively couple longer spans when their shared pieces survive.
- Chain DP remembers the last active class/winding through disabled spans. A different next retained label pays the break cost; each disabled run pays opening once. No latent-label multiplicity, including all-disabled chains.
- Original hard adjacent active/active continuity remains: a winding break requires a disabled gap, not two differently labeled adjacent retained pieces.
- Chains split at geometry gaps and different effective integer gauges; prepass-Defect nodes do not introduce gauge links. No new measurements, prepass changes or gauge merges.
- Constant-window decoding monotonically disables conflicting labels, preserving gauges/fixed states. These additional projection defects are reported, not hidden as successful joint optimization.
- Exact conditional sequence messages do not make the overall loopy BP globally exact. The old overlapping extraction has a different prepass/aggregation; these are not claimed to be identical formulations.
- Objective includes memory factors. Reported active_components in JSON describes ordinary retained factors, not additional memory connectivity.
- Pilot implementation uses O(N*K²) native sequence messages, parallel across factors; effective overlapping-coupling span distributions are not yet instrumented.
- One run per setting; wall time is observational, not an isolated speed benchmark. Full ladders, missing/tied refs and provenance in adjacent JSON.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_source_memory_v1`. Approved runner action `iso20_source_memory`.

## Result

No improvement. The rebuilt control exactly reproduces the prior fine-piece states. Every memory variant loses reference8 in addition to already missing4/12, increases disabled length, and worsens both benchmark scores. Extra source-memory feasibility projection disabled0pieces in every run, so the regression is not caused by that new final projection. All prepass assignments remain identical. Opening1000 worsens the chains further. Stronger correlation alone does not recover the old overlapping model behavior on this fine-piece graph. This does not prove all hierarchical models fail: coarse extraction/prepass is different and was not replaced in this trial. No production defaults promoted; these remain opt-in experiments.

All six new variants took35.1–36.6seconds end-to-end versus32.7seconds for control (one run/configuration). No larger iteration budgets or rejected repair were used.
