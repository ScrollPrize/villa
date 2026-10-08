# Plan

1. Lock first two failing manifest paths from0139, first from1299, first two Paris4; record original trace paths. Preserve directed pieces/hash/length, model and evaluation protocol.
2. Existing benchmark-full runner, subset manifest, fiberlet backend,128-base chunks,224 workers, four iterations/gamma1/q2/local sigma1.5/damping0.5. Reuse inference; separate construction identity. No center-only or algorithm changes.
3. Independent plan review, then approved runner with demand-only GPUs. Finish and report actual wall time;2–3minutes is not guaranteed.
4. Compare per-piece restarts/coverage and pooled mean/median/P05/P95 with existing metrics. Historical baseline is not matched speed measurement.

## Spec update
No production/default changes; locked downstream pilot only.

## Docs and changelog
Archive previous planning; persist subset/config, comparison/commands in benchmark docs and update log/status/changelog. Failure-selected pilot is not representative overall quality.

## Validation
Runner verifies annotation/model/normal hashes, case count and length. Check resolved weighted settings and distinct construction identity. Prior native/Python suites passed; no executable changes.
