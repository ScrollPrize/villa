# Plan

1. Preserve completed correction planning in archive. Freeze one-case manifest/config from mini5v3 with identical snapshot/cache/settings. Keep user companion annotation as a separately hashed auxiliary hypothesis, not ground truth.
2. Use native replay-references for both CP-bounded eligible lines; no extrapolated tails. Reuse benchmark-full for baseline render. Add optional pair diagnostic that freezes companion and records native plan, exact point-to-segment distances and arclength profiles against GT and first failed trace. Dense profiles only on this small case; no volume-wide scan or new solver.
3. Inspect existing render and compare sampled geometry near merge and divergent end. Distinguish geometric corroboration from physical fiber identity; no assertion that CT establishes identity without evidence. Save inputs and report under dedicated benchmark output.
4. Regression-test distance helper, run approved tests/benchmark, validate unchanged one-case replay vs mini5v3. Log result, command and limitations.

## Spec update
Single merge/split benchmark stores primary GT, auxiliary candidate (not an independent GT label), frozen hashes and native eligibility, baseline model/cache provenance.

## Docs updates
New benchmark report with paths, geometric evidence, reproduction command and future many-to-many assignment motivation. Changelog entry; update active log/status.
