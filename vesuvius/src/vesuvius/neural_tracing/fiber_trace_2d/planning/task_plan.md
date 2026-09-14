# Plan

1. Add training-local geometry filtering shared by file-backed train/test 3D
   loaders. Keep maximal contiguous cspline/trace runs between exact CP line
   indices; split at rejected spans, never bridge gaps; trim exterior tails.
2. Preserve NML node/edge supervision (synthetic legacy metadata is not real
   provenance). Missing/legacy JSON modes are ineligible. Report retained runs
   and skipped files; fail clearly if a dataset has no eligible data.
3. Keep scroll-balanced arclength sampling over retained lengths and existing
   optimized queued loading/targets/manager. Independent runs cannot bridge gaps.
4. New JSON-only five-scroll config from regular EMA comparison: LR0.01,
   2M updates,batch128,no embeddings,normal test/snapshot intervals. Remove
   historical auxiliary dataset, whose provenance is unqualified.
5. CPU tests: geometry, gap-free deterministic sampling, train/test loader
   integration, actual corpus coverage. No GPU kernels or separate evaluator
   changes. Independent plan/implementation review.

## Spec update
Document training-only eligibility, exact CP trimming, gap splitting and NML
exception; no length cutoff. Changed data identity invalidates old comparisons.

## Docs updates / changelog
Document new config/start command, changed test population, corpus statistics,
tests and limitations. Update status/task_log and concise changelog entry.
