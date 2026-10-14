# Bending/smoothness sweep — 2026-10-14

Task: test whether increased existing regularization repairs local folded quads
in both joint-sliding surface fits, before introducing any repair mechanism.

1. Reuse original/deformed inputs, identical frozen retained H mask, initialization,
   grid100,600steps, meshLR5/UVLR.1 and attachment1→20 ramp. Baseline bend5,
   smooth.5, step.5. Trial bend15 and50, smooth unchanged, on both layouts.
2. If folds remain, test smooth1.5 and5 with the better bending setting for each
   input (fewer folded quads; inspect fit error). No automatic default promotion.
3. Use existing approved runner, shared8GiB GPU grants on0–5 only. Fresh outputs;
   do not overwrite prior results or touch upstream inputs.
4. Compare XYZ mean/p95/max, normal RMS, local opposing-triangle and degenerate
   counts, shift statistics and timings. Audit folds in target-containing cells
   versus cells without attachments; this is not a global intersection audit.
5. Validate unchanged input hashes/selection and report every candidate, not only
   winners. Store exact configs/results and commands.

## Spec update

No mathematical/implementation change to solver or sampling. Only explicit
experimental regularization weights vary. No new collision/fold penalty yet.

## Docs, tests and changelog

Independent plan review; rerun existing regression tests if code changes. Save
smoothing_results.md and update task log/status/changelog. Existing baselines
remain versioned and unchanged. This is quality testing, not isolated speed profiling.
