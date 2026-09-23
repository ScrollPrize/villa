# Plan

1. Reuse existing independent-reference config, changing only trace input to reverse-v2 and output prefix.
2. Independently review config/plan. Use approved crop benchmark runner; no inference/retracing/build needed.
3. Verify completed unpruned initial BP and independently ranked references; report rank1 accuracy, zero-hard-sign errors and per-reference ladder against7/26baseline. Check initial states match preceding reverse BP run if available.
4. Persist exact config/results in dated benchmark report, task_log/status/changelog.

## Spec update
No spec.md exists. No algorithm changes; initial BP frozen and references independently estimated, with existing global gauge alignment for scoring. Defect-state inactive evidence remains as before.

## Docs updates
Extend reverse consistency report with unassisted results and command, retain removal metric distinction. Record convergence/iteration cap.
