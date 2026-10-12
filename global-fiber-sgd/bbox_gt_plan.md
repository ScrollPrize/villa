# Four reviewed fibers: standalone solve

User clarification supersedes splitting/rejection: FOUR INTACT TRUSTED fibers.
No splitting, no broken category available in UV. Existing preprocessing retained
all4; assert that instead of silently reviving any discarded evidence. Old split
trial stopped; new output uv_whole, no reuse of its tensor cache or state.

Use the four JSON curves in data/bbox_gt, unchanged and without invented links,
as an isolated dataset. Convert through the existing AFV writer, retaining exact
native XYZ samples and a filename/ID manifest. Reuse native local-event Lasagna
extraction, normal orientation, normal-sign, H/V and direction stages, then the
current joint UV baseline weights. Do not mix these fibers into the AFV corpus.

Use the approved crop-benchmark entrypoint with a small orchestration adapter.
No duplicated extraction, tensor preparation or optimization implementation.
Single GPU is sufficient for four fibers; retain the baseline state/width LRs
and 10080 updates, but batch all available constraints rather than duplicating
them to fill a 24576 batch. Report this small-dataset difference explicitly.
No spiral-reference initialization or supervision. Report constraints per pair,
stage exclusions, final winding/phase, broken probabilities and violations.
If extraction is disconnected/empty, report that rather than inventing evidence.

## Spec update
Add JSON-curve dataset orchestration only. All existing solver semantics remain.
Automatic map inference is not manual adjacency GT. Absolute winding origin is
not established by these four fibers alone.

## Validation / docs
Validate finite XYZ, scan shape, no hidden links, AFV roundtrip and ID mapping.
Run adapter tests and existing solver regressions. Record exact command, timings,
parameters and results in bbox_gt_results.md and task_log.md. Preserve baseline.
