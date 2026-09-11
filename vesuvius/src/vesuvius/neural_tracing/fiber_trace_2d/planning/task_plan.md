# Plan

1. Add dataset dense_supervision boolean (JSON default false, NML stays dense). Add loader sampling_strategy=scroll_balanced_arclength (default control_points) and dataset scroll_id. Require explicit IDs, dense labels and positive finite fiber lengths for balanced mode; combine entries sharing an ID.
2. Precompute segment and per-scroll fiber length CDFs. Seed draws from global sample index independent of worker/rank. Use a typed continuous location for crop, tangent, targets and prefetch, not a fabricated CP. Preserve flat CP traversal for trace2cp and legacy defaults.
3. Opt-in dense JSON supervises the selected complete polyline, including re-entry, via existing segment clipping/target materialization. NML behavior stays unchanged.
4. Enable all train/test collection entries, grouping both Paris4 folders, and correct PHerc1229 folder identity to PHerc1299. Keep fixed test seed/index range and existing sample budget (not an exhaustive traversal under this strategy).
5. CPU tests: distribution, continuous interpolation, repeated vertices, invalid data, reordered deterministic requests, dense selected-only supervision at stride1/4, re-entry, legacy behavior, test config inheritance and prefetch alignment. Use approved configurable test launcher.
6. Focused GPU integration: acquire exclusive non-pausable/non-adaptable manager allocation through the same launcher. Compare CPU/CUDA dense targets and run the collection U-Net at stride 4 on a continuous sampled patch, BF16 and FP32, forward/backward/update and evaluation with finite gradients/loss. Single GPU, batch 1, capped at 4 GiB allocator memory. No checkpoint or production-run edits; manager handles pause/resume.

## Spec update

Update planning/specs.md for dense JSON opt-in, continuous scroll-balanced strategy and deterministic evaluation; preserve compatibility.

## Docs updates

Document config and changed scientific meaning of collection metrics in docs. Update status/task_log/changelog. No checkpoint edits.
