# Plan

1. Commit the completed merge/split/fiberlet diagnostics before the new runs.
2. Freeze and identify the `062136` best/current EMA999 snapshots.
3. Run both with the identical original-GT 50-fiber/180-direction protocol,
   embeddings and rendering disabled, all eight GPUs, and fresh model caches.
4. Validate global and per-scroll metrics and compare against the previous
   refinement leader.
5. Select the actual tracing winner as the refinement default. Point the
   canonical training config at the source checkpoint using explicit
   weights-only initialization semantics so its sibling geometry archive remains
   available without pretending an EMA artifact is an optimizer resume; record
   an immutable machine-readable manifest with both training and frozen-inference
   paths and hashes.
6. Validate configs, snapshot/archive hashes and initialization metadata; test
   weights/state initialization and the resume/initialize exclusion, document,
   and commit the evaluation plus default selection.

## Spec update

None. The evaluator and refinement algorithm do not change. The canonical
refinement starting weights change to the current tracing-selected winner.

## Docs updates

Add the `062136` benchmark report, current-default model documentation, and a
changelog entry. Replace the active status/log with this task's provenance.

## Testing

Parse all JSON, verify checkpoint and refinement archive hashes, load checkpoint
metadata, validate the configured initialization/state relationship, run the
focused refinement and checkpoint-source tests, and rely on both completed
180-direction runner summaries for evaluation validation.
