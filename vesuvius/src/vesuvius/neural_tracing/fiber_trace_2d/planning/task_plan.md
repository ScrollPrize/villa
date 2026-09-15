# Merge plan

1. Commit current embedding/evaluation implementation and docs (de6fc71e8).
2. Merge local fiber-training-refinement at c3e0d21a8. Only active planning/changelog conflicts: archive both task histories, use this merge task as active, concatenate independent changelog entries. Preserve automatically merged specs and all source features.
3. Independent review of merge plan and feature preservation. Run existing native replay/evaluation suite and incoming refinement suite through reusable runners; inspect semantic overlap in train/loader/evaluation/shared formats. No fresh training or full eval.
4. Record tests/resolutions; commit merge and verify parents/clean worktree.

## Spec update

None beyond combining existing specifications from both branches; no new behavior requested.

## Docs updates

Preserve both original planning sets under planning/merge_refinement_20260915; merge changelog and record validation here. Existing refinement and embedding docs retained.
