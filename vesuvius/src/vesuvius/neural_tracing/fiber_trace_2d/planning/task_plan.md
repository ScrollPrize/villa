# Plan

1. Extend fiber target materialization with an opt-in swept rectangle in the
   local tangent/surface-normal/in-sheet frame. Use half-width 64 and
   half-height 16 base voxels, excluding the existing nonzero Gaussian support.
   Track left and right masks separately.
2. Add left/right sideband BCE-to-zero sums and counts to the existing
   target-only, globally synchronized loss-statistics protocol. Normalize each
   side independently and add their mean at weight 0.1 relative to the presence
   ridge term. Preserve one forward/backward pass and all existing losses.
3. Reuse the shared cross-section sampler during configured tests. Accumulate
   raw-model predictions over the deterministic complete test set on every
   rank, merge bounded sum/count/histogram state, and log/write float mean and
   `q0.1` exceedance images. Do not alter crops, targets, refinement, metrics,
   or checkpoint selection.
4. Add focused CPU tests for swept-sideband geometry, exclusion of Gaussian
   support, independent left/right statistics, disabled behavior, distributed
   merge semantics, and image generation. Run the relevant loss, target,
   refinement, and cross-section suites plus static validation.
5. Add a derived resume config enabling the sideband loss and periodic
   cross-section visualization. After validation, stop any surviving prior
   run cleanly and launch a managed resume from its latest checkpoint.

## Spec update

Document the optional local-sideband target geometry, independent global
normalization, and test-only cross-section visualization settings in
`planning/specs.md`. No GPU-manager protocol change.

## Docs updates

Record the settings, validation, source checkpoint, replacement run, artifact
locations, and any measured test overhead in the task log and changelog.

## Plan review

The repository process requests an independent agent review. The active
orchestration policy forbids spawning sub-agents unless the user explicitly
requests delegation, so this plan received an explicit self-review against
`specs.md`, `plan.md`, and the task before implementation; this is the only
process deviation.
