# Plan

1. Stop the active refinement run and verify its cooperative allocation is released.
2. Add a distinct full-resolution config derived from the regular eligible five-scroll JSON-only config. Change only `model_3d.output_stride` from 4 to 1 and `training.run_name`; retain LR 0.01, global batch 128, GroupNorm/BF16, EMA, data, augmentation, tests, and fresh initialization.
3. Add a regression asserting that the full-resolution config differs from the stride-4 baseline only in output stride and run name, and explicitly has neither refinement nor enabled embeddings nor a resume source.
4. Run the focused CPU configuration tests. Use the existing stride-1 model/target coverage as the numerical implementation validation; no unmanaged GPU test.
5. Point the approved production launcher at the new config, launch through the cooperative manager, and verify the emitted config/run identity and initial health.

## Spec update

None. `model_3d.output_stride: 1` and fresh training are already specified.

## Docs updates

Add the full-resolution comparison config and launch identity to the training-span documentation and current task log/status.

## Changelog

No feature changelog entry: this is a configuration experiment using existing behavior.
