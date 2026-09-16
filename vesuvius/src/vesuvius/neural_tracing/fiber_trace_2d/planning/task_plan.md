# Plan

1. Stop the active full-resolution run and identify its latest durable `current.pt` checkpoint.
2. Add a distinct full-resolution refinement config derived from the existing reviewed stride-4 refinement config. Change only output stride 4->1, resume source to the stopped full-resolution checkpoint, and run name. This preserves normal manifests, bounded accumulation, refinement normal/tangent/direction weights 0.1/10/0.1, LR0.01, global batch128, EMA and all data/augmentation settings.
3. Add a regression asserting that the new config differs from the existing refinement config only in those three fields, that its architecture matches the stopped stride-1 checkpoint config, and that refinement is enabled while embeddings remain disabled.
4. Run the approved CPU training-span and refinement suites; no unmanaged GPU test.
5. Point the approved production launcher at the continuation config, commit the reproducible state, launch through the cooperative manager, and verify checkpoint step, fresh refinement geometry, calibration/allocation and resumed progress.

## Spec update

None. Resume and full-resolution refinement are already specified behaviors.

## Docs updates

Record the full-resolution refinement continuation and launch identity in refinement documentation and current task log/status.

## Changelog

No feature changelog entry: this is a configuration continuation using existing behavior.
