# Plan

1. Merge retrain commit, retaining sample-based console/TB/checkpoint progress and cooperative handoff plus embedding supervision.
2. Keep both durable changelog/spec additions. Replace active task documents for this integration.
3. Derive a separate embedding config from train_s1a_nml_8gpu_fast.json; change only run name, embedding settings and matching train/test normals paths. Verify local metadata and data globs.
4. Run CPU embedding, sample-progress, accumulation and cooperative regression tests. Add sample-progress checkpoint coverage for shared snapshot helpers.

## Spec update
Preserve both merged contracts; cumulative sample counts also survive RAM handoff. No loss, scheduling or precision changes.

## Docs updates
Document exact embedding config/command and LR caveat. Keep retraining and embedding docs.

## Changelog
Record integration and tested launch config. No GPU benchmark or training run in this task.
