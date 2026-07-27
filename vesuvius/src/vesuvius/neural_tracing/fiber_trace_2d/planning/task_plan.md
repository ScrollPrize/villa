# 128 sd2 Single-Output 3D Fiber Config Plan

## Scope

Targets:

- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all_128_sd2.json`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md`
- planning/status/task-log files for this current task

## Plan

1. Confirm that `conditioned_decoder_enabled: false` still builds a legacy
   grouped-output model and routes training through `compute_losses(...)`.
2. Update the 128 sd2 config to emit one seven-channel prediction:
   `direction_branch_count: 1`, `output_channels: 7`, and
   `conditioned_decoder_enabled: false`.
3. Remove conditioned-only training query weights from that config because they
   are ignored outside conditioned mode.
4. Keep BF16 mixed precision, BatchNorm, dataset, augmentation, and optimizer
   settings unchanged.
5. Change the 128 sd2 run name away from the conditioned run.
6. Update specs/docs that describe which sd2 config uses conditioned training.

## Spec Update

- Change the S1A sd2 config note so only the 64 sd2 config is still described
  as conditioned, while the 128 sd2 config is documented as single-output
  legacy/single-branch supervision.

## Docs Updates

- No separate `docs/` update is needed because `docs/code_structure.md`
  already documents both model modes and does not list this specific config as
  conditioned.

## Tests

- Run `python -m py_compile` on the 3D model/train files.
- Run a config smoke check that builds the 128 sd2 model and verifies:
  conditioned mode is false, output channels are seven, branch count is one,
  and `_forward_loss_tensors(...)` can execute the non-conditioned loss path on
  a tiny synthetic batch.

## Changelog Update

- Add a one-line changelog note that the 128 sd2 config was switched back to
  single-output legacy supervision for comparison against the conditioned
  experiment.
