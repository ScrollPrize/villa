# 3D Resume Optimizer Config Override

Change `vesuvius.neural_tracing.fiber_trace_3d.train` resume behavior so a run
can resume from a checkpoint while still honoring optimizer settings from the
current JSON config.

- Keep restoring model weights from the checkpoint.
- Keep restoring optimizer state as much as possible, including AdamW moment
  buffers and step counters.
- After restoring optimizer state, reapply current config optimizer
  hyperparameters supported by the trainer.
- The known supported optimizer config values are:
  - `training.learning_rate`
  - `training.weight_decay`
- Log or print the effective optimizer hyperparameters so resume behavior is
  visible.
- Add a regression test that resumed optimizer state is preserved while current
  config LR/weight decay win over checkpoint param-group values.
