# Task: Restore Exhaustive Deterministic Random Sample Order

Recover the intended training and prefetch sample order:

- Training and prefetch must use deterministic pseudo-random CP ordering, not
  flat sequential ordering.
- The deterministic random order must go through all configured CPs before
  repeating, so a full prefetch can cover the same dataset order training will
  consume.
- Changing `training.max_steps` must only truncate or extend the consumed
  prefix; it must not change the earlier sample order.
- Explicit `--prefetch-steps N` must override configured training step counts.
- Explicit `--prefetch-steps 0` must prefetch every configured training/test CP
  once in the same deterministic random order.
- `training.max_steps = 0` must mean indefinite training over repeated
  deterministic random full-dataset passes.
- Invalid CP-local samples must be skipped consistently in training and
  prefetch without hiding fatal infrastructure/programming errors.
