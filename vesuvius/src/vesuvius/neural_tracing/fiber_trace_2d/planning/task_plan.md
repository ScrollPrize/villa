# Task Plan: Restore Exhaustive Deterministic Random Sample Order

## Scope

Replace the recent flat sequential training/prefetch order with deterministic
random-without-replacement ordering. The order must cover every configured CP
once per pass before repeating, and prefetch must use the same order as
training.

## Plan

- Replace training's current `flat` sample mode with an exhaustive
  deterministic random mode:
  - build one deterministic random permutation of all flat CP indices for a
    dataset pass;
  - map global sample position `i` to
    `permutation[i % sample_count]`, with pass number
    `i // sample_count` selecting the repeated pass;
  - use a stable seed derivation from config seed plus pass number so repeated
    passes are deterministic but not necessarily identical unless the
    implementation already expects identical repeats.
- Make the random order stable for the same CP set:
  - avoid making `training.max_steps`, `batch_size`, or
    `control_points_per_step` part of the random-order seed;
  - prefer stable CP identity when constructing/order-keying the permutation so
    moving fibers between train/test directories does not unnecessarily reshuffle
    unchanged CPs beyond the changed dataset membership.
- Thread the exhaustive random sample mode through:
  - `descriptor_for_sample_index`;
  - `build_strip_source`;
  - `build_sample`;
  - `load_batch`;
  - dependency generation and `prefetch`;
  - `iter_batches` and the trainer.
- Keep the existing skip semantics:
  - invalid CP-local data errors such as missing Lasagna samples or
    `grad_mag == 0` are data skips;
  - fatal infrastructure/programming errors still stop the run;
  - batch loading advances the deterministic random stream until the requested
    number of valid control-point samples is loaded or the configured skip
    guard is exceeded.
- Align prefetch with training:
  - omitted `--prefetch-steps` uses configured `training.max_steps`;
  - explicit positive `--prefetch-steps N` overrides config and covers the same
    deterministic random prefix that `N` training steps would consume;
  - explicit `--prefetch-steps 0` overrides config and covers one complete
    deterministic random pass over all configured training CPs;
  - full prefetch also covers one complete deterministic random pass over
    configured `test_datasets` CPs.
- Update prefetch/training console text to name the mode as deterministic random
  order, not flat order.
- Remove or demote the previous flat mode if it is no longer useful. If retained
  only for tests/debug, make sure the trainer and prefetcher do not use it.

## Spec Update

Update `planning/specs.md` to replace the flat CP-order semantics with
deterministic random-without-replacement semantics:

- training/prefetch consume the same deterministic random CP order;
- each pass covers all configured CPs before repeating;
- `max_steps` affects only how much of the deterministic random stream is
  consumed;
- `--prefetch-steps 0` means one complete deterministic random pass over train
  and configured test CPs;
- invalid samples are skipped while preserving deterministic stream advancement.

## Docs Updates

Update `docs/code_structure.md` and `planning/local_development.md` where they
currently describe flat prefetch/training order. The docs should explain the
training command still uses deterministic random order and how to prefetch the
same prefix or all CPs.

## Testing

- Add or update focused tests for:
  - deterministic random order is not flat sequential order for datasets with
    enough CPs;
  - the first `K` sample descriptors are identical whether
    `training.max_steps` is small or large;
  - the deterministic random order covers every CP exactly once before
    repeating;
  - after one full pass, the repeated pass is deterministic and has the expected
    configured behavior;
  - explicit positive `--prefetch-steps N` covers exactly
    `N * training.control_points_per_step` positions from the training random
    stream;
  - explicit `--prefetch-steps 0` covers exactly one full training/test pass;
  - invalid data skips advance through the same deterministic random stream in
    batch loading and prefetch.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a changelog entry because this corrects public training/prefetch ordering
semantics and replaces the recently introduced flat training order.
