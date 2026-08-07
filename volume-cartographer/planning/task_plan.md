# Task Plan

## Scope

1. Add a native replay-engine protocol command that accepts per-thread
   Callgrind events plus the frozen event-cost model and returns deterministic
   per-thread and total modeled costs.
2. Implement the existing supported feature bases, validation, overlap rule,
   and dot-product arithmetic in the shared C++ replay library. Use
   overflow-safe interaction arithmetic, fixed coefficient and numeric-thread
   accumulation order, finite-value checks, and tight parity against frozen
   representative profiles.
3. Bump the replay protocol for the new native scoring command. Make the
   rendering gate use native cost calculation for both serial and
   parallel cases. Remove its runtime import of the NumPy calibration helper,
   and extract the dependency-free DRD parser into a shared module used by both
   CLIs.
4. Keep Python as orchestration and artifact parsing only; do not add NumPy to
   the CI image.

## Specification Updates

- Require the CI scoring path to use the native replay engine for event-cost
  calculation and replay without third-party Python runtime dependencies.

## Documentation Updates

- Clarify the native/Python boundary and document the clean-environment smoke
  check.

## Testing And Validation

1. Add C++ parity/validation tests for supported event features, malformed
   profiles/models, and modeled costs, plus protocol/client tests for native
   profile scoring.
2. Add a `python3 -S` runtime test that imports the CI driver, exercises the
   shared DRD parser, and performs real profile scoring through the native
   replay binary without NumPy or other site packages. The workflow's full
   matrix continues to execute the actual callgrind, DRD, and evaluate commands.
3. Run the native tests, full benchmark Python suite, full eight-case gate,
   workflow YAML parsing, formatting/lint, and `git diff --check`.
4. Run the pinned CI container when a usable local container runtime is
   available; otherwise explicitly report that limitation and rely on the
   dependency-free `python3 -S` smoke plus the next GitHub Actions run.

## Changelog Update

- Record native event-cost scoring and removal of NumPy from the CI runtime
  path.
