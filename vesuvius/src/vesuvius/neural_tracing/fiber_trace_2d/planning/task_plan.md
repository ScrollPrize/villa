# Recovery plan

1. Inventory main versus fiber-exp5 by symbol/behavior, not filename alone. Preserve extracted replay, normal sampling and chunk route APIs.
2. Port missing reusable chunk options, seed refinement, initial matching and joint joins; recover normal-aware conflict filtering.
3. Recover optional CUDA beam construction, fixed-plane online refinement, native pooling and shared sampling/scoring helpers. Optional algorithms disabled by default; avoid obsolete duplicated internals.
4. Expose through current bindings/CLI and generic benchmark commands with cache-only execution; no hard-coded cohorts/models/machine paths required.
5. Validate deterministic synthetic tests, native chunk/replay/scoring tests and Python tests; check cache formats without overwriting caches or silently recomputing. CPU build independent of CUDA. No dependency installation.
6. Selective commits of reusable code/tests/docs; experiment scripts, results and configurations stay out of commits.

## Spec update
No spec.md exists on main. Document opt-in capabilities, ownership/cache invariants, coordinate units and portability in focused recovery documentation; do not import obsolete global specs.

## Docs updates
Recovery inventory, validation commands, supported entrypoints, cache limitations and explicit outstanding gaps. Update status/task_log/changelog.

## Validation
Use existing build environment/presets. Native chunk, reverse-consistency, scoring and synthetic filter/pooling tests; benchmark parsing and metrics tests. GPU execution only under managed grants. Cached cohort reruns once integration works.

## Changelog
Record recovery onto main and excluded data/experiment artifacts.
