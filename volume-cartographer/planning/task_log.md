# Task Log

## Findings

- GitHub Actions run `31173510984`, job `92850416859`, failed in `Install
  deterministic profiler`; configure and benchmark steps were skipped.
- The Ubuntu 26.04 container repository supplies Valgrind 3.26.0 while the
  frozen reference records Valgrind 3.25.1.
- The workflow rejected that difference before collection, and the evaluator
  would reject it again because `valgrind_version` is part of exact reference
  identity equality.
- Same-run Callgrind/DRD version equality is a separate consistency check and
  remains useful.

## Decisions

- Keep profiler versions in all artifacts and references for diagnosis.
- Ignore only reference-versus-observed Valgrind version when comparing
  environment/workload identity.
- Keep the modeled score, checksum, model, and all other identity checks strict.

## Plan Review

- Require non-empty version metadata even though it is excluded from cross-run
  identity equality.
- Add explicit coverage for same-run Callgrind/DRD version consistency.
- Populate reference and observed profiler versions before score validation so
  a failed score artifact remains diagnosable.
- Rename and update workflow/documentation wording that implied a pinned
  profiler version.

## Validation

- Focused driver unit tests: 19 passed.
- Full benchmark Python suite: 102 passed.
- Focused CTest `test_render_valgrind_ci_driver`: passed.
- Full eight-case local Valgrind gate: passed; serial ratios were 1.000 and
  parallel ratios ranged from 0.992 to 1.030.
- Workflow YAML parsing, Python byte compilation, Ruff formatting/correctness
  lint, and `git diff --check`: passed.
- A broad Ruff style run exposed pre-existing `EXE001` file-mode findings and
  one pre-existing `SIM117` nested-context finding in the touched test file;
  those unrelated style issues were not changed.
- Local collection used Valgrind 3.25.1. Cross-version behavior is covered by
  unit tests; GitHub Actions will provide the first full 3.26.0 matrix.
