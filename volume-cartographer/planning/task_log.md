# Task Log

## Findings

- The uploaded GitHub Actions artifact used GCC 15.2.0 and Valgrind 3.26.0,
  while the historical reference recorded GCC 15.3.0 and Valgrind 3.25.1.
- The renderer checksum was unchanged and the failing
  `parallel/fallback_3` modeled score was only 0.97% above its reference, but
  exact historical identity comparison failed before the 10% performance gate.
- The current reference check also rejects model-hash and checksum changes,
  making it broader than the requested performance-only regression check.

## Decisions

- Historical reference comparison will gate only the modeled-runtime score.
- Identity, model, checksum, and workload fields remain in artifacts for
  diagnosis but will not affect pass/fail.
- Same-run artifact integrity and Callgrind/DRD consistency checks remain;
  removing them would allow malformed inputs to produce meaningless scores.

## Plan Review

- Make the performance gate one-sided so improvements never fail.
- Treat schema validation and case lookup as structural requirements for
  reading the baseline, not as historical workload-identity gates.
- Preserve profiler-version diagnostics without validating historical values.
- Reject non-finite or non-positive scores before calculating the ratio.

## Validation

- All 19 focused `run_render_valgrind_ci` unit tests pass.
- The `python3 -S` dependency-free scoring smoke test passes against the
  existing GCC 15 native replay binary.
- Re-evaluated every completed evaluation in the uploaded GitHub Actions
  artifact. All six available cases pass, including the formerly failing
  `parallel/fallback_3` case at 1.009663x reference. The two remaining parallel
  evaluations were not produced because Ninja stopped after the original gate
  failure; the next CI run remains the full eight-case confirmation.
- Ruff check passes for the touched Python files when ignoring the branch's
  pre-existing `EXE001` file-mode findings and existing `SIM117` nested-context
  finding. Ruff format and `git diff --check` pass.
