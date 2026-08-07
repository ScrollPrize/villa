# Task Plan

## Scope

1. Make the checked reference's `tolerance` field the CI gate's single source
   of truth. Add `freeze-reference --tolerance`, validate `0 <= tolerance < 1`,
   and retain an optional evaluation override only for tests/diagnostics. Add a
   `freeze-model` command that validates synthetic-only provenance and the
   selected seven-feature basis before producing the compact CI model; require
   an explicit flag to promote an unaccepted experimental calibration. Add an
   atomic `set-tolerance` command that preserves every model/case field, so a
   policy-only update cannot accidentally refresh variable parallel scores.
2. Change the GitHub workflow to compute available logical CPUs with `nproc`
   and use that value for both benchmark compilation and the Ninja Valgrind
   graph. Keep renderer worker count and replay core count fixed because those
   are workload/model inputs, not host build concurrency.
3. Update the benchmark specification and documentation to distinguish:
   synthetic-only model recalibration, eight-case reference refresh, and
   tolerance-policy changes. Include exact commands, required review checks,
   activation conditions, and the meaning of increasing/decreasing tolerance.
   Add normal VC3D usage: local execution, artifact locations, failure
   diagnosis, and the distinction between running the check and requiring it
   through repository rules.
4. Add focused tests for reference-owned tolerance, explicit overrides, invalid
   tolerance values, and freeze behavior. Preserve existing score/reference
   values unless intentionally regenerated.

## Specification Updates

- Replace fixed four-job wording with one independent Valgrind collector per
  available host CPU, bounded naturally by the artifact graph.
- Express expected duration for the reference four-core runner while allowing
  other runner sizes, rather than making four cores a general requirement.
- Define the checked reference as the tolerance source and require intentional
  review for model, baseline, or tolerance changes.

## Documentation Updates

- Add separate procedures for recalibration, rebaselining, and tolerance
  adjustment, including frequency setup/restore and exact validation commands.
- Explain that merging into `main` activates the existing workflow job for
  core/scripts/workflow changes selected by `changes.outputs.test_core`, while
  documentation-only changes skip the rendering job. Explain that repository
  branch protection/rulesets are separately required to make the check
  mandatory for merges.

## Testing And Validation

1. Run the focused CI-driver unit tests and the full benchmark Python tests.
2. Parse the workflow YAML and verify the `render-benchmark` job's two Ninja
   builds use the detected runner CPU count; unrelated jobs are out of scope.
3. Run Python format/lint/byte compilation and `git diff --check`.

## Changelog Update

- Record dynamic runner parallelism and the documented calibration/reference/
  tolerance maintenance workflow.
