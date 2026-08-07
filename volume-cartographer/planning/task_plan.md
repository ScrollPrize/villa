# Task Plan

## Scope

1. Remove the workflow's exact Valgrind-version assertion while retaining the
   normal package installation.
2. Keep the collected Valgrind version in manifests, evaluation output, and
   frozen references, but exclude only that field from reference identity
   equality. Require both versions to be present and non-empty. Report the
   reference and observed versions in checked output before score validation so
   failed score artifacts retain the diagnostic.
3. Preserve strict checks for model hash, renderer checksum, compiler, build
   configuration, workload, cache geometry, schema, and matching profiler
   versions between paired Callgrind and DRD artifacts from one run.

## Specification Updates

- Replace the exact reference Valgrind-version requirement with a requirement
  to record both versions and let the modeled-runtime tolerance gate changes.

## Documentation Updates

- Document that profiler updates are diagnostic metadata, not an automatic
  identity failure, and that significant event-count changes should fail the
  score comparison.

## Testing And Validation

1. Add a regression test proving a reference/observed Valgrind-version change
   is accepted when the modeled score remains within tolerance and is exposed
   in the result.
2. Preserve tests for all remaining identity checks and add coverage for
   same-run Callgrind/DRD profiler consistency, required version metadata, and
   version diagnostics on score failure.
3. Run the focused driver tests, full benchmark Python tests, workflow YAML
   parsing, formatting/lint checks, and `git diff --check`.

## Changelog Update

- Record that Valgrind updates no longer require an immediate reference refresh
  unless they materially move the modeled regression score.
