# Task Plan

## Scope

1. Make `check_reference` compare only the observed modeled-runtime score with
   the stored score using the configured tolerance. Use a one-sided regression
   bound: any lower score passes, and only a score above
   `reference * (1 + tolerance)` fails. Keep reference-schema validation and
   case lookup only because they are required to interpret and select the score
   baseline. Remove reference gates for model hash, renderer checksum, compiler,
   profiler, cache, fixture, and workload identity.
2. Continue recording all existing identity, model, and checksum fields in
   evaluation and frozen-reference artifacts for diagnosis. Preserve
   collection-time validation that Callgrind/DRD inputs are complete and belong
   to the same run; those checks protect score construction rather than compare
   the run with a historical environment.
3. Preserve existing profiler-version diagnostics without validating or
   comparing their values. Require finite positive observed/reference scores so
   a malformed score cannot bypass the sole remaining gate.

## Specification Updates

- Define the historical reference gate as a one-sided performance-regression
  check while retaining raw artifact and same-run consistency validation.

## Documentation Updates

- Update failure diagnosis and maintenance documentation so environment,
  workload, model, and checksum changes are diagnostic and never gate a run.

## Testing And Validation

1. Replace identity-rejection tests with a regression test that changes every
   non-performance reference/observed field and still passes at the same score.
2. Test the inclusive upper tolerance boundary, rejection above it, acceptance
   of arbitrarily large improvements, malformed score rejection, missing
   reference cases, and collection-time Callgrind/DRD consistency.
3. Re-evaluate the downloaded GitHub Actions artifact against the updated gate,
   run the focused benchmark-driver tests, and run `git diff --check`.

## Changelog Update

- Record that historical environment and correctness identity are now
  diagnostic-only and the CI reference gate checks only modeled performance.
