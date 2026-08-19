# Plan: Native Trace2CP benchmark record

1. Copy the user-supplied historical benchmark document exactly.
2. Add the supplied 2026-08-19 aggregate results to the existing main table
   with the code revision used for both runs.
3. Add only the new snapshot and settings rows required to define that result.
4. Validate Markdown formatting, arithmetic, and repository diff cleanliness.

## Spec update

None. This records measurements without changing runtime behavior.

## Documentation update

Restore the exact supplied `docs/native_trace2cp_benchmarks.md` and extend its
existing tables with the supplied run.

## Changelog update

None. A benchmark-record update does not change behavior.
