# Task plan

1. Separate skipped inference tiles into empty-input and resume-complete
   categories at the existing work-discovery points.
2. Estimate remaining inference work by applying only the observed empty-input
   fraction to the remaining candidate tiles. Do not extrapolate resume skips.
3. Keep resume classification on the existing output checks; add no eager scan
   and no source read solely for ETA reporting.
4. Add focused unit coverage for clean sparse runs and resumed runs, including
   the case where many resume skips precede observed inference work.
5. Run the focused Lasagna test module and `git diff --check`.

## Testing

- Unit-test ETA arithmetic with distinct processed, empty and resume counts.
- Exercise the existing small inference fixtures if practical.
- Run `git diff --check`.

## Spec update

Specify that whole-volume inference ETA extrapolates empty-input sparsity but
does not interpret already-complete output as evidence of future emptiness.

## Docs updates

No user-facing workflow changes; progress-line behavior is self-describing.

## Changelog

Not required for this focused reporting correction.
