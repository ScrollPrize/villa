# Plan: reliable and parallel on-demand fiber replay

## 1. Reproduce and isolate

1. Use the captured full replay thread state to identify the actual stalled
   scheduler invariant.
2. Publish ready-cell and tile completion while holding the condition
   variable's mutex, eliminating the demonstrated lost wakeup without polling
   or timeout recovery.
3. Preserve exact nested dependency errors, including dataset kind, chunk key,
   status, and underlying error.
4. Add focused multi-tile parallel extraction coverage and run the existing
   anchor scheduler tests.

## 2. Progress contract

5. Introduce replay-specific greedy progress containing the restart segment,
   local native step/budget, matched global reference arc/fraction, and current
   state. Do not present the local safety budget as a whole-replay target.
6. Add fiberlet graph progress on the same global reference-arc basis.
7. Precompute the replay chunk schedule. Report a stable schedule index,
   nearest global reference arc, monotone generated count, and scheduled count
   for anchor and fiberlet preprocessing; retain per-chunk key and input/output
   diagnostics. Do not count disk cache hits as newly generated chunks.
8. Print explicit completion for each evaluator independently, so the CLI says
   which future remains active before joining both results.

## 3. On-demand fiberlet parallelism

9. Parallelize deterministic candidate enumeration inside one
   `traceFiberletPaths` call using source-index work partitions and per-source
   output/counters. Concatenate in canonical source order so output and floating
   point work are unchanged.
10. Keep preparation, sampling, materialization, and DP execution on their
    existing parallel paths. Release each prepared candidate on its search
    worker so multi-gigabyte vector teardown is not serialized after search.
    Add phase progress and measured CPU use to the on-demand chunk row.

## 4. Validation

11. Add focused tests for parallel/serial candidate identity, replay progress
    monotonicity, and the multi-tile scheduler. Exercise independent evaluator
    terminal output in the CLI replay validation.
12. Build affected targets with `-j32`; run anchor, fiberlet-path, graph replay,
    cache, and CLI-focused tests.
13. Re-run an uncached Paris4 chunk, verify exact prefix/route bytes, and
    measure wall/CPU time and effective core usage before/after. Leave the
    multi-hour full-reference validation to the existing user run after this
    focused regression passes.

## Spec update

Document that replay progress separates restart-local work from global
reference progress, preprocessing uses scheduled-work counters, cache waits do
not use polling/timeouts, and generated dependency failures preserve their
exact cause. No storage-format change is required.

## Documentation update

Update `volume-cartographer/docs/fiberlets.md` with the revised progress fields,
evaluator completion, and per-chunk parallel phase diagnostics.

## Changelog update

Record the on-demand replay stall fix, parallel chunk generation, and coherent
restart/global progress.
