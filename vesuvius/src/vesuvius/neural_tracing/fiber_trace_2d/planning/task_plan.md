# Plan: concise replay progress and latest fiberlet optimizations

## Output

1. Add one thread-safe replay progress reporter. Its global completion is the
   minimum monotone reference-arc fraction reached by the greedy and fiberlet
   evaluators, because the command is complete only where both were evaluated.
2. Render a bounded-width terminal bar with percentage, elapsed time, and ETA.
   ETA remains unavailable until measurable global progress exists. Clamp and
   monotonically accumulate each concurrent evaluator independently so stale,
   non-finite, or restart-local callbacks cannot move the bar backward.
3. Suppress stage, chunk, restart, evaluator, cache, and visualization detail in
   the default run. Extend `--stats` to replay and preserve those diagnostics
   there. Reserve terminal progress until visualization/publication completes,
   always preserve the terminal result summary and thrown errors, and always
   finish the progress line before either result or error output.

## Merge and performance

4. Commit the focused progress/output change before merging.
5. Merge the unmerged `fiber-lets2` anchor and path-search optimizations. Resolve
   transient task/status/log conflicts in favor of this active task, merge
   cumulative specifications and documentation semantically, and preserve the
   exact float-cache graph contract.
6. Check whether optimized anchor/path APIs are used by on-demand chunk
   generation. Port shared optimizations into that path if the merge does not
   naturally cover it; do not duplicate implementations.

## Verification

7. Exercise default and `--stats` output, concurrent/stale progress, unavailable
   ETA, final newlines, error handling, and absence of suppressed default detail.
   Run the fiberlet storage, path, anchor, replay, and on-demand tests after a
   `RelWithDebInfo` `-j32` build.
8. Run cold eager and cached 5,000-base-voxel Paris4 replay with 32 threads.
   Compare float-cache payload populations and output artifacts exactly. Measure
   repeated wall time, process CPU time, effective cores, and sampled process CPU
   utilization before/after; summarize mean and min/median/max and record cache
   state. Capture a profiler hotspot sample if the merge does not improve the
   cold-cache path as expected.

## Spec update

Specify the default concise global progress contract, its reference-arc
semantics, and the opt-in detailed `--stats` diagnostics. Preserve float-cache
equivalence requirements.

## Documentation updates

Update `volume-cartographer/docs/fiberlets.md` with the default display and the
diagnostic option. Add a changelog entry and record commands/results in the
current task log.
