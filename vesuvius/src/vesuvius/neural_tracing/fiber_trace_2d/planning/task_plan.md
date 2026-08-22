# Plan: separate fiber replay progress

## Output contract

1. Track cache/preprocessing and evaluator progress as independent monotone
   fractions. Cache progress retains the existing anchor weight one and
   fiberlet-prefix weight sixteen. Trace progress remains
   `min(greedy_fraction, fiberlet_fraction)` and therefore denotes actual
   progress along the selected reference interval.
2. Render both labeled values in compact mode while preprocessing and tracing
   overlap. Give each its own ETA but show one overall replay elapsed time on the line;
   never combine them into a weighted percentage. Remove the cache/prep field
   once its scheduled fraction reaches 100%, and clear any remainder from the
   previously longer terminal line.
   Render both values on one terminal line so event output can close and redraw
   it atomically. Omit cache for eager replay.
3. Cache timing starts when the deterministic schedule is attached, trace
   timing immediately before evaluator launch, and output timing after both
   futures join. Never force unresolved cache work to 100% at trace completion.
4. Once both tracers complete, close the cache/trace display. Report overview,
   per-failure visualization, and bundle publication as named output stages.
   Do not reuse the old unexplained 82/98/99-percent estimates. Only the
   visualization stage has a real completed/total denominator and ETA.
5. Preserve immediate failure lines, periodic redraws during long cache work,
   `--stats` behavior, cache scheduling, and all numerical replay behavior.
   Cache progress intentionally excludes data-dependent neighbor-prefix and
   committed-route reads, which remain part of tracing.
6. Alongside the whole-trace average ETA, show `eta_current` from a rolling
   ten-second trace-fraction window. Ticker samples with no progress must lower
   the measured recent rate; report `n/a` when the window has no positive
   progress.
7. Extend fiberlet replay progress with diagnostics from its latest completed
   bounded lookahead decision. `fiberlet_rollout_expansions` is the total number
   of states whose successors were enumerated across all intermediate fronts.
   `fiberlet_local_cutoff_loss_per_vx_min` is the minimum applied final-front
   cutoff after subtracting the input route's loss at the front start and
   dividing by that front's prediction-voxel length. Publish a cutoff only when
   the existing strict queued-lower-bound stop actually fires. Omit both values
   in exact-search mode.

## Implementation

1. Refactor the local replay progress reporter to hold independent cache,
   trace, and output state and independent phase start times.
2. Keep existing call sites and callbacks narrowly adapted to the new reporter
   contract. Quantization replay uses its fiberlet evaluator fraction as trace
   progress and routes failure events through the reporter.
3. Carry bounded expansion count and local cutoff density through the existing graph replay
   progress callback and detailed `--stats` row. Do not infer search internals
   in the CLI.
4. Document the terminal output semantics and remove the obsolete 95/5
   composite formula from the specification and user documentation.

## Testing

1. Build `vc_fiberlets` with `-j32`.
2. Run a short hot-cache replay in compact mode and verify that output contains
   separately labeled cache/prep and trace progress with the trace fraction
   matching the detailed reference fraction.
3. Run the focused fiber replay tests to ensure progress-only changes do not
   alter replay results.
4. Exercise cold-cache overlap, cache below 100% at trace completion, monotone
   trace high-water updates, eager cache omission, event-line redraw, output
   stage transitions, cache-field removal without stale terminal text, and
   error termination without false completion. Verify a `--stats` run emits no
   compact progress labels.
5. Add focused replay checks that bounded running-progress events expose the
   same total expanded-state count and minimum applied final-front local cutoff
   density as decision/front diagnostics. Verify exact mode omits them and that
   expansion diagnostics persist until the next completed decision.

## Spec update

- Replace the single weighted replay progress contract with independent
  cache/prep and trace progress, plus a separate output phase after tracing.
- Define recent-speed ETA, bounded rollout expansions, and local pruning-cutoff
  semantics.

## Docs update

- Update `volume-cartographer/docs/fiberlets.md` with the new labels, fraction
  definitions, and overlapping-phase behavior.

## Changelog

- Record the correction from composite to explicit replay progress.
