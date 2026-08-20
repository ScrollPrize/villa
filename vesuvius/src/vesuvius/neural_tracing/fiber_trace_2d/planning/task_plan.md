# Plan: show live cached replay preprocessing progress

## Resolution accounting

1. Extend the generated fiberlet chunk-cache helper with an optional resolved-
   chunk callback carrying chunk kind, key, and final status. Invoke it exactly
   once for both persisted-cache reads and newly generated chunks, after fetch
   resolution, without changing payload generation or cache scheduling.
   Contain observer exceptions separately so they cannot change a successful
   fetch into a cache error.
2. Forward resolved anchor and prefix events through the on-demand
   preprocessor. In the replay CLI, derive the unique expected anchor dependency
   keys and scheduled fiberlet prefix keys from the existing deterministic
   schedule. Install expected sets before prefetch begins. Count each successful
   resolved key at most once under one mutex, including later reloads, and
   ignore failed or out-of-model keys. Resolution callbacks capture only a
   shared accounting state, not CLI stack objects; disabling the state makes
   late worker callbacks harmless.

## Overall estimate and repaint

3. Give the single overall progress display an explicit preprocessing model.
   Treat extraction as 95% of tracing work, matching the measured workload
   where graph traversal is small. Define scheduled cache fraction as
   `(resolvedAnchors + 16 * resolvedPrefixes) /
   (expectedAnchors + 16 * expectedPrefixes)`; the 16:1 ratio reflects measured
   roughly one-second anchor chunks versus roughly 15-20-second fiberlet
   chunks. Tracing fraction is
   `0.95 * cacheFraction + 0.05 * min(greedyFraction, fiberletFraction)` for
   cached replay, or the existing tracer minimum for eager replay. Zero
   scheduled work has cache fraction one. Clamp and retain every component
   monotonically, and reserve final completion for actual tracer,
   visualization, and publication completion. Prefix reach-neighborhood reads
   outside the scheduled prefetch set and committed route reads remain covered
   by the final 5% tracer term: they are data-dependent and historically small
   compared with extraction, so they are deliberately not represented as
   expected preprocessing keys.
4. Run a private timer while the concise display is active and repaint at a
   bounded interval even when no worker callback fires. ETA remains a live
   elapsed/fraction estimate and may increase while the estimated fraction is
   stationary. Signal and join the timer without holding the render mutex,
   then terminate the line; success, error, and destructor shutdown are
   idempotent and prevent output after the line closes. `--stats` creates no
   ticker and remains callback-driven and unchanged.

## Verification

5. Add a focused cache test proving resolution callbacks fire for both a newly
   generated chunk and a persisted cache hit without rerunning the generator,
   and that observer exceptions cannot alter successful cache results.
6. Build with `-j32`; run fiberlet storage and replay tests. Run a fresh-cache
   radius-768 replay under a short timeout and verify the captured default line
   contains increasing elapsed values, a nonzero estimated percentage, and a
   finite ETA while CPU work continues. Also verify `--stats` contains no
   concise timer line and interrupt cleanup leaves one terminated line.

## Spec update

Document that concise replay progress incorporates weighted scheduled cache
resolution and is timer-refreshed; the fraction is an estimate and remains
monotone, while completion still reflects real workflow completion.

## Documentation updates

Update `volume-cartographer/docs/fiberlets.md`, planning status/task log, and the
changelog with the estimate, timer behavior, validation, and limitations.
