# Task log: separate fiber replay progress

## Finding

- The compact percentage was not actual reference-fiber progress. Cached replay
  computed `0.95*cache_fraction + 0.05*min(greedy,fiberlet)` and then reserved
  another one or twenty percent for publication/visualization. This made a
  displayed 95.9% compatible with only about 37% reference progress after the
  scheduled cache work had completed.
- A diagnostic `--stats` run reached reference arc `21851.462`, fraction
  `0.369495`, confirming that the earlier interpretation of the aggregate bar
  as near-end fiber progress was wrong.

## Constraints

- Do not change cache scheduling, preprocessing, tracing, replay output, or
  numerical results.
- Do not hide concurrently advancing trace progress behind the cache estimate.

## Implementation

- Compact cached replay renders concurrent `cache/prep` and `trace` bars. Each
  has an independent clock and ETA. Eager replay renders only `trace`.
- Trace completion closes those bars at the actual final fractions; output and
  visualization then use a separately labeled bar.
- The compact line shows elapsed time once, removes cache/prep at 100%, and
  pads shortened redraws so stale cache text cannot remain visible.
- `eta_current` uses a rolling ten-second trace-fraction window. Each completed
  bounded lookahead reports its total expanded-state count across all fronts.
  If the final-front queue is actually stopped by its strict loss cutoff, it
  also reports the minimum cutoff increment over that front normalized by the
  front's prediction-voxel length. This normalization is diagnostic only.

## Independent review

- Do not force scheduled cache work to completion when the evaluators finish.
- Start timers at schedule attachment, evaluator launch, and output-stage
  transition; keep ETAs independent and `n/a` before measurable progress.
- Render concurrent cache/trace values on one line and preserve atomic event
  interruption/redraw.
- Replace arbitrary post-trace percentage estimates with named output stages.
- Quantization replay must use the same reporter for failure lines, and
  `--stats` must remain free of compact output.

## Validation

- `cmake --build volume-cartographer/build --target vc_fiberlets
  test_fiberlet_paths -j32` succeeds.
- A cold cached 1,700-base-voxel replay showed cache/prep advancing separately
  through 6%, 23%, 36%, 86%, and 100%, while trace remained at 0% and then
  advanced through 8%, 29%, 52%, 71%, 91%, and 100%. The existing greedy
  failure line interrupted and redrew the active line correctly.
- A 500-base-voxel eager replay displayed only trace progress. The matching
  `--stats` run emitted the existing machine-readable stage/chunk/evaluator
  rows and no compact progress labels.
- Earlier hot and detailed smoke runs validated cache removal, line clearing,
  and both ETA variants. They used superseded rollout-width and cumulative
  cutoff labels; final expansion-count and local-cutoff results follow below.
- A hot 1,700-base-voxel replay emitted per-decision expansion counts from 18
  to 311 and applied local cutoff densities from 0.131224 to 0.188406 loss per
  prediction voxel. Cache/prep disappeared after completion, the line contained
  one overall elapsed field, and both trace ETAs continued to update.
- A detailed 500-base-voxel `--stats` replay emitted the same diagnostic field
  names and no compact progress line. Its bounded decisions reported 156 to
  239 expanded states and local cutoff densities from 0.101771 to 0.163650.
- The bounded replay unit fixture now forces an actual strict cutoff stop and
  checks that callback expansion/cutoff diagnostics match the recorded
  decision. Exact-search callbacks explicitly omit both diagnostics.
- Both smoke runs completed at the reference end with unchanged zero fiberlet
  failures. The 1,700-base-voxel run retained its known single greedy failure.
- `test_fiberlet_paths` still reports exactly the pre-existing 298 float
  bitwise and Q4 extraction fixture checks; this change introduced no new
  failure locations.
