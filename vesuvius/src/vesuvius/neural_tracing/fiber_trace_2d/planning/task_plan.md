# Plan: Post-solve perpendicular consensus

## Contract

- Add opt-in direction-ablation controls `--post-iterations N` and
  `--post-influence I`. Zero iterations remains the default and writes no post
  artifacts. Influence must be finite in `(0,1]` and defaults to `1` when post
  filtering is requested.
- Require the final selected checkpoint to have exactly one MILP piece for
  every represented input fiber, with trace indices forming a complete unique
  `[0,N)` set. Initialize H=`1`, V=`0`, and Broken=`0.5`. Split, missing, or
  duplicate source-fiber pieces fail explicitly; non-admitted source fibers are
  absent from the checkpoint and never enter post-filter output.
- Build unique source-fiber adjacency from the exact retained solver links.
  This post-filter is accepted only with `--perpendicular-only`; hard
  same-source links are ignored. Every neighbor therefore contributes
  `1-neighbor_value`, reflecting perpendicular H/V opposition.
- For influence `I`, define neighbor confidence
  `clamp((abs(v-0.5)-0.5*(1-I))/(0.5*I), 0, 1)`. Thus `I=1` rises linearly from
  zero at `0.5` to one at the extrema, while `I=0.5` is zero through
  `[0.25,0.75]` and rises linearly over the remaining quarter intervals.
- Every iteration is a synchronous Jacobi update. A fiber becomes the weighted
  mean of transformed neighbor values. A fiber with no positive total weight
  keeps its previous value. Stable trace-index and neighbor ordering make the
  result independent of thread scheduling.
- Divide whole source fibers into ten fixed H-value bands and write
  `<base>_p0.obj` through `<base>_p9.obj`: `p0=[0,0.1)`, ..., `p8=[0.8,0.9)`,
  and `p9=[0.9,1]`. Exact internal boundaries belong to the higher band, and
  equal values remain together. Every file is written even when empty. Print
  count and min/mean/max value for every group. Also stratify the existing
  gauge-aligned comparison by band, reporting H, V, and Mixed populations and
  errors without defining a new post-filter error threshold.
- Post-filtering does not modify the MILP labels, error statistics, existing
  label/constraint/initial OBJ outputs, solver objective, or selected broken
  cost. It is a diagnostic transformation only.

## Implementation

1. Add a reusable core post-filter helper operating on no-split constraint
   pieces, retained constraint indices, and MILP labels, returning one value
   per represented trace.
2. Add reusable fixed-band classification and a writer for the ten whole-fiber
   OBJ layers with short stable suffixes.
3. Parse and validate the CLI controls, run the helper only for the final
   selected direction-ablation checkpoint, and print its fixed-band summary.
4. Extend the stable runner with explicit post-filter arguments and rerun the
   selected no-split, perpendicular-only, 40-mixed, cost-0.2435 experiment.

## Spec Update

Document no-split/represented-fiber scope, initialization, perpendicular
inversion, confidence weighting, synchronous iteration, no-evidence behavior,
and value-band output ownership in `planning/specs.md`.

## Docs Updates

Document the CLI controls, formula, and `_p0` through `_p9` output layers in
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Unit-test initialization, rejection of split/missing/duplicate fibers,
  unique-neighbor deduplication, perpendicular inversion, both influence
  examples, synchronous multi-iteration behavior, no-weight retention, exact
  value-band boundaries/ties, and ten output names including empty groups.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused suite, and run `git diff --check`.
- Rerun the stable centered-384 experiment with three post iterations and
  influence 1.0; confirm the MILP statistics remain identical, exactly 135
  fibers are partitioned, and all ten short-named OBJ layers are written.
- Repeat with 100 iterations to measure long-run consensus behavior and report
  the complete per-band H, V, and Mixed error table for both iteration counts.

## Changelog

Record the opt-in post-solve perpendicular consensus diagnostic.
