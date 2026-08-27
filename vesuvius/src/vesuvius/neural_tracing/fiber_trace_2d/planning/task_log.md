# Task Log: Post-solve perpendicular consensus

## Decisions

- Perpendicular graph links invert neighbor H values before averaging.
- Confidence uses the exact influence-dependent central dead-band formula in
  the task plan.
- Iterations are synchronous and preserve the prior value when no neighbor has
  positive confidence.
- Post-filter input/output is exactly the represented final checkpoint and
  requires one piece per fiber; non-admitted fibers are excluded.
- Fixed 0.1-value OBJ bands contain complete source fibers, not extracted
  pieces. Exact ties stay together and internal boundaries use the higher bin.

## Plan Review

- Removed invented split/absent-fiber averaging semantics; the diagnostic now
  enforces the requested no-split checkpoint population.
- Replaced ambiguous rank percentiles with explicit fixed value bands so equal
  values are never split arbitrarily.

## Deviations

- None.

## Validation

- Built `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` from the local
  Release build with `-j32`.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace` passed all 49 cases.
- Stable run:
  `/tmp/vc_direction_ablation_runner.sh 0.2435 40 perpendicular 3 1.0`.
- The run retained the same 135 fibers/pieces and 855 constraints. MILP stayed
  at objective 103.912955, 2/95 trusted errors, 30/40 mixed active errors, and
  32/135 total errors. Post-filtering did not modify solver output.
- Three influence-1 iterations partitioned exactly 135 represented fibers:
  `p0=65`, `p1=2`, `p2=0`, `p3=0`, `p4=0`, `p5=2`, `p6=0`, `p7=0`, `p8=1`,
  and `p9=65`. Values ranged `[0,0.041099]` in p0 and `[0.933945,1]` in p9.
- All ten `$VES/data/workdir3/384/384_pN.obj` files were overwritten, including
  valid comment-only files for empty bands. The stable current log contains
  the per-band population, error, and min/mean/max table.
- The per-band error report preserves the existing gauge-aligned comparison.
  Across both runs it reconciles to 50 H references, 45 V references, and 40
  Mixed references, with 0 H errors, 2 V errors, and 30 Mixed errors.
- At three iterations, errors by band were `p0=(H0,V0,M14)`,
  `p5=(H0,V1,M0)`, and `p9=(H0,V1,M16)`; all other bands had zero errors.
- A second stable run used
  `/tmp/vc_direction_ablation_runner.sh 0.2435 40 perpendicular 100 1.0`.
  Errors moved to `p2=(H0,V0,M4)`, `p3=(H0,V0,M10)`,
  `p5=(H0,V1,M0)`, `p6=(H0,V0,M8)`, and `p7=(H0,V1,M8)`.
  The 100 synchronous iterations moved the two main populations inward to
  `[0.282380,0.385884]` and `[0.616010,0.719646]`; they did not change the
  underlying 32/135 MILP error count.
