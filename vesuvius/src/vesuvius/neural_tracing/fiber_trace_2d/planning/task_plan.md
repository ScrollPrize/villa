# Task Plan

## Shared model

1. Add a public Ceres solver API beside the BP API. Extract and reuse the exact
   public prepared winding model as the single source of materialized
   dominant hypothesis, scale-first canonical target, effective magnitude
   coefficient, sign coefficient/hardness, and endpoint ordering.
2. Optimize `horizontalness h in [0,1]`, `activity a in [0,1]`, and real
   winding `z` per piece. Use the existing orientation scores as paired
   parallel/perpendicular residuals, so binary endpoints reproduce the
   existing same/different orientation costs.
3. Gate cross-piece residuals by endpoint activity. Use degree-scaled
   `--winding-defect-cost` for the activity unary and
   `--piece-break-cost` for activity discontinuity across split-continuation
   links. Preserve exact split-continuation H/V and winding behavior through a
   strong least-squares residual when hard continuation is enabled.
4. Use the five existing magnitude weights and two sign weights after current
   confidence attenuation. Map every effective coefficient `c` to residual
   scale `sqrt(c)`, so Ceres contributes `c*r^2`. Represent finite and promoted
   hard signs by one-sided residuals. Reuse the existing mixed/defect,
   continuation, sign, magnitude, confidence, and scale controls without adding
   a separate Ceres-only tuning surface.
5. Fix one deterministic central H/V gauge and one winding gauge per
   otherwise-unfixed corresponding component without fixing activity.
   Initialize H/V from PCA direction groups, activity from the orientation
   prepass, and winding at zero without running discrete BP.

## CLI and artifacts

1. Extend `--winding-solver` with `ceres`; retain `joint-grid` as the default.
2. In Ceres mode, run the usual orientation prepass only as initialization,
   then publish fractional H/V/activity/winding through the existing OBJ/CSV
   artifact path. Threshold only where legacy discrete files require a label;
   preserve the fractional values in the report/CSV.
3. Print Ceres termination, iterations, initial/final cost, solve time, and
   residual-class summaries. Existing BP-only component selection and
   constraint extraction remain shared.

## Reference solve

1. Reuse the existing reference-to-crop cross-constraint extraction. For each
   annotated reference source, build the subproblem containing its pieces,
   their hard continuation links, and only cross constraints to solved crop
   pieces.
2. Fix every crop endpoint to its main Ceres horizontalness, activity, and
   winding. Optimize only the selected reference source with the identical
   residual construction and settings; do not use the BP reference voting
   scorer to infer its state.
3. Aggregate each reference source's piece results by arc-length weight. Report
   raw horizontalness, activity, winding, residual cost, and usable connected
   constraints. Fit one global sign and half-step offset maximizing exact
   filename-order ladder matches, with squared error as deterministic tie
   break, then report calibrated winding and error.

## Validation

- Unit-test a synthetic parallel/perpendicular system, fractional activity,
  weight reuse, hard-continuation behavior, deterministic gauge fixing, fixed
  endpoint solving, and per-reference recovery.
- Build Release `vc_fiber_trace_chunk` and the focused solver test.
- Run the focused test and a short 1024-crop Ceres command using the approved
  runner. Record exact command, input, build type, runtime, convergence, and
  reference table.
- Confirm the established joint-grid command still takes the old path and its
  defaults are unchanged.

## Spec Update

- Specify the continuous variables, residual definitions, activity gating,
  gauge, hard-continuation approximation, sign handling, output thresholds,
  and per-reference fixed-source benchmark.
- State explicitly that least squares changes the residual norm relative to
  discrete BP while reusing its materialized targets and coefficients.

## Docs Updates

- Document `--winding-solver ceres`, its reused controls, output
  fields, and the reference benchmark in
  `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog Update

- Add one entry for the experimental continuous Ceres solver and fixed-source
  reference evaluation.
