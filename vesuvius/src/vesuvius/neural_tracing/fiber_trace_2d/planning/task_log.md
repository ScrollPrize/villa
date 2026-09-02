# Task Log

## Discovery

- The repository already requires and links Ceres in several targets, but
  `vc_fiber_tracer` does not currently link it.
- The existing deferred proposal in
  `volume-cartographer/docs/fiber_winding_ceres_proposal.md` defines continuous
  orientation, activity, and winding variables. This task promotes a scoped,
  fixed-calibration version of that proposal into an experimental solver.
- Printable factor diagnostics were not sufficient solver input. The private
  scale-first materialization path was extracted as
  `prepareFiberTraceWindingModel`; BP diagnostics and Ceres now consume that
  same prepared model.
- The existing reference cross report preserves one piece per input line and
  contains only reference-to-crop constraints. Reference-source continuity is
  available separately from the reference-only constraint report.

## Deliberate semantic difference

- Ceres minimizes squared residuals. Existing joint-grid BP ranks discrete
  states using predominantly weighted absolute residuals. The new solver
  shares targets and coefficients, not the residual norm; this is the point of
  the experiment and will be explicit in diagnostics and documentation.
- Residuals use the differentiable product `a_i*a_j`; because Ceres squares
  residuals, pair energy fades as `a_i^2*a_j^2`. No hidden epsilon or robust
  loss is applied.
- BP hard continuation and promoted hard signs are large finite residuals in
  Ceres. Exact discrete feasibility is intentionally not claimed.
- The first implementation uses the existing PCA orientation split as the H/V
  initializer and zero winding. Exporting BP's private continuous initializer
  was deferred because it would couple the independent experiment to another
  solve.

## Independent plan review

- Incorporated: expose the exact prepared model; apply measurement scale once;
  use exact prepared incidence and winding gauges; use a positive sign margin;
  report unidentifiable references as `NA`; avoid hidden robust losses and
  ladder priors; distinguish finite hard-constraint approximations.
- Clarified: an H/V component also requires one orientation-label gauge. It
  fixes horizontalness only, while the winding gauge fixes winding only;
  neither gauge forces activity.

## Validation

- Release build: `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk -j 8`.
- Focused tests: `volume-cartographer/build/bin/test_fiber_trace_winding_bp`;
  all 80 cases passed, including three new Ceres regression cases.
- Clang system-dependency build of `vc_fiber_trace_chunk` and the focused test
  succeeded; the same 80 cases passed. Clang reported one pre-existing ignored
  `[[nodiscard]]` warning in `FiberReplay.cpp`, outside this change.
- 1024-crop Release validation input: `crop_traces.zarr`, 500/1998
  quality-selected fibers, 1360 retained pieces, 69,232 prepared constraints,
  32 workers, and the default 500-iteration cap. Main Ceres solve converged in
  47 iterations and 4.9 seconds.
- Initial result deliberately uses current production weights and scale 0.822.
  It retained fractional activity near one; the per-reference fixed-source
  solve produced 24 usable estimates, matched 9/24 filename-order windings,
  and reported two sources with no usable winding evidence as `NA`. This is a
  functioning experimental baseline, not a tuned quality result.
