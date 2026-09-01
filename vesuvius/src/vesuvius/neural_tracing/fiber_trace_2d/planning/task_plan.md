# Plan: reference calibration of winding phase

## Model and implementation

1. Add a reusable reference-phase calibration helper beside the existing
   reference scale calibration. It consumes the ordered reference-to-reference
   constraint rows directly; production solver weights must not affect this
   reference measurement.
2. Classify every row by dominance and reference parity. Only dominant, signed
   perpendicular observations between opposite parities identify phase and may
   enter the fit, each with unit weight. Count same-parity perpendicular rows as
   phase-independent, same-parity parallel rows as non-identifying, and
   opposite-parity parallel rows as contradictions of the assumed H/V model.
3. Enumerate the four unobservable gauges: increasing/decreasing winding
   direction `d` and even reference source mapped to H/V. For source `i`, parity
   `p=i mod 2`, and phase `phi`, use exactly:
   - even-to-H: `y_i=d*(floor(i/2)+p*phi)`;
   - even-to-V: `y_i=d*(ceil(i/2)-p*phi)`.
   Minimize
   `sum(abs(predicted_delta / fixed_scale - raw_signed_delta))` over phase
   in `[0,0.5]`. Enumerate the exact weighted-L1 breakpoints plus both bounds;
   do not use an arbitrary sampling grid. Use source-oriented `rawStep`, never
   the BP-calibrated global sign or `calibratedStep`.
4. Select by unweighted L1, then smaller phase, increasing direction, and
   even-to-H. Sign penalties are deliberately excluded because production sign
   semantics reject the zero predicted delta that phase zero is intended to
   test; report sign disagreement descriptively instead. Report all four gauge
   rows with total/used/identifying and parity-class counts, effective weight,
   losses at phase `0` and `0.5`, optimum phase/loss, and percentage reduction.
   No signed identifying rows produces `NA/unidentifiable`, not phase zero.
   Use the run's finite positive output measurement scale and label it explicitly
   so phase is conditional on that scale.
5. Print the phase table with the existing reference diagnostics. Do not feed
   the fitted phase back into BP and do not change defaults.
6. Independently summarize the raw signed reference measurements. After mapping
   even/odd reference parity through each fitted H/V gauge, report H-to-V and
   V-to-H perpendicular rows in nominal 0.5, 1.5, and 2.5+ bands, and H-to-H
   and V-to-V parallel rows in nominal 1, 2, and 3+ bands. Each row reports
   count, minimum, mean, median, and maximum. This table is unweighted and does
   not use canonicalized targets.

## Tests

- Synthetic alternating references recover a known interior phase and gauge.
- Phase-zero data prefers the proposed zero/one alternating ladder without
  being rejected by production hard-sign semantics.
- Reversed direction and H/V parity ambiguity are enumerated deterministically.
- Same-parity perpendicular and opposite-parity parallel rows are classified;
  parallel, unsigned, empty, and malformed inputs have explicit behavior.
- Directional raw-step summaries preserve H/V transition direction, distance
  bands, signed values, and deterministic median behavior.
- Raw source-oriented signs do not reuse the BP global sign; fixed scale changes
  the optimum predictably; L1 interval and boundary ties are deterministic.
- Build Release `vc_fiber_trace_chunk`, run the focused winding BP suite, then
  run the established 1024 reference command and record the four-gauge result.

## Spec update

Document that reference-only phase calibration uses raw signed perpendicular
measurements, fixed scale, and explicit H/V/direction gauge enumeration. It is
diagnostic and cannot silently alter solver calibration or defaults.

## Docs update

Document the output table, objective, ambiguity columns, and why parallel
constraints carry no phase information.

## Changelog

Record the diagnostic and measured 1024 result after validation.
