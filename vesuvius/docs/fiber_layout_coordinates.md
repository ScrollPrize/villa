# N/U semantics — authoritative model contract

- **N:** real-valued center of a probability distribution over integer labels k.
  Its fractional part changes probabilities, not geometric phase.
- **U:** unwrapped phase/progression. **U mod 1** is the umbilicus phase.
- **Realized position:** **k+U**, evaluated for an integer hypothesis. N+U is
  not a realized position and must not be used as one in sign/reference checks.
- Example: N=5.5,U=2.3 gives equal probability to 7.3 and 8.3 (k=5,6),
  both with phase .3. It does not put a fiber at 7.8.
- Expected losses integrate hypotheses with their probabilities. Single-layout
  reports choose the maximum-probability integer label (MAP); exact ties select
  the lower label deterministically. This is label selection, not rounding a
  measured constraint or a predicted endpoint into its own target.
- Angular prior compares U modulo one with measured angle, not (N+U) modulo one.
- N+=integer,U-=integer is an equivalent relabeling. Fractional transfer is NOT:
  it changes the phase and the discrete probability distribution.
- Arc-length-to-phase mapping can depend on the width of the hypothesized
  winding. Evaluate that existing mapping for the selected k; do not add the
  fractional part of N to the resulting phase.
- Shared knots store U/V. Each hypothesis has physical X = width_integral(k+U).
  Interpolate/extrapolate that physical curve separately per integer label.
  N is not an input to that fixed-label geometry. Length, direction, alignment,
  bending and phase-prior energies are expectations over integer hypotheses,
  not energies evaluated at the fractional mean N. N gradients through their
  probabilities remain valid; a shift of N can change expected energy but must
  not deform any fixed-label curve.
- New snapshots mark `shared_curve_integer_geometry=true`. Old snapshots without
  the marker retain historical readout; they are not silently reinterpreted.
  `shared_curve_integer_geometry=false` exists for historical replay only.

GT training marginalizes order-magnitude and wrong-order-mass penalties over
integer hypotheses. It never trains through a hard MAP selection or evaluates
the fractional mean as a realized label. Existing weights remain unchanged:
the magnitude weight is soft_pair_weight + probability_pair_weight; the
wrong-order-mass weight is probability_pair_weight. MAP is only a report.
Reference scoring keeps one global additive alignment, with no per-fiber fit.
Historical continuous-center metrics must be labeled historical, not silently
compared as the same metric. Earlier continuous-N phase-prior tests are invalid.
