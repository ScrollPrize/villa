# Task: reference calibration of winding phase

Test whether the perpendicular-constraint problem comes from the fixed winding
phase. In the proposed physical model, H and V on the same sheet have zero
latent separation, while the transition from V on one sheet to H on the next
has separation one. This corresponds to phase zero rather than the current
fixed phase `0.5`.

Fit phase from the ordered tagged reference fibers and report how relevant it
is. The reference H/V mapping and winding direction are ambiguous, so enumerate
both choices for each. This is a reference-only diagnostic; do not change the
production phase default from this experiment alone.

Do not collapse the two alternating perpendicular transitions into one opaque
fit. Report direct, unweighted raw-step statistics separately for H-to-V and
V-to-H transitions, split by nominal reference separation 0.5, 1.5, and 2.5+
windings. Report the corresponding H-to-H and V-to-V parallel statistics at
integer separations 1, 2, and 3+. These measurements, not solver-weight sweeps,
are the primary result.
