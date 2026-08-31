# Task: estimate-first reference winding calibration

Infer each reference winding from its admitted constraints before using any
ground-truth calibration. Calibrate the global sign and per-component
half-integer offsets from those raw estimates, maximizing the number of exact
matches to the known filename-ordered reference windings. Apply that calibration
to the estimates and all individual constraint diagnostics. Keep the existing
`+/-0.5` tolerance only for the final right/wrong reporting counts.
