# Task

Correct whole-volume inference ETA estimation so it extrapolates the observed
empty-input tile fraction. Resume skips caused by already-complete output must
not be treated as empty tiles or bias that fraction. Preserve the fast resume
path: the correction must not add a whole-volume pre-scan or extra source I/O.
