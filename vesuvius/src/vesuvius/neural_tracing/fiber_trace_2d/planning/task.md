# Native C++ Trace2CP Python-Parity Fixes

Fix the remaining native C++ Trace2CP parity issues identified against the
Python tracer:

- Match Python beam pruning and reached-target selection semantics.
- Replace the C++ compact normal principal-axis power iteration with the same
  symmetric eigensolver convention used by Python.
- Preserve and test the existing candidate-loss all-pairs formula for the
  `candidate_substeps=1` path used by the native metric command.
