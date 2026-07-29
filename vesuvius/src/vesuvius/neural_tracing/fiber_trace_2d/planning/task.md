# Native C++ Trace2CP Parallel Candidate Scoring

Speed up the native C++ Trace2CP tracer by parallelizing independent candidate
scoring inside each beam generation while preserving deterministic Python-style
beam ordering and trace results.

The whole-fiber segment chain remains sequential because each segment starts
from the previous segment's trace/restart state.
