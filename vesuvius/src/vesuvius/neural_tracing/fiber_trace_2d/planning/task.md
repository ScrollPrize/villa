# TTA Reference Mapping Performance

Fix Trace2CP median-TTA tracing hanging on large strips. The current path maps
each reference trace point into each TTA field by scanning the whole TTA
output-to-reference coordinate image. That must be replaced with direct lookup
through the augmentation forward map so per-step mapping is constant-time.
