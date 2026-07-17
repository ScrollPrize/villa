# Native 3D Trace2CP Beam Search

Replace the native 3D Trace2CP greedy one-step candidate choice with a small
beam search so tracing can keep multiple plausible continuations over several
steps. Reduce the native cone candidate density from the current opaque
`25x25` grid to explicit 5-degree angular steps inside the configured cone.
