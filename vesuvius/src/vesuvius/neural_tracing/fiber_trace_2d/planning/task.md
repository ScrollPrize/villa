# Task: accelerate fiberlet tracing

Accelerate anchor and fiberlet extraction used by `vc_fiberlets`, especially
`fiberlet-replay`, while preserving stable geometric behavior and deterministic
output. Float32 tube-containment geometry is sufficiently accurate; exact
numeric identity with the current double-precision scan is not required.

Profiling identified fiberlet node enumeration and tube containment as the first
optimization target. Replace repeated linear replay-tube segment scans with a
reusable float32 Boost segment R-tree, and test each candidate's adjacent local
corridor segment before scanning the remainder. The geometric containment rule
is whether any continuous segment lies within the configured radius. Measure
and report changed classifications near the radius boundary; changes away from
the float32 boundary tolerance are not allowed.

Use the supplied 5,000-base-voxel `fiberlet-replay` command as the canonical
before/after workload. Record the measured anchor-fitting follow-up separately;
do not mix anchor changes into this optimization.
