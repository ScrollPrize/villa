# Status: continuous deterministic crop tracing and zero-copy graph access

- [x] Record the active task.
- [x] Inspect scheduler, graph interfaces, cache leases, and hot query paths.
- [x] Establish the current 500-attempt crop baseline.
- [x] Write the implementation and validation plan.
- [x] Complete independent plan review and incorporate requested scheduler,
  indexed-handle, and lease-lifetime clarifications.
- [x] Add directional borrowed graph views and contiguous immutable storage.
- [x] Make cache-backed compatibility queries return one owned aggregate view.
- [x] Port crop lookahead to allocation-free view traversal.
- [x] Implement bounded continuous computation with ordered finalization.
- [x] Add exact-output and lifetime regressions.
- [x] Build and test with GCC and Clang.
- [x] Benchmark the same 500-attempt crop and compare exact outputs.
- [x] Finalize specifications, documentation, changelog, and task log.
