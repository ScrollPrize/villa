# Task: remove obsolete interactive chunk-cache compatibility APIs

Remove the implicit `beginViewRequest()` frame-epoch model and other dead
compatibility surfaces left by the retired per-workspace private decoded-cache
architecture.

- Keep explicit `(view ID, view version)` demand publication and cancellation
  as the only interactive scheduling model.
- Keep context-free chunk access as explicit background/batch work because it
  is used by Python bindings, CLI tools, slicing, and blocking samplers.
- Keep cache-wide scheduler group epochs used by `invalidate()`.
- Remove dead VC3D cache-policy routing, refresh hooks, surface-view generation
  plumbing, and private-pool footprint helpers.
- Preserve rendering values, cache residency, background access, invalidation,
  queue fairness, and active-view priority.
