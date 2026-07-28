# Task: Native 3D Trace2CP GPU-Centric Beam Acceleration

Speed up native 3D Trace2CP while preserving the current metric quality and
semantics.

Current measured state on the approved whole-fiber benchmark:

- Sparse corner/tensor Lasagna normals are the default and match the baseline
  quality: 3 restarts on the reference fiber.
- Runtime improved from roughly 472s to roughly 201s, but the remaining hot
  path is still not aligned with the intended GPU-centric tracer.
- Remaining issues to fix:
  - block routing/grouping for point lookup is still CPU-side,
  - beam tracing is still step-wise Python orchestration,
  - candidate scoring is vectorized only inside each expansion, not across
    broader trace work,
  - field lookup uses many small GPU calls instead of fewer larger calls.

Requirements:

- Keep the same benchmark command and report before/after timings after each
  implementation step.
- Keep sparse corner/tensor normals as the default normal sampler and keep
  baseline normals as an explicit debug/fallback mode.
- Do not reintroduce raw compact `nx`/`ny` interpolation or dense inverse/search
  normal decoding.
- Preserve deterministic tracing and the current metric semantics.
