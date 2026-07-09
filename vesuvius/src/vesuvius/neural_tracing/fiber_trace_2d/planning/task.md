# Current Task

Add temporary debug profiling to 2D fiber-strip training prefetch.

- Keep the current shared augment-vis/training/prefetch loading behavior
  unchanged.
- Add prefetch timing output that shows where a slow prefetch spends time:
  source descriptor/window/Lasagna-normal/source-coordinate construction,
  per-offset envelope coordinate generation, sampler/cache prefetch time, and
  dependency/download/cache stats reported by the sampler.
- Print a short per-patch start line before the potentially blocking sampler
  call so hangs inside VC3D/cache reads are visible.
- Run the focused fiber_trace_2d tests and lightweight syntax/help checks.
