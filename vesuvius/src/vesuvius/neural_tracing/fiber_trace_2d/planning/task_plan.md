# Task Plan: Prefetch Debug Profiling

## Scope

Add temporary observability to the existing 2D fiber-strip prefetch path without
changing loading, coordinate generation, augmentation, cache semantics, or model
training.

In scope:

- Time source construction, including descriptor lookup, line-window creation,
  Lasagna-normal sampling, and source-strip coordinate generation.
- Time per-strip-offset prefetch-envelope coordinate generation.
- Print a per-patch start line before the blocking sampler prefetch call.
- Time sampler/cache prefetch and report sampler stats such as dependency count,
  downloaded chunks, bytes, and mode.
- Keep the existing aggregate progress bar and summary.

Out of scope:

- Optimizing prefetch.
- Changing VC3D/cache APIs.
- Changing augmentation ranges or sampled coordinates.
- Adding parallel prefetch workers.

## Spec Update

No normative spec change. This is debug/profiling output for diagnosing current
runtime behavior.

## Docs Updates

- Update `planning/task_log.md` with the instrumentation and validation result.
- Update `planning/status.md` for this profiling task.

## Testing

- Run Python compile checks for touched code.
- Run the focused 2D fiber-strip loader tests.
- Run the training CLI help smoke check.
