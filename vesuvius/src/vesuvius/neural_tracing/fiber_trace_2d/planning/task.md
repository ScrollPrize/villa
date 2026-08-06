# Task: manager no-prefetch download behavior

When managed automatic prefetch is enabled, keep passing `--no-download` to
the Fiber or Lasagna inference backend. When the user explicitly passes
`las_manager inference run --no-prefetch`, omit backend `--no-download` so the
normal inference auto-download path remains available.
