# Task: adaptive remote download parallelism

Adapt normal interactive VC3D remote downloads to measured bandwidth and
encoded chunk size.

- Keep up to 64 source-fetch workers available.
- Start at two admitted downloads.
- Estimate bandwidth from the latest `parallelism * 4` successful encoded
  chunk downloads.
- Set the next parallelism to
  `ceil(bandwidth * 0.25 seconds / average encoded chunk bytes)`, clamped to
  `[2, 64]`.
- Use the same estimate in the existing network status display.
- Do not add exploratory concurrency changes.
- Preserve fixed concurrency for explicit callers, tests, and prefill jobs.

Changing admission must not change chunk priority order, cache behavior,
decoding, or rendered values.
