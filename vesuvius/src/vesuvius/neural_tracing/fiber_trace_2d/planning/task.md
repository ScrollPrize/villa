# Task: preprocess a whole fiber volume

Add a `vc_fiberlets` preprocessing command that generates anchors and
fiberlets for every relevant spatial chunk of a fiber inference volume.

- Scan only the canonical presence array to determine sparse eligibility.
- Do not process an output chunk when every overlapping input presence chunk
  is missing or decodes to all zero.
- Generate a durable, resumable float anchor-cache Zarr first.
- Keep that anchor cache after successful completion.
- Generate a second, final Zarr containing anchors, fiberlet prefixes, and
  routes together using the compact default representation: float positions,
  compact directions, and fixed sqrt-density `uint16` costs.
- Schedule work in Z/Y/X order. Parallel work may complete out of order.
- Reuse the existing anchor extraction, fiberlet extraction, dependency,
  serialization, and generated-cache implementations.
- Retain `--presence-floor` as the observation threshold. A cell with no
  usable observation at or above the floor must exit before fitting.
- Do not persist an active-chunk index or dataset/per-chunk completion markers.
  Reconstruct expected work from the input presence volume on every run and
  determine cache/output completeness solely from the payload chunks found.
- Keep every individual cache/output payload publication atomic. Recover
  partial final triples by reusing matching files and generate missing files.
- Remove stale atomic-write temporary files from the anchor cache and final
  output on resume and after preprocessing finishes.
- Report anchor and final-output progress on one live terminal line refreshed
  about once per second, with a persistent newline at least once per minute.
  Include completed/total chunks, percent, elapsed time, rate, ETA, and
  current/projected payload size.
