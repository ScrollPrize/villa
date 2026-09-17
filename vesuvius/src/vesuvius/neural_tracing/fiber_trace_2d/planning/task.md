# Task

Run the established full 50-fiber/five-scroll evaluation on the best EMA999
checkpoint from training run
`5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema_20260916_213753`.
Use the current no-embedding tracer protocol and the existing pre-approved
evaluation runner, persist the immutable checkpoint/config/result provenance,
and report combined and per-scroll metrics.

The full-resolution run must also replace its disposable full-volume sparse
accumulators with a reusable slot arena. Keep a fixed, large hot pool in shared
memory. Spill only evicted incomplete output chunks into reusable fixed-size
slots in large file-backed mmap arena segments under the existing `/ephemeral`
cache. Spill addressing is by allocated slot, never by global volume
coordinate. Reuse freed hot and spill slots after publication/promotion, grow
the spill arena only beyond its prior concurrent high-water mark, and impose no
hard limit on the number of incomplete chunks. Do not create one backing file
per output chunk and do not move scratch to another filesystem.

## Completion

The approved v22 full run completed all 180 directions. The final transport
keeps 96 small input slots in POSIX shared memory and places the 32 large,
reusable GPU result slots in one direct regular-file mmap, avoiding host-global
shared-memory exhaustion without staging or additional copies. Complete metrics,
provenance, timings and validation are recorded in
`../docs/benchmarks/fullres_refinement_213753_best_ema999_20260917.md`.
