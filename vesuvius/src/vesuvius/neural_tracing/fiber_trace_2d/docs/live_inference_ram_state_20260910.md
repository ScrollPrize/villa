# RAM-only demand inference bookkeeping

`lasagna/demand_predict3d.py` no longer imports or opens SQLite. The coordinator's
existing dictionaries, sets and deques own tile state, chunk dependencies,
pending/applied contributions, priorities and publication acknowledgements.
Workers still exchange shared-memory slots and acknowledgements. No persistent
queue, per-tile transaction, replacement journal or per-chunk done marker exists.

## Persistent output versus temporary state

The cache retains ordinary Zarr output and static model/source/settings metadata.
`cache_identity.json` is written once on creation; it contains no scheduling state.
The writer lock remains. RAM state starts empty each session. Requested cached
chunks are checked lazily: all sibling files must exist and decode with the
expected shape, dtype and chunk geometry. A successful check is remembered in
RAM. Explicit zero chunks and padded edge chunks are supported; absent files do
not count as Zarr fill. There is no whole-volume scan.

An unidentified cache with payloads is refused, not assigned current provenance.
Legacy database-only caches are also refused without opening/deleting their
database or outputs. Use a fresh identity-keyed cache root. Metadata-only new
Zarr groups can be initialized normally. No legacy migration was implemented.

Legacy temporary chunk-major/flat mmap accumulation may remain under an output
product's hidden `.live_tmp/session-*` directory. New reusable result-slot
backing is anonymous/unlinked where supported. Fresh sparse files read as zero
without reservation or initial clearing. No scratch fsync, staged NPZ, or
on-disk contribution records. After writer ownership is acquired, startup
scans known channel `.live_tmp` directories and removes only direct stale
`session-*` directories; symlinks, unrelated files, and published chunks are
left untouched.

Publication still uses the shared finalizer and atomic Zarr writer, with final
file/directory synchronization before notifying readiness. When replacing an
incomplete sibling group, all paths are validated and old siblings removed
before writing the first replacement. This prevents a process interruption from
leaving old/new siblings that falsely appear complete. Ready groups are immutable.

Limitations: this is not an arbitrary power-loss transaction across multiple
files. Without persistent checksums, same-size corruption that remains decodable
is not cryptographically detectable. Incomplete inference is recomputed, not
resumed from scratch buffers. Numerical inference and arrival-order accumulation
were not changed by removing SQLite.

## Validation and profiling

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' \
  -q lasagna/tests/test_demand_predict3d.py lasagna/tests/test_inference_profiling.py \
  vesuvius/tests/neural_tracing/test_live_inference_profile.py

/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

CPU regression result: **60 passed in 24.79s**. Tests prohibit SQLite connections,
check restart/zero/edge chunks, corruption, interrupted sibling publication,
unidentified payload preservation, symlink safety, priority, exact identity-model
outputs, shared slots and multiworker backpressure. Independent review passed
after correcting two startup/publication path-safety findings.

The short benchmark uses the same locked PHerc0139 reference-intersection workload
and classic checkpoint as the previous instrumented SQLite run. Eight H100s,
512-cubed input tiles, 128 readers, 256 input slots, 128 result slots,
chunk-major accumulation, no prefault, bf16 model and float16/float32 accumulation
remain unchanged. Model-only controls/warmup precede a 40-second measurement.
This is not the full 100-direction tracing evaluation. See the managed evaluation
document for launch/queue details. The same host/environment/native build is used;
this change is Python bookkeeping, with no C++ rebuild or kernel changes.

SQLite baseline `20260910T145617665960`: 177.9999 fully accumulated 128-cubed
input-equivalent tiles/s. SQLite updates consumed 27.5349s wall but only 0.0354s
calling-thread CPU, so the instrumented wait was near the database calls without
proving disk I/O as its cause.

RAM-only run `20260910T151437745034` completed and released all eight GPUs.
Reports are under `/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/`.
Each result below is one warmup followed by one approximately 40-second window,
not a repeated-trial statistical speedup claim. Completed means fully accumulated
input tiles, not fully published Zarr throughput. Initial/final backlogs are in
the report and differ between runs.

| Metric | SQLite baseline | RAM-only |
|---|---:|---:|
| Measured seconds | 40.2697 | 40.6604 |
| Fully accumulated 512³ tiles | 112 | 113 |
| Fully accumulated 128³-equivalent tiles/s | 177.9999 | 177.8633 |
| GPU results received (512³ tiles) | 207 | 210 |
| Published output chunks | 22 | 20 |
| Coordinator admission seconds | 27.6494 | 6.5181 |
| Coordinator background seconds | 3.0077 | 19.3782 |
| Coordinator event handling seconds | 9.2100 | 14.3581 |
| Tile-state update seconds | 27.5349 | 0.000275 |
| Accumulation task mean / p50 / p95, ms | 2.547 / 0.890 / 6.328 | 2.244 / 0.826 / 5.247 |

**No end-to-end throughput gain was measured.** RAM updates are negligible
(286 calls, mean0.960µs, p500.862µs, p951.324µs), but the coordinator still stalls:
executor submission6.439s elapsed versus0.0054s calling-thread CPU; budget checks
10.431s inclusive elapsed; background completion/submission handling19.378s.
These nested numbers must not be added together. Lazy cache completeness checks
took0.225s total across11,386 calls, so they are not the dominant measured phase.
Four publication threads completed20 chunks; inclusive publication mean6.744s,
p507.065s,p9512.191s includes all finalization, scheduling and storage work, not
pure Zarr encoding or fsync time. Final-sync time summed across those calls3.078s.

The shifted stalls are consistent with wider thread/lock/GIL contention, but
this profile does not isolate which mechanism. Do not interpret the old SQLite
wall time as proof of slow database disk traffic. Next profiling should isolate
coordinator waits and reader/native allocation contention. The current task
does not alter reader conversions, native kernels, executor design or budget
checking policy. Full100-direction evaluation remains unrun.
