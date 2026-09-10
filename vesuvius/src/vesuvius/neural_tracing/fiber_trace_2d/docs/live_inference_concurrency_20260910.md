# Concurrency instrumentation (paused)

User paused this work to commit and integrate the GPU manager branch.
Current shared queue snapshot and short-profile250ms series/distributions are
implemented, but no instrumented benchmark has run and live JSONL is not yet
implemented. Actual timestamps/gap distributions reveal coordinator sampling
delays; sampled means are not guaranteed time-weighted. GPU submitted includes
queued/executing/completed-uncollected results, not hardware utilization.

Confirmed loader: NativeTileReader → Vc3dCoordinateSampler.sample_block_zyx →
VC3D native block sampler/cache. This reuses the native sampling backend, not
the entire training prefetch/batching pipeline. It returns sampled float32 and
converts back to source uint8/uint16 before shared-slot dispatch.

Current profile has8 inputs (8devices×1slot),8 reader threads and128 result
slots. Read submission reserves an input slot first. Hundreds of unique queued
tiles therefore do not imply hundreds of concurrent reads. Increasing result
slots alone does not change that cap. Native cache operations/HTTP requests
inside one tile read are not counted by the Python queue snapshot.

The short profile prequeues1599 reference-intersecting output chunks; it does
not run100 actual trace directions. Real trace RPC waiters, deduplicated chunks,
unique queued inference tiles and submitted tile reads are different units.
Read futures are classified once as queued/running/done-awaiting-dispatch.
Input accounting: free + loading + GPU-owned = input capacity.
Result accounting: free + GPU submitted/uncollected + accumulating = result capacity.
Snapshot reads metadata only, without ready()/fingerprinting/future.result/qsize.

Resume plan: integrate snapshots into live1s bufferedJSONL including drain/failure,
then run the existing approved classic profile command and analyze admission
versus loading/dispatch. Do not infer a measured concurrency improvement yet.
