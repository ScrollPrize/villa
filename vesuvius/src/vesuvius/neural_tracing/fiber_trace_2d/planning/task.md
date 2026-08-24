# Task: make shared tiled inference band scheduling reliable

Replace the current loosely coupled multi-GPU event/accumulator/flush state
machine in `lasagna.tiled_predict3d.run_tiled_inference_3d` with an explicit
bounded Z-band lifecycle.

The shared Fiber and Lasagna runner must retain lazy tile generation,
TensorStore read-ahead, multi-GPU inference, process-parallel accumulation,
sparse/resume skipping, circular mmap storage, and overlap of exactly one
completed band's flush with computation of the following band. It must not
copy a full band or accumulator into RAM or a second mmap.

Every inference/accumulation result slot must have explicit ownership and be
returned before its Z band completes. A completed flush batch must be finalized
and release its ring generations immediately rather than depending on a later
tile-commit transition. Quiescent states with unresolved work must fail with a
complete stage/slot/band diagnostic instead of polling an unrelated queue
forever.

Production rerun evidence shows the first implementation can still become
quiescent mid-band: every GPU and accumulator worker is alive and blocked
waiting for new input while the coordinator polls indefinitely. A Python-aware
production stack and locals capture proves that no completion was lost: the
canonical accumulation frontier was a ready tile, while all 16 result slots
were owned by later GPU-complete tiles. Those later tiles could not accumulate
before the frontier, and the frontier could not enter the GPU without a result
slot. Fix this circular resource dependency at admission rather than adding
completion retries, acknowledgements, queue replacement, or a watchdog.

Keep the established bounded process queues. Reserve one input/result slot pair
for the canonical accumulation-frontier tile whenever that tile has not yet
entered GPU processing; later ready tiles may consume all remaining slots.
Restore the normal local-input path's pre-live-fetch atomic discovery/read
submission. Keep live fetching as a bounded upstream materialization ledger
feeding the same reader/GPU/accumulator/flush stages. Track exact per-worker
task and shared-slot ownership, validate it continuously, and make a violation
of the frontier-capacity invariant fail immediately with exact state.

Remove the incorrect elapsed-time/process-limit diagnosis and flush-only
watchdog. Preserve the established single native BLAS/OpenMP/codec thread and
bounded Zarr executor per process so process workers cannot multiply the host's
thread count.
