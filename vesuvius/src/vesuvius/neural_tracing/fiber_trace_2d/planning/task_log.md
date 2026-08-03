# Task log: TensorStore whole-volume inference prefetch

## Findings

- Shared whole-volume inference currently opens input with Python `zarr.open`
  and reads each tile with one `np.asarray(array[z0:z1,y0:y1,x0:x1])` call.
- Multi-GPU mode uses at most `devices * slots_per_gpu` total events. With eight
  GPUs and two slots this means active GPU/result work and read-ahead share only
  sixteen events; there is usually about one extra tile per busy GPU.
- The existing `lasagna.tensorstore_omezarr` module provides proven local Zarr
  driver/context construction with explicit cache, file-I/O, and data-copy
  concurrency controls. TensorStore `.read()` returns pollable asynchronous
  futures and preserves source dtype.
- Prefetch-array memory is dtype dependent: one 256-cubed uint8 tile is 16 MiB,
  uint16 is 32 MiB, and an eight-byte dtype is 128 MiB. With four tiles per
  eight GPUs the completed/outstanding-array bound is therefore 0.5, 1, or
  4 GiB respectively, separate from shared slots and TensorStore cache.

## Review, implementation, and validation

Independent review required and incorporated: context construction must follow
all child starts; single-device inference also needs bounded read-ahead; every
outstanding read counts as a full-tile memory allocation; backend failures are
explicit rather than silent fallback; slicing oddities, local Zarr-v2
eligibility, cancellation/drain, diagnostic wall metrics, spawn portability,
and real controlled before/after measurements must be covered.

## Implementation

- Added exact shared clipped-bbox/read-finalization helpers and a lazy
  `_TensorStoreTileReader` using the existing `TensorStoreConfig`, context, and
  local Zarr-v2 opener. TensorStore import/open failures are explicit.
- Fiber and Lasagna front ends default to TensorStore with four prefetch tiles
  per selected GPU, 4 GiB cache, 16 file-I/O threads, and four copy/decode
  threads. Python Zarr remains explicit fallback.
- Multi-GPU read events are bounded independently of the two GPU/result slots
  per GPU. Reads do not reserve shared slots; ready arrays enter shared memory
  only when both slot types and a GPU queue are available. Single-device mode
  maintains the same bounded future window while model forwards run.
- Added backend/read byte and latency totals, live/ready high-water marks, copy
  time, and input-starvation time. Removed the coordinator's GPU-result
  get-and-requeue operation so it handles the descriptor directly.
- Queue feeder threads are now closed and joined on normal completion and
  startup failures, preventing the shared inference queues from being reported
  as leaked multiprocessing semaphores at interpreter shutdown.
- Added a reusable reader benchmark script with warmups and mean/p50/p95/min/max
  output.

## Validation and limitations

- Compilation passed for the shared runner, both front ends, benchmark helper,
  and touched tests. `git diff --check` passes.
- Five focused tests passed in 5.39 s: TensorStore/Python-Zarr bbox equivalence
  with reflect padding, serial output equivalence, spawned multi-device output
  equivalence, the existing serial/multi-device exactness test, and Lasagna CLI
  forwarding. TensorStore exercised little-endian uint16 and real Zarr-v2 raw
  chunks.
- Fiber CLI help exposes every new option; its pytest forwarding test compiles,
  but pytest is absent from the active venv.
- The benchmark command against the representative 2.2 TiB local level was
  attempted with 16 tiles, one warmup, and three iterations. In this sandbox,
  Python Zarr `zarr.open` stalled before printing metadata (even tiny newly
  created Zarr arrays exhibit this sandbox-only stall), so no honest controlled
  before/after timing could be collected here. The benchmark script and normal
  input metrics are ready for the user's unsandboxed run. This is an explicit
  deviation from the requested performance protocol.
- Fully-outside reads preserve historical uint8-zero behavior by construction.
  Size-one reflect-axis error behavior follows the shared NumPy finalizer but
  does not yet have a dedicated test. Import/open/read-error and interrupt
  cancellation paths are implemented but not exhaustively fault-injected.
