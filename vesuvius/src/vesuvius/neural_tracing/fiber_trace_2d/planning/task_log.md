# Task Log: Bound BLAS Threads in Pyramid Workers

## Finding

- `workers=0` becomes `multiprocessing.cpu_count()` (128 here).
- Every process can initialize an OpenBLAS pool of 128 threads, creating a
  theoretical 16,384 tasks and exhausting `RLIMIT_NPROC`.
- The process count itself is retained by request; native numerical libraries
  must be constrained independently to one thread in every worker.
- Independent review required exception-safe environment/native-limit and
  progress-thread restoration, a worker-lifetime controller, the same policy
  for serial work, macOS/NumExpr environment variables, and actual pool-size
  logging.

## Implementation

- Added an exception-safe native-runtime context covering serial and parallel
  pyramid work. It temporarily sets common native thread environment variables
  to one and uses `threadpoolctl` for already loaded runtimes.
- Pyramid pools initialize every worker with a lifetime-retained native thread
  limit. Pool process selection is unchanged and logs the actual pool size.
- Pyramid progress threads now always stop and join when a worker raises.
- Added `threadpoolctl>=3.1` to Lasagna preprocessing dependencies.

## Validation

- Four focused tests passed: parent restoration after failure, progress-thread
  cleanup, unchanged pool-size selection, and a real two-process native-runtime
  check reporting no loaded runtime above one thread.
- Python compilation and `git diff --check` passed.
- Two pre-existing Zarr-backed pyramid worker tests did not reach assertions in
  this checkout and timed out after 60 seconds, consistent with the previously
  observed local Zarr test-fixture stall. No large inference was started.
