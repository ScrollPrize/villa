# Plan: Single-Threaded BLAS in Pyramid Processes

## Implementation

1. Add a top-level, picklable pyramid worker initializer that forces common
   BLAS/OpenMP environment variables to `1` and applies
   `threadpoolctl.threadpool_limits(1)` to already loaded native runtimes.
2. Before creating a multiprocessing pool, temporarily set those environment
   variables in the parent and enter a threadpoolctl limit context. This
   covers both fork inheritance and spawn-time library initialization.
3. Construct the pool with the initializer, then restore the parent environment
   and native thread limits after the pool closes, including worker exceptions.
   Retain the worker controller for the worker lifetime, and apply the same
   native limit to serial pyramid work. Do not change the requested or automatic
   process count.
4. Make progress-thread shutdown exception-safe and print the actual pool size
   plus the one-native-thread policy when pyramid work starts.
5. Add `threadpoolctl` to Lasagna's preprocessing dependencies.

## Testing

- Unit-test that a multiprocessing worker reports BLAS/OpenMP environment
  limits of one and, where a supported native runtime is loaded, no configured
  thread count above one.
- Verify the parent environment/native limits and progress thread are restored
  after success and worker failure.
- Verify the requested worker/process count still reaches the pool unchanged.
- Run the focused pyramid tests and Python compilation checks.

## Spec update

Specify that pyramid process parallelism and native BLAS threading are separate:
the worker count may use all available CPUs, while each process is constrained
to one native compute thread and parent limits are restored afterward.

## Docs update

Document automatic worker selection, the one-native-thread-per-worker policy,
and `--pyramid-workers` as a performance override rather than a correctness
requirement.

## Changelog

Record elimination of process-by-BLAS-thread oversubscription during pyramid
construction.
