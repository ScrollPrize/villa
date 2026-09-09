# TODO: debug Fiberlet cross-toolchain differences

## Observed divergence

- One focused GCC Release run generated 2,526 incident Fiberlets.
- Clang Debug consuming the same persisted anchors generated 2,528.
- Both staged reductions retained 2,275 Fiberlets.
- The exact two candidates and the accept/reject predicates that diverged were
  not identified. Do not describe the cause more specifically than this until
  it has been reproduced and instrumented.

## Investigation

1. Reproduce the focused workload with GCC and Clang from the same source data
   and generation settings.
2. Diff canonical Fiberlet IDs and isolate every candidate present in only one
   output.
3. For those candidates, record the raw value and threshold margin at every
   binary decision, including corridor, support, angular, merge, and route
   acceptance predicates.
4. Check compiler contraction, evaluation order, vectorization, and library
   math only after locating the first divergent decision.
5. Replace sensitive decisions with an explicitly specified portable domain,
   preferably quantized integer or fixed-point comparisons with deterministic
   tie behavior. Do not use an unspecified epsilon band as the cache contract.
6. Verify equivalent candidate IDs and payloads across GCC/Clang,
   QuickBuild/Release, amd64, and arm64 before declaring the producer portable.

## Current policy

Producer compiler and build configuration remain diagnostic metadata but are
excluded from Fiberlet cache fingerprints and compatibility. Version-3 caches
are intentionally reused across those producers until this investigation is
completed.
