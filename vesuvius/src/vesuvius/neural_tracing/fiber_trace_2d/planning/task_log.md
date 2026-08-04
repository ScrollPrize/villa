# Task log: process-parallel native accumulation

## Baseline

- Current accumulation is synchronous in the coordinator. Real-run
  `commit_sum` was 14.1 s for 294 processed tiles within 68.6 s inference.
- Controlled four-channel 32x256x256 ring, eight-addition, five-iteration
  benchmark: NumPy float32 0.0443 s mean with 32 MiB backing; NumPy float16
  0.2931 s with 16 MiB backing (6.6x slower).
- Target CPU is dual-socket Intel Xeon Platinum 8480+ (224 logical CPUs) with
  AVX-512F, F16C, and AVX-512-FP16. The implementation must remain portable to
  Ubuntu/macOS and amd64/arm64 through runtime dispatch and scalar fallback.

## Plan review

Independent review required precise half-to-float add semantics, ISA-matched
runtime dispatch, stable per-worker FIFO ownership, nonblocking submission with
ack pumping, provisional activity failure safety, coordinator-owned ring
generations, all-scale slot reference counts, flush-frontier invariants,
strided/unaligned/tail validation, inspectable/forceable native backends,
explicit cache/queue cleanup, backpressure diagnostics, adversarial ordering and
failure tests, and separate kernel/pipeline benchmarks. The plan incorporates
these requirements.

## Implementation

- Added `accumulator_add.cpp` as a second pybind11 extension. It accepts
  arbitrary positive Y/Z strides with contiguous X rows, releases the GIL,
  supports float16 and float32 destinations, and has forceable `scalar`,
  `avx512`, and `auto` dispatch modes. AVX-512 is isolated behind GCC/Clang
  target attributes and runtime AVX-512F+F16C detection; no global ISA flag is
  used.
- Added persistent spawn-context accumulator processes to the authoritative
  shared runner. Queue items contain only shared-slot/mmap descriptors and
  slices. Stable integer spatial ownership plus one bounded FIFO per worker
  serializes every output chunk without locks.
- The coordinator owns ring generation and activity metadata, retains result
  slots until all task acknowledgements, pumps acknowledgements under queue
  backpressure, and gates reservation at Z-row transitions. Activity is
  committed only after successful worker completion, before canonical progress
  and flush handling.
- Added `--accumulator-workers` to both Fiber and Lasagna frontends, defaulting
  to `min(CPU count, 32)`; zero uses the synchronous path. Added startup/backend
  and task/work/queue/wall/rate diagnostics.
- Product rings now default to float16; weights remain float32. Float32 remains
  selectable explicitly.

## Validation and measurements

- Manual baseline-ISA build (Python 3.14, GCC, pybind11 headers from the local
  build cache) succeeded; runtime backend on the Xeon 8480+ is `avx512`.
- Exhaustive 65,536-value binary16 input validation for one float32 addition
  matches NumPy bit-for-bit for all finite outputs in both forced scalar and
  automatic AVX-512 modes; NaN masks also match. This found and corrected an
  initial scalar subnormal exponent error.
- Strided Y/Z, unaligned X, and 37-element tail views match NumPy for float16
  and float32.
- Focused shared-runner regression passed: synchronous versus two CPU device
  workers plus two accumulator processes produced exactly equal output chunks
  with float16 product rings; scratch mmaps were cleaned.
- Kernel benchmark command used a `(16,512,512)` float16 destination and
  float32 source, eight repeated adds, seven iterations. NumPy median was
  142.507 ms (mean 142.316, min 141.391, max 143.150); AVX-512 median was
  7.653 ms (mean 7.654, min 7.644, max 7.663), an 18.6x median kernel speedup.
  Forced portable scalar median was 331.671 ms. This is a kernel benchmark, not
  an end-to-end volume throughput claim.

## Deviations and limitations

- The serial baseline continues through the existing `_accumulate_group`
  coordinator routine instead of being mechanically expressed as process task
  descriptors; it does use the same native add primitive. This keeps the
  zero-worker diagnostic path small while exact serial/process output coverage
  validates the two planners.
- The environment lacks `pytest`, so pytest suites were not run. Focused
  `unittest` cases and Python compile checks passed. A full unittest-module run
  was stopped after it made no progress in an unrelated early test; the focused
  affected tests were rerun individually.
- No representative full-volume eight-GPU run was launched by the agent. The
  new accumulator diagnostics are intended for the user's next real inference
  comparison; no end-to-end speedup is claimed here.
