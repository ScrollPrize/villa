# CT judge data preparation: September 25, 2026

## September 26 worker crash fix

The `direct_ct_judge_run2` worker core dump after step 12,700 located SIGSEGV
in `interpolate_supported`, reading a point through an uninitialized bucket
permutation entry. Its saved input coordinates and chunk assignments were valid,
but the internal bucket counts disagreed with those assignments. Replaying the
inputs alone did not reproduce the corruption. The sampler now visits contiguous
runs of points sharing a chunk, eliminating that scratch permutation, and enables
native bounds checks. Loading the chunk once per run avoids the per-pixel typed
list lookup overhead of the initial fix.
Chunk discovery, corner order, precision, and missing-support semantics are unchanged.
The historical chunk-order timings below therefore describe the earlier kernel.

Validation used the recovered 129-by-129 plane (16,641 points, six 128-cubed
chunks) from local `s1_ds2.zarr/0` in the production Python 3.14 environment.
Patched values and support were bit-identical to the prior kernel. Production
Python 3.14 / Numba 0.66, fast math disabled, 50 warmup calls followed by 1,000
serial timed calls per variant gave the following milliseconds (mean / p50 / p95):

| Measurement | Original kernel | Initial fix | Contiguous runs with bounds checks |
| --- | ---: | ---: | ---: |
| Interpolation kernel | 0.356 / 0.356 / 0.361 | 0.833 / 0.835 / 0.839 | 0.322 / 0.322 / 0.324 |
| Full `sample_supported`, warm chunk cache | 0.590 / 0.590 / 0.599 | 1.089 / 1.078 / 1.192 | 0.557 / 0.557 / 0.567 |

These measure CPU sampling, not complete training throughput. A cProfile pass of
500 calls per variant attributed 93–96% of time to `sample_supported` itself
(including its native kernels); Python chunk-cache lookup took about 1% or less.
The temporary harness, recovered inputs, JSON report, and profiles are under
`/tmp/fiber-worker-core-42bvz8q9`. Reproduce on this machine with:

```bash
AGENTS_AGENT_MODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/direct-numba \
MPLCONFIGDIR=/tmp/direct-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python \
  /tmp/fiber-worker-core-42bvz8q9/benchmark_runs.py
```

Another 3,000 real-CT calls passed with three threads and a one-chunk reader cache
to force concurrent mmap eviction (1.225 seconds). The targeted suite passed 94
tests with one CUDA skip, including invalid-index rejection and concurrent mmap
sampling:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_CACHE_DIR=/tmp/direct-numba \
MPLCONFIGDIR=/tmp/direct-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest \
  tests/test_ct_judge.py tests/test_fast_sample.py -q -o cache_dir=/tmp/ct-judge-pytest
```

Restart existing workers to load this fix. The crash dump identified the failing
access; the bounded checks above do not establish long-run stability.

The reported 30-second training stalls were caused by CPU slice gathering.
A profile of eight production microbatches spent 111.36 of 114.56 seconds in
`sample_supported`, including 91.19 seconds in `numpy.unique` (85.07 seconds
sorting). Each plane separately sorted all pixel chunk coordinates for each
of eight interpolation corners.

The sampler now buckets required corners by chunk in a Numba loop, reuses the
previous chunk lookup for adjacent corners, and gathers each chunk's requested
values together. The existing scalar interpolation kernel and accumulation order
remain unchanged; fast math is disabled. The reader still requests only chunks
containing positive-weight interpolation corners and preserves missing-data masks.

## Measurements

Production Python 3.14 environment, same local `s1_ds2.zarr/0` CT, 257×257 planes,
0.5 trace-voxel pixel spacing, batch 8 / microbatch 2, seed 0, synthetic fraction
0.25, and the fixed recovery bank from `output/direct_ct_judge_run1/config.json`.
No CT upsampling, resolution reduction, label changes, or sampling changes.
These are CPU data-preparation timings, not complete GPU training-step timings.
The existing training process was left running during measurements. OS page caches
were not flushed; reader caches started empty in each process.

| Eight identical serial microbatches | Before | After |
| --- | ---: | ---: |
| Mean | 14.320 s | 1.351 s |
| Median | 17.955 s | 0.780 s |
| p95 | 27.270 s | 3.645 s |
| Minimum | 0.020 s | 0.019 s |
| Maximum | 31.164 s | 4.635 s |
| Previously slowest microbatch | 31.164 s | 1.805 s |

The after-run includes the new Numba kernel compilation in its first microbatch
(4.635 seconds). All eight complete batch tensor hashes match exactly, including
follower images, judge images, targets, metadata, masks, and allocation counters.
Both runs sampled 38,638,665 pixels and loaded 410 chunks, with zero downloads.

A separate four-worker run consumed 32 microbatches (eight effective updates).
After 2.208 seconds starting workers, data wait per effective update had a mean
of 1.306 seconds, median 1.350, p95 1.808, minimum 0.662, and maximum 1.809.
This measures waits in `next(loader)`; tensor hashing between reads gives workers
some additional prefetch time. No GPU work is included in this benchmark.

## Reproduction

From `fiber_follow`, using the existing production environment:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR=/tmp/ct-judge-numba MPLCONFIGDIR=/tmp/ct-judge-mpl \
PYTHONPATH=../../.. /home/sean/Documents/villa4/vesuvius/.venv/bin/python \
  scripts/benchmark_ct_judge_data.py \
  --config output/direct_ct_judge_run1/config.json \
  --workers 4 --count 32 --out /tmp/ct-judge-data-four-workers.json
```

Use `--workers 0 --count 8` for a serial profile and per-batch hashes. The script
writes a JSON report and a sibling `.prof` file without modifying the run.
The original before/after serial measurements used the equivalent temporary
harness `/tmp/bench_ct_judge_data.py`, with `--baseline --count 8 --out
/tmp/ct-judge-data-before` and `--count 8 --out /tmp/ct-judge-data-final`, respectively.
That harness loads the preserved pre-change sampler from
`/tmp/ct-judge-volume-before.py` for the baseline. Both commands used the same
Python, PYTHONPATH and cache variables above, plus `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

Regression coverage: 104 passed, 3 CUDA tests skipped across `test_direct.py`,
`test_ct_judge.py`, `test_fast_sample.py`, `test_decode_store.py`, and
`test_compilation.py`. The multiprocessing decode test was rerun outside the
sandbox after the sandbox blocked its worker socket. New scalar-reference tests
cover irregular chunk dimensions, volume edges, exact voxel/chunk boundaries,
missing chunks, zero-weight corners, and empty input, with exact output/mask and
requested-chunk comparisons.

Existing training workers retain the old imported sampler. A restart or resume
is required to use this change.

## Second pass: September 25, 2026 (evening)

After the Numba bucketing change the trainer still waited on data (about 0.25 s
per update beside a 0.20 s GPU step). Changes, all with bit-identical batches:

- `sample_supported` finds chunks with a single-chunk fast path, then reads
  corners directly from the loaded chunks, visiting points chunk by chunk
  (no per-corner request arrays). 8.4 → 2.4 ms per 257² plane on 426 captured
  production planes; outputs and support masks are exactly equal to the previous kernel.
- The three planes of each record are sampled on a per-process three-thread pool
  (kernels are `nogil`). Reader counters are updated under the reader lock.
- `synthetic_contact` skips fiber pairs whose bounding boxes are at least 6 apart
  (no contact below 6 is possible), caches per-fiber KD-trees, and queries with
  `distance_upper_bound=6`. 516 → 6–11 ms per call; identical contexts and RNG state.
- Plane grids and markers are cached per slice geometry.

| Measurement (production config, 4 workers) | Before | After |
| --- | ---: | ---: |
| Loader data wait per update, `benchmark_ct_judge_data.py --workers 4 --count 48` | 1.31 s | 0.22 s |
| Trainer update, including GPU (steps 9–520, instrumented short run) | ≈1.5 s (estimated: data wait + 0.2 s GPU) | 0.32 s |
| Trainer startup to update 5 | 129 s | 34 s |

Sixteen serial microbatch hashes (`--workers 0 --count 16`) match the pre-change run.
