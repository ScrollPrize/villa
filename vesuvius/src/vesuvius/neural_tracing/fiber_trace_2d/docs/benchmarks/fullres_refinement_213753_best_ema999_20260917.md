# Full-resolution refinement best EMA999 evaluation — 2026-09-17

## Result

The five-scroll, 50-fiber no-embedding evaluation completed at native model
resolution. It produced all 180 directed case artifacts and exactly matched the
prepared 6,867.627253 mm directed-reference workload.

| Collection | Directions | Restarts | Restarts/mm | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 62 | 64 | 0.054489 | 18.070 | 3.191 | 0.149 | 45.742 |
| PHerc0332 | 6 | 4 | 0.009408 | 85.032 | 16.777 | 2.736 | 129.836 |
| PHerc1229 | 10 | 9 | 0.022107 | 40.711 | 20.519 | 0.713 | 39.817 |
| PHerc1451 | 4 | 0 | 0.000000 | 48.369 | 12.092 | 11.129 | 13.056 |
| PHercParis4 eval | 82 | 27 | 0.007387 | 130.539 | 21.083 | 0.397 | 109.763 |
| PHercParis4 fibers | 16 | 23 | 0.019873 | 48.223 | 5.578 | 0.386 | 120.454 |
| **Combined** | **180** | **127** | **0.018493** | **53.653** | **8.922** | **0.241** | **88.464** |

The mean uses `combined-directed-distance-per-restarts-plus-one-v1`; quantiles
are over the 307 reliability segments. Trace replay took 106.502 s.

## Reproduction

Checkpoint:

`/ephemeral/hendrik/fiber3d_train2/evaluations/fullres_refinement_213753_best_ema999_off_20260917_v1/frozen/live_inference/_models/5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e/snapshot.pt`

SHA-256:

`5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`

Config:

`vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json`

Command:

```bash
/home/hendrik/.venv_las/bin/python \
  /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py \
  full \
  --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
```

Result:

`/tmp/fullres_refinement_213753_best_ema999_off_20260917_v22_chunk64_file_results/full`

Persistent inference namespace:

`fullres_refinement_213753_best_ema999_20260917_v2_chunk64`

## Inference and accumulator behavior

This successful resume reused already published cache chunks from earlier
diagnostic attempts and inferred only missing work, so its wall time is not a
cold-cache throughput benchmark. It inferred 1,066 tiles in 382 GPU batches and
published 412 new chunks in 143.056 s. Batch sizes were 112x1, 40x2, 46x3 and
184x4. All eight GPUs stayed within 48,932,847,616 reserved CUDA bytes each.

Runtime settings were 256-cubed input tiles, 64-cubed output chunks, BF16 GPU
inference, float16 accumulation, 128 input workers, 128 accumulator/publication
workers, 96 input slots, 32 result slots and a maximum GPU batch of four.

The measured partial-chunk working set exceeded the former 131,072-record
(576-GiB virtual) hot arena during a diagnostic run. That caused approximately
19,000 records to enter the sparse-file spill path and made GPUs bursty while
the spill copies proceeded. The final run used 163,840 lazily committed hot
records (720-GiB virtual capacity) and did not spill.

The first attempts also exposed a separate failure: all eight GPU workers could
SIGBUS simultaneously while copying output into 128 large POSIX shared-memory
result slots. The host already retained about 987 GiB of shared memory from a
paused training job, so POSIX shared-memory capacity was not dependable. The
corrected transport uses 32 reusable result slots in one ordinary sparse
file-backed mmap under the cache's hidden accumulator directory. Workers map
the same fixed offsets directly; there is no serialization, per-tile file,
explicit fsync or copy through the coordinator. Thirty-two slots preserve the
maximum useful concurrency of eight GPUs times four tiles. The 96 smaller input
slots remain in POSIX shared memory. The successful summary records
`result_slot_backing=file-mmap` and a total interprocess slot footprint of
18,924,699,648 bytes.

## Validation

```text
native replay groups: 23 passed; 85 passed; 17 passed
Python replay suite: 283 passed, 8 skipped in 153.41 s
focused file-mapped/arena tests: 4 passed, 70 deselected in 8.98 s
git diff --check: passed
py_compile for changed inference/evaluation modules: passed
```

The broader multiprocessing backpressure selection was attempted separately but
was interrupted while its setup was blocked on this heavily loaded host. It is
not counted as passed; the approved complete replay suite above is the release
gate and completed successfully.
