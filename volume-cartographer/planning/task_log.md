# Task Log

## Boundary

- Three feature bases are predeclared: data-read cost only, split read/write
  misses only, and their combination.
- Every fit keeps the existing zero-overlap soft-L1 objective and differs only
  in named passive features.
- Renderer observations cannot fit or select a feature basis.
- Headline renderer reporting separates worker 1 from pooled workers 2--7.
- CPU frequency restoration remains deferred at the user's request.

## Crossed Access-Subset Experiment

- Added 96 generic cache-line cases: one/eight read and one/eight write pairs,
  plus crossed `r1w1`, `r8w1`, `r1w8`, and `r8w8` subsets. Together with the
  original 36 density cases, the clean fit contains 132 new cases and 261
  total synthetic records.
- The combined nine-feature matrix is rank 9/9 with maximum correlation 0.475
  and condition 3.93. All nine coefficients are positive, so the former
  read/split collapse is removed.
- The combined access coefficients are `Dr=0.091087 ns`, `Dw=0.072237 ns`,
  `D1mr=0.066331 ns`, `D1mw=0.152123 ns`, `DLmr=0.042316 ns`, and
  `DLmw=0.047179 ns`.
- Fixed-frequency monitoring passed. Maximum five-run range was 33.63% because
  isolated native samples were slow; the worst case had 2.52% MAD and the
  fit uses medians. This remains a diagnostic rather than a promoted model.

| Frozen pipeline | Max worker 1 | Max workers 2--7 | Runtime median/RMS | <=20% | Max speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous data reads | -19.22% | -21.27% | 17.45% / 16.82% | 27/35 | 18.74% |
| Crossed matched legacy | -13.55% | -18.12% | 11.11% / 11.68% | 35/35 | 17.97% |
| Crossed data reads | +7.03% | +16.80% | 4.42% / 6.77% | 35/35 | 16.44% |
| Crossed split misses | -11.76% | -16.92% | 9.04% / 10.17% | 35/35 | 17.80% |
| Crossed combined | +7.61% | +17.28% | 3.92% / 6.70% | 35/35 | 16.47% |

- Combined is no longer numerically identical to split-only. Compared with
  read-only it improves median absolute error by 0.49 points and RMS by 0.07,
  but worsens the prioritized maxima by 0.58 points for worker 1 and 0.48
  points for workers 2--7.
- Read-only has the best prioritized maximum-runtime result. Combined has the
  best median/RMS, and every expanded-model row is below 20%.
- Fit observations SHA-256: `0bb952a552629629ec7638e5e593fec551ddfa3e490d1fe6117c4f81484af5ed`.
- Fit report SHA-256: `e22df535a5fe9c8e3cd6bb0c1ba5e471db4067a12871c3d704d35100ba7880b3`.
- Renderer comparison SHA-256: `6d902e2d36e6926e645f8732354a943f1eedbb91c5835f7188224a4fb7e7d8da`.

## User-Directed Renderer Evaluation

- The user does not require the coefficient-bound or leave-one-family-out gates
  to block renderer evaluation. They remain reported diagnostics only.
- The fitted coefficients are frozen from `fit_report.json` before opening the
  renderer comparison. No event or synchronization coefficient may be changed
  afterward.

## Frozen Current Baseline

- Worker 1 maximum runtime error: +29.51%, `fallback_3`.
- Workers 2--7 maximum runtime error: +30.49%, `mixed_shuffled`/7.
- Overall maximum speedup error: 16.68%.

## Opened-Fit Identifiability

- `data_reads`: rank 7/7, condition 26.83, maximum correlation 0.9875 between
  data reads and non-data instructions. It fails the existing 0.98 gate.
- Split read/write misses: rank 8/8, condition 5.82, maximum correlation 0.8525.
- Combined: rank 9/9, condition 27.10, maximum correlation 0.9871.
- The data-read and combined candidates therefore require a generic
  higher-read-density calibration kernel. The gate will not be relaxed.

## Plan Review

- Independent review required schema-based dispatch because new candidates can
  have the same feature count as the existing serialization model.
- Exact feature equations, per-thread nonlinear extraction, fairness controls,
  and synchronization refit parity are now frozen in the plan.
- Two independent read-density families are added so omitting either does not
  remove all evidence for the read coefficient. A separate dense writer
  overconstrains write misses.
- The existing random `cache-write` kernel is not promoted: prior fixed-host
  samples contain approximately 3x timing modes. The new deterministic dense
  writer must pass native stability checks before holdout collection.
- Fresh holdout ordering is implement/test, collect opened fit extension,
  verify identifiability, freeze all inputs, then collect once. No model change
  is permitted after opening it.

## Discarded Opened Fit

- The first density-fit extension reduced maximum parameter correlation for the
  data-read candidate from 0.9875 to 0.6972, proving the kernels provide the
  missing independent event ratios.
- Its shortest native cases were below one millisecond and reached 27.23%
  five-process range, exceeding the predeclared 10% stability gate. No holdout
  was collected.
- Calibration iteration counts are increased about eightfold to put every
  native sample in a multi-millisecond regime. Feature equations and gates are
  unchanged; the corrected opened fit uses a new output directory.

## Stabilized Opened Fit

- Thirty-six cases: `read-four`, `read-eight`, and `write-eight`, each crossing
  16 KiB / 256 KiB / 4 MiB / 12 MiB and three work counts. Five native
  processes and one Callgrind profile were collected per case sequentially.
- Maximum five-run native range: 7.64%, below the predeclared 10% gate.
- Frequency validation passed: target 3,401,000 kHz, mean 3,394,968 kHz,
  observed range 3,347,319--3,401,000 kHz, 15,687 samples.

| Basis | Rank | Correlation | Condition | Bounds | Fit median/RMS/max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched legacy | 6/6 | 0.6527 | 3.55 | 0 | 21.98% / 23.87% / 67.34% |
| Data reads | 7/7 | 0.6988 | 5.07 | 1 | 20.85% / 24.04% / 67.80% |
| Split misses | 8/8 | 0.7573 | 5.04 | 3 | 20.21% / 23.83% / 66.76% |
| Combined | 9/9 | 0.7312 | 5.70 | 3 | 20.21% / 23.83% / 66.76% |

- Data-read candidate coefficients include `data_reads = 0.004352 ns` and
  `data_writes = 0`. Omitting `read-four` moves the read coefficient 346.7%;
  omitting `read-eight` drives it to zero. It fails bound and stability gates.
- Split candidates fit `data_writes = 0`, `D1mw = 0`, and `DLmw = 0`.
  Omitting a read family moves `DLmr` by 45.5--140.6%; omitting the dense writer
  destabilizes boundary write terms. Both fail bound and stability gates.
- The combined candidate also fits `data_reads = 0` and otherwise matches the
  split-only candidate. It adds no explanatory value.
- No candidate passed opened-fit gates. The fresh 40-case holdout was not
  collected. At the user's direction, the gate results were retained as
  diagnostics and all fitted bases were frozen for renderer evaluation.

## Actual Renderer Result

| Pipeline | Max worker 1 | Max workers 2--7 | Runtime median/RMS | <=20% | Max speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous current | +29.51% | +30.49% | 19.22% / 19.99% | 19/35 | 16.68% |
| Density-matched legacy | -19.83% | -21.58% | 17.98% / 17.34% | 24/35 | 18.83% |
| Data reads | -19.22% | -21.27% | 17.45% / 16.82% | 27/35 | 18.74% |
| Split misses | -20.62% | -22.32% | 18.79% / 18.04% | 21/35 | 18.83% |
| Combined | -20.62% | -22.32% | 18.79% / 18.04% | 21/35 | 18.83% |

- The matched legacy refit supplies most of the improvement versus the current
  model, showing that the three generic density families materially improve
  calibration even without changing the feature basis.
- `data_reads` is best on actual renderer measurements: another 0.61-point
  worker-1 and 0.31-point workers-2--7 improvement over the matched refit.
- Split-only and combined predictions are identical and worse than matched
  legacy. The split feature basis should not be adopted from these results.
- The new worst rows are underpredictions: `full_res` for worker 1 and worker 5
  for the pooled headline. All 35 rows are retained in the comparison artifact.
- Frozen model manifest SHA-256: `3c045a612601bbe5cfeebd797f5a0faabab033b255f74ab10d0d1975a6b1a485`.
- Renderer comparison SHA-256: `adfe6fa79448217b92255f0b9a0aef3045ec0998c442776165fab96ef3dc5ba2`.

## Commands And Verification

- Build: `cmake --build volume-cartographer/build-release --parallel 32 --target bench_thread_pool_dispatch`.
- Fit: `python3 volume-cartographer/scripts/calibrate_synthetic_event_features.py --phase fit --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch --base-observations /tmp/synthetic-event-costs-fixed-3401-v10-overlap/observations.json --output-dir /tmp/synthetic-event-features-v2 --native-trials 5` (expected exit 2).
- Python: 82 tests passed.
- CTest: 22 focused tests passed.
- Fit observations SHA-256: `a35387343988bc0875b03fe0f767818fb6c7cc3f054afa0704db765d475194d8`.
- Fit report SHA-256: `09d00ccc0640b72927096a466d31421e599a8c17304852eb94e3a3e402b7af4a`.
- CPU frequency restoration remains deferred at the user's request.
