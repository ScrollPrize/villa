# Changelog

## 2026-08-07

- Made the synthetic rendering historical gate performance-only and one-sided.
  Compiler, profiler, cache, workload, model, checksum, and other identity
  changes remain diagnostic but cannot fail the reference comparison; only a
  modeled score more than the configured tolerance slower than its case
  reference fails.
- Removed NumPy from the synthetic rendering gate's runtime path. Event-feature
  construction, per-thread cost scoring, and replay now run in the versioned
  C++ engine; standard-library Python only coordinates Valgrind and artifacts.
  Added native parity and validation coverage plus a `python3 -S` CI smoke test.
- Made Valgrind version changes diagnostic rather than automatic rendering-gate
  identity failures. Both versions remain required and recorded, paired
  Callgrind/DRD collections must still match, and material profiler effects are
  enforced by the unchanged modeled-score tolerance.
- Documented VC3D rendering-gate use, activation, artifact diagnosis, model
  recalibration, reference refresh, and tolerance-policy changes. GitHub
  Actions now derives this gate's Ninja concurrency from `nproc`; the checked
  reference is the single normal CI source for tolerance.
- Added a runner-sized Ninja artifact graph for the complete eight-case rendering
  Valgrind matrix. It collects fresh separate-thread Callgrind profiles and
  complete parallel DRD graphs, applies the frozen synthetic-only native replay
  score, verifies exact environment/workload identity and checksum, and gates
  every case symmetrically within 10% of its checked reference.
- Replaced production Python passive event playback with a persistent,
  versioned C++ replay engine. Graphs and cost attributions are cached and
  replay requests are batched while the Python implementation remains only as
  a compatibility oracle. On a 132,405-event shuffled renderer trace, one warm
  replay improved from 1.380 s median to 15.5 ms with identical output, an
  88.9x speedup.

## 2026-08-06

- Added paired cache-line access-density kernels and crossed one/eight
  read/write subsets. The expanded synthetic matrix identifies all combined
  read and split-miss coefficients without collapse. On 35 frozen renderer
  cases, read-only produced the best one/multiple-worker maximum errors at
  7.03%/16.80%; combined reached 7.61%/17.28% and the best runtime RMS at
  6.70%. Every expanded-model row stayed within 20%.
- Tested explicit data-read cost and split read/write cache-miss bases using
  three new generic density kernels. Added read coverage resolved the original
  collinearity, but the read coefficient was unstable and all write-side costs
  reached zero in split fits. A user-requested frozen renderer diagnostic found
  that the density-augmented legacy refit reduced headline maxima to
  19.83%/21.58%; `data_reads` improved them slightly further to 19.22%/21.27%,
  while split-only and combined bases were worse.
- Tested an unweighted minimax objective on the unchanged six-feature passive
  event model. It reduced maximum training error but increased fresh synthetic
  maximum error from 37.66% to 47.85% and frozen renderer maxima from
  29.51%/30.49% to 57.45%/54.39% for one/multiple workers, so it was rejected.
- Rejected an L1-miss-density cache-serialization proxy. It reduced fresh
  pointer-holdout maximum error from 93.66% to 63.81% but failed its 30% gate
  and worsened frozen renderer maximum runtime error for both one and multiple
  workers.
- Added a frozen-trace attribution sensitivity evaluator. Front/back event
  concentration improved renderer speedup RMS by about two points but did not
  improve the prioritized maximum runtime error, so no renderer-derived policy
  was selected. Maximum speedup error is monitor-only while it remains below
  20% for every frozen renderer row.
- Tested a hard-lower-bound-preserving dependency-excess scale with matched-work
  synchronization-depth pairs. It fitted to one, added no synthetic or renderer
  improvement, and was rejected while retaining backward-compatible model
  schema handling.
- Added deterministic phase-separated and interleaved trilinear-gather
  calibration kernels, cache/branch fit and holdout matrices, per-task skew,
  and a faster equivalent FIFO replay implementation.
- Tested bounded branch/cache overlap and replay-idle terms using synthetic
  observations only. Both fitted to zero and were rejected; expanded event
  calibration improved the five-scenario renderer diagnostic, but absolute
  timing remains outside its acceptance bound and timing claims stay disabled.

## 2026-08-05

- Established specification-driven planning documents for
  `volume-cartographer` and added a required synthetic rendering regression
  benchmark covering serial and fixed-thread parallel fine-to-coarse sampling.
- Corrected the synthetic rendering measurement boundary and added a versioned
  Callgrind event calibration with matched worker measurements, held-out
  locality validation, one shared modeled-cycle-to-time conversion, and
  estimated Mpx/s in serial and parallel benchmark reports.
- Added a native thread-pool dispatch diagnostic and identified coarse
  one-range-per-worker batches as the primary cause of the synthetic renderer's
  parallel scaling gap.
- Added an experimental passive Valgrind scheduler/futex stream extractor and
  replay diagnostic. It captures complete future wait/wake dependencies and
  ranks task-granularity changes correctly, while explicitly disabling timing
  claims after the coarse-task held-out error exceeded the acceptance bound.
- Identified the passive replay's residual as native queue-tail sleeping and
  scheduler delay plus a separate whole-process warmup/startup intercept. A
  reference-host model using steady parallelism, task-time tail penalties, and
  the process intercept predicted fresh coarse/fine speedups within about 2.3%,
  while documenting that Valgrind alone cannot supply these scheduler terms.
- Extended native dispatch measurements through 24 worker threads. A fixed
  short-task workload peaked at 16 physical workers; adding SMT siblings
  reduced throughput by 7% at 17 workers, 11% at 23, and 28% at 24.
- Added guarded fixed-frequency calibration controls and rejected the first
  cross-worker shared timing model on untouched holdouts. Effective runnable
  width depends on task duration and queue depth, while futures submission
  overlaps worker queue draining; worker count alone cannot model either term.
- Restricted the host dispatch calibration to one CCD and at most seven workers,
  with independent topology/frequency validation. Round and lifecycle holdouts
  stayed within 3.3%, but timing claims remain disabled because the per-worker
  startup coefficient collapsed to its lower bound; eight-worker CCD saturation
  is explicitly outside the model domain.
- Replaced the rejected six-coefficient host model with a three-value synthetic
  calibration and added gate, idle-history, clock-overhead, and lifecycle
  diagnostics. Five-process holdouts stayed within 13.34% by median and 14.71%
  per individual run, satisfying the provisional 20% threading bound; actual
  renderer timing claims remain disabled pending renderer holdouts.
- Added passive renderer replay using DRD vector-clock dependencies and fixed
  its native topology to reserve a caller CPU. Nine actual renderer cases were
  within 15.8--20.94% median speedup error without fitted renderer contention;
  the three five-worker cases narrowly failed the 20% per-case gate, so timing
  claims remain disabled.
- Extended passive renderer diagnostics to every worker count 1--7 after a
  uniform Release rebuild. The replay's parallel-runtime residual is an almost
  pure 9.59 ms additive term; a worker-count slope is negligible. No correction
  was adopted pending generic repetition-count calibration and new renderer
  holdouts.
- Added a renderer-independent passive replay calibration using only generic
  ALU work and `utils::ThreadPool` futures. Its three shared terms predicted a
  third untouched synthetic holdout matrix within 15.45%, but the calibration
  remains rejected because fit and leave-one-case stability narrowly missed
  their predeclared gates; no renderer data or coefficient was used.
