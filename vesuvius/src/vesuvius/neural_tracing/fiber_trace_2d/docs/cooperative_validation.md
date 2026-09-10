# Cooperative training validation — 2026-09-10

Status: implementation/reference validation complete under the user's revised
short-test scope; protocol v1 locked. Extended timing repetitions were deliberately
stopped, not passed. No further long benchmark is required or started.

## Reproduction and scope

All tests use the approved no-argument launcher:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
```

Its adjacent `fiber3d_cooperative_tests.json` selects named suites and reference
phases. Each output directory retains the exact control, command, source revision,
logs and results. Python is `/home/hendrik/.venv_las/bin/python`. Artifacts below
are relative to `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/`.

Reference data/test selection: archived `training_baseline_s1a_20260729.json`,
S1A NML annotations; current GroupNorm, quarter-resolution output, patch 128,
effective batch 128, seed 0, compilation off. Only GPUs 0/1 (H100 80 GB) are used.
The test requests eight CPU cores (six loading workers with two GPU workers),
microbatch 2/GPU normally and 1/GPU under reduced VRAM. CUDA model computation
uses BF16 unless an explicitly labelled FP32 oracle is selected. Native code is
QuickBuild `-O3`, without LTO or fast-math, loaded from the isolated build rather
than installed over the user's venv.

A 16 GiB/GPU test allowance includes a 4 GiB non-allocator reserve; the outer
process-tree monitor aborts above 20 GiB/GPU. Observed reference lifecycle peak
was 8.23 GiB/GPU. Calibration covers these two devices and measured microbatches
1/2, **not** arbitrary configurations or an eight-GPU run. Permission to share
devices is not evidence that concurrent GPU contention actually occurred.

## Correctness evidence

| Check | Result | Artifact directory |
| --- | --- | --- |
| CPU regression | 344 passed, 18 GPU skips; 71.31 s | `20260910_100827_372654` |
| Manager policy/protocol/recovery/failure cleanup | 61 passed; 12.29 s | `20260910_100827_372654` |
| Reference harness assertions | 6 passed; 1.61 s | `20260910_093209_505125` |
| Native prerequisites | 3/3 passed; 2.51 s | `20260910_090759_469557` |
| CUDA FP32/FP16/BF16 numerical checks | 12 passed | `20260910_083405_608714` |
| CUDA two-to-one cohort restart | 1 passed | `20260910_083405_608714` |
| Actual S1A FP32 same-allocation restart | byte-exact model/optimizer/test metric | `20260910_083025_287860` |
| Actual S1A BF16 same-allocation restart | byte-exact model/optimizer/test metric | `20260910_083716_992318` |
| Actual S1A frozen FP32 gradient sums | 612 component/layer checks pass | `20260910_082534_937046` |
| Two-update reference profile | passed; 161.41 s including initialization/export | `20260910_100958_644509` |

Same-allocation tests run six B128 updates and include a GPU-free pause during
testing. The frozen-gradient oracle observes the actual FP32 bank contributions:
one- and two-GPU FP64 shadow sums match, sample IDs/counts match exactly, and all
FP32 sums satisfy their computed summation-error bound. Maximum bound fractions
are 0.059152 (one GPU) and 0.032104 (two GPUs). FP64 is test-only. This isolates
accumulation roundoff, not every BF16 batch-shape-dependent kernel effect.

Changing physical batch/rank partition is not bitwise-identical. The original
pointwise trajectory criterion failed for a small subset of weights; it was
**not widened**. Six-update fixed/dynamic relative model L2 differences are
1.39e-5 (FP32) and 5.90e-5 (BF16). Final test losses are
2.242254823/2.242277861 (FP32) and 2.239006564/2.239104435 (BF16).
These are trajectory diagnostics, separate from the gradient semantic checks.

The first 110-update BF16 repetition shows amplified trajectory drift: fixed
test loss 1.582692 versus dynamic 1.606135; model relative L2 difference 0.010365,
optimizer moment relative L2 difference 0.114254. This is **not** a long-run
numerical-equivalence pass. Fixed versus unmanaged remains byte-exact. Frozen
gradient checks establish the objective/accumulation semantics but cannot prove
equal convergence after repeated batch-shape-dependent BF16 rounding.

## Throughput method

Artifacts: `20260910_093209_505125`, explicitly reusing the two completed first
baseline cases in `20260910_085311_079168`. The latter's dynamic case failed due
to a stale test-only pause timer and was rerun from initialization after the fix.
Ten warmup plus 100 measured updates per case. One full unmanaged/managed-fixed/
dynamic comparison completed. The user stopped the second repetition and cancelled
the remaining repetitions in favor of tests that finish in minutes.
Cases run sequentially. The **unmanaged comparator is the same bounded
production trainer/cohort without manager admission/RPC**, not the previous
graph-retaining trainer or its larger microbatch performance ceiling.

Dynamic cases include GPU-free pauses, two-GPU partial VRAM, one-GPU shrink and
expansion. Saved grants plus completed updates must prove each transition was
applied. Report active-update time separately from wall time including handoff.
The separate `profile` phase measures operator/stage costs without contaminating
the throughput sample; host stage durations are not CUDA kernel durations.

| Completed case | Wall update mean / p50 / p95 (s) | Min / max (s) | Global samples/s | Per GPU samples/s |
| --- | --- | --- | --- | --- |
| Same bounded trainer, unmanaged admission | 8.010 / 7.974 / 8.093 | 7.719 / 12.183 | 15.980 | 7.990 (2 GPUs) |
| Managed, fixed allocation | 7.945 / 7.903 / 8.074 | 7.434 / 12.325 | 16.111 | 8.056 (2 GPUs) |
| Managed, changing allocation incl. handoffs | 11.030 / 8.078 / 16.317 | 7.766 / 36.613 | 11.605 | not a fixed GPU count |

These single-comparison results do not establish a statistical speedup. Mean
active dynamic update time is 10.219s; mean wall overhead above active time is
0.811s/update across the deliberately frequent handoffs. By active grant:
two GPUs at microbatch 2 average 8.058 s/update (15.885 global samples/s), two GPUs
at microbatch 1 average 10.610 s (12.065 global), one GPU at microbatch 1 average
16.310 s (7.848 samples/s). Two recorded repeated pause latencies were 5.50/6.75 s
from the boundary request to driver-confirmed zero owned GPU processes.

## Short profile

Two B128 updates, one traced update, same two-GPU configuration. Artifact
`20260910_100958_644509` includes host stage JSON, CUDA traces and operator tables.
The full profile invocation took 161.41 s, including initial testing and trace
export; this is not a training-step time. Owned driver peaks were 8.03/7.78 GiB.

For the traced update, rank 0/rank 1 host stage totals respectively:

- forward/loss: 1.235/1.331 s;
- component backward: 4.864/4.950 s;
- final statistics/gradient reduction: 2.634/0.038 s;
- input wait: 0.004/2.407 s.

The reduction imbalance coincides with rank 1 waiting for inputs. Do not label
rank 0's entire 2.6 s as communication bandwidth cost. On GPU 0 the operator table
records convolution backward 1.518 s CUDA total, GroupNorm backward 0.942 s and
GroupNorm forward 0.628 s. Nested/custom ranges overlap: do not sum all profiler
rows. BF16 convolution kernels remain in the trace; production model compute
was not switched to FP32. This short trace is diagnostic, not a warmed throughput
benchmark or proof of the optimal configuration.

## Limits

No one-hour convergence claim follows from this validation. Only two H100 GPUs
were calibrated/tested; other configurations need calibration. Multi-user OS
credential tests use fake identities rather than provisioning extra local users.
Host queues/parking are structurally tested; the planned extended pause-RAM
settling study was omitted when the user requested short tests. The venv was not
reinstalled. Default CPU/manager tests take about 84 s here, and every normal
launcher invocation is capped at 300 s total. Long sweeps are explicitly rejected.
