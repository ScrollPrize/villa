# Local windows, batches and exclusive profiling

## Method

Paris4 CCW151, corrected quadratic objective, fresh radial initialization, seed0,
no reference initialization. All parameters jointly trainable from start.
Batch sampling with replacement, not exhaustive epochs. Reference metrics use
one median winding per original retained source fiber and global offset correction.
No loss/default changes beyond opt-in windows. N selects bucket-index distance
from maximum nonbroken probability, recomputed each update. Local probabilities
renormalized to original retained mass; distance/V objective is approximate.
Sign penalty remains mathematically exact over all hypotheses via sorted prefix
moments, including phase/family corrections, unknown sign and unordered offsets.
Floating-point reductions are not bitwise identical. No placeholders/proposal windows.

## Timing/memory protocol

H10080GB,FP32,compiledpair backend. One exclusive manager allocation for suite,
independent GPU lanes; timing matrix runs sequentially on GPU5 (UUID
GPU-09395a0d-28a4-4b5a-cc3e-d3e024d05c36).105production updates,first5excluded,
100CUDAevent timing samples. Compilation,initialization,finalaudit excluded from
steady timing. Memory is peak PyTorch allocated—not process VRAM/reserved memory.
Quality recorded solver elapsed includes compilation, prior diagnostics and the
final hard audit, but excludes final reference-metric calculation and saving;
also excludes manager queue/preparation. CPU resources/sharedhost can affect it.
Shared pilots and interrupted separate-exclusive attempts are not used below.

| Window N | Batch | Mean ms | p50 ms | p95 ms | k constraints/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| Full | 4096 | 22.392 | 22.390 | 22.475 | 182.9 | 3.985 |
| 2 | 4096 | 24.241 | 24.232 | 24.347 | 169.0 | 3.305 |
| Full | 8192 | 35.303 | 35.301 | 35.453 | 232.0 | 5.137 |
| 2 | 8192 | 36.224 | 36.217 | 36.364 | 226.1 | 3.536 |
| Full | 16384 | 60.007 | 60.005 | 60.136 | 273.0 | 7.430 |
| 2 | 16384 | 61.789 | 61.622 | 63.024 | 265.2 | 3.990 |
| Full | 32768 | 116.370 | 116.338 | 116.696 | 281.6 | 12.027 |
| 2 | 32768 | 117.173 | 117.166 | 117.478 | 279.7 | 4.898 |

## Radius quality, B4096

16130updates,66,068,480 sampled constraints (~10.001passes). Final global sign
audit over6,606,100events; sign percent denominator is retained known-sign events.
Captured mass is mean selected/fullretained probability per piece, not retained
length. Fit q10/q90=49/122 for all; matched-reference q10/q90=42/125 at final.

| Window N | Seconds | Ref L1 | Within2 % | Retained length % | Sign violations % | Captured mass % |
|---|---:|---:|---:|---:|---:|---:|
| Full | 519.61 | 11.723346 | 13.2134 | 66.0169 | 3.3308 | 100 |
| 2 | 530.38 | 11.761666 | 13.2501 | 64.7337 | 3.1214 | 93.5415 |
| 4 | 545.26 | 11.735520 | 13.2343 | 65.8070 | 3.2934 | 98.3075 |
| 8 | 549.85 | 11.724982 | 13.2247 | 66.0178 | 3.3716 | 99.9212 |
| 16 | 540.00 | 11.723144 | 13.2165 | 66.0248 | 3.3429 | 99.9979 |

No meaningful global winding improvement. Narrow windows reduce memory but are
slower and reject more length. Wide windows closely reproduce the baseline.

## Complete update profile (B4096 full)

| CUDA phase | Mean ms |
|---|---:|
| Sampling/coordinates |1.108|
| Pairforward |0.518|
| Otherlossforward |3.939|
| Allbackward |15.589|
| Gradientcoalescing+optimizer |1.239|
| Total |22.392|

Separate pairforward+backward1.718ms (~7.7% of wholeupdate),vs local windows
3.947–4.023ms. Fullpair incrementaltemporary1.402GiB,local~.067–.080GiB.
Do not confuse temporarypair memory with total model/optimizer allocations.
Complete Torch trace highlights aten::_index_put_impl_~10.212msGPU full,
~11.345msN2; corresponding indexing-backward kernels9.485/10.185ms. These are
nested operator/kernel views of the same work: do not add them together.
Coordinate maps index winding widths/boundaries repeatedly; deterministic
indexed backward is the dominant hotspot, not pair arithmetic.
Profiler times are instrumented snapshots, not steady throughput numbers.
Lastprofiledupdate excluded from timing statistics; unprofiled matrix supplies
memorycomparison. Traces: data/uv_update_trace_{full,n2}_exclusive_v1/update_trace.json.

## Larger batch quality

Reference L1 units are windings; within2 means within +/-2, inclusive.

Completed exit0; full B8192/16384/32768 and N2B32768 with unchanged LR.
8k matches eventbudget exactly;16/32k process8,192fewer constraints (.0124%).

| Window | Batch | Updates | Seconds | Ref L1 | Within2 % | Retained length % | Sign violations % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 4096 | 16130 | 519.61 | 11.723346 | 13.2134 | 66.0169 | 3.3308 |
| Full | 8192 | 8065 | 429.02 | 11.737049 | 13.1468 | 66.3490 | 3.4707 |
| Full | 16384 | 4032 | 399.82 | 11.745175 | 13.1990 | 66.6851 | 3.6633 |
| Full | 32768 | 2016 | 377.11 | 11.758667 | 13.1480 | 66.8429 | 4.0057 |
| 2 | 32768 | 2016 | 378.93 | 11.789237 | 13.1514 | 65.1803 | 3.7175 |

Larger batches improve constraints/sec but reduce update count and converge
slightly less at this fixed data budget. B16k is a reasonable speed/memory
tradeoff, not a proven accuracy improvement; no default changed. B32k uses
1.62x memory vs16k for only~3.1% additional steady throughput. Local windows
do not help quality/speed in these trials; retain as opt-in memory experiment.
All per-diagnostic records are in uv_local_window_progress.md. Initial/final
global reference offset remains+12; fit range49/122 barely moves.

## Reproduction

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_exclusive_matrix.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_batch_quality_matrix.json
```

Configs point to completed output directories; change output paths before rerun.
Matrices run existing solver subprocesses under one exclusive lease, with
per-lane sequential jobs, isolated CUDA_VISIBLE_DEVICES and cancellation cleanup.
No installation/native build changes. Portable eager implementation retained;
GPU performance measured only on this CUDA H100 system.

Validation:91tests passed; CPUfloat64/float32 signloss/gradient oracles, full
window coverage, boundarymass/zero handling, CUDA compiled/eager all4radii,
existing loss/optimizer regression suite, managed matrix launch regression.
Single seed per quality experiment; no statistical significance claimed.
