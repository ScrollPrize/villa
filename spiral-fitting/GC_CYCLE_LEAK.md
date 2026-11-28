# fit_spiral: device memory grows with the iteration count (uncollected reference cycles)

## What happens
Every step builds new transform objects (slice-to-spiral transform, the integrated-flow diffeomorphism and its cached-integrator closures). They sit in reference cycles, and each
holds a full gap-logit lattice copy and a full-resolution flow copy. Reference counting never frees them; only Python's cyclic collector does. That collector runs on
container-allocation counts, not on device memory, and a failed CUDA allocation does not start it. So on a long z range the garbage fills the card before a full collection runs.
It is delayed freeing, not a true leak: the memory returns once a collection runs (live-tensor census, no fix: gap-logit-sized copies 7 -> 89 -> 67 at iterations 13 / 149 / 272).

## The flag: FIT_SPIRAL_GC_POLICY
| policy | what it does | torch peak (z 9000-10500) | it/s | ramifications |
|---|---|---|---|---|
| `freeze` (default) | `gc.freeze()` once after the first step, then a full collection after every `FIT_SPIRAL_GC_EVERY_STEPS`-th step (default 1) | **2.81 GiB** | 13.71 | lowest peak; no measurable cost (a collection scans only what was created after the freeze). Objects alive at the first step are never collected afterwards (they live for the run anyway); the model-stage rebuild un-freezes first. `gc.freeze` is process-wide: cyclic garbage of other libraries created before the first step is never freed. |
| `every-step` | a full collection after every step, no freeze | 2.73 GiB | 2.15 (shared card) | each collection scans the whole heap (about 0.3 s here: tracks, pools, configuration), 3x slower on a small window; on a full-height fit (about 1 s steps) it would add about 12-18 %. Only useful when the heap is small. |
| `pressure` | a full collection only when the device memory allocated since the last one grew by more than `FIT_SPIRAL_GC_GARBAGE_FRACTION` (default 0.2) of the card | 5.58 GiB (4.28 at 0.05) | 13.91 (6.24 at 0.05, shared card) | no cost for fits that fit; the peak is the live set plus that fraction of the card, so it depends on the card size and a card smaller than live set + 20 % can still run out. |
| `off` | Python's own cyclic collector only (the previous behaviour) | 8.44 GiB | 13.56 | runs out of memory on long z ranges (z 1200-17500, 32-voxel flow grid: out of memory at iteration 18). |

`pressure` uses CUDA allocation counters and does not trigger explicit collections
on MPS or CPU. `FIT_SPIRAL_GC_EVERY_STEPS=0` disables explicit collection for
`freeze` and `every-step`; `FIT_SPIRAL_GC_GARBAGE_FRACTION` applies to `pressure`.

The first three rows above were measured on one V100-SXM2-32GB, PHerc0343, shell 250, flow 16-voxel, seed 1, 600 steps; the `freeze`, `pressure` (0.2) and `off` rows in the same
session on an otherwise idle card (13.6-13.9 it/s, within 2.5 % of each other), the `every-step` row earlier on a card shared with an inference job (compare it only with the 6.5 it/s of that session).

## On this branch's fitter with object-store inputs (2026-11-16)
PHerc0490B z 7000-7200 read from an ingested object store (557,248 tracks, 678,404 crossings), 32-voxel flow grid, 1,000-step
configuration, one dedicated hub V100-16GB, collector timed with `gc.callbacks`. After startup the Python heap holds 50.8 M objects, so
a full collection costs 22-30 s (against about 0.3 s on the older fitter above), and the policy decides the fit's speed:

| policy | s/it (it/s) | runs |
|---|---|---|
| `freeze` | 0.058 / 0.060 (17.2 / 16.6); the 1,001 collections after the freeze: 0.06-0.18 s in total | 2 |
| `every-step` | 28.9 / 27.2 (0.035 / 0.037): collections of 27-30 s | 2 (capped at 62 and 15 steps) |
| every 10 steps (`every-step`, `FIT_SPIRAL_GC_EVERY_STEPS=10`) | 2.81 (0.36) | 1 (capped at 151 steps) |
| `pressure` (0.2) | 0.058 (17.4) | 1 |
| `off` | 0.050 / 0.052 (20.1 / 19.4) | 2 |

Torch peak during the optimisation was 0.24 GiB in every arm (this window's per-step garbage is small; the memory effect is the
z 9000-10500 table above). `off` stays faster per step than `freeze` here (about 15 %) although the frozen collections themselves take
under a millisecond; `off` is the policy that runs out of memory on long z ranges.

## Whole scroll
With `freeze`, one fit over z 1200-17500 (16,300 slices), 32-voxel flow grid, 300 steps: 18.57 GiB torch peak (26.7 GB nvidia-smi on a card shared with a 7 GB job), 0.92 it/s, no
out-of-memory. Without any collection the same fit ran out of memory at iteration 18. There is no matched unpatched control at this size (it cannot run), so the per-step cost there is not isolated.

## Limits
One scroll, one seed, one GPU type. The 16-voxel full-height case was only measured with an older, external hook (28.2 GiB). A cleaner fix is to break the cycles (drop the cached
integrator and sampler at the end of `step()`, or hold the transform weakly in the closures); not attempted here.

## Reproduce
Run any 1500-slice window with `FIT_SPIRAL_GC_POLICY=off` and with the default and compare `torch.cuda.max_memory_allocated()` at iteration ~600.

## Synthetic reproduction and what the growth depends on (2026-11-16)
The growth can be reproduced without the fit (`pool_offload/gc_cycle_synthetic.py DEVICE HEAP STEPS TENSOR_MiB POLICIES`): a heap of N small objects, then per step a reference cycle that holds a 64 MiB tensor and survives long enough to reach the oldest generation, then is dropped (40 steps, V100):
| interpreter | heap | default collector | `freeze` (gc.freeze once, then a full collection every step) |
|---|---|---|---|
| Python 3.11 | 1e5 and 2e7 objects | 2,560 MiB (all 40 steps kept) | 64 MiB |
| Python 3.14 (incremental collector) | 1e5 objects | 256 MiB (no growth) | 64 MiB |
| Python 3.14 | 2e7 objects | 2,560 MiB | 64 MiB (1.2 s of collector time in total; collecting every step with no freeze: 49 s) |
With CPU tensors (each policy in a fresh process, 2e7-object heap, 30 steps of 64 MiB) the process grows by 1,963 MiB under the default collector and 110 MiB under `freeze` (collecting every step without a freeze: 110 MiB, 34 s of collector time): the effect is the Python collector's, not the GPU's. So the growth appears when the heap is large (the fit's is tens of millions of objects once the tracks are loaded)
or on interpreters before 3.14; a small-heap fit on 3.14 (45 k tracks, 60 iterations, `off`) showed none, with the same torch peak as `freeze`.
