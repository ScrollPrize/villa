# fit_spiral: device memory grows with the iteration count (uncollected reference cycles)

## What happens
Every step builds new transform objects (slice-to-spiral transform, the integrated-flow diffeomorphism and its
cached-integrator closures). They sit in reference cycles, and each holds a full gap-logit lattice copy and a
full-resolution flow copy. Reference counting never frees them; only Python's cyclic collector does. That collector runs on
container-allocation counts, not on device memory, and a failed CUDA allocation does not start it. So on a long z range the
garbage fills the card before a full collection runs. This is delayed freeing, not a true leak: the memory returns once a
collection runs (live-tensor census, no fix: gap-logit-sized copies 7 -> 89 -> 67 at iterations 13 / 149 / 272).

## Fix
After each optimiser step: `gc.collect()`; plus a one-time `gc.freeze()` after the first step. A full collection scans every
tracked object, and the setup heap (tracks, brick pools, configuration) is large and never dies, so a plain collection per step
costs about 0.3 s (3x slower on a small window). Freezing that heap removes it from every later scan.
`FIT_SPIRAL_GC_EVERY_STEPS=N` collects every N-th step (0 disables). The model-stage rebuild unfreezes first so it can still free
what it owns (that path was not exercised).

## Measurements (PHerc0343; V100-SXM2-32GB; seed 1; shell 250; flow 16-voxel unless stated; 600 steps)
Window z 9000-10500, alternating arms on one card shared with an inference job (it/s is noisy, about 3 %):

| arm | torch peak | it/s | GC time |
|---|---|---|---|
| unpatched (2 runs) | 8.68 / 8.60 GiB | 6.65 / 6.45 | 62 s of 292 s (automatic collections) |
| `gc.collect()` every step, no freeze (2 runs) | 2.73 / 2.73 GiB | 2.15 / 2.13 | about 270 s of 475 s |
| every step + `gc.freeze()` (1 run) | 2.81 GiB | 6.43 | 62 s of 291 s (846 full collections, 73 ms each) |
| collect only when 20 % / 5 % of the card was allocated since the last one | 5.50 (2 runs) / 4.28 GiB | 6.46 6.55 / 6.24 | |

Whole scroll z 1200-17500 (16,300 slices), 32-voxel flow grid, 300 steps: with the patch, 18.57 GiB torch peak
(26.7 GB nvidia-smi on a card shared with a 7 GB job), 0.92 it/s, no out-of-memory. Without any collection the same fit ran out
of memory at iteration 18 (an earlier, otherwise identical run). There is no matched unpatched control for this size (it cannot
run), so the per-step cost at full height is not isolated: the 534 full collections of that run took 156 s in total, of which
the share after the freeze is not separated.

## Limits
One scroll, one seed, one GPU type. Window it/s were measured on a shared card. The 16-voxel full-height case was only measured
with an older, external hook (28.2 GiB). A cleaner fix is to break the cycles (drop the cached integrator and sampler at the end
of `step()`, or hold the transform weakly in the closures); not attempted here.
