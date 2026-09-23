# Pinned spiral: status of implementation and testing

Companion to `pinned_spiral_plan.md`, which holds the design and the per-stage
status notes. This file is the short version: what exists, what has been
measured, and what the numbers currently say. Last updated 2026-09-23, branch
`spiral-gap-pin-change-radii`, HEAD `ebb4c5bda`.

## What is implemented

**Stage 2a (pinned winding radii).** `pins.py`, `PinnedGapExpandingTransform` in
`transforms.py`, `FitContext` hooks in `fit_spiral.py`. Registry of hard-constraint
points (verified-patch quad centres, cross-patch PCL points, unattached strips) with
per-component fractional targets `T`, CSR pin table with per-pin footprints, tridiagonal
radius blend, and a **localised** ordering guard (running max plus re-blend; the plan's
cumulative-sum guard made almost every pin inexact once a few inner anchors conflicted).

**Stage 2b (performance).** Three Triton kernels in `gap_triton.py` (affine interval map,
CSR anchor walk with recompute backward, batched tridiagonal solve with adjoint backward),
amortised pin-table rebuilds, and patch-driven pin sampling: each training step pins
every pin of the patches its losses sample (patch loss and PCL winding rows) plus all
chain pins, so the pinned map is exact where the losses evaluate it. Pinned steps run
at roughly 4x the unpinned cost on the real fit (about 1 it/s vs 10, B-spline flow).

**Stage 3 (strain loss).** `losses.get_pin_strain_loss`: the free map's miss at each pin,
hinged at `loss_margin_pin_strain`, replacing the patch-radius, rel/abs-winding and
unattached-strip radius losses while pins are active (soft losses stay on for unpinned
inputs). Targets detached by default.

**Registry hygiene found by the diagnostics.**
- Points outside the flow z domain are not pins (whole-scroll PCLs and fiber chains
  extended far beyond a z-range fit; 18.8k of 23k chain pins were being pinned at
  clamped z with targets tens of windings off).
- Resume re-estimates `T` when the checkpoint was written before activation.
- Excluded patches keep the `-1` patch component and get no pinned DT target.
- Integer targets (`model_pin_targets_integer`, optional) are re-applied on resume and
  run-boundary updates.

**Conflict-based demotion** (`model_pin_demote_conflicting_patches`, on by default).
Cross-component pin pairs within 30 voxels are compared the way the ray map treats
slots; a patch that disagrees with its neighbours more than it agrees (ratio 0.5, at
least 5% of its pins) is left unpinned and gets no soft loss. Re-checked every
`model_pin_demote_recheck_interval` steps against the current free map, reinstating
patches that no longer conflict. Every decision is appended to `pin_demotion.jsonl` in
the run directory with patch ids, pair counts and conflicting neighbours (a review queue
for the annotations). Replaces an earlier free-map spread cap (retired key).

**Opt-ins, measured but off by default.** `loss_weight_pair_agreement` (warm-up pair
agreement on the free map, see below), `model_pin_targets_integer` (snap and freeze
targets), `model_pin_targets_joint` (joint integer assignment over coincident
components), `model_pin_shift_patch_offsets` (pairwise target shift: correct a patch's
integer offset by a whole winding when that makes it agree with its neighbours).

**Measurement tooling.** The export prints `fractional_satisfied_*` (each patch against
its own fractional median, no integer snap) next to the strict lines and can colour the
overlays by it (`output_satisfaction_overlay_profile`); `output_satisfaction_log_interval`
logs both during the fit; the headless log prints pin diagnostics (inexact fraction,
guard-active rays, strain quantiles) every 200 steps.

## Test coverage

`tests/test_pins.py`, `tests/test_pins_transform.py`, `tests/test_pins_stage2b.py`,
`tests/test_pin_strain.py`, `tests/test_pair_agreement.py`, plus additions to `tests/test_checkpoint_load.py`,
`tests/test_packed_patch_satisfaction.py`. About 130 tests across these; the wider
touched suites (config, winding supervision, loss maps, SDT, fiber links) also pass.
Covered: exactness and monotonicity of the pinned map, guard semantics, transparency of
zero-mass anchors, Triton vs eager (values and gradients, whole transform), subsampling
and layout reuse, z-domain filtering, spread/demotion/subset registry logic, integer
target handling on resume, strain loss (zero iff satisfied, descent, gradcheck), joint
assignment, offset shift, fractional satisfaction profile.

Known test caveats: the torch.jit flow backward's NVRTC fusion fails on this machine
(torch cu130 vs nvrtc 12.8) after a few backward calls, so CUDA tests disable GPU
fusion via a fixture. Gradients into ray angles and pin positions through the singular
kernel are fp32-ill-conditioned in both eager and fused paths and are compared at
1e-2 relative.

Synthetic benchmarks: `tests/benchmark_pins_scale.py` (production-scale registry, no
data) and `tests/benchmark_pins_stage2b.py` (tiny, launch-bound).

## What the real fit says (z 10000-11000, 9309 verified patches)

Protocol: 5000 unpinned warm-up steps (B-spline flow, lattice 24 unless stated), then
300 more steps in each configuration from the same checkpoint, same seed. Runs from
2026-09-23 are under `spiral-out/pins-5k-warmup/eval_0922/`.

**Yardstick.** Two metrics are exported. The *strict* metric is the one main reports:
each patch's median shifted winding is snapped to the nearest integer and quad centres
must lie within the tolerance of that integer. The *fractional* metric scores each patch
against its own unsnapped median. The fractional metric measures within-patch flatness
only: a patch pinned exactly at winding 12.4 scores 100% on it and 0% on the strict one,
and the exported mesh sheets sit at integer windings. Earlier versions of this file used
the fractional metric as the yardstick and reported the strict one for reference; that
was the wrong way round. The fractional metric remains a useful flatness diagnostic.

| configuration (300 steps from the warm-up) | strict area | strict patches | strict patches, area-weighted | fractional area | fractional patches |
|---|---|---|---|---|---|
| unpinned continuation (main behaviour) | 58.1% | 23.6% | 24.6% | 91.0% | 68.6% |
| pinned, fractional trainable targets (branch default) | 60.7% | 35.5% | 38.4% | 94.9% | 80.1% |
| pinned, `model_pin_targets_integer` | **90.8%** | **68.8%** | **72.0%** | 92.1% | 70.1% |
| pinned, integer + joint + offset shift | 89.8% | 66.1% | 70.5% | 91.5% | 67.7% |

The same picture from a mature checkpoint (25,000-step main-style fit, cartesian lattice
16, from 2026-09-09; 300 steps each): unpinned 66.5% / 24.8%, pinned fractional 68.0% /
39.4%, pinned integer 90.3% / 67.5% (strict area / strict patches). The full 30,000-step
main-style run of 2026-08-26 ended at 73.3% / 38.7%.

**Pinning is an instant re-mapping.** Satisfaction logged at the activation step, with
zero pinned training steps, equals the value after 300 steps and after 10,000 steps to
within 0.5 points in every configuration above. The pinned steps do not change what the
export metric sees. They do improve the field: over the 10,000-step fractional run the
Dirichlet term fell from 17.6 to 9.1 and the normals term from 6.7 to 3.8, ending below
the unpinned fit's levels (about 11 and 3.9).

**Why fractional targets stall.** The plan relies on the DT loss to pull each
component's target `T` to an integer. `loss_start_patch_dt` is 25000 in this config, so
every pinned experiment recorded before 2026-09-23 (steps 5000-15000) ran with DT off.
`T` is trainable and does move (mean 0.12 winding between steps 5300 and 15000, 71% of
components by more than 0.05), but only under the image and smoothness terms through the
pinned anchors, which carry no integer or inter-component preference: the mean distance
of `T` from the nearest integer stayed at 0.234-0.241 throughout, and the inconsistent
pair rate drifted from 9.0% to 9.5%. The strain loss detaches the targets by default
(`loss_pin_strain_detach_targets`), so it contributes nothing to `T` either.

**DT started early (2026-09-23).** `loss_start_patch_dt` 5000, 2000 steps from the same
warm-up, fractional trainable targets:

| | strict area | strict patches | strict patches, area-weighted | T distance from integer |
|---|---|---|---|---|
| unpinned, DT from 5000 | 58.1% -> 73.4% | 23.9% -> 41.8% | 48.8% | |
| pinned fractional, DT from 5000 | 60.6% -> 78.8% | 36.0% -> 46.8% | 56.8% | 0.234 -> 0.174 |

`T` drifts toward integers monotonically but slowly (about 0.033 per 1000 steps under a
0.01 winding-per-step learning-rate ceiling), and the pinned metric was still rising at
the end (72.4% area at step 6000). This is the first pinned configuration in which pinned
steps move the export metric. Pairwise consistency did not improve (inconsistent pairs
10.4% -> 11.6%, demotions 346 -> 367): DT snaps each component to its own nearest
integer independently. DT alone at step 5000 also recovers most of the area gain without
pins, so DT at 25000 is late in the main schedule regardless of pinning.

**Conflicts at activation** (pairs diagnostic, B-spline 24 warm-up): about 10% of
cross-component pin pairs within 30 voxels are inconsistent. By cause: same slot but the
free map more than half a winding apart 66%, different slots with inverted radial order
21%, different slots with the gap compressed below the minimum rise 12%. Demotion removes
300-430 patches per run to keep the pinned set clean; since `T` is read off the free map
at activation, these are all inconsistencies the warm-up leaves in the free map (see the
warm-up proposals at the end of `pinned_spiral_plan.md`).

**Warm-up length (2026-09-23).** Same B-spline 24 warm-up continued unpinned (DT off) and
pins activated for one step at each length: 5000 steps, 10.37% inconsistent pairs, 346
demoted; 10000 steps, 9.92%, 314; 15000 steps, 9.83%, 306. Unpinned strict area is flat at
58% throughout (patches 24% -> 30%). A longer warm-up buys a few percent fewer conflicts
and flattens after 10k; the warm-up objective has no term that acts on them.

**Pair-agreement warm-up loss (2026-09-24, `loss_weight_pair_agreement`, off by default).**
Implemented as proposal 1 of the plan's warm-up section: nearby cross-component quad
centres must differ by a whole number of windings under the free map. A/B from the
B-spline 24 warm-up, 3000 unpinned steps (5000 -> 8000, DT off, same schedule), then pins
activated for one step:

| warm-up loss weight | inconsistent pairs at activation | demoted | unpinned strict area |
|---|---|---|---|
| 0 (control) | 1,314,310 | 348 | 58.7% |
| 8 | 1,268,152 (-3.5%) | 335 | 59.1% |
| 32 | 1,177,107 (-10.4%) | 326 | 60.4% |
| 128 | 1,067,991 (-18.7%) | 279 | 62.3% |

Weight is the limiter, not time: at 128 the loss was still falling (median residual
0.067 -> 0.044 winding, pairs over the 0.05 margin 57% -> 46%) with the other terms
unchanged (patch radius 22.2 vs 21.6, Dirichlet 11.2 vs 10.5). By cause, at weight 32
the same-slot class fell 18% (982k -> 804k) while the inverted and compressed classes
grew by 10-17%. The residual's geometry (control checkpoint, 15.1M pairs): pairs under
10 voxels apart agree almost perfectly (median |dW| 0.01, 6% of pairs); the half-winding
offsets sit on pairs 10-30 voxels apart, where the adjacent-sheet difference smears from
0.4 to 1.2 windings instead of 1; within a patch pair the offset is coherent (std 0.065),
so it is a patch-level radial-scale error, fixable by the gap logits and the flow.

Note the comparison across warm-up *lengths* is confounded by the exponential LR
schedule, which decays to `optimizer_lr_final_factor` at `optimizer_num_training_steps`:
the 8000-step control (1.31M) ends below the 10000- and 15000-step warm-ups (1.50M,
1.48M) despite fewer steps. Arms within one table share a schedule and are comparable.

**Joint integer assignment at a wide tolerance.** `model_pin_targets_joint` uses pairs
within `model_pin_targets_joint_tolerance_voxels` (default 3), which are exactly the pairs
that already agree. At 30 voxels on the same checkpoints it re-rounds 7.7k of 9.2k
components and halves the demoted set (348 -> 195 on the control, 326 -> 157 with weight
32) while the pin-pair inconsistency count barely moves (1.31M -> 1.29M) and strict area
drops 1-3 points (91.7% -> 89.9%). It trades demotions for guard inexactness rather than
removing conflicts.

**Cost.** Pinned steps run at about 1 it/s against 6 it/s unpinned on this fit; the
pinned satisfaction export takes 55-65 s against 1.3 s unpinned.

## Open items

- Decide the target policy. Frozen integer targets give the goal numbers today but commit
  at activation; DT-driven fractional targets commit gradually but need a much higher `T`
  learning rate or an explicit tent loss on `T` to converge in a useful number of steps.
  Neither addresses pairwise consistency.
- Warm-up changes so the free map arrives at activation with fewer conflicts: a modular
  pairwise agreement loss on nearby cross-patch quads, order/spacing penalties against a
  detached provisional `T`, a lower patch-radius weight plus overlap linking. Written up
  at the end of `pinned_spiral_plan.md`; read conflicts and the demotion count at
  activation, not the export metric.
- The patch DT term with pins active is 50-70 on the mature checkpoint against 26
  unpinned, even with integer targets where every pin sits on an integer. Unexplained;
  possibly the DT target selection for pinned patches or the training-step pin subset
  not covering the DT samples.
- Review the demotion queue (`pin_demotion.jsonl`) against the annotations: the argument
  for demotion rests on demoted patches being annotation problems.
- The demotion thresholds (30 voxels, 5%, ratio 0.5) are hand-set on this slab only.
- Demoted patches receive no supervision at all.
- Step time and export time (above). The anchor walk is linear in pins per step.
- Housekeeping: a checkpoint written with pins enabled cannot resume into a fit with
  `model_pins_enabled` false (optimiser group count); use an unreachable
  `model_pins_warmup_steps` for unpinned continuations. Checkpoints from before
  2026-09-14 store `influence_disable_dt_frac`, which main renamed without a retired-key
  entry, and need the key stripped to load.

Runs, checkpoints and diagnostic scripts from this work are under
`/home/paul/projects/vesuvius-scrolls/spiral-out/pins-5k-warmup/` (2026-09-23 runs in its
`eval_0922/` subdirectory, drivers `run_rest.sh` and `run_dt_early.sh`).
