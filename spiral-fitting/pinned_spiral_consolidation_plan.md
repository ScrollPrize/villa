# Pinned spiral: consolidation plan

Companion to `pinned_spiral_plan.md` (design) and `pinned_spiral_status.md` (measurements).
Written 2026-09-24 after the evaluation of that week. The pinned fit has accumulated
switches, modes and diagnostic knobs faster than it has accumulated results. This file
lists what the measurements say to keep, what to retire, and the one decision still open.
Numbers are from the z 10000-11000 slab (9309 verified patches, B-spline lattice 24) and
are detailed in the status file.

## What moves the strict metric

In order of effect on the integer-snapped satisfaction main reports:

1. **Integer-frozen pin targets** (`model_pin_targets_integer`): 58% -> 91% of area at
   activation, instantly. A hard commitment at activation (see the open decision).
2. **Pair-agreement warm-up loss** (`loss_weight_pair_agreement`): at weight 512 the
   unpinned map alone reaches 69% of area (from 58%) and activation conflicts halve
   (1.50M -> 0.73M pairs); it also holds the conflict count flat during the pinned phase.
3. **DT started before or at activation** (`loss_start_patch_dt`): +5 to +15 points of
   area over 1000-2000 steps, and the only term that moves fractional targets toward
   integers.
4. **B-spline flow, lattice 24** (`model_flow_field_type`, `model_flow_voxel_resolution`).

Everything else measured is second-order or negative.

## Retire

Each of these was built to test a hypothesis; the test is done. Retiring means: remove
the code path, add the key to `RETIRED_CONFIG_KEYS` so old checkpoints still load, drop
the tests that exist only for it, and delete its paragraph from the README.

| key | why it goes |
|---|---|
| `model_pin_targets_joint`, `model_pin_targets_joint_tolerance_voxels` | At the default 3-voxel tolerance it links only pairs that already agree. At 30 voxels it halves demotions but leaves the pair conflict count unchanged and costs 1-3 points of strict area. |
| `model_pin_shift_patch_offsets` | 57-66 patches shifted per run, no measurable change in any metric. |
| `loss_pin_strain_detach_targets` | Keep the detached behaviour (the only one ever run), drop the switch. |
| `output_satisfaction_overlay_profile` | The fractional metric stays as a log line and export field; the overlay colouring by it was a one-off diagnostic. |
| `model_pin_demote_recheck_interval` | Keep the re-check at a fixed 1000 steps; the 100-step variant made no difference. |
| `model_pin_max_patch_spread_windings` | Already retired; listed so it is not resurrected. |

Also delete the per-experiment diagnostic scripts under
`spiral-out/pins-5k-warmup/diag_*.py` once `diag_pairs_ckpt.py` (checkpoint-parameterised
pairs breakdown) is moved into `tests/` or `scripts/` as the one supported diagnostic.

## Keep, and change the defaults

- `model_flow_field_type = "bspline"`, `model_flow_voxel_resolution = 24` for pinned fits.
- `loss_weight_pair_agreement` on during warm-up at a large weight (128-512; 512 was
  still improving). Cost: Dirichlet 10.3 -> 12.0, within-patch flatness down a few
  points. Step rate 4.5 it/s against 6.
- `loss_start_patch_dt` at or before `model_pins_warmup_steps`. The current default of
  25000 leaves every pinned experiment shorter than that without integer pressure.
- Demotion (`model_pin_demote_conflicting_patches`) with its current thresholds; keep
  `pin_demotion.jsonl` as the annotation review queue.
- The strain loss with detached targets, as is.

## The open decision: how targets become integers

The exported mesh sheets sit at integer windings, so the targets must end up integer.
Two ways, measured on the same warm-up:

| | strict area | commitment | speed |
|---|---|---|---|
| integer-frozen at activation | 91% | irrevocable per component; a wrong rounding can only be demoted, never corrected | instant |
| fractional trainable + DT | 79% after 2000 steps and rising | revisable while the field settles | T moves ~0.012-0.03 winding per 1000 steps at `optimizer_lr_pin_targets` 0.01 |

The fractional route has the right semantics but converges too slowly to be usable. One
experiment decides it, not a new mode: raise `optimizer_lr_pin_targets` by 10x (and/or
add the DT loss's tent directly on `T` with a weight that ramps up) and measure whether
`|T - round(T)|` reaches ~0.05 within a few thousand steps without losing the pair
consistency the warm-up bought. If it does, fractional targets become the default and
`model_pin_targets_integer` is retired too. If it does not, integer targets become the
default and the fractional machinery (trainable `T`, its optimiser group, the slot
rebuild on `round(T + n)` changes) is retired instead. Either way one of the two paths
goes.

A middle policy worth one run if integer targets win: keep them frozen but re-round a
component from the current free map after it has stayed demoted for several re-checks,
so a wrong activation rounding has a way back.

## Order of work

1. Decide the target policy (one experiment above).
2. Retire the table's keys and the losing target path in one commit, with the retired
   keys registered.
3. Change the defaults listed under Keep.
4. Re-run the three-arm check (unpinned / pinned default / long warm-up) on the slab and
   update the status file; that becomes the baseline for the next round.
