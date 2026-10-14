# Joint sliding surface comparison — 2026-10-14

Both full supplied regional layouts completed. This is an initial 600-iteration
experiment, not a convergence or intersection-free claim.

## Inputs and protocol

- Original: villa2/global-fiber-sgd/data/implicit_match_weights_v1/same10.
- Deformed: villa2/global-fiber-sgd/data/exact_fixed_offsets_aligned_v1.
- AFV: villa2/global-fiber-sgd/data/Paris4_v1.afv.
- Piece tensor: villa2/global-fiber-sgd/data/uv_reference_w14_20_z30k50k_v1/tensors.npz.
- Identical deformed retained selection: 849 H pieces. Original/deformed state
  and widths exactly equal; deformations only change their input UV paths.
- 100-base-voxel mesh; <=50-base-voxel arc subdivisions plus curve/offset knots and
  initial grid crossings. Sample counts differ slightly because crossings differ.
- Existing Lasagna scale-space Adam, 600 joint steps, mesh LR5, UV LR0.1 grid units;
  smooth/step/bend weights .5/.5/5, attachment group ramp1→20. Within attachment
  group: weak displacement .01, displacement-slope1, stretch1, soft-limit10.
- H100 GPUs0/1, non-exclusive manager grants, 8GiB/8CPU each. No GPUs6/7 requested.
  Shared execution timings are not isolated hardware performance benchmarks.

## Fit and geometry

Distances below in base voxels. Mean/RMS are physical-arc weighted; median/p95/max
are sample statistics. Normal error ignores tangential separation, so full XYZ
error is reported alongside it. Initial→final values:

| Metric | Original | Deformed |
|---|---:|---:|
| Attachments | 43,478 | 43,496 |
| Mesh rows × columns | 232 × 1712 | 243 × 1712 |
| Mean XYZ error | 37.5455 → 8.1029 | 27.5918 → 7.9165 |
| RMS XYZ error | 87.8490 → 11.5178 | 84.9410 → 11.3534 |
| Median XYZ error | 11.0096 → 6.4136 | 6.6134 → 6.2837 |
| p95 XYZ error | 190.8873 → 21.3797 | 134.8230 → 20.6078 |
| Maximum XYZ error | 3105.3486 → 931.5852 | 3892.0603 → 890.8326 |
| Normal RMS | 14.9375 → 2.5214 | 13.4053 → 2.5064 |
| Opposing-triangle quads | 630 → 301 | 513 → 369 |
| Degenerate quads / attachment normals | 0 / 0 | 0 / 0 |
| Initialization/export setup | 8.57s | 10.05s |
| Model/solve stage | 7.50s | 7.48s |
| Surface stage total | 16.16s | 17.63s |

No nonfinite vertices. Local fold counts remain nonzero; not a global collision
audit. Only ~29.8% / 28.9% of initialization vertices are bracketed H support;
~45.4% are empty-U-column interpolation. Remaining area includes transverse and
boundary extrapolation. Unsupported regions should not be mistaken for evidence.
Adapter loading and manager startup are outside the surface-stage timer.

## Additional UV motion during this fit

These shifts are relative to each input's mapping, not relative to the other input.

| Absolute shift, base voxels | Original | Deformed |
|---|---:|---:|
| U median / p95 / maximum | 40.30 / 167.19 / 875.63 | 30.69 / 151.81 / 863.85 |
| V median / p95 / maximum | 61.00 / 316.23 / 769.70 | 63.22 / 295.64 / 683.34 |
| Attachments beyond either nominal soft allowance | 0% | 0.0920% |

Deformed slightly improves attachment errors, but has more remaining local folds.
Neither is unambiguously better geometrically. Loss oscillates late under the
rising attachment weight; this first trial is not established as converged.
The old fixed-UV pilot used a different subset/model and is not a matched control.

## Outputs and reproduction

Both contain initial.tifxyz, surface.tifxyz, targets.npz, attachments.npz,
config.json with input/code hashes, progress.json and summary.json:

- `/home/hendrik/villa4/lasagna/data/joint_uv_original_gpu_v1/`
- `/home/hendrik/villa4/lasagna/data/joint_uv_deformed_gpu_v1/`

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_joint_original.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa4/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa4/lasagna/configs/fiber_surface_joint_deformed.json
```

Choose fresh output roots for repetitions; completed outputs are protected.
Earlier CPU v1 attempts were interrupted when the user requested shared GPU runs.
Their partial directories are not completed results. Production runs above finished
with manager grants released on child exit. No source datasets were modified.
