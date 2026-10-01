# Plan

1. Reconcile the crop/BP `fiber-exp5` history with the latest `fiber-exps4`
   training history on a new `exps5-train` branch; preserve both durable
   changelog histories and replace only active task documents.
2. Trace the existing deterministic test path, compact fiber geometry, native
   Lasagna normals, augmentation transforms, model-output coordinates, EMA
   evaluation, and TIFF visualization helpers. Reuse shared implementations.
3. Define an opt-in cross-section diagnostic. At eligible supervised fiber
   centers, construct tangent, transformed surface-normal, and in-sheet
   transverse axes. Sample predicted presence across a fixed 160 x 80
   base-voxel field without changing the evaluated test crop or refinement.
4. Accumulate sum/count and a fixed bounded per-pixel histogram on each rank;
   reduce accumulators deterministically after evaluation. Derive per-pixel
   0.001, 0.01, and 0.10 quantiles without retaining all sample planes.
5. Write model/snapshot-qualified float TIFFs plus compact metadata describing
   axes, physical extent, resolution, histogram bins, sample count, and source
   checkpoint. Keep raw and EMA outputs separate and overwrite-safe.
6. Add CPU regression tests for orientation, coordinate scaling, histogram
   quantiles, distributed merge semantics, empty/invalid normals, and unchanged
   evaluation metrics when disabled. Run focused existing test suites.
7. Add a reusable approved-runner experiment mode/config, run best/current raw
   and EMA snapshots from the latest run through the cooperative GPU manager,
   and report the resulting image paths and numeric profile comparisons.

## Spec update

No protocol/spec change: this is an opt-in evaluation artifact only.

## Docs updates

Document settings, TIFF semantics, commands, snapshot identities, results, and
limitations in the task log and durable changelog.

