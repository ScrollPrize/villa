# Plan: parallel-separate-winding labeling ablation

1. Add an explicit labeling configuration and constraints-mode CLI flag that
   excludes only non-hard links satisfying `parallel_score > 0.5` and
   `winding_distance >= 0.5` from model construction.
2. Build degree penalties, adjacency, component gauges, triangles, objective
   terms, and reported costs from the retained labeling graph. Continue to
   write all four constraint OBJ classes from the complete extracted report.
   Implement this inside the labeling API as a retained constraint vector/view;
   the original report and complete piece vector remain immutable and available
   to visualization writers.
3. Report excluded and retained labeling-link counts. Preserve hard continuity
   links even if their numeric values happen to satisfy the measured-class
   predicate.
4. Replace the single perpendicular constraint OBJ with disjoint
   `_perpendicular_same_winding.obj` and
   `_perpendicular_separate_winding.obj` outputs. Perpendicular score must be
   strictly greater than `0.5`; winding below `0.5` is same and exact `0.5`
   through the extraction cutoff is separate.
5. Make LP CSV and thresholded OBJ names use the supplied basename directly:
   `_values.csv`, `_h_even.obj`, `_h_odd.obj`, `_v_even.obj`, `_v_odd.obj`, and
   `_broken.obj`. Existing extension stripping remains unchanged. The benchmark
   must pass the explicit basename `$VES/data/workdir3/384/384`; it must not use
   the trace-derived default.
6. Add focused regression coverage for the exact threshold ownership, hard
   continuity preservation, model dimensions, and concise output names.
7. Build with GCC Release using `-j32` and run focused tests. Before rerunning,
   archive the complete committed baseline artifacts under
   `$VES/data/workdir3/384/full/`. Rerun the same centered 384 artifact with the
   explicit basename `$VES/data/workdir3/384/384`, updating the root artifacts
   without losing the baseline comparison set.
8. Compare the five thresholded class counts, objective components, retained
   links, triangle count, solve time, wall time, CPU, and peak RSS against the
   committed full-constraint baseline. Use one Release run, host-default 32
   threads, `--lp-relaxation --lp-parallel`, trace artifact
   `/tmp/crop_traces_central_384.zarr`, and normal manifest
   `$VES/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json` for both.

## Spec update

Document the opt-in solver-only exclusion predicate, hard-link exception,
reporting, and concise LP artifact suffixes. The default full-constraint solve
must remain unchanged.

## Documentation update

Add the ablation flag and explain that constraint visualization remains
complete even though the labeling model receives a filtered link set.

## Testing

Use a small graph containing `parallel_score == 0.5`, winding `==0.5`, excluded
measured, and predicate-matching hard links. Assert default-off model identity,
exact excluded/retained counts, reduced model dimensions/graph structure, and
concise output paths. Run `test_fiberlet_crop_trace` and the representative 384
Release command.

## Changelog

Record the explicit solver-only ablation and direct concise artifact naming.
