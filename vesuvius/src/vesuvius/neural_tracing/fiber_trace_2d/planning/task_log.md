# Task Log: Iterative H/V consensus growing

- The exact-perpendicular HiGHS experiment was stopped after 490.78 seconds at
  effectively one core and 509,964 KiB peak RSS without reaching a solution.
  Its implementation remains in the worktree as a separate opt-in diagnostic.
- The requested consensus path is separate from HiGHS and reuses the canonical
  constraint extractor.
- The phrase "maximum constraint score" together with distance/count evidence
  is implemented as maximum `constraint_count / mean_distance`, so more and
  closer links have stronger priority. This interpretation is explicit in the
  plan and documentation.
- Independent review clarified that the seed must measure straightness rather
  than chord alone, source-pair multiplicity and summation order must be exact,
  and greedy broken cost is order-sensitive. The plan now specifies
  straightness-first seeding, stable per-piece evidence, irreversible
  current-active broken costs, degenerate handling, and exact milestone rules.
- Added the consensus core API, separate CLI mode, full-trace H/V OBJ writer,
  and scheduled snapshot writer. HiGHS-only flags are rejected by consensus.
- Validation: `cmake --build volume-cartographer/build --target
  vc_fiber_trace_chunk test_fiberlet_crop_trace -j32` succeeded and
  `volume-cartographer/build/bin/test_fiberlet_crop_trace` passed 32 cases.
  A CLI smoke check confirmed that consensus rejects `--hv-only` as a
  HiGHS-only option before opening input data.
- Centered-384 diagnostic used `/tmp/crop_traces_central_384.zarr`, the
  `las008_s1_full/las_008.lasagna.json` normal manifest, unsplit fibers,
  maximum distance 256, default exclusive winding cutoff 1.5, broken cost
  0.25, and host-default 32 threads. Extraction retained 3,776 of 11,840
  measured candidates for 179 traces. Consensus produced H=52, V=25,
  broken=102, orientation cost 90.292411, broken cost 200, and objective
  290.29241. Total command time was 0.12 s wall / 0.68 s CPU with 87,072 KiB
  peak RSS. Final and milestone OBJ files were written below
  `$VES/data/workdir3/384/`.
- Added a console table for the first ten consensus choices with trace,
  component, seed status, label, evidence, distance/priority, H/V/broken costs,
  and selected incremental cost.
- Formatted that table into fixed-width columns and limited floating values to
  three decimal places for terminal readability.
- Follow-up seed refinement: the primary seed will be restricted to traces
  longer than half the smallest stored crop extent, then ranked by
  straightness, exact distance from the crop center to the polyline, arc
  length, and trace index. Later disconnected-component seeds deliberately
  relax only the length cutoff so labeling remains complete.
- Independent review confirmed that crop center and nominal side must come
  from the artifact bounds, the half-side threshold must remain strict, and
  distance must be the minimum over full 3D polyline segments rather than
  vertices. The implementation and regression tests use those definitions.
- The centered-384 rerun selected trace 7 as the primary seed with arc length
  237.196 base voxels (strict cutoff 192), straightness 0.997479, and exact
  crop-center distance 175.157 base voxels. It produced H=67, V=70, and
  broken=42 in 0.13 s wall / 0.57 s CPU with 86,524 KiB peak RSS. The previous
  short perfectly straight seed is no longer eligible.
- Expanded the detailed console choice table from 10 to 100 assignments and
  moved the complete consensus count/cost summary to the final output block.
- Replaced the overly wide final summary with an aligned two-column metric and
  value table; floating costs use three decimal places.
- Added broken-fiber OBJ output beside H and V for both final labels and every
  existing milestone. Valid empty files are always emitted; degenerate
  non-assignment inputs remain excluded from all three class layers.
- Validation: the focused suite still passes 32 cases. The centered-384 output
  was regenerated in 0.14 s wall time and now contains 42 complete broken
  fibers in `384_broken.obj`, with matching broken milestone files through
  `384_step_100_broken.obj`.
