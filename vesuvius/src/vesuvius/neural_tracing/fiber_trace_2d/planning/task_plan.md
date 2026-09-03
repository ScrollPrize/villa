# Plan

1. Generate a new immutable crop-trace artifact with the three staged filter
   passes and no `--max-fibers` or `--max-attempts` limit. Confirm tracing ends
   because every candidate is covered or attempted, not because of a cap.
2. Run the canonical supervised oracle-inlier pruning benchmark against that
   exact staged trace cohort, preserving current defaults and the established
   reference set, quality fraction, and 512-base piece split.
3. Run `reference-replay-benchmark` directly against the source Fiberlet Zarr
   with the identical ordered stage list, crop, references, and physical voxel
   size. Confirm the stage options are accepted and applied before replay graph
   materialization.
4. Capture Git revision, Release build, source identities, commands, stage
   populations, output hashes, wall/user/system time, maximum RSS where
   available, and the headline metrics in dedicated benchmark run records.
5. Update the benchmark result index and plot data only with measured values,
   regenerate plots, and validate their consistency with the run records.

The records must freeze effective scientific defaults, distinguish the new
uncapped cohort from the old capped cohort, identify aggregate overlapping
stage counts as non-unique, hash durable artifacts, and disclose that the
local source is an unverified `build_state=partial` mirror. Successful sparse
reads do not prove parity with the unavailable authoritative remote inventory.

## Spec Update

No behavioral specification change is planned. Benchmark records must treat
the staged schedule and uncapped trace cohort as distinct provenance from the
earlier unstaged capped cohort.

## Docs Update

Add the complete staged commands and link the new run records from the Fiber
benchmark result index.

## Changelog

Record the first complete staged crop and staged reference-replay benchmark.

## Validation

Verify both benchmark commands exit successfully, inspect their JSON/Zarr
metadata, cross-check headline counts against detailed reports, regenerate the
two benchmark plots, and run the focused crop-trace and storage tests.
