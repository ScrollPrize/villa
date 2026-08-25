# Plan: durable Fiberlet crop trace artifacts

## Contract

- The authoritative crop output is a `traces` dataset kind in the existing
  Fiberlet Zarr v2 envelope. It uses the same strict metadata fingerprint,
  opaque sparse chunks, checksummed field blocks, and field-wise Zstd codec as
  preprocessing/cache datasets.
- A whole accepted crop trace is not a short preprocessing Fiberlet. Store its
  arbitrary float64 base-XYZ polyline directly in a trace payload; re-encoding
  it through the curved transverse-lattice codec would change geometry.
- Store the original deterministic result ordinal, seed base position and
  presence, total accumulated edge/join metric cost, prediction-space traced
  length, and all polyline points. Do not store transient scheduler counters or
  termination debug strings.
- Use a trace-only `float64_traces` storage profile with canonical otherwise
  unused codec fields. Trace metadata owns a crop-local ZYX chunk grid aligned
  to the source spatial chunk side in integer base coordinates. A trace is
  owned only by `floor((seed_base_zyx-origin_zyx)/chunk_side)`; a seed exactly
  on a boundary belongs to the upper chunk.
- Accumulate cost from the exact selected edges and joins. Crop-clipped edge
  cost and prediction length use the same retained fraction already used by
  lookahead. A bidirectional line includes its central join cost once when the
  graph defines one; the existing independent-side fallback has no central
  transition cost.
- Visualization must reopen and strictly decode the published dataset. It may
  not consume the in-memory trace result directly.
- Quality is `total_metric_cost / path_length_prediction_voxels`; lower is
  better. Stable ordering is quality then stored ordinal. Ten rank-decile OBJ
  files partition all lines exactly once, and a CSV plus console table records
  count and min/mean/max density for each bin.
- Only committed route cost is stored: complete selected edges contribute
  their full cost, only a crop-exit edge is prorated by the existing retained
  fraction, and internal joins contribute once. A bidirectional trace's
  reversed-negative-to-positive central join contributes once when defined.
  Speculative
  lookahead/horizon costs never enter the artifact.
- Publication is all-or-nothing. Write and fully validate a unique sibling
  temporary Zarr root, then rename it atomically to an output path that must not
  already exist. A visible output is therefore never a partial trace artifact.

## Storage implementation

1. Extend the shared Fiberlet payload codec with float64 fields and a strict
   trace-record payload containing structure-of-arrays metadata plus flattened
   XYZ point arrays. Validate finite/nonnegative costs and lengths, finite
   points, at least two points, valid offsets, unique ordinals, and a seed that
   belongs to the stored path. Add the trace-only storage profile and reject it
   for anchor/prefix/route payloads (and reject other profiles for traces).
2. Add `Traces` as a Fiberlet dataset kind and `FiberTraces` as a payload kind.
   Extend the common dataset metadata, Zarr array descriptor, chunk path,
   publish/read/decode path, and decoded-payload accounting without changing
   anchor/prefix/route behavior. Canonical trace metadata sets endpoint reach
   to zero and fixes unused bit-width fields rather than inheriting misleading
   preprocessing settings.
3. Add shared crop-artifact helpers that derive trace metadata from the source
   Fiberlet metadata, crop box, normal-manifest content, and effective tracing
   parameters. Include a trace contract version, every output-affecting trace
   parameter, source dataset fingerprint, and normalized Lasagna manifest
   content; exclude paths and thread count. Spatially own each line by its seed
   in crop-local chunks and write only nonempty chunks.
4. Store the exact populated chunk inventory and record count in trace metadata.
   Reopening enumerates the trace directory and rejects missing/unexpected or
   malformed chunk names, duplicate/missing global ordinals, records in the
   wrong owner chunk, and a count mismatch before restoring ordinal order.

## Tracing and CLI

5. Extend crop side tracing to accumulate authoritative selected edge and join
   costs and retained prediction length. Combine both sides plus any defined
   central join into each accepted line without changing selection, geometry, coverage,
   or scheduling.
6. Refactor `vc_fiber_trace_chunk` into explicit modes. The exact forms are
   `vc_fiber_trace_chunk trace INPUT_FIBERLETS ... --output TRACES.zarr
   [--obj LINES.obj]` and `vc_fiber_trace_chunk visualize TRACES.zarr --output
   LINES.obj`. Trace mode always reloads and visualizes; absent `--obj`, it uses
   the trace root with `.zarr` removed and `.obj` appended. The old mode-less
   invocation is intentionally removed. `visualize` accepts only an existing
   trace Zarr and regenerates OBJ/CSV
   artifacts without source Fiberlets, normals, or CT input. CT crop-face output
   remains a trace-time optional artifact because it depends on an external CT
   volume, but line visualization never bypasses stored traces.
7. Keep the existing all/direction/anchor OBJ naming and add ten quality files
   named `_quality_00_10` through `_quality_90_100` plus
   `_quality_histogram.csv`. For sorted rank `r` among `N`, bin is
   `min(9, floor(10*r/N))`; this also defines occupied bins for `N<10`. Empty
   bins produce valid empty OBJ files and blank numeric CSV cells. CSV and the
   console table report count and min/mean/max for both total cost and cost
   density.

## Tests

- Round-trip trace payloads with multiple records/point counts and reject
  corrupt offsets, nonfinite values, invalid costs/lengths, and wrong payload
  kinds.
- Create/reopen a sparse trace dataset and verify absent chunks are empty,
  present chunks are strict, records restore ordinal order, and stored float64
  point/cost bits round-trip exactly.
- Reject interrupted/partial roots, unexpected chunk files, duplicate/gapped
  cross-chunk ordinals, wrong seed ownership, and count/inventory mismatches.
  Cover zero traces, boundary seeds, and paths crossing owner chunks.
- Add crop tests proving one-sided and bidirectional global cost accumulation,
  including internal joins, the central join, and a crop-clipped edge.
- Add visualization tests for deterministic classification, histogram/decile
  partitioning, stable ties, fewer than ten lines, and OBJ generation from a
  reopened dataset.
- Add CLI parse/smoke coverage for trace publication, overwrite rejection,
  visualization-only regeneration, and an empty trace dataset.
- Build GCC Release and Clang, run focused crop/storage suites, run the canonical
  crop command, then compare the old and reloaded all/direction/anchor OBJ bytes
  where names are unchanged. Run `git diff --check`.

## Spec update

Replace OBJ-authoritative crop output with the durable trace-dataset contract,
exact stored fields/cost definition, reopen-before-visualize requirement, and
quality-density decile artifacts.

## Documentation updates

Document the `trace` and `visualize` commands, trace Zarr layout and provenance,
quality definition, histogram CSV, decile filenames, and the distinction
between complete crop traces and short preprocessing Fiberlets.

## Changelog

Add durable crop trace Zarr output and stored-data-driven quality visualization.
