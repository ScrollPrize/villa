# Plan: canonical anchor reuse across quantized fiberlets

## Contract

- Canonical anchor extraction, fitting, filtering, endpoint scoring, serialized
  chunks, and cache identity are independent of every evaluation quantization.
- Position and fitted-direction transforms are derived from canonical anchors
  inside the scenario context. Position transforms resample endpoint prediction
  and normal metadata; direction-only transforms retain canonical endpoint
  scoring. Transformed endpoints must be shared consistently by fiberlet DP,
  route reconstruction, replay seeding/arcs, transitions, and cost ownership.
- Fiberlet cache identity continues to include position/direction geometry and
  the opaque legacy u8 tag, preserving existing Q4 fiberlet cache reuse.
- Cost decoding remains graph-only and shares the matching fiberlet geometry.
- Keep strict metadata validation. Do not accept an arbitrary incompatible
  canonical anchor or fiberlet cache merely because its files exist.

## Implementation

1. Build anchor metadata/root only from the canonical profile. Build fiberlet
   metadata/root from the selected geometry profile. Allow the paired datasets
   to have different fingerprints while strictly checking source, grid, scale,
   chunk layout, and extraction compatibility.
2. Move position/direction evaluation transforms out of anchor chunk generation
   into a shared preprocessor endpoint-view helper. Batch-resample each
   position-transformed anchor chunk once per scenario. Retain transformed
   chunks through a bounded LRU with single-flight construction so overlapping
   fiberlet chunks share work without accumulating the whole corridor.
3. Make the chunk graph consume the same transformed anchor view for chunk
   enumeration, individual anchors, edges, route reconstruction, transitions,
   and compact-cost ownership.
4. Keep the exact anchor cache root common across baseline and every scenario,
   including explicit `--anchor-cache`. Keep scenario fiberlet roots unchanged
   so completed Q4 data is reused.
5. Document the one-anchor-cache plus per-geometry-fiberlet-cache model.
6. Add explicit `compact_axis_cost_u8` and `compact_axis_cost_u16` scenario
   descriptors. Keep their geometry key exactly `(position quantum 0, compact
   direction true)` so both reopen the existing compact-axis fiberlet cache and
   differ only in graph-side cost decoding.
7. Update the deterministic standard matrix to 18 total scenarios and 17
   non-baseline `--scenario all` rows. Preserve selected cost bits `0/8/16`
   independently of the compact-axis geometry namespace's opaque historical
   u8 compatibility tag.

## Testing

- Unit-test endpoint transformation for exact, compact-direction, and Q4
  position views and ensure canonical input is unchanged.
- Test graph anchor/edge/route consumers with a transformed anchor callback.
- Run the focused storage and replay tests.
- Run short cold/warm `compact_axis` and `position_q4` comparisons against one
  output root. Verify both use the same canonical anchor root, the second run
  does not modify anchor files, and scientific outputs match the earlier
  scenario implementation.
- Open the existing radius-768 scenario fiberlet caches through canonical
  anchors and run both requested full-corridor comparisons. Explicitly cancel
  and drain batch-owned speculative cache work before process-static worker
  teardown, and preserve each completed result line immediately.
- Assert the two compact-axis cost scenarios map to the same geometry-cache
  profile, algorithm identity, and anchor/fiberlet roots as `compact_axis`,
  while each graph sees its selected cost bits. Add a no-mutation sequence over
  float/u8/u16/float views and an existing-cache replay whose generators fail
  if invoked.
- Run focused tests, then both full radius-768 comparisons with the canonical
  anchor and baseline fiberlet overrides. Snapshot the canonical anchor and
  compact-axis prefix/route cache trees before and after. Require the same
  cache namespaces and no rewriting of existing payloads; record any missing
  on-demand compact-axis chunks completed while forming stable per-owner cost
  ranges. Report failure counts plus Euclidean/normal/tangential summaries.

## Spec Update

- State that anchor cache identity is always canonical and evaluation geometry
  begins only at the fiberlet layer.
- Define all transformed-anchor consumers and position-rescoring behavior.
- Add the two compact-axis cost views and update the standard matrix count.
- Define their fixed matrix order and distinguish selected cost bits from the
  opaque geometry-cache compatibility tag.

## Docs Update

- Document one shared anchor cache, scenario fiberlet caches, and what work is
  repeated for position/direction variants.
- Document that compact-axis `uint8`/`uint16` costs reuse compact-axis geometry.

## Changelog

- Record elimination of repeated anchor extraction across geometry scenarios.
- Record the added compact-axis replay-cost comparisons.
- Record exact commands, cache paths, revision/build mode, cache snapshots,
  wall/CPU/RSS, failure counts, and distance summaries in `task_log.md`; finish
  the matching `status.md` checklist.
