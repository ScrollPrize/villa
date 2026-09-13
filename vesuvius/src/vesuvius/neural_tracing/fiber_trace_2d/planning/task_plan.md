# Embedding consistency analysis
1. Inspect shared live inference model, native input, tile geometry and blending.
   Freeze a completed raw snapshot; record step, SHA256, config and precision.
2. Deterministically select held-out fiber segments across five scrolls; dense
   physical GT sampling through tile overlap. Distinguish inference tile edges,
   dominant-contributor switches and arbitrary stored Zarr chunk boundaries.
3. Measure within-tile cosine versus distance, same-location cross-tile cosine,
   blended vector norm/cancellation and along-GT cosine across boundaries versus
   distance-matched interiors. Include negative/off-fiber similarity distributions
   and presence so collapse or background cannot masquerade as consistency.
4. Reuse native reads, model construction/preprocessing and blend weights.
   Analysis-only bounded embedding tiles need not change production publication.
   Include all contributing tiles for blended points; disclose storage precision
   differences and sampling limitations explicitly.
5. CPU synthetic regression tests, then managed real measurement through approved
   experiment caller. Preserve running training and verify ownership release.
6. Save reproducible JSON and Markdown results per scroll plus equal-scroll
   aggregate. Stop for user review before implementing tracer changes.

## Independent review refinements
Blend every contributing tile at all eight query corners before interpolation.
Measure cancellation as norm(sum(w*e))/sum(w*norm(e)), separately from amplitude.
Match boundary/interior pairs by arclength lag. Random off-GT controls are not
guaranteed distinct fibers; report presence strata and cosine quantiles. Initial
controls use conservative distance >80 base voxels from the full GT polyline,
which guarantees exclusion from its radius20/80 normal ellipse without adding
normal loading. This is not exactly the training candidate distribution.

## Spec update
Analysis-only diagnostic, no training/loss, cache-format or manager-protocol change.

## Docs updates
Report commands, data, checkpoint, metrics, caveats and timing; replace active log/status.

## Validation
Independent plan review against task/spec/plan. Synthetic identities, boundaries
and inconsistent embeddings; real manager-exclusive measurement.

## Changelog
Record diagnostic and findings when complete.
