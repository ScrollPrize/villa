# Plan: fiberlet storage quantization experiment

## Scope and experiment contract

1. Add a dedicated `vc_fiberlets quantization-benchmark` command using the
   existing manifest, fiber JSON, normal manifest, extraction tube, beam, and
   replay options. It performs one anchor extraction, runs fiberlet DP once per
   distinct quantized endpoint geometry, and does not write a persistent
   quantized artifact.
2. Evaluate an unmodified float32 baseline plus isolated and combined storage
   scenarios. The exact matrix has 16 rows: float baseline; position-only
   `q=1,2,4`; compact-axis-only; `q=1,2,4` plus compact axis at float cost;
   cost-only `uint8` and `uint16`; and all six `q=1,2,4` plus compact-axis plus
   `uint8,uint16` cases. The storage chunk side defaults to 512 base voxels and
   is reported. Combined cost rows are compared both with the global baseline
   and their matching float-cost geometry row.

## Quantized geometry extraction

3. Quantize the extracted anchors before DP. Each distinct combination of
   endpoint position quantum and fitted-axis encoding is passed through the
   regular candidate generation, curved-domain construction, dense sampling,
   and DP solve. Do not transplant the baseline DP route onto changed endpoint
   geometry.
4. Cache each distinct geometry result. Cost-only scenarios reuse the baseline
   DP result, while combined cost scenarios reuse the corresponding
   position-plus-axis DP result.
5. Quantize canonical anchor positions globally in base coordinates using
   nearest integer `floor(value / q + 0.5)` (positive volume coordinates make
   the half-tie rule explicit), then assign storage chunks by the decoded
   coordinate. Recanonicalize every endpoint from its decoded key. Assign
   deterministic variants `0`/`1` by the persisted compact-axis bytes and then
   original `FiberletAnchorId` order at each coincident coordinate, including
   position-only scenarios whose DP still uses float axes. Resolve and validate
   the schema's local-position and endpoint-delta scalar widths for each
   scenario.
6. Round-trip fitted axes through the existing compact Lasagna encoder/decoder.
   The normal DP candidate setup sign-aligns the unoriented result to each edge
   chord. Any collision requiring more than two variants, duplicate or
   unresolvable decoded key, out-of-volume endpoint, endpoint collapse, or
   local-position/delta scalar overflow invalidates the whole scenario and
   suppresses graph/replay results. Ordinary candidate/DP rejection after an
   otherwise valid quantization is a measured tracing result, not a format
   failure.

## Cost and graph evaluation

7. Quantize successful total edge costs independently within the storage chunk
   containing the canonical first endpoint. Map each finite chunk minimum and
   maximum affinely onto the complete `uint8` or `uint16` range with no clipping;
   preserve constant-cost chunks exactly. Store `offset = float32(min)` and
   `scale = float32((max - offset) / max_code)`. Encode by float32 evaluation of
   `(cost - offset) / scale`, nearest-integer `floor(code + 0.5)`, and an exact
   maximum-to-`max_code` case; any other out-of-range value is an error rather
   than clipped. Decode with float32 `offset + scale * code` order. The decoded
   scalar is authoritative for beam ranking; do not reconstruct it by summing
   retained components.
8. Let the regular DP extraction resample dense prediction and Lasagna-normal
   values for changed paths and endpoints. Preserve stable original candidate
   identity for comparison, then map successful endpoints to their decoded
   coordinate-plus-variant keys before graph construction.
9. Build each valid scenario graph from its regular DP result. Recompute
   entry/exit directions, strict join-angle eligibility, and join costs. The
   beam denominator is that scenario DP's newly computed float32 path length;
   cost variants reuse the matching geometry length unchanged. Report its
   delta from baseline, added/removed transitions, and join-angle/join-cost
   error. Run existing replay over the same reference interval and
   configuration. Do not rerun anchor fitting, and do not rerun identical
   geometry DP work across cost widths.

## Report

10. Emit one stable key/value result row per scenario containing:
    - baseline and scenario replay failure counts, their signed difference, and
      symmetric maximum baseline-to-scenario Euclidean, Lasagna-normal, and
      Lasagna-tangential line distances in base voxels, plus invalid-normal
      sample count;
    - position quantum, axis mode, cost width, and chunk side;
    - anchor-key collisions, maximum variants, unresolved/duplicate original
      anchor IDs, and scalar-width failures;
    - accepted/rejected fiberlet counts and graph node/edge/transition
      counts;
    - position, endpoint-tangent, path-length, and decoded-cost error summaries;
    - absolute/relative decoded-cost error, global pairwise ordering inversions
      and top-k agreement over common successful candidates, plus per-chunk
      ordering diagnostics for cost-quantized rows;
    - replay completed fraction and selected edge count as diagnostics, without
      treating anchor, candidate, or point-index identity as quality metrics;
    - scenario wall time.
11. Keep the existing extraction profiling output so the input population and
    workload remain auditable. Quantization rows must be deterministic and
    machine-readable.

The maximum line distance is evaluated symmetrically over the disconnected
replay segment sets. Sample each source polyline at no more than one base voxel
between samples and project those samples onto the other set's actual line
segments in both directions. Do not connect across replay resets. Decompose
each closest-line displacement using the existing replay threshold measurement
and the Lasagna normal sampled at the projected target-line point.

## Testing

12. Add focused unit tests for quantized-anchor DP reruns, unchanged baseline
    extraction, position rounding and cross-chunk coordinate keys, strict
    two-variant collision handling, resolved uint8/uint16 and int8/int16 scalar
    boundaries, compact-axis round-trip/sign alignment, constant and ranged cost
    quantization, changed DP paths or acceptance, decoded graph/join
    recomputation, authoritative scalar cost, and deterministic reporting.
13. Build `vc_fiberlets`, `test_fiberlet_paths`, and `test_fiber_replay` with 32
    jobs; run the focused tests.
14. Run the canonical Paris4 5,000-base-voxel experiment with 32 threads and
    record every scenario's graph/replay changes and runtime. Quantization is
    experimental: do not select a production layout solely from unit tests.

## Spec update

Add the quantization-benchmark contract, quantized-anchor DP reruns, strict
failure behavior, scenario matrix, and report fields to `planning/specs.md`.
Do not specify a persistent storage version or accepted quantization yet.

## Documentation update

Update `volume-cartographer/docs/fiberlets.md` with the command, interpretation
of baseline/scenario rows, and canonical invocation. Update
`docs/fiberlet_storage.md` with measured results only after the benchmark runs.

## Changelog update

Add one entry describing the storage-quantization experiment and its measured
outcome. Do not describe any quantization as adopted until the user reviews the
results.

## Wide-corridor focused validation

15. Add `--scenario` selection to run baseline plus one named quantization case,
    while retaining the complete matrix as the default. A selected combined
    case must execute only its one additional position/direction DP geometry and
    cost decode. Update the durable spec and CLI documentation with accepted
    scenario names, two-row selected-run semantics, and report fields.
16. Measure baseline and scenario replay lines against the selected reference
    interval independently and directionally: resample each disconnected replay
    segment at no more than one base voxel spacing, project samples onto actual
    reference segments, and use the Lasagna normal at the projected reference
    point. Report total sample count; mean, median, and maximum Euclidean
    distance; valid-normal mean, median, and maximum absolute normal/tangential
    components; and invalid-normal count. Empty replay produces an unavailable
    distribution. Reuse the existing normal decomposition.
17. Emit progress for extraction, the selected geometry DP, and quality
    comparison so a wide-corridor run never appears stalled. Quality loops must
    emit initial, at-least-once-per-second periodic, and terminal progress with
    completed/total counts and ETA-equivalent elapsed/rate data.
18. Run baseline versus `P4+D+C8` over the complete reference interval with a
    768-base-voxel corridor and 32 threads. Record runtime, peak workload
    diagnostics, failure counts, quantized-to-baseline separation, and both
    reference-distance distributions.
19. Add focused tests proving selected runs execute only the required geometry,
    reference statistics ignore restart connections and point density, invalid
    normals are counted/excluded from component summaries, medians are correct,
    and progress contains initial and terminal events.

## Wide-corridor extraction scalability

20. Replace serial partition interval expansion and global sorting with a
    deterministic Z-slab preparation. Precompute exact row counts, fill and
    sort disjoint slabs using the configured anchor workers, merge intervals
    within each slab, and compact them in place. Preserve the exact sampled
    coordinate union and canonical ordering.
21. In diagnostics-free extraction, retain full cell results only when they
    contain anchors needed by selection and NMS. Tally per-cell diagnostics in
    worker-local counters and discard empty results immediately. Keep the
    diagnostics-enabled artifact path unchanged.
22. Account for raw interval storage and sparse result handles in the live-byte
    budget and expose interval-preparation wall/CPU time in the extraction
    profile. Keep progress visible while partitions are prepared.
23. Regression-test identical final anchors across worker counts, identical
    diagnostics-free versus diagnostic-enabled anchors, exact one-time shared
    sampling, bounded partition behavior, and canonical error precedence.
24. Benchmark the preserved pre-change binary and the optimized binary on the
    same bounded radius-768 interval, recording wall time, effective CPU use,
    and peak RSS. Only then resume the full-fiber radius-768 comparison.

## Float32 cache and Zarr-backed storage extension

25. Keep one logical fiberlet dataset format and add two explicit numeric
    encoding profiles: `float32_cache` and `compact_quantized`. Both profiles use
    the same logical anchor/fiberlet/route fields, chunk envelope, codec, reader,
    and graph path. The float32 profile stores stable cell/component endpoint
    identity, bit-preserved float32 anchor positions/axes, float32 total edge
    cost/path length, and the same entry/exit/middle integer lattice route as the
    compact profile. Do not store a redundant candidate index, expanded XYZ
    route, or individual cost components. Mixed profiles within one dataset are
    invalid, but they are not separate storage formats.
    Refactor the shared graph edge before cache integration so live and cached
    paths both use endpoint-pair/`FiberletId` identity and an authoritative
    scalar total cost. Keep candidate indices and decomposed costs as optional
    transient extraction diagnostics; cache-backed artifacts must not fabricate
    them. Prove uncached beam/replay behavior is unchanged by the refactor.
26. Add `separate` and `inline` anchor layouts. Separate layout publishes an
    `anchors` spatial array independently from the `fiberlets` spatial array;
    inline layout embeds the owning anchors in each fiberlet payload. Use
    separate layout for adaptive extraction, cache invalidation, and all partial
    or on-demand caches. Permit inline layout only for finite complete datasets,
    so a source edge can never reference a missing destination-anchor chunk.
27. Use a Zarr v2 root group and sparse `[1,1,1]` object-array chunks over the
    spatial chunk grid. Register one `vc-fiberlet-chunk` object codec with
    generic anchor/edge/inline `sample_format` identifiers; choose physical field
    encodings through dataset profile and field descriptors. Keep the outer
    compressor null because payload fields are transformed and compressed
    independently. Define its encoded bytes as an uncompressed fixed header and
    descriptor prefix followed by range-readable field blocks. Stock decoders
    may reject the custom array; metadata/store tools may inspect JSON and
    transport raw keys without decoding them. Before any writer lands, freeze a
    byte-level little-endian/IEEE-float codec appendix covering magic, enum and
    field IDs, packed offsets, uint64 bounds, checksum scope, and every logical
    anchor/edge/inline sample format plus profile encoding; do not serialize
    native C++ layouts.
28. Replace the proposed offset index with ordinary Zarr `z.y.x` ownership.
    Put source identities, algorithm/configuration fingerprint, coordinate
    space, encoding profile, anchor layout, base/prediction/grid shapes, float64
    prediction-to-base scale, complete anchor/path/graph settings, endpoint
    reach, anchor-refinement displacement, owner-halo bounds, and field-codec
    versions in root/array attributes. Put kind, encoding profile, key, uint64
    counts/offsets, fingerprint, field descriptors, and checksum in every custom
    chunk header. Include the producer toolchain/FP fingerprint for claims of
    bit-exact cross-build evaluation. Persist an exact reduced
    prediction-to-base ratio and integral prediction-grid origin; derive exact
    anchor ownership from global cell index using checked integer floor division
    and store the chunk's global owned-cell origin. Keep extraction diagnostics
    in benchmark/build output rather than duplicating them in graph chunks.
29. Build the adaptive cache spatially: publish anchors first; load anchor
    halos for bounded sets of first-endpoint owner chunks; prepare/sample/solve
    under a memory budget; commit completed fiberlet chunks; then release DP
    geometry. Preserve deterministic single ownership of cross-chunk edges and
    use the existing dense-volume cache plus a build-scoped, spillable scoring-
    page cache so interpolation corners are sampled once across work batches.
    Accumulate results in bounded per-field spool runs, externally merge them in
    canonical order, and stream compression so even one persisted owner chunk
    may exceed RAM without exceeding the working-memory budget.
30. Make resume strict. Reuse only final chunks with matching fingerprint and
    valid descriptors/checksum, compute missing expected chunks, and reject
    malformed or conflicting chunks. For finite builds, derive exact expected
    keys from root `build_domain`; for open on-demand caches, permit an arbitrary
    valid subset. Use missing, header-only, and nonempty final chunks to mean
    uncached, computed-empty, and computed-present without a duplicate coverage
    tensor. An inline anchor-present/fiberlet-empty chunk remains nonempty. Do
    not repair, migrate, or mix experimental profiles.
31. Make the spatial grid the on-demand index. Map an area directly to chunk
    keys; load anchors independently; load endpoint/cost/length/entry/exit field
    blocks over the endpoint halo for graph construction; and defer trailing
    middle-route blocks until selected geometry is needed. Use a bounded LRU for
    graph prefixes and selected route blocks. Beam/reset
    state stores stable IDs, not pointers. Before expanding an anchor or running
    a seed query, load every possible owner chunk in the declared endpoint halo
    so incident edges are complete, resolve separate or destination-inline
    anchor prefixes, sort identically to uncached search, and permit transparent
    eviction/reload without changing ties or results. Sum global anchor,
    fiberlet, and route-point record counts from headers without loading routes;
    extraction diagnostics remain build output. Publish local final files by temp-file,
    validation, file `fsync`, rename, and parent-directory `fsync` under a
    single-writer build.

## Cache validation

32. Round-trip float32 separate/inline datasets and compare anchor and endpoint-
    pair edge identities, float32 anchor/cost/length bit patterns, integer route
    choices, reconstructed route points, joins, graph replay, and failure metrics
    with uncached extraction under the matching producer numeric fingerprint.
    Test the shared scalar-cost graph adapter against the pre-refactor uncached
    graph and require identical ordering, beam decisions, and replay output.
33. Test cross-chunk ownership, sparse/missing chunks, strict fingerprint and
    checksum failures, interrupted local publication, finite-domain key
    derivation, partial-cache resume, prefix range reads,
    exact codec/sample-format identifiers,
    complete inline destination-anchor loading, strict rejection of partial or
    building inline datasets, and generic
    metadata/raw-chunk copying without the custom decoder. Check byte-identical
    golden payloads on supported OS/architectures and reject native-padding,
    endian, overflow, checksum-scope, and descriptor-overlap errors.
34. Measure a wide corridor larger than available working memory. Report peak
    RSS, wall/CPU time, encoded bytes by field, volume sample-call counts, cache
    reuse, and decoded graph equivalence. Include a synthetic owner chunk larger
    than RAM to validate streamed spooling independently of normal chunk density.
    Run graph replay with a cache budget smaller than its working graph and prove
    equality with uncached replay across repeated eviction/reload, then use the
    cache for the final P4+D+C8 evaluation.
35. Benchmark area-local access at dense `128`, `256`, and `512` base-voxel
    spatial chunk sides. Measure anchor-only, graph-prefix, and selected-route
    latency, decoded bytes, and peak memory for cold and warm local caches.
    Select the default chunk side from those measurements; add no finer persisted
    index unless a single dense chunk remains too expensive to parse.

## Extended spec update

Add the float32 encoding profile, separate/inline anchor layouts, Zarr v2
custom-object envelope, mandatory metadata/fingerprint rules, strict chunk
publication/resume behavior, spatial edge ownership, and cache-equivalence
requirements to `planning/specs.md` when implementation begins. Do not declare
the experimental sample formats stable or add backward compatibility.

## Extended documentation update

Keep `docs/fiberlet_storage.md` as the detailed format proposal. When
implemented, add float32 cache creation/resume/inspection commands and the custom
Zarr codec limitations to `volume-cartographer/docs/fiberlets.md`.

## Extended changelog update

When code lands, add one entry for bounded float32 fiberlet caching and its Zarr
spatial envelope. Planning-only schema edits do not claim implementation.
