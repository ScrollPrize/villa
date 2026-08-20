# Task log: fiberlet storage quantization experiment

## Findings

- Endpoint position and fitted-axis quantization change the cubic-Hermite curve,
  transported transverse planes, layer count, sampled dense values, and DP
  optimum. A baseline interior route is therefore not meaningful after either
  endpoint representation changes.
- The graph beam ranks stored edge and transition costs and uses the path length
  computed by that geometry's DP as its denominator. It does not rescore
  interior samples while expanding the graph.
- The existing compact Lasagna normal encoder/decoder is suitable for testing
  fitted-axis storage. Its measured mean/max unoriented angular errors were
  0.490/5.325 degrees on the canonical interval.
- The first full-fiber radius-768 attempt reached roughly 54.5 GiB RSS and
  heavy swap before completion. A 5,000-base-voxel probe showed the anchor
  fitter using its configured workers, followed by a one-thread partition
  setup phase.
- Partition setup expands every tile into row intervals, repeatedly reserves
  the growing vector, globally sorts it on one thread, and only then starts the
  parallel sampler/fitter. Diagnostics-free extraction also allocates a
  1,456-byte `FiberCellAnchorResult` slot for every work cell even though the
  overwhelming majority contain no retained anchor.
- The optimized 1,000-base-voxel radius-768 probe completed 155,028 anchor work
  cells but generated 3.82 million fiberlet candidates. The complete command
  took 10m33.82s, averaged 10.07 CPU cores, and peaked at 59,354,960 KiB RSS.
  The remaining peak comes from retaining prepared geometry/index data for all
  candidates, so the full-fiber corridor is not feasible as one in-memory run.
- The storage proposal now defines one logical format with float32-cache and
  compact-quantized encoding profiles. Both use the same Zarr v2 spatial object
  arrays, logical fields, route lattice, codec, reader, and graph path. The
  adaptive cache stores anchor chunks separately and publishes bounded fiberlet
  owner chunks.
- An intermediate draft expanded float32 routes and individual loss components,
  but those values are redundant for replay. The final plan keeps the compact
  layout's exact integer route choices and stores only float32 anchor geometry,
  total cost, and path length where compact storage quantizes them. The current
  whole-volume raw float32-cache estimate is therefore about 131.4 GiB before
  field compression, not 459 GiB.
- The revised plan derives finite expected keys from root build metadata and
  uses chunk absence/header-only/nonempty state directly, avoiding a redundant
  coverage tensor. It also uses bounded per-field spool/merge encoding for owner
  chunks larger than RAM and validated local temp-file publication. Inline
  anchor-only chunks retain their anchor arrays.
- Spatial keys provide direct area lookup. Anchors, graph-prefix fields, and
  selected route geometry can be loaded independently; chunk-side benchmarks
  decide whether the spatial chunk alone is a sufficiently fine index.
- The implementation must make the existing graph path consume scalar total
  edge cost and stable endpoint-pair/`FiberletId` identity. Candidate indices,
  decomposed costs, and rejection diagnostics stay transient rather than being
  added to the cache solely to satisfy the old in-memory graph structure.
- Partial/on-demand caches require separate anchor chunks. Inline layout is
  accepted only when a finite dataset is complete, preventing cross-chunk edges
  from referencing anchors in absent inline chunks without duplicating anchors.

## Implementation

- Added `vc_fiberlets quantization-benchmark`. It performs one production anchor
  extraction, quantizes copied anchors, and reruns the ordinary candidate
  generation, dense sampling, curved-plane construction, and DP once for each
  distinct endpoint geometry.
- Cached the seven additional geometry solves and reused them for matching
  `uint8`/`uint16` cost scenarios. The command evaluates 16 deterministic
  baseline, isolated, and combined rows.
- Added strict coordinate-key, two-variant, volume-bound, endpoint-collapse,
  local-position-width, and endpoint-delta-width validation. Ordinary DP
  rejection remains a measured result rather than a representation error.
- Added per-first-endpoint-chunk affine cost quantization, exact float32 decode
  arithmetic, global and within-chunk ordering diagnostics, and comparisons
  against both the float baseline and each cost row's matching geometry.
- Added focused regression tests and documented the command and measured result.

## Validation

- Built with 32 jobs:
  `cmake --build volume-cartographer/build --target vc_fiberlets test_fiberlet_paths test_fiber_replay -j32`
- Focused tests:
  `ctest --test-dir volume-cartographer/build -R 'test_fiberlet_paths|test_fiber_replay' --output-on-failure`
  passed 2/2 in 0.12 seconds.
- Canonical benchmark:
  `volume-cartographer/build/bin/vc_fiberlets quantization-benchmark /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --length 5000 --threads 32 --radius 64`
- The final run extracted 26,494 successful baseline fiberlets over 2,563 graph
  anchors in 4.87 seconds. All 16 scenarios were representable and the complete
  command took 33.19 seconds (`user=531.44`, `sys=4.44`).
- No scenario added a replay tracing failure on the canonical interval; the
  baseline and every scenario reported zero failures.
- Position `q=1/2/4` produced maximum Euclidean line separations of
  10.15/9.07/11.46 base voxels. Maximum Lasagna-normal separations were
  4.78/4.89/5.11 and tangential separations were 10.00/9.05/10.92 base voxels.
- Compact fitted axes alone produced 3.56 base voxels maximum Euclidean line
  separation: 0.60 normal and 3.56 tangential.
- `uint8` costs caused 861,684 global and 113,370 within-chunk ordering
  inversions, retained 89/100 top entries, and produced 5.35/2.43/5.22
  Euclidean/normal/tangential maximum line separations.
  `uint16` costs caused 3,463 global and 488 within-chunk inversions, retained
  all top-100 entries, and reproduced the exact baseline line.
- Position quantization created 8/16/26 coincident coordinate groups for
  `q=1/2/4`; none exceeded the two-variant limit. With 512-base-voxel chunks,
  `q=1` selected 16-bit local positions and endpoint deltas, while `q=2/4`
  selected 8-bit fields for this interval.

## Outcome

- No persistent format or production quantization has been selected.
- The experiment now measures the complete tracing consequence of decoded
  endpoint data. The results make compact axes and `uint16` costs plausible,
  while even one-base-voxel endpoint quantization needs further quality study.
- Replay candidate identity and corresponding point-index displacement were
  removed as quality metrics. The final decision inputs are tracing failures
  and symmetric maximum Euclidean/normal/tangential line separation.
