# Task: fiberlet storage quantization experiment

Add an experiment to the current C++ fiberlet tracer that measures when the
numeric quantizations proposed in `docs/fiberlet_storage.md` change graph search
or replay quality. Test anchor-position quanta of 1, 2, and 4 base voxels,
existing two-byte fitted-axis encoding, and per-spatial-chunk `uint8` and
`uint16` total fiberlet costs before implementing compression or a persistent
format.

The quality decision must be based on whether quantization increases graph
replay tracing failures and on the maximum geometric line-to-line distance
between the baseline and quantized replay. Report Euclidean distance and its
Lasagna-local normal and tangential components separately. Exact anchor,
candidate, point-count, or point-index agreement is not a quality requirement.

Validate the `P4+D+C8` case against baseline over the same complete reference
fiber with a 768-base-voxel extraction corridor. Avoid evaluating unrelated
quantization scenarios. In addition to failure count and quantized-to-baseline
line distance, report mean and median Euclidean, Lasagna-normal, and
Lasagna-tangential line distance from baseline and quantized replay to the
reference fiber. Keep command-line progress visible throughout the long run.

The wide-corridor run must use the configured worker count during anchor
partition preparation and must keep memory bounded by retaining only anchors
needed by later NMS and graph construction when diagnostic artifacts are
disabled. It must not materialize full diagnostic cell results for every empty
corridor cell.

The experiment must reuse one anchor extraction, quantize endpoint positions
and fitted directions before constructing and solving each distinct fiberlet DP
geometry, report invalid/colliding representations, and compare each quantized
graph replay with the float32 baseline. Cost variants may reuse the matching
geometry's DP result.

Extend the proposed fiberlet dataset with a non-quantized float32 cache encoding
profile so wide-corridor extraction and evaluation do not retain every fiberlet
in memory. It is a profile of the same logical format, loader, route arrays, and
chunk envelope as the compact encoding, not a separate format. Support both
inline anchors and anchors stored independently from fiberlet chunks; the
adaptive cache uses the independent layout. Store the dataset as Zarr metadata
and sparse spatial chunk keys. Chunk payloads remain custom structure-of-arrays
objects identified by a VC codec/sample format, so generic Zarr tooling can
inspect metadata and move raw chunks but cannot decode them as ordinary tensor
data.

The float32 profile stores stable endpoint identities, float32 anchor positions/
axes, float32 total cost and path length, and the same lossless integer route
lattice choices as the compact profile. It does not store redundant expanded
XYZ routes or unused individual cost components. Construction must remain
memory-bounded even when one persisted owner chunk exceeds RAM. Resume state
must be machine-readable and unambiguous. Local publication must
not expose partial chunks, and spatial chunks must support selective on-demand
anchor, graph-prefix, and selected-route loading.

The shared graph/replay implementation must consume the minimal persisted edge
directly: endpoint-pair/`FiberletId`, scalar total cost, length, tangents, and
lazy route. Candidate indices and decomposed costs remain transient diagnostics.
Partial/on-demand caches use separate anchors; inline anchors are supported only
for finite complete datasets so cross-chunk endpoint references cannot dangle.
