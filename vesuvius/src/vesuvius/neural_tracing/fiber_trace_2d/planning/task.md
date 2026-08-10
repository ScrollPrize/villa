# Task: integer-DP fiberlet paths

Extend the C++ anchor-seeded fiberlet pipeline with its first path-connection
stage.

For every retained cell anchor, inspect anchors in cells on a discrete
approximately spherical shell at a fixed radius of four cell widths. Reject
endpoint pairs whose unoriented axes are incompatible with their chord, but do
not impose a final graph degree or global assignment yet.

For each remaining pair, solve a short directed path through the dense stored
fiber-prediction volume. Search states must be integer prediction-grid voxels,
not coarse cells or continuous/sub-voxel positions. Preserve fitted sub-voxel
anchors as exact virtual endpoints. Use a deterministic 26-neighbour,
forward-only dynamic program in a bounded corridor between the anchors.

The initial objective must contain:

- low-presence cost;
- unoriented fiber-direction disagreement with a per-prediction-axis
  quantization floor, so the best direction representable by the discrete move
  stencil has zero direction penalty;
- direct one-step curvature, split into tangent-plane turn and normal tilt
  using a separate regular Lasagna normal manifest, with the existing native
  tracer equations shared rather than copied.

Weight data costs by edge length so axial and diagonal moves are comparable.
Leave cumulative/history smoothness out of this stage.

Consume the existing versioned `anchors.json` as the authoritative anchor
input and verify it against the supplied fiber manifest. Produce a versioned
machine-readable collection of successful/rejected candidate paths and an OBJ
with one base-coordinate line group per successful fiberlet. The immediate
validation target is manual inspection on a small crop.

All spatial coordinates exposed by the CLI or its generated artifacts must be
base-volume coordinates. Prediction-grid coordinates remain an internal solver
representation. Discrete cell indices and parameters expressed as counts of
prediction-grid cells or voxels remain in their native lattice units because
they are not spatial coordinate values.

Path-quality rejection beyond endpoint/path feasibility, overlapping-path
deduplication, extension, global graph construction, H/V and winding labels,
CUDA batching, and a production connection radius remain later stages.

Add a `paths --stats` diagnostic flag reporting anchor and candidate counts,
accepted fiberlets, and min/mean/max objective scores for every scored path and
for the accepted subset. The report must state how many candidates have no DP
score. Until a quality threshold exists, every successfully scored path is
accepted, so the two score summaries are expected to match.

Make diagnostic OBJ paths robust in MeshLab by emitting every adjacent path
edge as an explicit two-index OBJ `l` element instead of relying on a
multi-index polyline record.
