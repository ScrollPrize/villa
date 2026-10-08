# Shared BP / Lasagna constraint extraction

Inspected 2026-10-08 after merging `fiber-exp5` at `d7deec459` into
`global-fiber-sgd`. Last shared extractor/BP change on that lineage: `be3b45d36`.
No alternative extraction implementation was added.

## Inputs and reusable boundary

`extractFiberTraceConstraints` in
`volume-cartographer/core/src/fiber_tracer/FiberTraceConstraints.cpp` accepts
polylines (`FiberletCropTraceLine`), extraction settings, scalar/batched winding
distance callbacks, optional pair filtering, and an aligned normal field. It
returns piece geometry plus explicit pair observations (`FiberTraceConstraintReport`).
The extractor does not require the fibers to have been produced by the fiberlet
tracer, or to carry known H/V labels. The AFV reader can therefore feed it via a
small adapter; that adapter is not implemented by this merge.

The Lasagna inputs are **sheet normals** (`nx`, `ny`) and `grad_mag` winding
density, not the fiber-tangent `nx`/`ny` channels in the AFV's inference provenance.
Match the coordinate identity and source-to-base transform to Paris4 L0.
The AFV's fiber-prediction manifest alone is not a substitute for sheet Lasagna.

## Geometry and candidate pairs

Polylines are parameterized by arc length. A spatial RTree finds nearby resampled
points from different source traces; it is not an all-pairs search. Normal
integration uses shared batched C++ sampling with OpenMP and chunk caching.

There are two deliberately retained extraction modes, not two separate copies:

| Mode | Measurements | Piece layout |
| --- | --- | --- |
| Legacy (default) | Closest sampled encounter per piece pair, soft parallel/orthogonal scores | Nominal 512-base-voxel pieces, 128 overlap |
| Local events (opt-in) | Repeated localized encounters, side-consistency certification | Clean cuts between consecutive orthogonal events; no overlap by default |

Header defaults (individual benchmark configs can override): resampling32,
candidate distance128, tangent window32 and integration step8 base voxels;
maximum accepted winding distance strictly below1.5. Local settings: event
spacing64, one-sided tangent span16, parallel probe span64, reciprocal tolerance8,
angular tolerance30degrees, distance tolerance0.25windings and refinement2rounds.
These are inherited settings, not a newly selected best AFV configuration.

## Orientation and local certification

Legacy mode uses unoriented tangent agreement and correspondence walks along
the two curves to obtain normalized parallel/perpendicular evidence. Parallel
distance normally aggregates multiple connector measurements by median;
orthogonal distance uses the encounter connector.

Local mode compares before/after tangents independently on each source curve.
All available side combinations must agree: parallel within30degrees or
orthogonal within30degrees of90. Ambiguous encounters are rejected. Bounded
closest-point refinement and reciprocal projection checks improve localization.
Parallel events walk correspondence in both directions; orthogonal events probe
each trace independently around the crossing. The probes must agree on distance
within0.25windings, its rounded category (integer grid for parallel, half grid for
orthogonal), and signed order where applicable. Ordinary extraction requires
full side support; preserve-input/reference paths explicitly permit available
one-sided endpoint support. Repeated parallel evidence is normalized by the
represented arc interval, so denser sampling does not simply multiply its weight.
In local mode the default closest-distance estimator uses the base encounter's
distance for the final parallel measurement; the other probes certify consistency
rather than supplying a median replacement. This can be overridden independently.

## What Lasagna measures

For a straight connector from A to B, the shared sampler estimates

`d = integral rho(x) * abs(n(x) dot connector_unit_direction) ds`.

Here `rho = decoded grad_mag = encoded * grad_mag_factor / grad_mag_encode_scale`
is winding density per base voxel; `n` is the sheet-normal axis. Thus distance
along a sheet contributes little and distance across sheets contributes more.
This is a field-integral estimate, **not explicit counting of segmented sheets**.
Trapezoidal integration samples at up to8base-voxel intervals by default.
Missing/invalid samples invalidate the measurement rather than making it zero.
Normal axes use the existing sign-invariant compact-axis interpolation.

For order/sign, an additional binary BP aligns the otherwise sign-ambiguous
sheet normals on a lattice: choose keep/flip to favor neighboring agreement,
with one fixed orientation per connected component. A signed observation requires
A, midpoint and B to belong to the same aligned component. The sign is the
connector's dot product with the **aligned midpoint normal**. The code does not
integrate a signed gradient along the whole connector. Normal alignment magnitude
and component ID are retained as confidence/context. Disconnected components
do not establish a common global sign or winding origin.

## Splitting and solve are separate

In local mode observations are certified on the original curves first. Cuts are
then placed midway between consecutive orthogonal encounter arcs. The certified
measurements, support spans and weights are remapped unchanged to the owning
pieces: shortening a piece does **not** revalidate or discard its observation.
Adjacent pieces receive continuation links. Legacy/fine-piece and additional
scoring options remain available for comparisons.

Each observation records endpoints/arcs, parallel/perpendicular evidence,
unsigned and optional signed winding distance, normal alignment/component,
evidence weight and source support. These are directly useful inputs to a future
SGD objective without copying the extractor.

The winding solver is downstream: it interprets relative H/V evidence and
integer/half-step winding relationships, separates magnitude and order penalties,
and can disable pieces or soften continuation. Hard signs, class weights,
distance-scaled continuation costs and removal policies are **solver settings**,
not guarantees that the measured constraints are true. Current header defaults
include finite sign penalties with confidence-based hard-sign promotion; do not
assume every historical run used the same setting. No GT/reference assignments
are needed for ordinary extraction; benchmark reference filtering/readout are
separate operations.

## Sources

- `core/include/vc/fiber_tracer/FiberTraceConstraints.hpp`: API/config/report.
- `core/src/fiber_tracer/FiberTraceConstraints.cpp`: search, scoring, certification, cuts.
- `core/src/lasagna/LasagnaNormalSampler.cpp`: field sampling/integration.
- `core/src/fiber_tracer/LasagnaNormalAlignment.cpp`: normal keep/flip alignment.
- `core/src/fiber_tracer/FiberTraceWindingBeliefPropagation.cpp`: solver costs.
- `apps/src/vc_fiber_trace_chunk.cpp`: existing orchestration and callback wiring.

All paths above are relative to `volume-cartographer/`.

## Merge validation

One conflicted file: `volume-cartographer/python/CMakeLists.txt`. Retained both
`vc_fiber_tracer` (fiber branch) and `utils_volcomp_codec`, `volcomp` (main) in
the runtime-path and installation target lists. Core extraction, normal alignment
and winding BP source files are identical to the merged fiber-exp5 versions.

Approved build command attempted:
`bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build`.
Configuration stopped at missing `SQLite3_LIBRARY` / `SQLite3_INCLUDE_DIR`.
No packages installed; no old binary is presented as a rebuilt merge validation.
On Ubuntu the missing development package is `libsqlite3-dev`; installing it
requires separate approval, and later build requirements remain unverified.

Benchmark-support tests:
`/home/hendrik/.venv_las/bin/python -m pytest -q volume-cartographer/scripts/tests/test_plot_fiber_benchmarks.py volume-cartographer/scripts/tests/test_crop_fiber_jsons.py`
—12passed. This does not replace the blocked native BP/constraint tests.
