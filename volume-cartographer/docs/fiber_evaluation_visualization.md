# Aligned fiber evaluation visualization

The native APIs in `vc/fiber_tracer/FiberReplayVisualization.hpp` render
existing evaluation results without rerunning inference or changing tracing.
They are shared by dense/chunk replay inspection and graph-replay comparison.
They do not publish an evaluation bundle or start subprocesses.

## Views and coordinates

- `renderFiberReplayOverview` compares reference, direct replay and fiberlet
  graph replay in reference-aligned and traced-line-aligned top/side views.
- `renderFiberReplayTraceOverview` renders a single native replay, including
  disconnected components and restart markers.
- `renderFiberReplayInspection` pairs CT and presence views and adds transverse
  sections. Section centers map back to actual traced-line points; restart gaps
  are never interpolated into a continuous path. Breakoff/restart sections are
  retained even for singleton segments without longitudinal strips.
- `writeFiberletCropBoxVisualization` (in
  `vc/fiber_tracer/FiberletCropVisualization.hpp`) writes six CT TIFFs, six
  textured OBJ faces and one MTL for a base-coordinate bounding box.

All line and box coordinates are **base-volume XYZ voxels**. Normal and prediction
samplers have their own declared working-to-base scale. Texture callbacks receive
XYZ coordinates in the selected CT array's lattice, not in base coordinates.
`validateFiberReplayStripCtVolume` accepts one local, concrete uint8 OME-Zarr
array; it reads the parent ZYX multiscale scale/translation relative to the first
dataset. A whole pyramid, uint16 array, or missing coordinate metadata is rejected.
For another storage source, supply an explicit `FiberReplayStripTextureSource`
and rendering callback instead.

The inspection layout uses two base voxels per pixel. Its section axes/radii
come from the shared replay acceptance metric, including the replay's tangential
factor and invalid-normal fallback. An ellipse overlay describes the evaluator's
tolerance, **not** an inferred fiber width. Top/side ribbons use the shared VC3D
line-view builder and require usable line normals.

Presence uses the existing prediction source, including its readiness gate:
it converts CT-array coordinates back to base, then to prediction working
coordinates, and samples in bounded batches. Multiple prediction options are
shown as maximum presence, even if an option's direction is invalid. Values are
displayed as uint8 `255 * presence`; there is no per-image contrast stretch.
Missing CT samples remain zero, with coverage/error counts logged by the sampler.
An optional integer presence threshold adds binary rows without altering traces.

## Calling from an evaluation

Link against `vc_fiber_tracer`. Keep the volume, prediction source and normal
sampler alive for the duration of the synchronous call. Given an existing
`FiberReplayTraceResult replay` and its selected reference geometry:

```cpp
#include "vc/fiber_tracer/FiberReplayVisualization.hpp"
#include "vc/core/util/SurfaceTexture.hpp"

const auto source = vc::fiber_tracer::validateFiberReplayStripCtVolume(
    volume, ctArrayPath);
const auto ct = [&](const cv::Mat_<cv::Vec3f>& coordinates, int scale)
    -> cv::Mat_<uint8_t> {
    return vc::core::util::renderCoordsTextureFineToCoarse(
        coordinates, volume, 0, scale, "Evaluation CT");
};
const auto presence = [&](const cv::Mat_<cv::Vec3f>& coordinates, int scale) {
    return vc::fiber_tracer::renderFiberReplayPresenceTexture(
        coordinates, scale, source, predictionWorkingToBaseScale, predictions);
};
const auto inspection = vc::fiber_tracer::renderFiberReplayInspection(
    referenceBaseXYZ, replay, normals, normalWorkingToBaseScale,
    source, ct, presence, normalThresholdBaseVoxels, reverse, baseVoxelSizeUm);
```

Create the output directory, then save `inspection.pages` using OpenCV or
`writeUncompressedTextureTiff`, and save `inspection.metadata` as JSON.
Pages are BGR images. JSON records page/row/column placements, reference arcs
(base voxels and millimetres), original trace positions, section frames, and
restart identities. This mapping is intended for interactive inspection;
it is not a new evaluation score. The crop-face writer expects an existing
output directory and uses base-coordinate mesh vertices, allowing overlays
with the crop tracer's `pointsBaseXYZ` lines without rescaling.

## Verification

With an existing native build configured with `VC_TESTING=ON`:

```bash
cmake --build build --target test_fiber_replay_visualization
ctest --test-dir build -R '^test_fiber_replay_visualization$' --output-on-failure
```

The tests cover anisotropic transforms, metric ellipses, inverse strip mapping,
restart gaps/singletons, modality alignment, pagination, and invalid CT sources.
The optional real-data case reads these environment variables:

- `LASAGNA_TEST_VOLUME`: concrete local uint8 CT array.
- `VC_TEST_GRAPH_NORMALS`, `VC_TEST_GRAPH_PREDICTIONS`: normal/prediction manifests.
- `VC_TEST_WIDTH_FIBER`: VC3D JSON reference fiber.
- `VC_TEST_REPLAY_VIS_OUTPUT`: writable artifact directory.

The real fixture expects Paris4 CT level 1 (base-to-array scale 0.5) and samples
the first 128 base voxels of the reference. Prediction/normal working scale is 8.
It runs a short CPU beam replay, verifies exact CT/presence coordinate equality,
repeated image/metadata equality, nonempty data, and six correctly sized crop
textures. This tests rendering integration, not trace accuracy or a speedup.
No data, generated images or experiment runners are distributed with the API.
