#pragma once

// Skeleton tracing backend: builds the routable network for one crop and
// replays a directed reference through it.
//
// The evaluation protocol is not re-implemented here. The driver supplies one
// `ReplaySegmentRunner` to the shared reference replay core, which owns
// seeding on the reference, forward matching, the anisotropic threshold
// ellipse, excursion grace, failure records and the deterministic reset.

#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"
#include "vc/fiber_tracer/FiberSkeletonChoice.hpp"

#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{

struct FiberSkeletonTraceConfig {
    double tubeRadiusBase = 64.0;
    double cropMarginBase = 16.0;
    int thresholdU8 = 128;
    std::size_t minComponentVoxels = 8;
    double junctionMergeMaxBase = 64.0;
    double spurPruneMaxBase = 32.0;
    double entryOffsetBase = 32.0;
    // Port candidates: the section of the prediction at the split point,
    // measured within this radius, gives the centre and the two extremes
    // along the sheet; an extreme closer than the offset to the centre is the
    // same fibre. Each candidate is then centred by a small local centroid.
    double sectionRadiusBase = 24.0;
    double candidateMinOffsetBase = 6.0;
    double candidateRefineRadiusBase = 4.0;
    // Corridor around a skeleton piece for the port-to-port path search that
    // initialises every line and connection variant.
    double searchRadiusBase = 16.0;
    // Whether the centre candidate of a wide section is a routing target. It
    // is the seam between the fibres; when false a fibre only enters and
    // leaves through the extremes there, and the centre remains only where it
    // is a port's single candidate.
    bool routeThroughCentre = true;
    // Writes the thresholding input beside the other artifacts, as raw bytes in
    // the crop's own Z,Y,X order. It is a diagnostic: it lets the geometry be
    // re-measured against the prediction offline instead of by rebuilding.
    bool dumpCropPresence = false;
    // Lines whose every refined variant is written as a named polyline, with
    // its cost, next to the other artifacts. A diagnostic.
    std::vector<std::size_t> renderLineVariants;
    double stepBase = 8.0;
    double nominalStepBase = 48.0;
    FiberSkeletonCostWeights weights;
    FiberSkeletonChoiceConfig choice;
    FiberSkeletonSolverConfig solver;
    int workers = 1;
};

void validateFiberSkeletonTraceConfig(const FiberSkeletonTraceConfig& config);

struct FiberSkeletonStageTimings {
    double thresholdSeconds = 0.0;
    double thinningSeconds = 0.0;
    double graphSeconds = 0.0;
    double topologySeconds = 0.0;
    double lineRefineSeconds = 0.0;
    double choiceSeconds = 0.0;
    [[nodiscard]] double total() const noexcept;
};

// The routable network for one crop: reduced graph, split topology and the
// selected connection choice of every ambiguous region.
struct FiberSkeletonModel {
    FiberSkeletonGraph graph;
    FiberSkeletonTopology topology;
    std::vector<FiberSkeletonRegionChoice> choices;
    FiberSkeletonStageTimings timings;
    FiberSkeletonTraceConfig config;
    std::size_t portLimitRegions = 0;
};

// Runs stages 1 to 5 over the provider's crop.
[[nodiscard]] FiberSkeletonModel buildFiberSkeletonModel(
    FiberSkeletonFieldProvider& fields,
    const FiberSkeletonTraceConfig& config);

// Refines the annotated reference itself with the same solver and the same
// cost, endpoints fixed, and measures how far the routable network and the raw
// skeleton graph sit from that refined line. This separates "the network cannot
// represent the right answer" from "the choice picked the wrong branch".
struct FiberSkeletonReferenceCheck {
    std::vector<cv::Vec3d> refinedPointsBaseXYZ;
    FiberSkeletonCostBreakdown referenceCost;   // cost of the annotation as given
    FiberSkeletonCostBreakdown refinedCost;     // cost after refinement
    double referenceLengthBase = 0.0;
    double refinedLengthBase = 0.0;
    // Movement of the refinement away from the annotation.
    double medianShiftBase = 0.0;
    double p95ShiftBase = 0.0;
    double maxShiftBase = 0.0;
    // Distance from the refined reference to the nearest routable geometry.
    double medianNetworkGapBase = 0.0;
    double p95NetworkGapBase = 0.0;
    double maxNetworkGapBase = 0.0;
    // Same against every reduced-graph edge, before any region choice.
    double medianGraphGapBase = 0.0;
    double p95GraphGapBase = 0.0;
    double maxGraphGapBase = 0.0;
    int iterations = 0;
    bool converged = false;
    double seconds = 0.0;
};

[[nodiscard]] FiberSkeletonReferenceCheck checkFiberSkeletonReference(
    const FiberSkeletonModel& model,
    const std::vector<cv::Vec3d>& referencePointsBase,
    FiberSkeletonFieldProvider& fields);

// A final full-length refinement of the committed trace. The piecewise build
// freezes each port, so the emitted route is a chain of independently solved
// pieces; this re-solves each traced segment as one polyline under the same
// cost and reports how much closer to the reference that gets.
struct FiberSkeletonTraceRefinement {
    std::vector<std::vector<cv::Vec3d>> segmentsBaseXYZ;
    FiberSkeletonCostBreakdown initialCost;
    FiberSkeletonCostBreakdown finalCost;
    int maxIterations = 0;
    std::size_t cappedSegments = 0;
    // Distance from the traced geometry to the refined reference, before and
    // after this refinement.
    double medianBeforeBase = 0.0, p95BeforeBase = 0.0, maxBeforeBase = 0.0;
    double medianAfterBase = 0.0, p95AfterBase = 0.0, maxAfterBase = 0.0;
    double seconds = 0.0;
};

// Replays an already-refined trace against an already-refined reference
// through the shared core, purely to obtain a consistent match record for
// rendering and comparison. It performs no tracing of its own.
[[nodiscard]] FiberReplayTraceResult replayFiberSkeletonRefinedGeometry(
    const std::vector<std::vector<cv::Vec3d>>& refinedSegmentsBaseXYZ,
    const std::vector<cv::Vec3d>& refinedReferenceBaseXYZ,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, double nominalStepBase,
    double errorThresholdBaseVoxels, double excursionGraceBaseVoxels,
    double tangentialThresholdFactor, double matchRefineSteps);

[[nodiscard]] FiberSkeletonTraceRefinement refineFiberSkeletonTrace(
    const FiberSkeletonModel& model, const FiberReplayTraceResult& replay,
    const std::vector<cv::Vec3d>& refinedReferenceBaseXYZ,
    FiberSkeletonFieldProvider& fields);

struct FiberSkeletonReplayRequest {
    std::vector<cv::Vec3d> referencePointsBase;
    double errorThresholdBaseVoxels = 40.0;
    double excursionGraceBaseVoxels = 160.0;
    double tangentialThresholdFactor = 2.0;
    double matchRefineSteps = 1.0;
    double maxStepFactor = 3.0;
    std::optional<double> referenceBeginArcBase;
    std::optional<double> referenceEndArcBase;
};

// Per-segment routing outcome, recorded alongside the ordinary replay result.
struct FiberSkeletonSegmentReport {
    double startReferenceArcBase = 0.0;
    bool seeded = false;
    std::string terminationReason;
    std::size_t visitedLines = 0;
    std::size_t visitedRegions = 0;
    double routedLengthBase = 0.0;
};

struct FiberSkeletonReplayResult {
    FiberReplayTraceResult replay;
    std::vector<FiberSkeletonSegmentReport> segments;
    std::size_t unseededSegments = 0;
    double replaySeconds = 0.0;
};

// Termination reasons produced by the skeleton segment runner. They travel
// through the shared core into `FiberReplayFailure::reason`.
inline constexpr const char* kFiberSkeletonNoSeed = "no_skeleton_seed";
inline constexpr const char* kFiberSkeletonTerminal = "skeleton_terminal";
inline constexpr const char* kFiberSkeletonUnmatchedPort = "unmatched_port";
inline constexpr const char* kFiberSkeletonTubeBoundary = "tube_boundary";
inline constexpr const char* kFiberSkeletonRegionUnresolved =
    "region_unresolved";

[[nodiscard]] FiberSkeletonReplayResult traceFiberSkeletonReplay(
    const FiberSkeletonModel& model, const FiberSkeletonReplayRequest& request,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberTraceProgressCallback& progress = {},
    const FiberReplayFailureCallback& failureCallback = {});

// `skeleton.json`: configuration, stage statistics and timings, the reduced
// graph, ports, lines, and every region's connections and scored matchings.
[[nodiscard]] nlohmann::json fiberSkeletonModelJson(
    const FiberSkeletonModel& model, const FiberSkeletonCrop& crop,
    const FiberSkeletonReplayResult* replay,
    const FiberSkeletonReferenceCheck* reference = nullptr,
    const FiberSkeletonTraceRefinement* traceRefinement = nullptr);

void writeFiberSkeletonArtifacts(
    const std::filesystem::path& directory, const FiberSkeletonModel& model,
    const FiberSkeletonCrop& crop, const FiberSkeletonReplayResult* replay,
    const FiberSkeletonReferenceCheck* reference = nullptr,
    const FiberSkeletonTraceRefinement* traceRefinement = nullptr);

}  // namespace vc::fiber_tracer
