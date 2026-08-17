#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/types.hpp>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

namespace vc3d::line_annotation
{

enum class FiberOptimizationMode {
    Lasagna,
    NativeFiberTrace3d,
};

inline constexpr FiberOptimizationMode kDefaultNewFiberOptimizationMode =
    FiberOptimizationMode::NativeFiberTrace3d;

[[nodiscard]] bool shouldRunNativeSeedTrace(
    FiberOptimizationMode mode,
    bool hasSelectedFiberInferenceDataset,
    size_t attachedFiberInferenceDatasetCount) noexcept;

enum class SegmentInterpolationGoal {
    Global,
    Cspline,
    Lasagna,
    Trace,
};

enum class SegmentInterpolationMode {
    Cspline,
    Lasagna,
    Trace,
};

[[nodiscard]] std::string segmentInterpolationGoalToString(SegmentInterpolationGoal goal);
[[nodiscard]] SegmentInterpolationGoal segmentInterpolationGoalFromString(const std::string& value);
[[nodiscard]] std::string segmentInterpolationModeToString(SegmentInterpolationMode mode);
[[nodiscard]] SegmentInterpolationMode segmentInterpolationModeFromString(const std::string& value);
[[nodiscard]] char segmentInterpolationModeMarker(SegmentInterpolationMode mode) noexcept;
[[nodiscard]] SegmentInterpolationMode resolveSegmentInterpolationMode(
    SegmentInterpolationGoal goal,
    FiberOptimizationMode globalMode,
    double endpointDistanceBaseVoxels);

[[nodiscard]] std::string fiberOptimizationModeToString(FiberOptimizationMode mode);
[[nodiscard]] FiberOptimizationMode fiberOptimizationModeFromString(const std::string& value);

struct FiberTraceSegmentMetadata {
    enum class Outcome {
        AcceptedNative,
        LasagnaFallback,
    };

    static constexpr int MetadataVersion = 3;
    static constexpr int TracerVersion = 2;

    SegmentInterpolationGoal interpGoal = SegmentInterpolationGoal::Global;
    SegmentInterpolationMode interpMode = SegmentInterpolationMode::Lasagna;
    std::optional<double> metric;
    std::string message;
    Outcome outcome = Outcome::AcceptedNative;
    std::string normalManifestLocation;
    std::string fiberManifestLocation;
    double traceToBaseScale = 1.0;
    vc::fiber_tracer::FiberTraceConfig config;
    std::optional<double> meetingErrorBaseVoxels;
    std::optional<double> meetingErrorRatio;
    std::string meetingSource;
    std::string failureCode;
    std::string failureDetail;
    std::string lasagnaFailureCode;
    std::string lasagnaFailureDetail;
};

[[nodiscard]] bool isAcceptedNativeTrace(const FiberTraceSegmentMetadata& metadata) noexcept;
[[nodiscard]] bool isAcceptedNativeTrace(
    const std::optional<FiberTraceSegmentMetadata>& metadata) noexcept;

[[nodiscard]] FiberTraceSegmentMetadata fiberTraceSegmentMetadataForResult(
    std::string normalManifestLocation,
    std::string fiberManifestLocation,
    double traceToBaseScale,
    const vc::fiber_tracer::FiberTraceConfig& config,
    const vc::fiber_tracer::FiberTraceSegmentResult& result);

[[nodiscard]] FiberTraceSegmentMetadata fiberTraceSegmentMetadataForException(
    std::string normalManifestLocation,
    std::string fiberManifestLocation,
    double traceToBaseScale,
    const vc::fiber_tracer::FiberTraceConfig& config,
    std::string detail);

struct LineControlPoint : vc::lasagna::LineControlPoint {
    std::optional<FiberTraceSegmentMetadata> segmentToNext;

    LineControlPoint() = default;
    LineControlPoint(double linePositionValue, cv::Vec3d volumePointValue, bool isSeedValue, int optimizedIndexValue)
    {
        linePosition = linePositionValue;
        volumePoint = volumePointValue;
        isSeed = isSeedValue;
        optimizedIndex = optimizedIndexValue;
    }
    explicit LineControlPoint(const vc::lasagna::LineControlPoint& value) : vc::lasagna::LineControlPoint(value) {}
};

struct ControlPointCollapseResult {
    std::vector<LineControlPoint> controlPoints;
    std::vector<size_t> oldToNewIndices;
    std::vector<size_t> collapsedOldIndices;
    std::vector<size_t> dirtySegmentIndices;
    size_t replacementIndex = std::numeric_limits<size_t>::max();

    [[nodiscard]] bool replacedExisting() const noexcept
    {
        return !collapsedOldIndices.empty();
    }
};

[[nodiscard]] ControlPointCollapseResult collapseControlPointsAtClick(
    const std::vector<LineControlPoint>& controls,
    std::vector<size_t> collapsedIndices,
    double clickedLinePosition,
    const cv::Vec3d& clickedPoint);

struct StoredControlPoint : cv::Vec3d {
    std::optional<FiberTraceSegmentMetadata> segmentToNext;

    StoredControlPoint() = default;
    explicit StoredControlPoint(const cv::Vec3d& position) : cv::Vec3d(position) {}
};

// Ordinary free-form fiber tag carrying the human review verdict; also the
// default publish gate of scripts/vc_sync.py hfsync — keep the literal in
// sync. A toolbar interpolation-mode switch strips it on the save after a
// successful re-optimization; nothing else touches it programmatically.
inline constexpr const char* kReviewedTag = "reviewed";

enum class FiberTraceState {
    Legacy,       // no prediction-traced spans in the stored geometry
    Predictions,  // traced spans, fiber-global mode native_fiber_trace3d
    Mixed,        // traced spans under a lasagna-global fiber
};

[[nodiscard]] bool hasAcceptedTraceSpan(
    const std::vector<LineControlPoint>& controls) noexcept;
[[nodiscard]] bool hasAcceptedTraceSpan(
    const std::vector<StoredControlPoint>& controls) noexcept;

[[nodiscard]] FiberTraceState deriveTraceState(
    FiberOptimizationMode mode,
    const std::vector<StoredControlPoint>& controls) noexcept;

struct FiberExtrapolationFallbackDiagnostic {
    enum class Side {
        Left,
        Right,
    };

    Side side = Side::Left;
    std::string reason;
    size_t tracePointCount = 0;
    bool fromException = false;
};

struct FiberModeOptimizationRequest {
    std::vector<LineControlPoint> controlPoints;
    std::vector<cv::Vec3d> linePointsBase;
    const vc::fiber_tracer::FiberPredictionSource* predictions = nullptr;
    const vc::lasagna::NormalSampler* baseNormalSampler = nullptr;
    const vc::lasagna::NormalSampler* traceNormalSampler = nullptr;
    vc::lasagna::LineOptimizationConfig lasagnaConfig;
    vc::fiber_tracer::FiberTraceConfig traceConfig;
    std::string normalManifestLocation;
    std::string fiberManifestLocation;
    double traceToBaseScale = 1.0;
    double extrapolationDistanceBaseVoxels = 0.0;
    FiberOptimizationMode globalMode = FiberOptimizationMode::NativeFiberTrace3d;
    std::optional<std::vector<size_t>> dirtySegments;
    bool globalGoalsOnly = false;
    bool retraceAll = false;
    std::function<void(const FiberExtrapolationFallbackDiagnostic&)>
        extrapolationFallbackCallback;
};

struct FiberModeOptimizationResult {
    std::vector<LineControlPoint> controlPoints;
    vc::lasagna::LineOptimizationResult optimization;
    int nativeSegments = 0;
    int lasagnaFallbackSegments = 0;
    int nativeExtrapolations = 0;
    int lasagnaFallbackExtrapolations = 0;
};

[[nodiscard]] FiberModeOptimizationResult optimizeFiberWithNativeFallback(
    FiberModeOptimizationRequest request);

[[nodiscard]] nlohmann::json fiberTraceSegmentMetadataToJson(const FiberTraceSegmentMetadata& metadata);
[[nodiscard]] FiberTraceSegmentMetadata fiberTraceSegmentMetadataFromJson(const nlohmann::json& json);
[[nodiscard]] nlohmann::json storedControlPointToJson(const StoredControlPoint& control);
[[nodiscard]] StoredControlPoint storedControlPointFromJson(const nlohmann::json& json, int fiberVersion);

void validateStoredControlPoints(const std::vector<StoredControlPoint>& controls);

[[nodiscard]] std::vector<cv::Vec3d> storedControlPointPositions(const std::vector<StoredControlPoint>& controls);
[[nodiscard]] std::vector<vc::lasagna::LineControlPoint> optimizerControlPoints(const std::vector<LineControlPoint>& controls);
[[nodiscard]] std::vector<LineControlPoint> mergeOptimizerControlPoints(
    std::vector<vc::lasagna::LineControlPoint> optimized, const std::vector<LineControlPoint>& original);
void invalidateSegmentsAdjacentToControl(std::vector<LineControlPoint>& controls, size_t controlIndex);
void invalidateSegmentSplitByInsertedControl(std::vector<LineControlPoint>& controls, size_t insertedIndex);

// Where to cut a fiber's stored geometry between adjacent control points a
// and a+1: the prefix keeps [0..a] and the suffix [a+1..end], and the dense
// line points of the removed a->a+1 span belong to neither half.
struct FiberSplitPlan {
    size_t splitAfterControlIndex = 0;  // a
    size_t prefixControlCount = 0;      // a+1 control points: [0..a]
    size_t prefixLineCount = 0;         // line points [0..lineIndex(a)]
    size_t suffixControlBegin = 0;      // a+1
    size_t suffixLineBegin = 0;         // lineIndex(a+1)
};

// Returns nullopt when splitAfterControlIndex is out of range, either half
// would keep fewer than 2 control points, or the control points are not an
// ordered subset of linePoints under the loader's exact-membership
// tolerance (vc::atlas::validateFiberInputControlPoints).
[[nodiscard]] std::optional<FiberSplitPlan> computeFiberSplitPlan(
    const std::vector<cv::Vec3d>& controlPoints,
    const std::vector<cv::Vec3d>& linePoints,
    size_t splitAfterControlIndex);

// Post-split home of a stored control index: {inSuffix, remappedIndex}.
// Prefix indices are unchanged; suffix indices shift down by
// suffixControlBegin. nullopt for negative indices.
[[nodiscard]] std::optional<std::pair<bool, int>> remappedSplitControlPointIndex(
    const FiberSplitPlan& plan, int controlPointIndex);

// Reversed stored geometry. Each span descriptor moves with its span:
// reversed CP j carries original CP (n-2-j)'s segmentToNext, and the new
// final CP has none (v3 contract).
[[nodiscard]] std::vector<StoredControlPoint> reversedStoredControlPoints(
    const std::vector<StoredControlPoint>& controls);

// End-to-end concatenation of two fibers whose inputs are pre-oriented so
// `a` ends at the join and `b` starts at it.
struct FiberMergeGeometry {
    std::vector<StoredControlPoint> controlPoints;  // a + b
    std::vector<cv::Vec3d> linePoints;   // a.line[0..idx(a.last)] + b.line[idx(b.first)..]
    size_t joinControlIndex = 0;         // a.controlPoints.size() - 1
};

// Slices both extrapolated tails at the boundary CPs via the same exact
// ordered-subset scan as computeFiberSplitPlan (raw concatenation leaves
// tail points between the join CPs and breaks the optimizer's strict
// line-order mapping), and synthesizes the join span's descriptor on a's
// last CP so in-memory consumers that dereference every non-final
// descriptor keep working. nullopt when either side has fewer than 2
// control points or fails the ordered scan.
[[nodiscard]] std::optional<FiberMergeGeometry> computeFiberMergeGeometry(
    const std::vector<StoredControlPoint>& aControls,
    const std::vector<cv::Vec3d>& aLine,
    const std::vector<StoredControlPoint>& bControls,
    const std::vector<cv::Vec3d>& bLine);

}  // namespace vc3d::line_annotation
