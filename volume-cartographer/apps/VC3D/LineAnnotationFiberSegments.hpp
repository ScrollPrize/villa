#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
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

struct StoredControlPoint : cv::Vec3d {
    std::optional<FiberTraceSegmentMetadata> segmentToNext;

    StoredControlPoint() = default;
    explicit StoredControlPoint(const cv::Vec3d& position) : cv::Vec3d(position) {}
};

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

}  // namespace vc3d::line_annotation
