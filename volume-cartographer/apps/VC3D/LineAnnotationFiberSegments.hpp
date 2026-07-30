#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/types.hpp>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

namespace vc3d::line_annotation
{

struct FiberTraceSegmentMetadata {
    static constexpr int MetadataVersion = 1;
    static constexpr int TracerVersion = 1;

    std::string normalManifestLocation;
    std::string fiberManifestLocation;
    double traceToBaseScale = 1.0;
    vc::fiber_tracer::FiberTraceConfig config;
    double maxEndpointErrorBaseVoxels = 0.0;
};

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
