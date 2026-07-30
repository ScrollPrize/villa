#include "LineAnnotationFiberSegments.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace vc3d::line_annotation
{
namespace
{

constexpr std::string_view kOptimizer = "native_fiber_trace3d";

void requireFinitePositive(double value, std::string_view name)
{
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be finite and positive");
    }
}

void requireFiniteNonNegative(double value, std::string_view name)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(std::string(name) + " must be finite and non-negative");
    }
}

void rejectUnknownKeys(const nlohmann::json& json, const std::unordered_set<std::string>& allowed, std::string_view context)
{
    for (const auto& [key, value] : json.items()) {
        (void)value;
        if (!allowed.contains(key)) {
            throw std::runtime_error(std::string(context) + " contains unknown field: " + key);
        }
    }
}

cv::Vec3d pointFromJson(const nlohmann::json& json)
{
    if (!json.is_array() || json.size() != 3) {
        throw std::runtime_error("control point position must contain exactly three numbers");
    }
    cv::Vec3d point;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!json[axis].is_number()) {
            throw std::runtime_error("control point position must contain exactly three numbers");
        }
        point[static_cast<int>(axis)] = json[axis].get<double>();
        if (!std::isfinite(point[static_cast<int>(axis)])) {
            throw std::runtime_error("control point position must be finite");
        }
    }
    return point;
}

nlohmann::json pointToJson(const cv::Vec3d& point)
{
    return {point[0], point[1], point[2]};
}

}  // namespace

nlohmann::json fiberTraceSegmentMetadataToJson(const FiberTraceSegmentMetadata& metadata)
{
    const auto& config = metadata.config;
    return {
        {"optimizer", kOptimizer},
        {"metadata_version", FiberTraceSegmentMetadata::MetadataVersion},
        {"tracer_version", FiberTraceSegmentMetadata::TracerVersion},
        {"normal_manifest", metadata.normalManifestLocation},
        {"fiber_manifest", metadata.fiberManifestLocation},
        {"trace_to_base_scale", metadata.traceToBaseScale},
        {"max_endpoint_error_base_voxels", metadata.maxEndpointErrorBaseVoxels},
        {"config",
         {
             {"step_voxels", config.stepVoxels},
             {"cone_angle_degrees", config.coneAngleDegrees},
             {"cone_angle_step_degrees", config.coneAngleStepDegrees},
             {"cone_grid_size", config.coneGridSize},
             {"beam_width", config.beamWidth},
             {"beam_prune_distance_voxels", config.beamPruneDistanceVoxels},
             {"beam_lookahead_steps", config.beamLookaheadSteps},
             {"smoothness_weight", config.smoothnessWeight},
             {"smoothness_normal_weight", config.smoothnessNormalWeight},
             {"smoothness_tangent_weight", config.smoothnessTangentWeight},
             {"smoothness_free_angle_degrees", config.smoothnessFreeAngleDegrees},
             {"cumulative_smoothness_steps", config.cumulativeSmoothnessSteps},
             {"cumulative_smoothness_tangent_weight", config.cumulativeSmoothnessTangentWeight},
             {"initial_free_angle_degrees", config.initialFreeAngleDegrees},
             {"max_step_factor", config.maxStepFactor},
             {"fusion_gap_factor", config.fusionGapFactor},
             {"endpoint_accept_threshold_base_voxels", config.endpointAcceptThresholdBaseVoxels},
         }},
    };
}

FiberTraceSegmentMetadata fiberTraceSegmentMetadataFromJson(const nlohmann::json& json)
{
    if (!json.is_object()) {
        throw std::runtime_error("segment_to_next must be an object");
    }
    rejectUnknownKeys(
        json,
        {"optimizer",
         "metadata_version",
         "tracer_version",
         "normal_manifest",
         "fiber_manifest",
         "trace_to_base_scale",
         "max_endpoint_error_base_voxels",
         "config"},
        "segment_to_next");
    if (json.at("optimizer").get<std::string>() != kOptimizer) {
        throw std::runtime_error("unsupported segment_to_next optimizer");
    }
    if (json.at("metadata_version").get<int>() != FiberTraceSegmentMetadata::MetadataVersion) {
        throw std::runtime_error("unsupported segment_to_next metadata_version");
    }
    if (json.at("tracer_version").get<int>() != FiberTraceSegmentMetadata::TracerVersion) {
        throw std::runtime_error("unsupported native fiber tracer version");
    }

    FiberTraceSegmentMetadata metadata;
    metadata.normalManifestLocation = json.at("normal_manifest").get<std::string>();
    metadata.fiberManifestLocation = json.at("fiber_manifest").get<std::string>();
    if (metadata.normalManifestLocation.empty() || metadata.fiberManifestLocation.empty()) {
        throw std::runtime_error("segment_to_next manifest locations must not be empty");
    }
    metadata.traceToBaseScale = json.at("trace_to_base_scale").get<double>();
    metadata.maxEndpointErrorBaseVoxels = json.at("max_endpoint_error_base_voxels").get<double>();

    const auto& configJson = json.at("config");
    if (!configJson.is_object()) {
        throw std::runtime_error("segment_to_next config must be an object");
    }
    rejectUnknownKeys(
        configJson,
        {"step_voxels",
         "cone_angle_degrees",
         "cone_angle_step_degrees",
         "cone_grid_size",
         "beam_width",
         "beam_prune_distance_voxels",
         "beam_lookahead_steps",
         "smoothness_weight",
         "smoothness_normal_weight",
         "smoothness_tangent_weight",
         "smoothness_free_angle_degrees",
         "cumulative_smoothness_steps",
         "cumulative_smoothness_tangent_weight",
         "initial_free_angle_degrees",
         "max_step_factor",
         "fusion_gap_factor",
         "endpoint_accept_threshold_base_voxels"},
        "segment_to_next config");
    auto& config = metadata.config;
    config.stepVoxels = configJson.at("step_voxels").get<double>();
    config.coneAngleDegrees = configJson.at("cone_angle_degrees").get<double>();
    config.coneAngleStepDegrees = configJson.at("cone_angle_step_degrees").get<double>();
    config.coneGridSize = configJson.at("cone_grid_size").get<int>();
    config.beamWidth = configJson.at("beam_width").get<int>();
    config.beamPruneDistanceVoxels = configJson.at("beam_prune_distance_voxels").get<double>();
    config.beamLookaheadSteps = configJson.at("beam_lookahead_steps").get<int>();
    config.smoothnessWeight = configJson.at("smoothness_weight").get<double>();
    config.smoothnessNormalWeight = configJson.at("smoothness_normal_weight").get<double>();
    config.smoothnessTangentWeight = configJson.at("smoothness_tangent_weight").get<double>();
    config.smoothnessFreeAngleDegrees = configJson.at("smoothness_free_angle_degrees").get<double>();
    config.cumulativeSmoothnessSteps = configJson.at("cumulative_smoothness_steps").get<int>();
    config.cumulativeSmoothnessTangentWeight = configJson.at("cumulative_smoothness_tangent_weight").get<double>();
    config.initialFreeAngleDegrees = configJson.at("initial_free_angle_degrees").get<double>();
    config.maxStepFactor = configJson.at("max_step_factor").get<double>();
    config.fusionGapFactor = configJson.at("fusion_gap_factor").get<double>();
    config.endpointAcceptThresholdBaseVoxels = configJson.at("endpoint_accept_threshold_base_voxels").get<double>();
    config.traceToBaseScale = metadata.traceToBaseScale;

    requireFinitePositive(metadata.traceToBaseScale, "trace_to_base_scale");
    requireFiniteNonNegative(metadata.maxEndpointErrorBaseVoxels, "max_endpoint_error_base_voxels");
    requireFinitePositive(config.stepVoxels, "step_voxels");
    requireFinitePositive(config.coneAngleDegrees, "cone_angle_degrees");
    requireFinitePositive(config.coneAngleStepDegrees, "cone_angle_step_degrees");
    requireFinitePositive(config.beamPruneDistanceVoxels, "beam_prune_distance_voxels");
    requireFiniteNonNegative(config.smoothnessWeight, "smoothness_weight");
    requireFiniteNonNegative(config.smoothnessNormalWeight, "smoothness_normal_weight");
    requireFiniteNonNegative(config.smoothnessTangentWeight, "smoothness_tangent_weight");
    requireFiniteNonNegative(config.smoothnessFreeAngleDegrees, "smoothness_free_angle_degrees");
    requireFiniteNonNegative(config.cumulativeSmoothnessTangentWeight, "cumulative_smoothness_tangent_weight");
    requireFiniteNonNegative(config.initialFreeAngleDegrees, "initial_free_angle_degrees");
    requireFinitePositive(config.maxStepFactor, "max_step_factor");
    requireFiniteNonNegative(config.fusionGapFactor, "fusion_gap_factor");
    requireFinitePositive(config.endpointAcceptThresholdBaseVoxels, "endpoint_accept_threshold_base_voxels");
    if (config.coneGridSize <= 0 || config.beamWidth <= 0 || config.beamLookaheadSteps < 0 || config.cumulativeSmoothnessSteps < 0) {
        throw std::runtime_error("segment_to_next config contains invalid integer values");
    }
    return metadata;
}

nlohmann::json storedControlPointToJson(const StoredControlPoint& control)
{
    nlohmann::json json{{"position", pointToJson(control)}};
    if (control.segmentToNext) {
        json["segment_to_next"] = fiberTraceSegmentMetadataToJson(*control.segmentToNext);
    }
    return json;
}

StoredControlPoint storedControlPointFromJson(const nlohmann::json& json, int fiberVersion)
{
    if (fiberVersion == 1) {
        return StoredControlPoint{pointFromJson(json)};
    }
    if (fiberVersion != 2 || !json.is_object()) {
        throw std::runtime_error("version-2 control point entries must be objects");
    }
    rejectUnknownKeys(json, {"position", "segment_to_next"}, "control point");
    StoredControlPoint control{pointFromJson(json.at("position"))};
    if (json.contains("segment_to_next")) {
        control.segmentToNext = fiberTraceSegmentMetadataFromJson(json.at("segment_to_next"));
    }
    return control;
}

void validateStoredControlPoints(const std::vector<StoredControlPoint>& controls)
{
    if (!controls.empty() && controls.back().segmentToNext) {
        throw std::runtime_error("the final control point cannot contain segment_to_next");
    }
}

std::vector<cv::Vec3d> storedControlPointPositions(const std::vector<StoredControlPoint>& controls)
{
    std::vector<cv::Vec3d> positions;
    positions.reserve(controls.size());
    for (const auto& control : controls) {
        positions.emplace_back(control);
    }
    return positions;
}

std::vector<vc::lasagna::LineControlPoint> optimizerControlPoints(const std::vector<LineControlPoint>& controls)
{
    std::vector<vc::lasagna::LineControlPoint> result;
    result.reserve(controls.size());
    for (const auto& control : controls) {
        result.push_back(control);
    }
    return result;
}

std::vector<LineControlPoint> mergeOptimizerControlPoints(std::vector<vc::lasagna::LineControlPoint> optimized, const std::vector<LineControlPoint>& original)
{
    std::vector<LineControlPoint> result;
    result.reserve(optimized.size());
    for (auto& control : optimized) {
        LineControlPoint merged{control};
        const auto found = std::find_if(original.begin(), original.end(), [&control](const LineControlPoint& candidate) {
            return candidate.volumePoint == control.volumePoint;
        });
        if (found != original.end()) {
            merged.segmentToNext = found->segmentToNext;
        }
        result.push_back(std::move(merged));
    }
    return result;
}

void invalidateSegmentsAdjacentToControl(std::vector<LineControlPoint>& controls, size_t controlIndex)
{
    if (controlIndex >= controls.size()) {
        return;
    }
    std::vector<size_t> order(controls.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&controls](size_t lhs, size_t rhs) {
        return controls[lhs].linePosition < controls[rhs].linePosition;
    });
    const auto found = std::find(order.begin(), order.end(), controlIndex);
    controls[controlIndex].segmentToNext.reset();
    if (found != order.begin() && found != order.end()) {
        controls[*(found - 1)].segmentToNext.reset();
    }
}

void invalidateSegmentSplitByInsertedControl(std::vector<LineControlPoint>& controls, size_t insertedIndex)
{
    if (insertedIndex >= controls.size()) {
        return;
    }
    std::vector<size_t> order(controls.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&controls](size_t lhs, size_t rhs) {
        return controls[lhs].linePosition < controls[rhs].linePosition;
    });
    const auto found = std::find(order.begin(), order.end(), insertedIndex);
    if (found != order.begin() && found != order.end()) {
        controls[*(found - 1)].segmentToNext.reset();
    }
}

}  // namespace vc3d::line_annotation
