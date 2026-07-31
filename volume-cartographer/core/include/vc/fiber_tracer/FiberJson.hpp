#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{
struct Vc3dFiberJson {
    int version = 1;
    std::string optimizationMode = "lasagna";
    std::vector<cv::Vec3d> linePoints;
    std::vector<cv::Vec3d> controlPoints;
    std::vector<nlohmann::json> segmentMetadata;
};

namespace detail
{

inline void requireExactKeys(const nlohmann::json& value, const std::unordered_set<std::string>& keys, const std::string& context)
{
    if (!value.is_object() || value.size() != keys.size()) {
        throw std::runtime_error(context + " has missing or unknown fields");
    }
    for (const auto& [key, item] : value.items()) {
        (void)item;
        if (!keys.contains(key)) {
            throw std::runtime_error(context + " contains unknown field: " + key);
        }
    }
}

inline void validateSegmentMetadata(const nlohmann::json& value)
{
    if (!value.is_object())
        throw std::runtime_error("segment_to_next must be an object");
    requireExactKeys(
        value,
        {"optimizer", "metadata_version", "tracer_version",
         "interp_goal", "interp_mode", "metric", "msg",
         "normal_manifest", "fiber_manifest", "trace_to_base_scale",
         "meeting_error_base_voxels", "meeting_error_ratio",
         "meeting_source", "failure_code", "failure_detail",
         "lasagna_failure_code", "lasagna_failure_detail", "config"},
        "segment_to_next");
    if (value.at("optimizer").get<std::string>() != "native_fiber_trace3d")
        throw std::runtime_error("unsupported segment_to_next optimizer");
    const int metadataVersion = value.at("metadata_version").get<int>();
    const int tracerVersion = value.at("tracer_version").get<int>();
    const bool current = metadataVersion == 3 && tracerVersion == 2;
    if (!current) {
        throw std::runtime_error("unsupported segment_to_next optimizer or version");
    }
    if (!value.at("normal_manifest").is_string() ||
        !value.at("fiber_manifest").is_string()) {
        throw std::runtime_error("segment_to_next manifests must be strings");
    }
    if (!value.at("trace_to_base_scale").is_number())
        throw std::runtime_error("segment_to_next scale is invalid");
    const double scale = value.at("trace_to_base_scale").get<double>();
    if (!std::isfinite(scale) || scale <= 0.0)
        throw std::runtime_error("segment_to_next scale is invalid");
    const std::string goal = value.at("interp_goal").get<std::string>();
    const std::string mode = value.at("interp_mode").get<std::string>();
    if (goal != "global" && goal != "cspline" && goal != "lasagna" && goal != "trace")
        throw std::runtime_error("segment_to_next interpolation goal is invalid");
    if (mode != "cspline" && mode != "lasagna" && mode != "trace")
        throw std::runtime_error("segment_to_next interpolation mode is invalid");
    const auto& metric = value.at("metric");
    if (!metric.is_null() && (!metric.is_number() ||
        !std::isfinite(metric.get<double>()) || metric.get<double>() < 0.0)) {
        throw std::runtime_error("segment_to_next metric is invalid");
    }
    if (mode == "cspline" && !metric.is_null())
        throw std::runtime_error("cspline segment_to_next cannot contain metric");
    for (const char* key : {"msg", "meeting_source", "failure_code",
                            "failure_detail", "lasagna_failure_code",
                            "lasagna_failure_detail"}) {
        if (!value.at(key).is_string())
            throw std::runtime_error("segment_to_next diagnostic is not a string");
    }
    const bool haveError = !value.at("meeting_error_base_voxels").is_null();
    const bool haveRatio = !value.at("meeting_error_ratio").is_null();
    if (mode == "trace") {
        if (metric.is_null() || !haveError || !haveRatio ||
            value.at("meeting_source").get<std::string>().empty() ||
            !value.at("failure_code").get<std::string>().empty() ||
            !value.at("failure_detail").get<std::string>().empty() ||
            value.at("normal_manifest").get<std::string>().empty() ||
            value.at("fiber_manifest").get<std::string>().empty()) {
            throw std::runtime_error("trace segment_to_next is inconsistent");
        }
        const double error = value.at("meeting_error_base_voxels").get<double>();
        const double ratio = value.at("meeting_error_ratio").get<double>();
        if (!std::isfinite(error) || error < 0.0 ||
            !std::isfinite(ratio) || ratio < 0.0) {
            throw std::runtime_error("trace meeting diagnostics are invalid");
        }
    } else if (haveError || haveRatio ||
               !value.at("meeting_source").get<std::string>().empty()) {
        throw std::runtime_error(
            "non-trace segment_to_next contains meeting diagnostics");
    }
    const auto& config = value.at("config");
    requireExactKeys(
        config,
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
         "meeting_accept_max_error_ratio",
         "endpoint_accept_threshold_base_voxels"},
        "segment_to_next config");
    for (const auto& [key, item] : config.items()) {
        if (!item.is_number() || !std::isfinite(item.get<double>())) {
            throw std::runtime_error("segment_to_next config field is invalid: " + key);
        }
    }
    for (const char* key : {"step_voxels", "cone_angle_degrees",
                            "cone_angle_step_degrees", "beam_prune_distance_voxels",
                            "max_step_factor", "endpoint_accept_threshold_base_voxels"}) {
        if (config.at(key).get<double>() <= 0.0)
            throw std::runtime_error(
                "segment_to_next config field must be positive: " + std::string(key));
    }
    for (const char* key : {"smoothness_weight", "smoothness_normal_weight",
                            "smoothness_tangent_weight", "smoothness_free_angle_degrees",
                            "cumulative_smoothness_tangent_weight",
                            "initial_free_angle_degrees"}) {
        if (config.at(key).get<double>() < 0.0)
            throw std::runtime_error(
                "segment_to_next config field must be non-negative: " + std::string(key));
    }
    for (const char* key : {"cone_grid_size", "beam_width", "beam_lookahead_steps",
                            "cumulative_smoothness_steps"}) {
        if (!config.at(key).is_number_integer())
            throw std::runtime_error(
                "segment_to_next config integer field is invalid: " + std::string(key));
    }
    if (config.at("cone_grid_size").get<int>() <= 0 ||
        config.at("beam_width").get<int>() <= 0 ||
        config.at("beam_lookahead_steps").get<int>() < 0 ||
        config.at("cumulative_smoothness_steps").get<int>() < 0) {
        throw std::runtime_error("segment_to_next config contains invalid integer values");
    }
    const double ratio = config.at("meeting_accept_max_error_ratio").get<double>();
    if (ratio < 0.0 || ratio > 1.0)
        throw std::runtime_error("segment_to_next meeting ratio config is invalid");
}

inline cv::Vec3d pointFromJson(const nlohmann::json& value, const std::string& context)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(context + " point must be [x, y, z]");
    }
    cv::Vec3d point{value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>()};
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        throw std::runtime_error(context + " point contains non-finite coordinates");
    }
    return point;
}

}  // namespace detail

inline std::vector<cv::Vec3d> vc3dFiberPointArrayFromJson(const nlohmann::json& root, const std::string& key, int version, const std::string& context)
{
    if (!root.contains(key) || !root.at(key).is_array()) {
        throw std::runtime_error(context + " is missing array " + key);
    }
    std::vector<cv::Vec3d> points;
    const auto& array = root.at(key);
    points.reserve(array.size());
    for (size_t index = 0; index < array.size(); ++index) {
        const auto& value = array[index];
        if (key != "control_points" || version == 1) {
            points.push_back(detail::pointFromJson(value, context));
            continue;
        }
        if (version != 3 || !value.is_object()) {
            throw std::runtime_error(context + " version-3 control point must be an object");
        }
        for (const auto& [field, item] : value.items()) {
            (void)item;
            if (field != "position" && field != "segment_to_next") {
                throw std::runtime_error(context + " control point contains unknown field: " + field);
            }
        }
        points.push_back(detail::pointFromJson(value.at("position"), context));
        if (index + 1 == array.size()) {
            if (value.contains("segment_to_next")) {
                throw std::runtime_error(context + " final control point cannot contain segment_to_next");
            }
        } else {
            if (!value.contains("segment_to_next")) {
                throw std::runtime_error(
                    context + " non-final control point is missing segment_to_next");
            }
            detail::validateSegmentMetadata(value.at("segment_to_next"));
        }
    }
    return points;
}

inline Vc3dFiberJson parseVc3dFiberJson(const nlohmann::json& root,
                                       const std::string& context)
{
    if (!root.is_object() || root.value("type", std::string{}) != "vc3d_fiber")
        throw std::runtime_error(context + " is not a vc3d_fiber JSON object");

    Vc3dFiberJson fiber;
    fiber.version = root.value("version", 1);
    if (fiber.version != 1 && fiber.version != 3)
        throw std::runtime_error(context + " has unsupported vc3d_fiber version");

    if (fiber.version == 3 && !root.contains("optimization_mode"))
        throw std::runtime_error(context + " version-3 fiber is missing optimization_mode");
    if (root.contains("optimization_mode")) {
        if (!root.at("optimization_mode").is_string())
            throw std::runtime_error(context + " optimization_mode must be a string");
        fiber.optimizationMode = root.at("optimization_mode").get<std::string>();
    }
    if (fiber.optimizationMode != "lasagna" &&
        fiber.optimizationMode != "native_fiber_trace3d") {
        throw std::runtime_error(context + " optimization_mode is invalid");
    }

    fiber.linePoints = vc3dFiberPointArrayFromJson(
        root, "line_points", fiber.version, context);
    fiber.controlPoints = vc3dFiberPointArrayFromJson(
        root, "control_points", fiber.version, context);
    fiber.segmentMetadata.resize(fiber.controlPoints.size());
    if (fiber.version == 3) {
        const auto& controls = root.at("control_points");
        for (size_t index = 0; index + 1 < controls.size(); ++index)
            fiber.segmentMetadata[index] = controls.at(index).at("segment_to_next");
    }
    return fiber;
}

inline nlohmann::json makeLasagnaSegmentMetadataJson(
    const std::string& interpolationGoal,
    const std::string& normalManifest,
    double traceToBaseScale,
    const nlohmann::json& config,
    const nlohmann::json& metric,
    const std::string& message = "lasagna")
{
    nlohmann::json result{
        {"optimizer", "native_fiber_trace3d"},
        {"metadata_version", 3},
        {"tracer_version", 2},
        {"interp_goal", interpolationGoal},
        {"interp_mode", "lasagna"},
        {"metric", metric},
        {"msg", message},
        {"normal_manifest", normalManifest},
        {"fiber_manifest", ""},
        {"trace_to_base_scale", traceToBaseScale},
        {"meeting_error_base_voxels", nullptr},
        {"meeting_error_ratio", nullptr},
        {"meeting_source", ""},
        {"failure_code", ""},
        {"failure_detail", ""},
        {"lasagna_failure_code", ""},
        {"lasagna_failure_detail", ""},
        {"config", config},
    };
    detail::validateSegmentMetadata(result);
    return result;
}

}  // namespace vc::fiber_tracer
