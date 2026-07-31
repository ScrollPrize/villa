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
    if (value.at("optimizer").get<std::string>() != "native_fiber_trace3d")
        throw std::runtime_error("unsupported segment_to_next optimizer");
    const int metadataVersion = value.at("metadata_version").get<int>();
    const int tracerVersion = value.at("tracer_version").get<int>();
    const bool previous = metadataVersion == 1 && tracerVersion == 1;
    const bool version2 = metadataVersion == 2 && tracerVersion == 2;
    const bool current = metadataVersion == 3 && tracerVersion == 2;
    if (!previous && !version2 && !current) {
        throw std::runtime_error("unsupported segment_to_next optimizer or version");
    }
    requireExactKeys(
        value,
        previous
            ? std::unordered_set<std::string>{
                  "optimizer", "metadata_version", "tracer_version",
                  "normal_manifest", "fiber_manifest", "trace_to_base_scale",
                  "max_endpoint_error_base_voxels", "config"}
            : version2 ? std::unordered_set<std::string>{
                  "optimizer", "metadata_version", "tracer_version", "outcome",
                  "normal_manifest", "fiber_manifest", "trace_to_base_scale",
                  "meeting_error_base_voxels", "meeting_error_ratio",
                  "meeting_source", "failure_code", "failure_detail", "config"}
            : std::unordered_set<std::string>{
                  "optimizer", "metadata_version", "tracer_version",
                  "interp_goal", "interp_mode", "metric", "msg",
                  "normal_manifest", "fiber_manifest", "trace_to_base_scale",
                  "meeting_error_base_voxels", "meeting_error_ratio",
                  "meeting_source", "failure_code", "failure_detail",
                  "lasagna_failure_code", "lasagna_failure_detail", "config"},
        "segment_to_next");
    if (!current && (value.at("normal_manifest").get<std::string>().empty() || value.at("fiber_manifest").get<std::string>().empty())) {
        throw std::runtime_error("segment_to_next manifest locations must not be empty");
    }
    const double scale = value.at("trace_to_base_scale").get<double>();
    if (!std::isfinite(scale) || scale <= 0.0)
        throw std::runtime_error("segment_to_next scale is invalid");
    if (previous) {
        const double error = value.at("max_endpoint_error_base_voxels").get<double>();
        if (!std::isfinite(error) || error < 0.0)
            throw std::runtime_error("segment_to_next endpoint error is invalid");
    } else if (version2) {
        const std::string outcome = value.at("outcome").get<std::string>();
        if (outcome != "accepted_native" && outcome != "lasagna_fallback")
            throw std::runtime_error("segment_to_next outcome is invalid");
        bool haveError = false;
        std::string source;
        if (outcome == "accepted_native") {
            haveError = !value.at("meeting_error_base_voxels").is_null();
            const bool haveRatio = !value.at("meeting_error_ratio").is_null();
            if (haveError != haveRatio) {
                throw std::runtime_error(
                    "segment_to_next meeting diagnostics are inconsistent");
            }
            if (haveError) {
                const double error =
                    value.at("meeting_error_base_voxels").get<double>();
                const double ratio = value.at("meeting_error_ratio").get<double>();
                if (!std::isfinite(error) || error < 0.0 ||
                    !std::isfinite(ratio) || ratio < 0.0) {
                    throw std::runtime_error(
                        "segment_to_next meeting diagnostics are invalid");
                }
            }
            source = value.at("meeting_source").get<std::string>();
        }
        const std::string failure = value.at("failure_code").get<std::string>();
        const std::string detail = value.at("failure_detail").get<std::string>();
        if (outcome == "accepted_native" &&
            (!haveError || source.empty() || !failure.empty() || !detail.empty())) {
            throw std::runtime_error("accepted segment_to_next outcome is inconsistent");
        }
        if (outcome == "lasagna_fallback" && failure.empty())
            throw std::runtime_error("fallback segment_to_next requires failure_code");
    } else {
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
         previous ? "fusion_gap_factor" : "meeting_accept_max_error_ratio",
         "endpoint_accept_threshold_base_voxels"},
        "segment_to_next config");
    for (const auto& [key, item] : config.items()) {
        if (!item.is_number() || !std::isfinite(item.get<double>())) {
            throw std::runtime_error("segment_to_next config field is invalid: " + key);
        }
    }
    if (!previous) {
        const double ratio = config.at("meeting_accept_max_error_ratio").get<double>();
        if (ratio < 0.0 || ratio > 1.0)
            throw std::runtime_error("segment_to_next meeting ratio config is invalid");
    }
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
        if ((version != 2 && version != 3) || !value.is_object()) {
            throw std::runtime_error(context + " version-2/3 control point must be an object");
        }
        for (const auto& [field, item] : value.items()) {
            (void)item;
            if (field != "position" && field != "segment_to_next") {
                throw std::runtime_error(context + " control point contains unknown field: " + field);
            }
        }
        points.push_back(detail::pointFromJson(value.at("position"), context));
        if (value.contains("segment_to_next")) {
            if (index + 1 == array.size()) {
                throw std::runtime_error(context + " final control point cannot contain segment_to_next");
            }
            detail::validateSegmentMetadata(value.at("segment_to_next"));
        }
    }
    return points;
}

}  // namespace vc::fiber_tracer
