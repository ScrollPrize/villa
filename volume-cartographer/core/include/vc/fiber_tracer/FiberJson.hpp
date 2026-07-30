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
    requireExactKeys(
        value,
        {"optimizer",
         "metadata_version",
         "tracer_version",
         "normal_manifest",
         "fiber_manifest",
         "trace_to_base_scale",
         "max_endpoint_error_base_voxels",
         "config"},
        "segment_to_next");
    if (value.at("optimizer").get<std::string>() != "native_fiber_trace3d" || value.at("metadata_version").get<int>() != 1 ||
        value.at("tracer_version").get<int>() != 1) {
        throw std::runtime_error("unsupported segment_to_next optimizer or version");
    }
    if (value.at("normal_manifest").get<std::string>().empty() || value.at("fiber_manifest").get<std::string>().empty()) {
        throw std::runtime_error("segment_to_next manifest locations must not be empty");
    }
    const double scale = value.at("trace_to_base_scale").get<double>();
    const double error = value.at("max_endpoint_error_base_voxels").get<double>();
    if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(error) || error < 0.0) {
        throw std::runtime_error("segment_to_next scale/error values are invalid");
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
         "fusion_gap_factor",
         "endpoint_accept_threshold_base_voxels"},
        "segment_to_next config");
    for (const auto& [key, item] : config.items()) {
        if (!item.is_number() || !std::isfinite(item.get<double>())) {
            throw std::runtime_error("segment_to_next config field is invalid: " + key);
        }
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
        if (version != 2 || !value.is_object()) {
            throw std::runtime_error(context + " version-2 control point must be an object");
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
