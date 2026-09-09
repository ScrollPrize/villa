#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer::cli
{

inline std::string requireValue(
    int& index, int argc, char** argv, const std::string& name)
{
    if (index + 1 >= argc)
        throw std::invalid_argument("--" + name + " requires a value");
    return argv[++index];
}

inline double parseDouble(const std::string& text, const std::string& name)
{
    size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value))
        throw std::invalid_argument("--" + name + " requires a finite number");
    return value;
}

inline int parseInt(const std::string& text, const std::string& name)
{
    size_t parsed = 0;
    const long long value = std::stoll(text, &parsed);
    if (parsed != text.size() || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("--" + name + " requires an integer");
    }
    return static_cast<int>(value);
}

struct SeenOptions {
    bool beamWidth = false;
    bool beamLookahead = false;
};

inline bool parseTraceOption(
    const std::string& argument,
    int& index,
    int argc,
    char** argv,
    FiberTraceConfig& config,
    SeenOptions* seen = nullptr)
{
    const auto value = [&](const char* name) {
        return requireValue(index, argc, argv, name);
    };
    if (argument == "--step-voxels")
        config.stepVoxels = parseDouble(value("step-voxels"), "step-voxels");
    else if (argument == "--cone-angle-degrees")
        config.coneAngleDegrees = parseDouble(value("cone-angle-degrees"), "cone-angle-degrees");
    else if (argument == "--cone-angle-step-degrees")
        config.coneAngleStepDegrees = parseDouble(value("cone-angle-step-degrees"), "cone-angle-step-degrees");
    else if (argument == "--cone-grid-size")
        config.coneGridSize = parseInt(value("cone-grid-size"), "cone-grid-size");
    else if (argument == "--beam-width") {
        config.beamWidth = parseInt(value("beam-width"), "beam-width");
        if (seen)
            seen->beamWidth = true;
    } else if (argument == "--beam-prune-distance-voxels")
        config.beamPruneDistanceVoxels = parseDouble(value("beam-prune-distance-voxels"), "beam-prune-distance-voxels");
    else if (argument == "--beam-lookahead-steps") {
        config.beamLookaheadSteps = parseInt(value("beam-lookahead-steps"), "beam-lookahead-steps");
        if (seen)
            seen->beamLookahead = true;
    } else if (argument == "--lookahead-parent-cap") {
        const int cap = parseInt(value("lookahead-parent-cap"), "lookahead-parent-cap");
        if (cap < 0)
            throw std::invalid_argument("--lookahead-parent-cap must be non-negative");
        config.lookaheadParentCap = static_cast<size_t>(cap);
    } else if (argument == "--lookahead-retry-parent-cap") {
        const int cap = parseInt(value("lookahead-retry-parent-cap"), "lookahead-retry-parent-cap");
        if (cap < 0)
            throw std::invalid_argument("--lookahead-retry-parent-cap must be non-negative");
        config.lookaheadRetryParentCap = static_cast<size_t>(cap);
    } else if (argument == "--exhaustive-lookahead")
        config.lazyLookahead = false;
    else if (argument == "--threads")
        config.parallelThreads = parseInt(value("threads"), "threads");
    else if (argument == "--smoothness-weight")
        config.smoothnessWeight = parseDouble(value("smoothness-weight"), "smoothness-weight");
    else if (argument == "--smoothness-normal-weight")
        config.smoothnessNormalWeight = parseDouble(value("smoothness-normal-weight"), "smoothness-normal-weight");
    else if (argument == "--smoothness-tangent-weight")
        config.smoothnessTangentWeight = parseDouble(value("smoothness-tangent-weight"), "smoothness-tangent-weight");
    else if (argument == "--smoothness-free-angle-degrees")
        config.smoothnessFreeAngleDegrees = parseDouble(value("smoothness-free-angle-degrees"), "smoothness-free-angle-degrees");
    else if (argument == "--cumulative-smoothness-steps")
        config.cumulativeSmoothnessSteps = parseInt(value("cumulative-smoothness-steps"), "cumulative-smoothness-steps");
    else if (argument == "--cumulative-smoothness-tangent-weight")
        config.cumulativeSmoothnessTangentWeight = parseDouble(value("cumulative-smoothness-tangent-weight"), "cumulative-smoothness-tangent-weight");
    else if (argument == "--max-step-factor")
        config.maxStepFactor = parseDouble(value("max-step-factor"), "max-step-factor");
    else
        return false;
    return true;
}

inline void validateTraceOptions(const FiberTraceConfig& config)
{
    if (!(config.stepVoxels > 0.0))
        throw std::invalid_argument("--step-voxels must be positive");
    if (!(config.coneAngleDegrees >= 0.0))
        throw std::invalid_argument("--cone-angle-degrees must be non-negative");
    if (config.coneGridSize < 1 || config.beamWidth < 1 ||
        config.beamLookaheadSteps < 1 || config.parallelThreads < 0 ||
        config.cumulativeSmoothnessSteps < 1) {
        throw std::invalid_argument("fiber trace integer options are outside their valid range");
    }
    const double values[]{
        config.beamPruneDistanceVoxels,
        config.smoothnessWeight,
        config.smoothnessNormalWeight,
        config.smoothnessTangentWeight,
        config.smoothnessFreeAngleDegrees,
        config.cumulativeSmoothnessTangentWeight,
    };
    for (const double value : values) {
        if (!(value >= 0.0) || !std::isfinite(value))
            throw std::invalid_argument("fiber trace weights and angles must be finite and non-negative");
    }
    if (!(config.maxStepFactor > 0.0) || !std::isfinite(config.maxStepFactor))
        throw std::invalid_argument("--max-step-factor must be positive");
}

inline nlohmann::json traceConfigJson(const FiberTraceConfig& config)
{
    return {
        {"step_voxels", config.stepVoxels},
        {"cone_angle_degrees", config.coneAngleDegrees},
        {"cone_angle_step_degrees", config.coneAngleStepDegrees},
        {"cone_grid_size", config.coneGridSize},
        {"beam_width", config.beamWidth},
        {"beam_prune_distance_voxels", config.beamPruneDistanceVoxels},
        {"beam_lookahead_steps", config.beamLookaheadSteps},
        {"lazy_lookahead", config.lazyLookahead},
        {"lookahead_parent_cap", config.lookaheadParentCap},
        {"lookahead_retry_parent_cap", config.lookaheadRetryParentCap},
        {"parallel_threads", config.parallelThreads},
        {"smoothness_weight", config.smoothnessWeight},
        {"smoothness_normal_weight", config.smoothnessNormalWeight},
        {"smoothness_tangent_weight", config.smoothnessTangentWeight},
        {"smoothness_free_angle_degrees", config.smoothnessFreeAngleDegrees},
        {"cumulative_smoothness_steps", config.cumulativeSmoothnessSteps},
        {"cumulative_smoothness_tangent_weight", config.cumulativeSmoothnessTangentWeight},
        {"max_step_factor", config.maxStepFactor},
    };
}

} // namespace vc::fiber_tracer::cli
