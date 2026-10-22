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
    else if (argument == "--width-relative-tolerance") {
        config.widthRelativeTolerance = parseDouble(value("width-relative-tolerance"), "width-relative-tolerance");
        if (!std::isfinite(config.widthRelativeTolerance) || config.widthRelativeTolerance < 0)
            throw std::invalid_argument("width relative tolerance must be finite and nonnegative (zero disables it)");
    }
    else if (argument == "--width-weighting") {
        const std::string shape = value("width-weighting");
        if (shape != "linear" && shape != "cosine")
            throw std::invalid_argument("width weighting must be linear or cosine");
        config.widthCosineWeight = shape == "cosine";
    }
    else if (argument == "--replay-presence-mode") {
        const std::string mode = value("replay-presence-mode");
        if (mode != "raw" && mode != "modulated")
            throw std::invalid_argument("replay presence mode must be raw or modulated");
        config.replayPresenceModulated = mode == "modulated";
    }
    else if (argument == "--replay-presence-cutoff") {
        config.replayPresenceCutoff = parseDouble(value("replay-presence-cutoff"), "replay-presence-cutoff");
        if (!std::isfinite(config.replayPresenceCutoff) || config.replayPresenceCutoff < 0 || config.replayPresenceCutoff > 1)
            throw std::invalid_argument("replay presence cutoff must be in [0,1]");
    }
    else if (argument == "--replay-reverse-interval-base") {
        config.replayReverseIntervalBase = parseDouble(value("replay-reverse-interval-base"), "replay-reverse-interval-base");
        if (!std::isfinite(config.replayReverseIntervalBase) || config.replayReverseIntervalBase <= 0)
            throw std::invalid_argument("replay reverse interval must be positive");
    }
    else if (argument == "--replay-reverse-normal-base" || argument == "--replay-reverse-tangential-base") {
        const double radius = parseDouble(value(argument.c_str()), argument);
        if (!(radius > 0.0))
            throw std::invalid_argument("replay reverse radius must be finite and positive");
        if (argument == "--replay-reverse-normal-base") config.replayReverseNormalRadiusBase = radius;
        else config.replayReverseTangentialRadiusBase = radius;
    }
    else if (argument == "--replay-reverse-to-start")
        config.replayReverseToStart = true;
    else if (argument == "--replay-reverse-all-endpoints")
        config.replayReverseAllEndpoints = true;
    else if (argument == "--replay-reverse-history-steps") {
        config.replayReverseHistorySteps = parseInt(value("replay-reverse-history-steps"), "replay-reverse-history-steps");
        if (config.replayReverseHistorySteps < 0 || config.replayReverseHistorySteps > 64)
            throw std::invalid_argument("reverse history steps must be in [0,64]");
    }
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
    else if (argument == "--embedding-weight")
        config.embeddingWeight = parseDouble(value("embedding-weight"), "embedding-weight");
    else if (argument == "--embedding-history-spacing-base-voxels")
        config.embeddingHistorySpacingBaseVoxels = parseDouble(value("embedding-history-spacing-base-voxels"), "embedding-history-spacing-base-voxels");
    else if (argument == "--embedding-history-window-base-voxels")
        config.embeddingHistoryWindowBaseVoxels = parseDouble(value("embedding-history-window-base-voxels"), "embedding-history-window-base-voxels");
    else if (argument == "--embedding-history-falloff")
        config.embeddingHistoryFalloff = value("embedding-history-falloff");
    else if (argument == "--embedding-history-count")
        config.embeddingHistoryCount = parseInt(value("embedding-history-count"), "embedding-history-count");
    else if (argument == "--embedding-reference-mode")
        config.embeddingReferenceMode = value("embedding-reference-mode");
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
        {"replay_reverse_interval_base", config.replayReverseIntervalBase},
        {"replay_reverse_normal_radius_base", config.replayReverseNormalRadiusBase},
        {"replay_reverse_tangential_radius_base", config.replayReverseTangentialRadiusBase},
        {"replay_reverse_to_start", config.replayReverseToStart},
        {"replay_reverse_all_endpoints", config.replayReverseAllEndpoints},
        {"replay_reverse_history_steps", config.replayReverseHistorySteps},
        {"embedding_weight", config.embeddingWeight},
        {"replay_presence_cutoff", config.replayPresenceCutoff},
        {"replay_presence_mode", config.replayPresenceModulated ? "modulated" : "raw"},
        {"width_relative_tolerance", config.widthRelativeTolerance},
        {"width_weighting", config.widthCosineWeight ? "cosine" : "linear"},
        {"embedding_history_spacing_base_voxels", config.embeddingHistorySpacingBaseVoxels},
        {"embedding_history_window_base_voxels", config.embeddingHistoryWindowBaseVoxels},
        {"embedding_history_count", config.embeddingHistoryCount},
        {"embedding_history_falloff", config.embeddingHistoryFalloff},
        {"embedding_reference_mode", config.embeddingReferenceMode},
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
