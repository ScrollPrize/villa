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
    else if (argument == "--replay-reverse-interval-base") {
        config.replayReverseIntervalBase = parseDouble(value("replay-reverse-interval-base"), "replay-reverse-interval-base");
        if (!std::isfinite(config.replayReverseIntervalBase) || config.replayReverseIntervalBase <= 0)
            throw std::invalid_argument("replay reverse interval must be positive");
    }
    else if (argument == "--sample-pool-radius-voxels")
        config.samplePoolRadiusVoxels = parseDouble(value("sample-pool-radius-voxels"), "sample-pool-radius-voxels");
    else if (argument == "--sample-pool-mode")
        config.samplePoolMode = value("sample-pool-mode");
    else if (argument == "--sample-pool-stencil")
        config.samplePoolStencil = value("sample-pool-stencil");
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
    else if (argument == "--embedding-anchor-count")
        config.embeddingAnchorCount = parseInt(value("embedding-anchor-count"), "embedding-anchor-count");
    else if (argument == "--embedding-anchor-ema-mix")
        config.embeddingAnchorEmaMix = parseDouble(value("embedding-anchor-ema-mix"), "embedding-anchor-ema-mix");
    else if (argument == "--embedding-anchor-aggregation")
        config.embeddingAnchorAggregation = value("embedding-anchor-aggregation");
    else if (argument == "--embedding-anchor-novelty")
        config.embeddingAnchorNovelty = parseDouble(value("embedding-anchor-novelty"), "embedding-anchor-novelty");
    else if (argument == "--embedding-anchor-touch")
        config.embeddingAnchorTouch = value("embedding-anchor-touch");
    else if (argument == "--embedding-anchor-temperature")
        config.embeddingAnchorTemperature = parseDouble(value("embedding-anchor-temperature"), "embedding-anchor-temperature");
    else if (argument == "--embedding-integration")
        config.embeddingIntegration.kind = value("embedding-integration");
    else if (argument == "--embedding-update-spacing-base-voxels")
        config.embeddingIntegration.spacing = parseDouble(value("embedding-update-spacing-base-voxels"), "embedding-update-spacing-base-voxels");
    else if (argument == "--embedding-update-gate-bias")
        config.embeddingIntegration.gateBias = parseDouble(value("embedding-update-gate-bias"), "embedding-update-gate-bias");
    else if (argument == "--embedding-update-gate-presence")
        config.embeddingIntegration.gatePresence = parseDouble(value("embedding-update-gate-presence"), "embedding-update-gate-presence");
    else if (argument == "--embedding-update-gate-distance")
        config.embeddingIntegration.gateDistance = parseDouble(value("embedding-update-gate-distance"), "embedding-update-gate-distance");
    else if (argument == "--embedding-update-cap-degrees-per-8-base-voxels")
        config.embeddingIntegration.capDegrees = parseDouble(value("embedding-update-cap-degrees-per-8-base-voxels"), "embedding-update-cap-degrees-per-8-base-voxels");
    else if (argument == "--embedding-update-long-half-life-base-voxels")
        config.embeddingIntegration.longHalfLife = parseDouble(value("embedding-update-long-half-life-base-voxels"), "embedding-update-long-half-life-base-voxels");
    else if (argument == "--embedding-update-mix-bias")
        config.embeddingIntegration.mixBias = parseDouble(value("embedding-update-mix-bias"), "embedding-update-mix-bias");
    else if (argument == "--embedding-update-mix-slope")
        config.embeddingIntegration.mixSlope = parseDouble(value("embedding-update-mix-slope"), "embedding-update-mix-slope");
    else if (argument == "--embedding-integration-window-base-voxels")
        config.embeddingIntegration.window = parseDouble(value("embedding-integration-window-base-voxels"), "embedding-integration-window-base-voxels");
    else if (argument == "--embedding-distance-threshold")
        config.embeddingDistanceThreshold = parseDouble(value("embedding-distance-threshold"), "embedding-distance-threshold");
    else if (argument == "--embedding-distance-temperature")
        config.embeddingDistanceTemperature = parseDouble(value("embedding-distance-temperature"), "embedding-distance-temperature");
    else if (argument == "--embedding-ema-half-life-base-voxels")
        config.embeddingEmaHalfLifeBaseVoxels = parseDouble(value("embedding-ema-half-life-base-voxels"), "embedding-ema-half-life-base-voxels");
    else if (argument == "--embedding-ema-delay-base-voxels")
        config.embeddingEmaDelayBaseVoxels = parseDouble(value("embedding-ema-delay-base-voxels"), "embedding-ema-delay-base-voxels");
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
    if (!(config.samplePoolRadiusVoxels >= 0.0) || !std::isfinite(config.samplePoolRadiusVoxels))
        throw std::invalid_argument("--sample-pool-radius-voxels must be finite and non-negative");
    if (config.samplePoolMode != "max" && config.samplePoolMode != "mean")
        throw std::invalid_argument("--sample-pool-mode must be max or mean");
    if (config.samplePoolStencil != "cross" && config.samplePoolStencil != "cube")
        throw std::invalid_argument("--sample-pool-stencil must be cross or cube");
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
        {"embedding_weight", config.embeddingWeight},
        {"embedding_integration", {{"kind",config.embeddingIntegration.kind},{"spacing",config.embeddingIntegration.spacing},
            {"gate_bias",config.embeddingIntegration.gateBias},{"gate_presence",config.embeddingIntegration.gatePresence},{"gate_distance",config.embeddingIntegration.gateDistance},
            {"cap_degrees",config.embeddingIntegration.capDegrees},{"long_half_life",config.embeddingIntegration.longHalfLife},
            {"mix_bias",config.embeddingIntegration.mixBias},{"mix_slope",config.embeddingIntegration.mixSlope},{"window",config.embeddingIntegration.window}}},
        {"embedding_distance_threshold", config.embeddingDistanceThreshold},
        {"embedding_distance_temperature", config.embeddingDistanceTemperature},
        {"embedding_history_spacing_base_voxels", config.embeddingHistorySpacingBaseVoxels},
        {"embedding_history_window_base_voxels", config.embeddingHistoryWindowBaseVoxels},
        {"embedding_history_count", config.embeddingHistoryCount},
        {"embedding_history_falloff", config.embeddingHistoryFalloff},
        {"embedding_reference_mode", config.embeddingReferenceMode},
        {"embedding_anchor_count", config.embeddingAnchorCount},
        {"embedding_anchor_ema_mix", config.embeddingAnchorEmaMix},
        {"embedding_anchor_aggregation", config.embeddingAnchorAggregation},
        {"embedding_anchor_novelty", config.embeddingAnchorNovelty},
        {"embedding_anchor_touch", config.embeddingAnchorTouch},
        {"embedding_anchor_temperature", config.embeddingAnchorTemperature},
        {"embedding_ema_half_life_base_voxels", config.embeddingEmaHalfLifeBaseVoxels},
        {"embedding_ema_delay_base_voxels", config.embeddingEmaDelayBaseVoxels},
        {"step_voxels", config.stepVoxels},
        {"sample_pool_radius_voxels", config.samplePoolRadiusVoxels},
        {"sample_pool_mode", config.samplePoolMode},
        {"sample_pool_stencil", config.samplePoolStencil},
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
