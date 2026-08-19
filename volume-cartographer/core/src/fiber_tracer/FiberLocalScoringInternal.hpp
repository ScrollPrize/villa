#pragma once

#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace vc::fiber_tracer::detail
{

inline constexpr float kFiberLocalEpsilon = 1.0e-6f;

static inline float clampFiberLocalUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, -1.0f, 1.0f);
}

static inline float clampFiberLocalPositiveUnit(float value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

static inline float clampFiberLocalFinitePositiveUnit(float value)
{
    return std::min(std::max(value, 0.0f), 1.0f);
}

static inline cv::Vec3f finiteFiberLocalOrZero(const cv::Vec3f& value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) ||
        !std::isfinite(value[2])) {
        return {0.0f, 0.0f, 0.0f};
    }
    return value;
}

static inline cv::Vec3f normalizeFiberLocalOrZero(const cv::Vec3f& value)
{
    const float length = std::sqrt(value.dot(value));
    if (!(length > kFiberLocalEpsilon) || !std::isfinite(length))
        return {0.0f, 0.0f, 0.0f};
    return value / length;
}

static inline float fiberLocalAngleBetweenUnit(
    const cv::Vec3f& left, const cv::Vec3f& right)
{
    return std::acos(clampFiberLocalUnit(left.dot(right)));
}

static inline float fiberLocalExcessAngleSquared(float angle, float freeAngle)
{
    const float excess = std::max(0.0f, angle - freeAngle);
    return excess * excess;
}

static inline float fiberLocalAlignmentLossInline(
    float presence,
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& currentPredictionDirection,
    const cv::Vec3f& candidatePredictionDirection)
{
    float score = clampFiberLocalPositiveUnit(presence);
    score *= clampFiberLocalPositiveUnit(
        previousStepDirection.dot(candidateStepDirection));
    score *= clampFiberLocalPositiveUnit(
        previousStepDirection.dot(currentPredictionDirection));
    score *= clampFiberLocalPositiveUnit(
        previousStepDirection.dot(candidatePredictionDirection));
    score *= clampFiberLocalPositiveUnit(
        currentPredictionDirection.dot(candidateStepDirection));
    score *= clampFiberLocalPositiveUnit(
        currentPredictionDirection.dot(candidatePredictionDirection));
    score *= clampFiberLocalPositiveUnit(
        candidateStepDirection.dot(candidatePredictionDirection));
    return 1.0f - score;
}

struct FiberLocalPreparedIncomingAlignment {
    cv::Vec3f previousDirection{0.0f, 0.0f, 0.0f};
    cv::Vec3f currentPredictionDirection{0.0f, 0.0f, 0.0f};
    float previousCurrentAlignment = 0.0f;
};

static inline FiberLocalPreparedIncomingAlignment
prepareFiberLocalIncomingAlignmentInline(
    const FiberLocalMetricSample* currentPrediction,
    const cv::Vec3f& previousStepDirection)
{
    FiberLocalPreparedIncomingAlignment prepared;
    prepared.previousDirection = previousStepDirection;
    prepared.currentPredictionDirection = previousStepDirection;
    if (currentPrediction != nullptr && currentPrediction->valid) {
        prepared.currentPredictionDirection = finiteFiberLocalOrZero(
            currentPrediction->direction);
        if (prepared.currentPredictionDirection.dot(previousStepDirection) <
            0.0f) {
            prepared.currentPredictionDirection *= -1.0f;
        }
    }
    prepared.previousCurrentAlignment = clampFiberLocalPositiveUnit(
        previousStepDirection.dot(prepared.currentPredictionDirection));
    return prepared;
}

enum class FiberLocalPreparedCandidateSmoothnessMode : std::uint8_t {
    InvalidDirection,
    IsotropicFallback,
    NormalAwareDegenerateTangent,
    NormalAware,
};

struct FiberLocalPreparedCandidateSmoothness {
    cv::Vec3f normal{0.0f, 0.0f, 0.0f};
    cv::Vec3f tangent{0.0f, 0.0f, 0.0f};
    float normalAngle = 0.0f;
    FiberLocalPreparedCandidateSmoothnessMode mode =
        FiberLocalPreparedCandidateSmoothnessMode::InvalidDirection;
};

struct FiberLocalPreparedCandidateMetric {
    cv::Vec3f direction{0.0f, 0.0f, 0.0f};
    cv::Vec3f predictionDirection{0.0f, 0.0f, 0.0f};
    float presence = 0.0f;
    float directionPredictionAlignment = 0.0f;
    FiberLocalPreparedCandidateSmoothness smoothness;
};

template <std::size_t Capacity>
struct FiberLocalPreparedCandidateAlignmentBatch {
    std::array<float, Capacity> directionX{};
    std::array<float, Capacity> directionY{};
    std::array<float, Capacity> directionZ{};
    std::array<float, Capacity> predictionX{};
    std::array<float, Capacity> predictionY{};
    std::array<float, Capacity> predictionZ{};
    std::array<float, Capacity> presence{};
    std::array<float, Capacity> directionPredictionAlignment{};
    std::array<std::uint8_t, Capacity> slotOfLane{};
    std::size_t count = 0;
};

template <std::size_t Capacity>
static inline void appendFiberLocalCandidateAlignmentInline(
    FiberLocalPreparedCandidateAlignmentBatch<Capacity>& batch,
    std::uint8_t slot,
    const FiberLocalPreparedCandidateMetric& candidate)
{
    assert(batch.count < Capacity);
    const std::size_t lane = batch.count++;
    const cv::Vec3f direction = finiteFiberLocalOrZero(candidate.direction);
    const cv::Vec3f prediction = finiteFiberLocalOrZero(
        candidate.predictionDirection);
    batch.directionX[lane] = direction[0];
    batch.directionY[lane] = direction[1];
    batch.directionZ[lane] = direction[2];
    batch.predictionX[lane] = prediction[0];
    batch.predictionY[lane] = prediction[1];
    batch.predictionZ[lane] = prediction[2];
    batch.presence[lane] = candidate.presence;
    batch.directionPredictionAlignment[lane] =
        candidate.directionPredictionAlignment;
    batch.slotOfLane[lane] = slot;
}

template <std::size_t Capacity>
static inline void fiberLocalAlignmentLossPreparedBatchInline(
    const FiberLocalPreparedIncomingAlignment& incoming,
    const FiberLocalPreparedCandidateAlignmentBatch<Capacity>& candidates,
    std::array<float, Capacity>& losses)
{
    for (std::size_t lane = 0; lane < candidates.count; ++lane) {
        const auto dot = [&](const cv::Vec3f& left,
                             const std::array<float, Capacity>& rightX,
                             const std::array<float, Capacity>& rightY,
                             const std::array<float, Capacity>& rightZ) {
            float result = 0.0f;
            result += left[0] * rightX[lane];
            result += left[1] * rightY[lane];
            result += left[2] * rightZ[lane];
            return result;
        };
        float score = candidates.presence[lane];
        score *= clampFiberLocalFinitePositiveUnit(dot(
            incoming.previousDirection,
            candidates.directionX,
            candidates.directionY,
            candidates.directionZ));
        score *= incoming.previousCurrentAlignment;
        score *= clampFiberLocalFinitePositiveUnit(dot(
            incoming.previousDirection,
            candidates.predictionX,
            candidates.predictionY,
            candidates.predictionZ));
        score *= clampFiberLocalFinitePositiveUnit(dot(
            incoming.currentPredictionDirection,
            candidates.directionX,
            candidates.directionY,
            candidates.directionZ));
        score *= clampFiberLocalFinitePositiveUnit(dot(
            incoming.currentPredictionDirection,
            candidates.predictionX,
            candidates.predictionY,
            candidates.predictionZ));
        score *= candidates.directionPredictionAlignment[lane];
        losses[lane] = 1.0f - score;
    }
}

static inline FiberLocalPreparedCandidateSmoothness
prepareFiberLocalCandidateSmoothnessInline(
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& normal,
    bool normalValid)
{
    FiberLocalPreparedCandidateSmoothness prepared;
    prepared.normal = normal;
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (candidateStepDirection.dot(candidateStepDirection) <= epsilon2)
        return prepared;
    if (!normalValid || normal.dot(normal) <= epsilon2) {
        prepared.mode =
            FiberLocalPreparedCandidateSmoothnessMode::IsotropicFallback;
        return prepared;
    }

    const float candidateNormal = clampFiberLocalUnit(
        candidateStepDirection.dot(normal));
    prepared.tangent = normalizeFiberLocalOrZero(
        candidateStepDirection - normal * candidateNormal);
    prepared.normalAngle = std::asin(candidateNormal);
    prepared.mode = prepared.tangent.dot(prepared.tangent) > epsilon2
        ? FiberLocalPreparedCandidateSmoothnessMode::NormalAware
        : FiberLocalPreparedCandidateSmoothnessMode::NormalAwareDegenerateTangent;
    return prepared;
}

static inline FiberLocalPreparedCandidateMetric
prepareFiberLocalCandidateMetricInline(
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& candidateStepDirection,
    const FiberLocalPreparedCandidateSmoothness& smoothness)
{
    FiberLocalPreparedCandidateMetric prepared;
    prepared.direction = candidateStepDirection;
    prepared.predictionDirection = candidatePrediction.direction;
    if (prepared.predictionDirection.dot(candidateStepDirection) < 0.0f)
        prepared.predictionDirection *= -1.0f;
    prepared.presence = clampFiberLocalPositiveUnit(
        candidatePrediction.presence);
    prepared.directionPredictionAlignment = clampFiberLocalPositiveUnit(
        candidateStepDirection.dot(prepared.predictionDirection));
    prepared.smoothness = smoothness;
    return prepared;
}

static inline FiberLocalPreparedCandidateMetric
prepareFiberLocalCandidateMetricInline(
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& normal,
    bool normalValid)
{
    return prepareFiberLocalCandidateMetricInline(
        candidatePrediction, candidateStepDirection,
        prepareFiberLocalCandidateSmoothnessInline(
            candidateStepDirection, normal, normalValid));
}

static inline float fiberLocalAlignmentLossPreparedInline(
    const FiberLocalPreparedIncomingAlignment& incoming,
    const FiberLocalPreparedCandidateMetric& candidate)
{
    float score = candidate.presence;
    score *= clampFiberLocalPositiveUnit(
        incoming.previousDirection.dot(candidate.direction));
    score *= incoming.previousCurrentAlignment;
    score *= clampFiberLocalPositiveUnit(
        incoming.previousDirection.dot(candidate.predictionDirection));
    score *= clampFiberLocalPositiveUnit(
        incoming.currentPredictionDirection.dot(candidate.direction));
    score *= clampFiberLocalPositiveUnit(
        incoming.currentPredictionDirection.dot(
            candidate.predictionDirection));
    score *= candidate.directionPredictionAlignment;
    return 1.0f - score;
}

static inline FiberLocalSmoothnessCost
fiberLocalSmoothnessCostCandidatePreparedInline(
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const FiberLocalPreparedCandidateSmoothness& candidatePrepared,
    const FiberLocalSmoothnessConfig& config)
{
    FiberLocalSmoothnessCost cost;
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 ||
        candidatePrepared.mode ==
            FiberLocalPreparedCandidateSmoothnessMode::InvalidDirection) {
        return cost;
    }

    if (candidatePrepared.mode ==
        FiberLocalPreparedCandidateSmoothnessMode::IsotropicFallback) {
        const float isotropicAngle = fiberLocalAngleBetweenUnit(
            previousStepDirection, candidateStepDirection);
        cost.isotropic = config.isotropicWeight *
                         fiberLocalExcessAngleSquared(
                             isotropicAngle, config.freeAngleRadians);
        cost.mode = FiberLocalSmoothnessMode::IsotropicFallback;
        return cost;
    }

    const float previousNormal = clampFiberLocalUnit(
        previousStepDirection.dot(candidatePrepared.normal));
    const cv::Vec3f previousTangent = normalizeFiberLocalOrZero(
        previousStepDirection - candidatePrepared.normal * previousNormal);
    const bool tangentValid =
        previousTangent.dot(previousTangent) > epsilon2 &&
        candidatePrepared.mode ==
            FiberLocalPreparedCandidateSmoothnessMode::NormalAware;
    const float tangentAngle = tangentValid
        ? fiberLocalAngleBetweenUnit(
              previousTangent, candidatePrepared.tangent)
        : fiberLocalAngleBetweenUnit(
              previousStepDirection, candidateStepDirection);
    const float normalAngle = std::abs(
        candidatePrepared.normalAngle - std::asin(previousNormal));
    cost.tangent = config.tangentWeight * fiberLocalExcessAngleSquared(
        tangentAngle, config.freeAngleRadians);
    cost.normal = config.normalWeight * fiberLocalExcessAngleSquared(
        normalAngle, config.freeAngleRadians);
    cost.mode = FiberLocalSmoothnessMode::NormalAware;
    return cost;
}

static inline FiberLocalSmoothnessCost fiberLocalSmoothnessCostInline(
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberLocalSmoothnessConfig& config)
{
    constexpr float epsilon2 = kFiberLocalEpsilon * kFiberLocalEpsilon;
    if (previousStepDirection.dot(previousStepDirection) <= epsilon2 ||
        candidateStepDirection.dot(candidateStepDirection) <= epsilon2) {
        return {};
    }
    return fiberLocalSmoothnessCostCandidatePreparedInline(
        previousStepDirection, candidateStepDirection,
        prepareFiberLocalCandidateSmoothnessInline(
            candidateStepDirection, normal, normalValid),
        config);
}

static inline FiberLocalMetricCost
fiberLocalMetricCostFromPreparedAlignmentInline(
    float alignmentLoss,
    const FiberLocalPreparedIncomingAlignment& incoming,
    float previousStepLength,
    float candidateStepLength,
    const FiberLocalPreparedCandidateMetric& candidate,
    const FiberLocalMetricConfig& config)
{
    FiberLocalMetricCost cost;
    cost.alignment = alignmentLoss * std::max(0.0f, candidateStepLength);
    const auto smoothness = fiberLocalSmoothnessCostCandidatePreparedInline(
        incoming.previousDirection, candidate.direction,
        candidate.smoothness, config.smoothness);
    const float effectiveLength = std::max(
        1.0f,
        (std::max(0.0f, previousStepLength) +
         std::max(0.0f, candidateStepLength)) * 0.5f);
    cost.isotropicSmoothness = smoothness.isotropic / effectiveLength;
    cost.tangentSmoothness = smoothness.tangent / effectiveLength;
    cost.normalSmoothness = smoothness.normal / effectiveLength;
    return cost;
}

static inline FiberLocalMetricCost
fiberLocalMetricCostFullyPreparedInline(
    const FiberLocalPreparedIncomingAlignment& incoming,
    float previousStepLength,
    float candidateStepLength,
    const FiberLocalPreparedCandidateMetric& candidate,
    const FiberLocalMetricConfig& config)
{
    return fiberLocalMetricCostFromPreparedAlignmentInline(
        fiberLocalAlignmentLossPreparedInline(incoming, candidate),
        incoming, previousStepLength, candidateStepLength, candidate, config);
}

static inline FiberLocalMetricCost
fiberLocalMetricCostCandidatePreparedInline(
    const FiberLocalMetricSample* currentPrediction,
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& previousStepUnitDirection,
    float previousStepLength,
    const cv::Vec3f& candidateStepUnitDirection,
    float candidateStepLength,
    const FiberLocalPreparedCandidateSmoothness& candidateSmoothness,
    const FiberLocalMetricConfig& config)
{
    FiberLocalMetricCost cost;
    if (!candidatePrediction.valid) {
        cost.invalidPrediction = config.invalidPredictionCostPerVoxel *
                                 std::max(0.0f, candidateStepLength);
        return cost;
    }
    const auto candidate = prepareFiberLocalCandidateMetricInline(
        candidatePrediction, candidateStepUnitDirection,
        candidateSmoothness);
    return fiberLocalMetricCostFullyPreparedInline(
        prepareFiberLocalIncomingAlignmentInline(
            currentPrediction, previousStepUnitDirection),
        previousStepLength, candidateStepLength, candidate, config);
}

static inline FiberLocalMetricCost fiberLocalMetricCostPreparedInline(
    const FiberLocalMetricSample* currentPrediction,
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& previousStepUnitDirection,
    float previousStepLength,
    const cv::Vec3f& candidateStepUnitDirection,
    float candidateStepLength,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberLocalMetricConfig& config)
{
    if (!candidatePrediction.valid) {
        return fiberLocalMetricCostCandidatePreparedInline(
            currentPrediction, candidatePrediction,
            previousStepUnitDirection, previousStepLength,
            candidateStepUnitDirection, candidateStepLength,
            {}, config);
    }
    return fiberLocalMetricCostCandidatePreparedInline(
        currentPrediction, candidatePrediction,
        previousStepUnitDirection, previousStepLength,
        candidateStepUnitDirection, candidateStepLength,
        prepareFiberLocalCandidateSmoothnessInline(
            candidateStepUnitDirection, normal, normalValid),
        config);
}

}  // namespace vc::fiber_tracer::detail
