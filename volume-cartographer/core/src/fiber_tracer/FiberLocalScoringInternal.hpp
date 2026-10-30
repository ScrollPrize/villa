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

using portable::kFiberLocalEpsilon;
using portable::clampFiberLocalUnit;
using portable::clampFiberLocalPositiveUnit;
using portable::clampFiberLocalFinitePositiveUnit;
using portable::finiteFiberLocalOrZero;
using portable::normalizeFiberLocalOrZero;
using portable::fiberLocalAngleBetweenUnit;
using portable::fiberLocalExcessAngleSquared;

using portable::fiberLocalAlignmentLossInline;

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

using portable::FiberLocalPreparedCandidateSmoothnessMode;
using FiberLocalPreparedCandidateSmoothness = portable::FiberLocalPreparedCandidateSmoothness<cv::Vec3f>;

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

using portable::prepareFiberLocalCandidateSmoothnessInline;

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

using portable::fiberLocalSmoothnessCostCandidatePreparedInline;
using portable::fiberLocalSmoothnessCostInline;

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
