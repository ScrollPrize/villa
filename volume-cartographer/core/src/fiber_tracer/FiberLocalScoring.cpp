#include "vc/fiber_tracer/FiberLocalScoring.hpp"

#include "FiberLocalScoringInternal.hpp"

#include <optional>

namespace vc::fiber_tracer
{
float fiberLocalAlignmentLoss(
    float presence,
    const cv::Vec3f& previousStepDirection,
    const cv::Vec3f& candidateStepDirection,
    const cv::Vec3f& currentPredictionDirection,
    const cv::Vec3f& candidatePredictionDirection)
{
    return detail::fiberLocalAlignmentLossInline(
        presence, previousStepDirection, candidateStepDirection,
        currentPredictionDirection, candidatePredictionDirection);
}

FiberLocalSmoothnessCost fiberLocalSmoothnessCost(
    const cv::Vec3f& previousStepDirection, const cv::Vec3f& candidateStepDirection, const cv::Vec3f& normal, bool normalValid, const FiberLocalSmoothnessConfig& config)
{
    return detail::fiberLocalSmoothnessCostInline(
        previousStepDirection, candidateStepDirection,
        normal, normalValid, config);
}

FiberLocalMetricCost fiberLocalMetricCostPrepared(
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
    return detail::fiberLocalMetricCostPreparedInline(
        currentPrediction, candidatePrediction,
        previousStepUnitDirection, previousStepLength,
        candidateStepUnitDirection, candidateStepLength,
        normal, normalValid, config);
}

cv::Vec3f prepareFiberLocalUnitDirection(const cv::Vec3f& direction)
{
    return detail::normalizeFiberLocalOrZero(direction);
}

FiberLocalMetricCost fiberLocalMetricCost(
    const FiberLocalMetricSample* currentPrediction,
    const FiberLocalMetricSample& candidatePrediction,
    const cv::Vec3f& previousStepDirection,
    float previousStepLength,
    const cv::Vec3f& candidateStepDirection,
    float candidateStepLength,
    const cv::Vec3f& normal,
    bool normalValid,
    const FiberLocalMetricConfig& config)
{
    std::optional<FiberLocalMetricSample> preparedCurrent;
    if (currentPrediction != nullptr) {
        preparedCurrent = *currentPrediction;
        preparedCurrent->direction =
            prepareFiberLocalUnitDirection(preparedCurrent->direction);
    }
    FiberLocalMetricSample preparedCandidate = candidatePrediction;
    preparedCandidate.direction =
        prepareFiberLocalUnitDirection(preparedCandidate.direction);
    return fiberLocalMetricCostPrepared(
        preparedCurrent.has_value() ? &*preparedCurrent : nullptr,
        preparedCandidate,
        prepareFiberLocalUnitDirection(previousStepDirection),
        previousStepLength,
        prepareFiberLocalUnitDirection(candidateStepDirection),
        candidateStepLength,
        normal,
        normalValid,
        config);
}

}  // namespace vc::fiber_tracer
