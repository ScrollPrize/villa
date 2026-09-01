#pragma once

#include "vc/fiber_tracer/FiberTraceConstraints.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberTraceBalanceMode : unsigned char {
    None,
    Soft,
    Tight,
};

enum class FiberTraceBeliefInference : unsigned char {
    MinSum,
    SumProduct,
    SumProductMixed,
};

struct FiberTraceBeliefTopology {
    std::vector<std::vector<std::size_t>> piecesByTrace;
    std::vector<std::size_t> hardConstraintIndices;
    std::vector<std::size_t> softConstraintIndices;
    std::vector<FiberletCropTraceLine> pieceLines;
    std::vector<double> pieceCenterDistanceBaseVoxels;
    std::vector<double> normalizedArcWeights;
    std::size_t centralSeedPiece = 0;
};

struct FiberTraceBeliefPropagationConfig {
    FiberTraceBalanceMode balanceMode = FiberTraceBalanceMode::None;
    double targetHorizontalFraction = 0.5;
    double softBalanceStrength = 1.0;
    double horizontalnessTemperature = 1.25;
    double mixedUnaryCost = 1.0;
    double messageDamping = 0.5;
    double messageResidualTolerance = 1.0e-8;
    double balanceTolerance = 1.0e-3;
    std::size_t maximumMessageIterations = 500;
    std::size_t maximumBalanceIterations = 64;
    cv::Vec3d cropMinimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d cropMaximumBaseXYZ{0.0, 0.0, 0.0};
};

struct FiberTraceBeliefPropagationReport {
    std::vector<double> horizontalness;
    std::vector<double> minMarginalAdvantage;
    std::vector<double> logOdds;
    std::vector<double> verticalProbability;
    std::vector<double> mixedProbability;
    std::vector<double> horizontalProbability;
    std::vector<double> normalizedArcWeights;
    std::size_t seedPieceIndex = 0;
    std::size_t factors = 0;
    std::size_t mergedMeasurements = 0;
    std::size_t neutralFactors = 0;
    std::size_t neutralMeasurements = 0;
    std::size_t connectedComponents = 0;
    std::size_t isolatedPieces = 0;
    std::size_t messageIterations = 0;
    std::size_t balanceIterations = 0;
    double messageResidual = 0.0;
    double balanceField = 0.0;
    double targetHorizontalFraction = 0.0;
    double achievedHorizontalFraction = 0.0;
    double solveSeconds = 0.0;
    bool messageConverged = false;
    bool balanceConverged = false;
    FiberTraceBeliefInference inference = FiberTraceBeliefInference::MinSum;
    double inferenceTemperature = 0.0;
    double mixedUnaryCost = 0.0;
    std::string status;
};

struct FiberTraceConstraintConsistency {
    std::size_t pieceIndex = 0;
    std::size_t degree = 0;
    std::size_t incidentMeasurements = 0;
    std::size_t resolvedDegree = 0;
    std::size_t unresolvedDegree = 0;
    std::size_t hardMismatches = 0;
    double totalStrength = 0.0;
    double resolvedStrength = 0.0;
    double unresolvedStrength = 0.0;
    std::optional<double> hardMismatchRate;
    std::optional<double> weightedHardMismatchRate;
    std::optional<double> softMismatchProxy;
    std::optional<double> neighborSupportBalance;
    std::optional<double> neighborCertainty;
};

struct FiberTraceConstraintConsistencyReport {
    std::vector<FiberTraceConstraintConsistency> pieces;
    double verticalThreshold = 0.25;
    double horizontalThreshold = 0.75;
};

[[nodiscard]] const char* fiberTraceBalanceModeName(
    FiberTraceBalanceMode mode) noexcept;

[[nodiscard]] const char* fiberTraceBeliefInferenceName(
    FiberTraceBeliefInference inference) noexcept;

[[nodiscard]] FiberTraceBeliefTopology prepareFiberTraceBeliefTopology(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const cv::Vec3d& cropMinimumBaseXYZ,
    const cv::Vec3d& cropMaximumBaseXYZ);

[[nodiscard]] FiberTraceBeliefPropagationReport
solveFiberTraceBeliefPropagation(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config);

[[nodiscard]] FiberTraceBeliefPropagationReport
solveFiberTraceSumProduct(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config);

[[nodiscard]] FiberTraceBeliefPropagationReport
solveFiberTraceMixedSumProduct(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config);

[[nodiscard]] FiberTraceConstraintConsistencyReport
analyzeFiberTraceConstraintConsistency(
    const FiberTraceConstraintReport& constraints,
    std::span<const double> horizontalness,
    double verticalThreshold = 0.25,
    double horizontalThreshold = 0.75);

[[nodiscard]] FiberTraceConstraintConsistencyReport
analyzeMixedFiberTraceConstraintConsistency(
    const FiberTraceConstraintReport& constraints,
    std::span<const double> verticalProbability,
    std::span<const double> mixedProbability,
    std::span<const double> horizontalProbability,
    double verticalThreshold = 0.25,
    double horizontalThreshold = 0.75);

}  // namespace vc::fiber_tracer
