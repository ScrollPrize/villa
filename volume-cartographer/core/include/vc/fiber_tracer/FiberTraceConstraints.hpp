#pragma once

#include "vc/fiber_tracer/FiberletCropTrace.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct LasagnaNormalAlignmentField;

enum class FiberTraceParallelCorrespondence {
    Distance,
    PerpendicularGrid,
};

struct FiberTraceConstraintConfig {
    double resampleSpacingBaseVoxels = 32.0;
    double targetPieceLengthBaseVoxels = 512.0;
    double pieceOverlapBaseVoxels = 128.0;
    double maximumDistanceBaseVoxels = 128.0;
    double tangentWindowBaseVoxels = 32.0;
    FiberTraceParallelCorrespondence parallelCorrespondence =
        FiberTraceParallelCorrespondence::Distance;
    double phaseRefinementStepFraction = 0.05;
    double phaseRefinementLimitFraction = 0.05;
    double correspondenceGridStepFraction = 0.05;
    double correspondenceGridLimitFraction = 0.25;
    double windingIntegrationStepBaseVoxels = 8.0;
    double maximumWindingDistance = 1.5;
    bool enforceMaximumWindingDistance = true;
    bool preserveInputLinesAsPieces = false;
    std::size_t parallelThreads = 0;
};

struct FiberTraceConstraintPiece {
    std::size_t traceIndex = 0;
    std::size_t pieceIndex = 0;
    double beginArcBaseVoxels = 0.0;
    double endArcBaseVoxels = 0.0;
    std::vector<double> sampleArcsBaseVoxels;
    std::vector<cv::Vec3d> samplePointsBaseXYZ;
};

struct FiberTraceConstraint {
    std::size_t pieceA = 0;
    std::size_t pieceB = 0;
    double arcABaseVoxels = 0.0;
    double arcBBaseVoxels = 0.0;
    cv::Vec3d pointABaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d pointBBaseXYZ{0.0, 0.0, 0.0};
    double closestDistanceBaseVoxels = 0.0;
    double parallelScore = 0.0;
    double perpendicularScore = 0.0;
    double windingDistance = 0.0;
    bool hardContinuity = false;
    std::optional<double> signedWindingDelta;
    std::optional<std::size_t> windingNormalComponent;
    double parallelWindingDistance = 0.0;
    std::optional<double> signedParallelWindingDelta;
};

struct FiberTraceConstraintReport {
    std::vector<FiberTraceConstraintPiece> pieces;
    std::vector<FiberTraceConstraint> constraints;
    std::size_t inputTraces = 0;
    std::size_t skippedDegenerateTraces = 0;
    std::size_t resampledPoints = 0;
    std::size_t spatialHits = 0;
    std::size_t measuredCandidates = 0;
    std::size_t rejectedTangents = 0;
    std::size_t rejectedWinding = 0;
    std::size_t rejectedWindingCutoff = 0;
    std::size_t hardConstraints = 0;
    std::size_t signedWindingConstraints = 0;
    std::size_t skippedSignedWindingConstraints = 0;
    double prepareSeconds = 0.0;
    double searchSeconds = 0.0;
    double orientationScoreSeconds = 0.0;
    double windingScoreSeconds = 0.0;
    double scoreSeconds = 0.0;
};

void orientFiberTraceConstraintWindings(
    FiberTraceConstraintReport& report,
    const LasagnaNormalAlignmentField& alignedNormals);

struct FiberTraceConstraintGraphStats {
    std::size_t traces = 0;
    std::size_t crossTraceConstraints = 0;
    std::size_t isolatedTraces = 0;
    std::size_t connectedComponents = 0;
    std::size_t minimumDegree = 0;
    double meanDegree = 0.0;
    double medianDegree = 0.0;
    std::size_t maximumDegree = 0;
};

struct FiberTraceConstraintPruningReport {
    std::size_t maximumConstraintsPerTrace = 0;
    std::size_t inputTotalConstraints = 0;
    std::size_t retainedTotalConstraints = 0;
    std::size_t hardConstraints = 0;
    std::size_t rejectedZeroStrength = 0;
    std::size_t rejectedNotMutual = 0;
    std::size_t recoveryCandidates = 0;
    std::size_t expectedRecoveryBridges = 0;
    std::size_t recoveryBridges = 0;
    std::size_t capRespectingRecoveryBridges = 0;
    std::size_t fallbackOverflowBridges = 0;
    std::size_t tracesAboveTargetDegree = 0;
    FiberTraceConstraintGraphStats before;
    FiberTraceConstraintGraphStats mutual;
    FiberTraceConstraintGraphStats after;
};

struct FiberTraceConstraintPruningResult {
    std::vector<FiberTraceConstraint> constraints;
    FiberTraceConstraintPruningReport report;
};

struct FiberTraceConstraintSubsetResult {
    FiberTraceConstraintReport report;
    std::vector<std::size_t> retainedPieceIndices;
    std::vector<std::size_t> retainedTraceIndices;
};

using FiberTraceWindingDistance = std::function<double(
    const cv::Vec3d& aBaseXYZ,
    const cv::Vec3d& bBaseXYZ,
    double stepBaseVoxels)>;

using FiberTraceWindingDistanceBatch = std::function<std::vector<double>(
    const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectorsBaseXYZ,
    double stepBaseVoxels,
    int parallelThreads)>;

using FiberTraceConstraintTracePairFilter = std::function<bool(std::size_t traceA, std::size_t traceB)>;

struct FiberTraceConstraintObjPaths {
    std::filesystem::path perpendicularSameWinding;
    std::filesystem::path perpendicularSeparateWinding;
    std::filesystem::path parallelSameWinding;
    std::filesystem::path parallelSeparateWinding;
};

struct FiberTraceConstraintObjReport {
    FiberTraceConstraintObjPaths paths;
    std::size_t perpendicularSameWinding = 0;
    std::size_t perpendicularSeparateWinding = 0;
    std::size_t parallelSameWinding = 0;
    std::size_t parallelSeparateWinding = 0;
};

[[nodiscard]] FiberTraceConstraintReport extractFiberTraceConstraints(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberTraceConstraintConfig& config,
    const FiberTraceWindingDistance& windingDistance,
    const FiberTraceWindingDistanceBatch& windingDistanceBatch = {},
    const FiberTraceConstraintTracePairFilter& tracePairFilter = {},
    const LasagnaNormalAlignmentField* alignedNormals = nullptr);

[[nodiscard]] double dominantFiberTraceConstraintWindingDistance(
    const FiberTraceConstraint& constraint) noexcept;

struct FiberTraceOrderedCrossSourceConstraint {
    std::size_t constraintIndex = 0;
    std::size_t ownerSource = 0;
    std::size_t targetSource = 0;
    bool perpendicularDominant = false;
};

[[nodiscard]] std::vector<FiberTraceOrderedCrossSourceConstraint>
orderMeasuredCrossSourceFiberTraceConstraints(
    const FiberTraceConstraintReport& report,
    const std::vector<std::size_t>& sourceIdsByTrace);

[[nodiscard]] std::vector<FiberletCropTraceLine>
makeFiberTraceConstraintPieceLines(
    const std::vector<FiberletCropTraceLine>& sourceLines,
    const FiberTraceConstraintReport& constraints);

[[nodiscard]] FiberTraceConstraintSubsetResult subsetFiberTraceConstraintReport(
    const FiberTraceConstraintReport& constraints, std::span<const std::size_t> retainedPieceIndices);

[[nodiscard]] FiberTraceConstraintPruningResult
pruneFiberTraceConstraintsByStrength(
    const FiberTraceConstraintReport& constraints,
    double maximumDistanceBaseVoxels,
    std::size_t maximumConstraintsPerTrace);

[[nodiscard]] FiberTraceConstraintObjPaths fiberTraceConstraintObjPaths(
    const std::filesystem::path& outputBase);

[[nodiscard]] FiberTraceConstraintObjReport writeFiberTraceConstraintObjs(
    const FiberTraceConstraintReport& report,
    const std::filesystem::path& outputBase);

}  // namespace vc::fiber_tracer
