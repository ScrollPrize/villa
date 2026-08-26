#pragma once

#include "vc/fiber_tracer/FiberletCropTrace.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberTraceConstraintConfig {
    double resampleSpacingBaseVoxels = 32.0;
    double targetPieceLengthBaseVoxels = 512.0;
    double pieceOverlapBaseVoxels = 128.0;
    double maximumDistanceBaseVoxels = 128.0;
    double tangentWindowBaseVoxels = 32.0;
    double phaseRefinementStepFraction = 0.05;
    double phaseRefinementLimitFraction = 0.05;
    double windingIntegrationStepBaseVoxels = 8.0;
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
    std::size_t hardConstraints = 0;
    double prepareSeconds = 0.0;
    double searchSeconds = 0.0;
    double orientationScoreSeconds = 0.0;
    double windingScoreSeconds = 0.0;
    double scoreSeconds = 0.0;
};

using FiberTraceWindingDistance = std::function<double(
    const cv::Vec3d& aBaseXYZ,
    const cv::Vec3d& bBaseXYZ,
    double stepBaseVoxels)>;

using FiberTraceWindingDistanceBatch = std::function<std::vector<double>(
    const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectorsBaseXYZ,
    double stepBaseVoxels,
    int parallelThreads)>;

struct FiberTraceConstraintObjPaths {
    std::filesystem::path perpendicular;
    std::filesystem::path parallelSameWinding;
    std::filesystem::path parallelSeparateWinding;
};

struct FiberTraceConstraintObjReport {
    FiberTraceConstraintObjPaths paths;
    std::size_t perpendicular = 0;
    std::size_t parallelSameWinding = 0;
    std::size_t parallelSeparateWinding = 0;
};

[[nodiscard]] FiberTraceConstraintReport extractFiberTraceConstraints(
    const std::vector<FiberletCropTraceLine>& lines,
    const FiberTraceConstraintConfig& config,
    const FiberTraceWindingDistance& windingDistance,
    const FiberTraceWindingDistanceBatch& windingDistanceBatch = {});

[[nodiscard]] FiberTraceConstraintObjPaths fiberTraceConstraintObjPaths(
    const std::filesystem::path& outputBase);

[[nodiscard]] FiberTraceConstraintObjReport writeFiberTraceConstraintObjs(
    const FiberTraceConstraintReport& report,
    const std::filesystem::path& outputBase);

}  // namespace vc::fiber_tracer
