#pragma once

// Fixed-endpoint non-linear least-squares refinement of a skeleton-initialized
// polyline. The objective mirrors the shared length-scaled discrete fiber
// metric (`FiberLocalScoring`): every residual is the square root of the
// corresponding discrete term, so squared residuals reproduce its weighting
// and order. The beam tracer's unscaled per-step form and its cumulative
// tangent term are deliberately not mirrored.

#include "vc/fiber_tracer/FiberSkeletonFields.hpp"

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberSkeletonCostWeights {
    double normal = 0.1;
    double tangent = 10.0;
    double isotropic = 2.0;
    double spacing = 1.0;
};

struct FiberSkeletonSolverConfig {
    // Measured on the case31 corridor: line refinements take up to about 1300
    // iterations and region connections up to about 800, so a low cap silently
    // truncates most solves mid-descent and leaves the geometry wrong.
    int maxIterations = 2000;
    double functionTolerance = 1.0e-6;
};

struct FiberSkeletonRefineConfig {
    FiberSkeletonCostWeights weights;
    FiberSkeletonSolverConfig solver;
    // Unit tangents that continue the line beyond its fixed endpoints. When
    // present they contribute the boundary smoothness terms that tie the
    // refined geometry to the port orientation.
    std::optional<cv::Vec3d> startTangent;
    std::optional<cv::Vec3d> endTangent;
    bool freeStart = false;
    bool freeEnd = false;
};

struct FiberSkeletonCostBreakdown {
    double alignment = 0.0;
    double normalSmoothness = 0.0;
    double tangentSmoothness = 0.0;
    double isotropicSmoothness = 0.0;
    double spacing = 0.0;

    [[nodiscard]] double total() const noexcept
    {
        return alignment + normalSmoothness + tangentSmoothness +
               isotropicSmoothness + spacing;
    }
};

struct FiberSkeletonRefineResult {
    std::vector<cv::Vec3d> pointsBaseXYZ;
    FiberSkeletonCostBreakdown initialCost;
    FiberSkeletonCostBreakdown finalCost;
    double lengthBase = 0.0;
    double initialLengthBase = 0.0;
    double spacingBase = 0.0;
    int iterations = 0;
    double solveSeconds = 0.0;
    bool converged = false;
    std::string terminationMessage;
};

// `initialPointsBaseXYZ` must already be uniformly resampled; its point count
// is preserved. Corner blocks for every vertex are reloaded from `fields`
// before each solver evaluation.
[[nodiscard]] FiberSkeletonRefineResult refineFixedEndpointPolyline(
    const std::vector<cv::Vec3d>& initialPointsBaseXYZ,
    FiberSkeletonFieldProvider& fields, const FiberSkeletonRefineConfig& config);

// The discrete cost of an existing polyline under the same term definitions,
// evaluated once without optimizing. Used for initial-loss comparisons and to
// report costs for geometry that is not refined.
[[nodiscard]] FiberSkeletonCostBreakdown evaluateFiberSkeletonPolylineCost(
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    FiberSkeletonFieldProvider& fields, const FiberSkeletonRefineConfig& config);

}  // namespace vc::fiber_tracer
