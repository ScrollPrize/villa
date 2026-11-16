#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

// Geometry-only strip normals for a traced fiber.
//
// A papyrus fiber is a flattened strip: it bends easily out of its plane and
// resists turning within the plane perpendicular to its normal. Each polyline
// vertex carries a unit normal n_i perpendicular to the fiber there (one angle
// theta_i in a parallel-transported frame). The solve minimizes, with Ceres,
//   twist:   w_t (theta_{i+1} - theta_i)^2            rotation of n around the fiber
//   lateral: w_l <l_i, t_i - t_{i-1}>^2               tangent turning inside n's plane, l = t x n
// where t_i is the direction of segment i of the smoothed, resampled curve. Twist is
// weighted far above lateral turning so vertex noise cannot rotate the normal.
// Nothing but the fiber's own geometry enters the energy: no umbilicus, no
// prediction field. The sign of a whole fiber stays undetermined here; the
// joint sign stage decides it.
namespace vc::fiber_tracer
{

struct FiberStripNormalOptions {
    double twistWeight = 100.0;
    double lateralWeight = 1.0;
    // The traced polyline is first smoothed along its arc with a Gaussian of
    // this sigma (voxels; 0 disables) and resampled at this fixed arc spacing
    // (voxels; 0 keeps the original vertices). Both lift the per-segment turn
    // above tracing jitter; the solve runs on the resampled curve and the
    // normals are interpolated back to the original vertices.
    double smoothingSigma = 32.0;
    double resampleSpacing = 32.0;
    int maxIterations = 50;
};

struct FiberStripNormalResult {
    std::vector<cv::Vec3d> normals;  // one unit normal per input vertex; empty when degenerate
    double initialCost = 0.0;
    double finalCost = 0.0;
    int iterations = 0;
    bool converged = false;
};

[[nodiscard]] FiberStripNormalResult solveFiberStripNormals(
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    const FiberStripNormalOptions& options = {});

// Per-fiber solves in parallel; a fiber with fewer than three distinct
// vertices gets an empty normal list.
[[nodiscard]] std::vector<FiberStripNormalResult> solveFiberStripNormals(
    const std::vector<std::vector<cv::Vec3d>>& fibers,
    const FiberStripNormalOptions& options,
    int threads);

}  // namespace vc::fiber_tracer
