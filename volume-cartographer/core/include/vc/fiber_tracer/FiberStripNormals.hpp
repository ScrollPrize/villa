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
// Optionally a predicted sheet axis a_i (sign-free) pulls the normal at a node:
//   axis:    w_a w_i |n_i - a_i|^2
// with a per-node weight w_i (for example from the prediction's presence). The
// geometry-only solution is computed first and orients every a_i (so the term
// has one minimum per node and the sign stays continuous along the fiber);
// nodes without a prediction (w_i = 0) are bridged by the geometry terms. No
// umbilicus enters the energy. The sign of a whole fiber stays undetermined
// here; the joint sign stage decides it.
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
    // Scale of the axis term; multiplied by each target's own weight.
    double axisWeight = 1.0;
    int maxIterations = 50;
};

// The smoothed, resampled curve a solve runs on: `points` with their arc
// positions, and the arc position of every original vertex. Axis targets are
// given per chain point.
struct FiberStripChain {
    std::vector<cv::Vec3d> points;
    std::vector<double> arc;
    std::vector<double> vertexArc;
};

[[nodiscard]] FiberStripChain fiberStripChain(
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    const FiberStripNormalOptions& options = {});

// A predicted sheet axis at one chain point. The sign of `axis` is irrelevant;
// weight 0 (or a zero axis) means no prediction there.
struct FiberStripAxisTarget {
    cv::Vec3d axis{0.0, 0.0, 0.0};
    double weight = 0.0;
};

struct FiberStripNormalResult {
    std::vector<cv::Vec3d> normals;  // one unit normal per input vertex; empty when degenerate
    double initialCost = 0.0;
    double finalCost = 0.0;
    int iterations = 0;
    bool converged = false;
};

// `targets`, when given, holds one entry per point of fiberStripChain(points, options).
[[nodiscard]] FiberStripNormalResult solveFiberStripNormals(
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    const FiberStripNormalOptions& options = {},
    const std::vector<FiberStripAxisTarget>* targets = nullptr);

// Per-fiber solves in parallel; a fiber with fewer than three distinct
// vertices gets an empty normal list. `targets`, when given, has one list per fiber.
[[nodiscard]] std::vector<FiberStripNormalResult> solveFiberStripNormals(
    const std::vector<std::vector<cv::Vec3d>>& fibers,
    const FiberStripNormalOptions& options,
    int threads,
    const std::vector<std::vector<FiberStripAxisTarget>>* targets = nullptr);

}  // namespace vc::fiber_tracer
