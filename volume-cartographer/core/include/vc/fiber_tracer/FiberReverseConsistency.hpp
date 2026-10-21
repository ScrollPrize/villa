#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

namespace vc::fiber_tracer {

// Backend-independent verification against the actual forward prediction.
// Coordinates and radii are base voxels. Geometry must come from
// makePolylineArcGeometry and outlive this object, as must normals.
class ReverseSpanVerifier {
public:
    ReverseSpanVerifier(const PolylineArcGeometry& geometry, double startArc,
        const cv::Vec3d& startPoint, const vc::lasagna::NormalSampler& normals,
        double normalWorkingToBaseScale, double normalRadius, double tangentRadius);
    // Stop at false: either the start was reached, or the first check failed.
    bool observe(const cv::Vec3d& point);
    double remainingArc() const { return arc_; }
private:
    const PolylineArcGeometry& geometry_;
    const vc::lasagna::NormalSampler& normals_;
    double scale_, normalRadius_, tangentRadius_, arc_;
    cv::Vec3d previous_;
};

// Return the earliest verified base arclength reached from startArc (zero means
// the whole span passed). No ground truth enters tracing or verification.
// Optional history seeds reverse smoothness from up to 64 forward edges and
// snaps the start to the next forward vertex. Normals use trace coordinates.
// For width gating, pass the forward result's widthMean as widthReferenceMean
// to keep the same fixed reference during the reverse check.
double verifyFiberReverseSpan(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase,
    int historySteps = 0);

} // namespace vc::fiber_tracer
