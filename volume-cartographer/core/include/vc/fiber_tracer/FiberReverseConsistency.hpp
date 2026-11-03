#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

namespace vc::fiber_tracer {

struct ReverseSpanOptions {
    // Incoming edges from the FUTURE side of the forward checkpoint, in reverse
    // travel order. Empty preserves legacy initialization.
    std::vector<cv::Vec3d> initialDirections;
    double verifiedThroughArc = 0;
    double minimumTravel = 0;
    double convergenceDistance = 4;
    double convergenceAngleDegrees = 5;
    int convergenceSteps = 4;
};
struct ReverseSpanResult {
    bool passed = false;
    bool converged = false;
    double remainingArc = 0;
    double travelled = 0;
};

std::vector<cv::Vec3d> fiberReverseFutureHistory(
    const PolylineArcGeometry&, size_t checkpoint, size_t count);

// Backend-independent verification against the actual forward prediction.
// Coordinates and radii are base voxels. Geometry must come from
// makePolylineArcGeometry and outlive this object, as must normals.
class ReverseSpanVerifier {
public:
    ReverseSpanVerifier(const PolylineArcGeometry& geometry, double startArc,
        const cv::Vec3d& startPoint, const vc::lasagna::NormalSampler& normals,
        double normalWorkingToBaseScale, double normalRadius, double tangentRadius,
        const ReverseSpanOptions& options = {});
    // Stop at false: either the start was reached, or the first check failed.
    bool observe(const cv::Vec3d& point);
    double remainingArc() const { return arc_; }
    ReverseSpanResult result() const { return {arc_ <= 1e-12 || converged_, converged_, arc_, travelled_}; }
private:
    const PolylineArcGeometry& geometry_;
    const vc::lasagna::NormalSampler& normals_;
    double scale_, normalRadius_, tangentRadius_, arc_;
    cv::Vec3d previous_;
    ReverseSpanOptions options_;
    double travelled_ = 0;
    int closeSteps_ = 0;
    bool converged_ = false;
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

ReverseSpanResult verifyFiberReverseSpanDetailed(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase,
    int historySteps, const ReverseSpanOptions& options);

} // namespace vc::fiber_tracer
