#pragma once

// Shared directed-reference replay driver. It owns the evaluation protocol
// that every replay backend must use identically: seeding each segment exactly
// on the reference, forward reference matching, the anisotropic threshold
// ellipse, excursion grace, failure records, and the deterministic reset that
// continues through the remainder of the directed run.
//
// A backend supplies one ReplaySegmentRunner. The runner traces from the given
// reference-derived start and reports every committed point to the observer;
// the observer returns false once the segment must stop.

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <functional>
#include <string>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

// (committed point in base XYZ, cumulative loss, step index) -> continue?
using ReplayPointObserver =
    std::function<bool(const cv::Vec3d&, double, int)>;

// (segment start on the reference, remaining base voxels, maximum steps,
//  point observer, progress callback) -> termination reason
using ReplaySegmentRunner = std::function<std::string(
    const PolylineArcSample&,
    double,
    int,
    const ReplayPointObserver&,
    const FiberTraceProgressCallback&)>;

struct ReplayConsistencyPolicy {
    // Current and previous committed base points; false filters this point.
    std::function<bool(const cv::Vec3d&, const cv::Vec3d&)> acceptPoint;
    double intervalBase = 0;
    bool toStart = false;
    bool allEndpoints = false;
    std::function<bool(const std::vector<cv::Vec3d>&)> verify;
    // Return prediction-derived seed, with its corresponding reference arc.
    std::function<std::optional<PolylineArcSample>(double)> nextSeed;
};

// Backend-independent reverse verification against the actual forward path.
// Coordinates are base XYZ; this object must not outlive geometry/normals.
class ReverseSpanVerifier {
public:
    ReverseSpanVerifier(const PolylineArcGeometry& geometry, double startArc,
        const cv::Vec3d& startPoint, const vc::lasagna::NormalSampler& normals,
        double normalWorkingToBaseScale, double normalRadius, double tangentRadius);
    bool observe(const cv::Vec3d& point);
    double remainingArc() const { return arc_; }
private:
    const PolylineArcGeometry& geometry_;
    const vc::lasagna::NormalSampler& normals_;
    double scale_, normalRadius_, tangentRadius_, arc_;
    cv::Vec3d previous_;
};

[[nodiscard]] FiberReplayTraceResult traceReferenceReplayCore(
    const PolylineArcGeometry& reference,
    double startArcBase,
    double endArcBase,
    double nominalStepBase,
    double maxStepFactor,
    double errorThresholdBaseVoxels,
    double excursionGraceBaseVoxels,
    double tangentialThresholdFactor,
    double matchRefineSteps,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const ReplaySegmentRunner& runSegment,
    const FiberTraceProgressCallback& progress,
    const FiberReplayFailureCallback& failureCallback,
    const ReplayConsistencyPolicy& consistency = {});

}  // namespace vc::fiber_tracer
