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
    double intervalBase = 0;
    std::function<bool(const std::vector<cv::Vec3d>&)> verify;
    // Return prediction-derived seed, with its corresponding reference arc.
    std::function<std::optional<PolylineArcSample>(double)> nextSeed;
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
