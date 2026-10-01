#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

namespace vc::fiber_tracer {
std::optional<PolylineArcSample> findFiberReplayRestartSeed(
    const FiberPredictionSource& predictions, const FiberReplayTraceRequest& request,
    const vc::lasagna::NormalSampler& normalSampler, double normalWorkingToBaseScale,
    const PolylineArcGeometry& reference, double fromArc, double endArcBase,
    double nominalStepBase);
// Return the earliest verified arc reached from startArc. No GT is involved.
double verifyFiberReverseSpan(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase);
}
