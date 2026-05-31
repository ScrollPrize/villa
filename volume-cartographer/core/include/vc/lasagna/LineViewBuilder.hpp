#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <memory>
#include <string>
#include <vector>

class PlaneSurface;
class QuadSurface;

namespace vc::lasagna {

struct LineViewConfig {
    // Non-positive values auto-size from the optimized control-point step and
    // crossSamples, so cross-strip spacing matches the line step.
    double surfaceHalfWidth = 0.0;
    double sideSliceHalfDepth = 0.0;
    int crossSamples = 21;
};

struct LineViewSurfaces {
    std::shared_ptr<QuadSurface> lineSurface;
    std::shared_ptr<QuadSurface> lineSideSlice;
    std::vector<std::shared_ptr<PlaneSurface>> lineZSlices;
    std::vector<cv::Vec3f> lineUpVectors;
};

struct LineViewFrameIssue {
    size_t index = 0;
    double rollDeltaRadians = 0.0;
    double normalContinuityDot = 1.0;
    double sideContinuityDot = 1.0;
    double sampledAxisContinuityDot = 1.0;
    double meshToSampledAxisDot = 1.0;
    std::string reason;
};

struct LineViewFrameDiagnostics {
    size_t frameCount = 0;
    double maxAbsRollDeltaRadians = 0.0;
    double minNormalContinuityDot = 1.0;
    double minSideContinuityDot = 1.0;
    double minSampledAxisContinuityDot = 1.0;
    double minMeshToSampledAxisDot = 1.0;
    std::vector<LineViewFrameIssue> issues;
};

LineViewSurfaces buildLineViewSurfaces(const LineModel& line,
                                       const LineViewConfig& config = {});

LineViewFrameDiagnostics diagnoseLineViewFrames(const LineModel& line,
                                                const LineViewConfig& config = {});

} // namespace vc::lasagna
