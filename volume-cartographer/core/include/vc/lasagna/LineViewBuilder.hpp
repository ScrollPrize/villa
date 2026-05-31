#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <memory>
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
};

LineViewSurfaces buildLineViewSurfaces(const LineModel& line,
                                       const LineViewConfig& config = {});

} // namespace vc::lasagna
