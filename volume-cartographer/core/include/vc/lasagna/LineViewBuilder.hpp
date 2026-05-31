#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <memory>
#include <vector>

class PlaneSurface;
class QuadSurface;

namespace vc::lasagna {

struct LineViewConfig {
    double surfaceHalfWidth = 50.0;
    double sideSliceHalfDepth = 50.0;
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
