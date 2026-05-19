#pragma once

#include "../volume_viewers/VolumeViewerBase.hpp"

#include <vector>

#include "vc/core/util/PlaneSurface.hpp"

namespace vc3d::segmentation {

inline void invalidateApprovalPlaneIntersections(const std::vector<VolumeViewerBase*>& viewers)
{
    for (auto* viewer : viewers) {
        if (!viewer) {
            continue;
        }
        if (dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            viewer->invalidateIntersect("segmentation");
            viewer->scheduleIntersectionRender("approval mask changed");
        }
    }
}

} // namespace vc3d::segmentation
