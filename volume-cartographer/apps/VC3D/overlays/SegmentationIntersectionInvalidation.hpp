#pragma once

#include "../volume_viewers/VolumeViewerBase.hpp"

#include <vector>

#include "vc/core/util/PlaneSurface.hpp"

namespace vc3d::segmentation {

enum class ApprovalIntersectionRefresh {
    Immediate,
    Deferred,
};

inline void invalidateApprovalPlaneIntersections(
    const std::vector<VolumeViewerBase*>& viewers,
    ApprovalIntersectionRefresh refresh)
{
    for (auto* viewer : viewers) {
        if (!viewer) {
            continue;
        }
        if (dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
            viewer->invalidateIntersect("segmentation");
            if (refresh == ApprovalIntersectionRefresh::Immediate) {
                viewer->renderIntersections("approval mask changed");
            } else {
                viewer->scheduleIntersectionRender("approval mask changed");
            }
        }
    }
}

} // namespace vc3d::segmentation
