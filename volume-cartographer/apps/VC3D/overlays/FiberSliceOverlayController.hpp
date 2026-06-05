#pragma once

#include "ViewerOverlayControllerBase.hpp"
#include "../FiberSliceGeometry.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

class VolumeViewerBase;

class FiberSliceOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct FiberData {
        uint64_t id{0};
        std::vector<cv::Vec3d> linePoints;
        std::vector<cv::Vec3d> controlPoints;
    };

    struct SliceData {
        std::string surfaceName;
        uint64_t selectedFiberId{0};
        vc3d::fiber_slice::Plane plane;
        std::vector<cv::Vec3d> fitSamples;
        std::vector<FiberData> fibers;
    };

    explicit FiberSliceOverlayController(QObject* parent = nullptr);

    void setSlice(VolumeViewerBase* viewer, SliceData data);
    void clearSlice();
    void detachViewer(VolumeViewerBase* viewer) override;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    QPointF projectedVolumeToScene(VolumeViewerBase* viewer, const cv::Vec3d& point) const;
    double currentViewportMinSpan(VolumeViewerBase* viewer) const;

    VolumeViewerBase* _activeViewer{nullptr};
    std::optional<SliceData> _slice;
};
