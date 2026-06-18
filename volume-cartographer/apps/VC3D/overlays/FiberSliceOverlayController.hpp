#pragma once

#include "ViewerOverlayControllerBase.hpp"
#include "../FiberSliceGeometry.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>

class VolumeViewerBase;

class FiberSliceOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct FiberStyle {
        OverlayStyle lineStyle;
        OverlayStyle controlStyle;
        OverlayStyle markerStyle;
    };

    struct FiberData {
        uint64_t id{0};
        std::vector<cv::Vec3d> linePoints;
        std::vector<cv::Vec3d> controlPoints;
        FiberStyle style;
    };

    struct ConnectionSegment {
        uint64_t sourceFiberId{0};
        uint64_t targetFiberId{0};
        cv::Vec3d sourcePoint{0.0, 0.0, 0.0};
        cv::Vec3d targetPoint{0.0, 0.0, 0.0};
        double maxDistanceVx{1.0};
    };

    struct FocusMarker {
        uint64_t fiberId{0};
        cv::Vec3d point{0.0, 0.0, 0.0};
        double radius{3.5};
        bool requirePlaneProximity{true};
    };

    struct SliceData {
        std::string surfaceName;
        uint64_t selectedFiberId{0};
        std::vector<uint64_t> fullLineFiberIds;
        vc3d::fiber_slice::Plane plane;
        std::vector<cv::Vec3d> fitSamples;
        std::vector<FiberData> fibers;
        std::optional<ConnectionSegment> connectionSegment;
        std::vector<FocusMarker> focusMarkers;
        bool showGenericCrossings{true};
        double focusMarkerViewportFraction{0.05};
    };

    explicit FiberSliceOverlayController(QObject* parent = nullptr);

    void setSlice(VolumeViewerBase* viewer, SliceData data);
    void clearSlice();
    void detachViewer(VolumeViewerBase* viewer) override;
    static FiberStyle sourceFiberStyle();
    static FiberStyle targetFiberStyle();
    static bool focusMarkerVisible(double distanceToPlane,
                                   double minVisibleViewportSpanVx,
                                   double viewportFraction = 0.05);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    QPointF projectedVolumeToScene(VolumeViewerBase* viewer,
                                   const SliceData& slice,
                                   const cv::Vec3d& point) const;
    double currentViewportMinSpan(VolumeViewerBase* viewer, const SliceData& slice) const;
    vc3d::fiber_slice::Plane currentPlaneForViewer(VolumeViewerBase* viewer,
                                                   const SliceData& slice) const;

    std::unordered_map<VolumeViewerBase*, SliceData> _slices;
};
