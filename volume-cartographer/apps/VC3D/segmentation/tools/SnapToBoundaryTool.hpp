#pragma once

#include "SegmentationTool.hpp"

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

class SegmentationEditManager;
class SegmentationModule;
class CAdaptiveVolumeViewer;
#ifndef CTiledVolumeViewer
#define CTiledVolumeViewer CAdaptiveVolumeViewer
#endif
class CState;
class Volume;
class PlaneSurface;

// Click-drag snap of the active QuadSurface onto the nearest iso-crossing in
// the volume. The user clicks on a misaligned intersection-line in a slice
// viewer (XY/XZ/YZ), drags to where the line should land, and on release every
// affected vertex of the active surface is pushed along its local normal until
// it crosses the iso threshold (volumeWindowLow). The drag direction
// disambiguates which iso crossing to prefer when several exist along the
// normal.
class SnapToBoundaryTool : public SegmentationTool
{
public:
    SnapToBoundaryTool(SegmentationModule& module,
                       SegmentationEditManager* editManager,
                       CState* state);

    void setDependencies(SegmentationEditManager* editManager,
                         CState* state);

    bool startStroke(CTiledVolumeViewer* viewer, const cv::Vec3f& worldPos);
    void extendStroke(const cv::Vec3f& worldPos);
    bool applyStroke();

    void cancel() override;
    [[nodiscard]] bool isActive() const override { return _active; }

    // Preview line (start → current cursor) in world coordinates. Empty when
    // the tool is inactive.
    [[nodiscard]] const std::vector<cv::Vec3f>& previewWorldPoints() const { return _previewPoints; }

private:
    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    CState* _state{nullptr};

    bool _active{false};
    cv::Vec3f _startWorld{0.0f, 0.0f, 0.0f};
    cv::Vec3f _currentWorld{0.0f, 0.0f, 0.0f};
    CTiledVolumeViewer* _viewer{nullptr};

    // Cached for the duration of a stroke
    std::shared_ptr<Volume> _volume;
    cv::Vec3f _planeNormal{0.0f, 0.0f, 1.0f};

    std::vector<cv::Vec3f> _previewPoints;
};
