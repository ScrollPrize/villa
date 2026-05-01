#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

class PatchGraphOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit PatchGraphOverlayController(QObject* parent = nullptr);

    void setPath(std::vector<cv::Vec3f> path);
    void setHoverPoint(std::optional<cv::Vec3f> point);
    void clearPath();
    [[nodiscard]] bool hasPath() const { return !_path.empty(); }

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    std::vector<cv::Vec3f> _path;
    std::optional<cv::Vec3f> _hoverPoint;
};
