#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <opencv2/core/matx.hpp>

#include <QString>
#include <vector>

// A Spiral fit's fibers at the revisions it pinned, read-only: each line is
// clipped to a slab around a plane viewer's slice and coloured by its fitted
// H/V family, as the fiber collection colours them.
class SpiralFiberOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    struct Fiber {
        std::vector<cv::Vec3d> line;  // volume voxels of the workspace, XYZ
        QString family;               // "H", "V" or empty
    };

    explicit SpiralFiberOverlayController(QObject* parent = nullptr);
    void setFibers(std::vector<Fiber> fibers);
    void setVisible(bool visible);
    // Half the slab depth, in volume voxels, around each plane viewer.
    void setSlabHalfDepth(double voxels);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    std::vector<Fiber> _fibers;
    bool _visible = false;
    double _halfDepth = 2.0;
};
