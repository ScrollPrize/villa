#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include "vc/atlas/Atlas.hpp"

#include <QRectF>

#include <memory>
#include <optional>

class QuadSurface;

class AtlasOverlayController : public ViewerOverlayControllerBase
{
public:
    explicit AtlasOverlayController(QObject* parent = nullptr);

    void setAtlas(vc::atlas::Atlas atlas,
                  std::shared_ptr<const QuadSurface> displaySurface,
                  vc::atlas::AtlasDisplayRange displayRange);
    void clearAtlas();

    [[nodiscard]] std::optional<QRectF> surfaceBounds() const;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    [[nodiscard]] std::optional<cv::Vec2f> atlasAnchorToSurface(
        const vc::atlas::AtlasAnchor& anchor,
        const vc::atlas::FiberMapping& fiber) const;

    std::optional<vc::atlas::Atlas> _atlas;
    std::shared_ptr<const QuadSurface> _displaySurface;
    vc::atlas::AtlasDisplayRange _displayRange;
};
