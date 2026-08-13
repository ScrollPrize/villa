#pragma once

#include "ViewerOverlayControllerBase.hpp"

class BBoxOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit BBoxOverlayController(QObject* parent = nullptr);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
};

