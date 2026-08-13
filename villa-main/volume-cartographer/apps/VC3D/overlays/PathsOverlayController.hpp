#pragma once

#include "ViewerOverlayControllerBase.hpp"

class PathsOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit PathsOverlayController(QObject* parent = nullptr);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
};

