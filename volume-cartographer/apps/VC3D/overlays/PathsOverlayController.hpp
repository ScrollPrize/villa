#pragma once

#include "ViewerOverlayControllerBase.hpp"

class PathsOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit PathsOverlayController(QObject* parent = nullptr);

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;
};

