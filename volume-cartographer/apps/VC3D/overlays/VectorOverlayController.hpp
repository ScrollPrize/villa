#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <functional>
#include <vector>

class CSurfaceCollection;

class VectorOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    using Provider = std::function<void(CVolumeViewer*, OverlayBuilder&)>;

    explicit VectorOverlayController(CSurfaceCollection* surfaces, QObject* parent = nullptr);

    void addProvider(Provider provider);

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;

private:
    void collectDirectionHints(CVolumeViewer* viewer, OverlayBuilder& builder) const;

    CSurfaceCollection* _surfaces;
    std::vector<Provider> _providers;
};
