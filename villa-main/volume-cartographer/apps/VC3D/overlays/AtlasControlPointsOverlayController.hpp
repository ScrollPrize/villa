#pragma once

#include "ViewerOverlayControllerBase.hpp"
#include "../AtlasControlPointsTypes.hpp"

#include <optional>

class AtlasControlPointsOverlayController : public ViewerOverlayControllerBase
{
public:
    explicit AtlasControlPointsOverlayController(QObject* parent = nullptr);

    void setResults(AtlasControlPointResults results);
    void clearResults();
    void setOverlayEnabled(bool enabled);
    [[nodiscard]] bool overlayEnabled() const { return _enabled; }
    void setSelectedPoint(const QString& fiberId, int controlIndex);
    void clearSelectedPoint();

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    AtlasControlPointResults _results;
    bool _enabled = false;
    std::optional<std::pair<QString, int>> _selected;
};
