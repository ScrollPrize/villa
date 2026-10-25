#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "ObjectStoreRegions.hpp"

#include <QPolygonF>
#include <QString>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class CState;
class SurfaceCatalog;

// Where the surface list's visible surfaces cross each plane viewer, read from
// the object store per view region (no surface is materialized). The surface
// being edited is drawn by the viewer itself from its live geometry.
class StoreSurfaceOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    StoreSurfaceOverlayController(CState* state, SurfaceCatalog* catalog, ViewerManager* manager, QObject* parent = nullptr);
    ~StoreSurfaceOverlayController() override;

    // The surfaces to draw (UUIDs) and those drawn highlighted.
    void setSurfaces(const std::vector<std::string>& uuids);
    void setHighlighted(const std::vector<std::string>& uuids);
    void detachViewer(VolumeViewerBase* viewer) override;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct Run {
        std::string uuid;
        std::vector<std::array<double, 3>> points;
    };
    struct View {
        vc3d::store::RegionRequests region;
        std::optional<vc3d::store::Region> displayed;
        QRectF displayedRect;
        std::vector<Run> runs;
        bool dirty{true};
    };
    std::optional<vc3d::store::Region> region(VolumeViewerBase* viewer) const;
    void request(VolumeViewerBase* viewer, const vc3d::store::Region& region);
    // Data changed: query every view again.
    void invalidate();
    QColor colorFor(const std::string& uuid);

    CState* _state;
    SurfaceCatalog* _catalog;
    std::vector<std::string> _surfaces;
    std::set<std::string> _highlighted;
    std::map<std::string, std::size_t> _colorIndex;
    uint64_t _revision{0};
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> _views;
};
