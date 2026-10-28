#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "ObjectStoreRegions.hpp"

#include <QPolygonF>
#include <QString>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class CState;

// Where surfaces cross each viewer, from the object store (stored surfaces and
// the session's drafts; no surface is materialized or indexed here):
//  - plane viewers: the surface list's visible surfaces and the viewer's
//    intersect targets (the active surface, workspace surfaces), queried per
//    view region;
//  - surface viewers: where the viewer's intersect planes ("seg xy/xz/yz") cut
//    its surface, drawn on the surface through each crossing's grid position.
// Drafts change while they are edited, so their lines are cut on the spot;
// stored surfaces are read on the thread pool.
class StoreSurfaceOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    StoreSurfaceOverlayController(CState* state, ViewerManager* manager, QObject* parent = nullptr);
    ~StoreSurfaceOverlayController() override;

    // The surfaces to draw in plane viewers (UUIDs) and those drawn highlighted.
    void setSurfaces(const std::vector<std::string>& uuids);
    void setHighlighted(const std::vector<std::string>& uuids);
    void detachViewer(VolumeViewerBase* viewer) override;

    // Where one surface crosses a view (volume voxels) and where on its grid.
    struct Run {
        std::string uuid;
        std::vector<std::array<double, 3>> points;
        std::vector<std::array<double, 2>> grid;
    };

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct Listing {
        std::vector<Run> runs;
        bool complete = true;
    };
    // A plane cutting a surface viewer's surface.
    struct PlaneCut {
        std::string plane;
        vc3d::store::Region slab;
    };
    struct View {
        // Plane viewers: stored targets in the view's region.
        vc3d::store::RegionRequests region;
        std::optional<vc3d::store::Region> displayed;
        QRectF displayedRect;
        std::vector<std::string> targets;
        std::vector<Run> runs;
        bool dirty{true};
        bool refreshPending{false};
        // Surface viewers: the planes' cuts of a stored surface.
        std::vector<PlaneCut> cuts;
        std::map<std::string, std::vector<Run>> planeRuns;
        bool cutsBusy{false};
        bool cutsStale{false};
    };
    void request(VolumeViewerBase* viewer, const vc3d::store::Region& region, const std::vector<std::string>& uuids);
    void requestCuts(VolumeViewerBase* viewer, const std::string& uuid, const std::vector<PlaneCut>& cuts);
    // Data changed: query every view again.
    void invalidate();
    // Stored geometry changed: keep displayed lines and let an in-flight read finish first.
    void refreshCommittedSurfaces();
    // The UUIDs a plane viewer shows: visible surfaces and its intersect targets.
    std::vector<std::string> targets(VolumeViewerBase* viewer) const;
    std::string activeUuid() const;
    std::vector<PlaneCut> planeCuts(VolumeViewerBase* viewer) const;
    // Drafts among `uuids`, cut synchronously.
    std::vector<Run> draftRuns(const std::vector<std::string>& uuids, const vc3d::store::Region& slab) const;
    void drawPlaneViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder);
    void drawSurfaceViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder);

    CState* _state;
    std::string _volume;
    std::vector<std::string> _surfaces;
    std::set<std::string> _highlighted;
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> _views;
};
