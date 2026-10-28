#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "ObjectStoreRegions.hpp"

#include <QPolygonF>
#include <QString>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class CState;
class PlaneSurface;
class QuadSurface;

// Where surfaces cross each viewer, from the object store (stored surfaces and
// the session's drafts; no surface is materialized or indexed here):
//  - plane viewers: the surface list's visible surfaces and the viewer's
//    intersect targets (the active surface, workspace surfaces), queried per
//    view region;
//  - surface viewers: where the viewer's intersect planes ("seg xy/xz/yz") cut
//    its surface, drawn on the surface through each crossing's grid position.
// Drafts change while they are edited, so their lines are cut on the spot;
// stored surfaces are read on the thread pool. Catalog placeholders reached
// are fetched one at a time in the background and drawn as each arrives.
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
    void clearOverlay(VolumeViewerBase* viewer) const override;
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct Listing {
        std::vector<Run> runs;
        bool complete = true;
        std::vector<std::pair<vc::objects::AttachmentIdentity, std::string>> remote;
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
        std::uint64_t runsVersion{0};
        // Stored lines are a group of their own, drawn for this camera and
        // style: a pan only moves them.
        // Drafts cut over the whole plane, at their revisions.
        std::optional<vc3d::store::Region> draftSlab;
        std::vector<std::string> drafted;
        std::vector<std::uint64_t> draftRevisions;
        std::vector<Run> draftRuns;
        struct Drawn {
            std::uint64_t runsVersion{0}, styleVersion{0};
            std::vector<std::uint64_t> draftRevisions;
            QPointF origin, axisU, axisV;
            cv::Vec3f planeOrigin, planeNormal;
            std::vector<float> offsets;
            std::string active;
            float opacity{0}, thickness{0};
            bool operator==(const Drawn&) const = default;
        };
        std::optional<Drawn> drawn;
        bool dirty{true};
        bool refreshPending{false};
        // Surface viewers: the planes' cuts of a stored surface.
        std::vector<PlaneCut> cuts;
        std::map<std::string, std::vector<Run>> planeRuns;
        std::uint64_t cutRevision{0};
        bool cutsBusy{false};
        bool cutsStale{false};
    };
    void request(VolumeViewerBase* viewer, const vc3d::store::Region& region, const std::vector<std::string>& uuids);
    void requestCuts(VolumeViewerBase* viewer, const std::string& uuid, const std::vector<PlaneCut>& cuts);
    // Fetch placeholders' geometry (not already fetching or failed); each
    // arrival is a commit, which reads the views again.
    void fetch(const std::vector<std::pair<vc::objects::AttachmentIdentity, std::string>>& remote);
    // Data changed: query every view again.
    void invalidate();
    // Stored geometry changed: keep displayed lines and let an in-flight read finish first.
    void refreshCommittedSurfaces();
    // The UUIDs a plane viewer shows: visible surfaces and its intersect targets.
    std::vector<std::string> targets(VolumeViewerBase* viewer) const;
    std::string activeUuid() const;
    std::vector<PlaneCut> planeCuts(VolumeViewerBase* viewer) const;
    // The slab of a whole plane (the cut is made in its mid-plane).
    static std::optional<vc3d::store::Region> planeSlab(PlaneSurface& plane);
    // How plane viewers draw lines: the active surface's colour and the
    // dotted copies at the flattened viewer's normal offset or slab bounds.
    struct LineStyle {
        std::string active;
        std::shared_ptr<QuadSurface> activeSurface;
        std::vector<float> offsets;
        bool volumetric{false};
        float opacity{0}, thickness{0};
    };
    LineStyle lineStyle(VolumeViewerBase* viewer) const;
    void addLines(VolumeViewerBase* viewer, const std::vector<Run>& runs, const LineStyle& style,
                  OverlayBuilder& builder) const;
    // Drafts among `uuids`, cut synchronously.
    std::vector<Run> draftRuns(const std::vector<std::string>& uuids, const vc3d::store::Region& slab) const;
    void drawPlaneViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder);
    void drawSurfaceViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder);

    CState* _state;
    std::string _volume;
    std::vector<std::string> _surfaces;
    std::set<std::string> _highlighted;
    std::uint64_t _styleVersion{0};
    std::set<std::string> _fetching;
    std::map<std::string, QString> _fetchFailures;
    std::shared_ptr<std::atomic_bool> _fetchCancel = std::make_shared<std::atomic_bool>(false);
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> _views;
};
