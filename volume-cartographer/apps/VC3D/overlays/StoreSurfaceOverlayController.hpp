#pragma once
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "ObjectStoreRegions.hpp"
#include "SurfaceCatalog.hpp"

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

// Where surfaces cross each viewer, using the store's shared intersection kernels:
//  - plane viewers: the surface list's visible surfaces and the viewer's
//    intersect targets (the active surface, workspace surfaces), queried per
//    view region;
//  - surface viewers: intersect the displayed immutable grid (current, historical
//    or draft) through the same resident/store intersection kernel.
// All selected geometry is cut on the thread pool. Catalog placeholders reached
// are fetched one at a time in the background and drawn as each arrives.
class StoreSurfaceOverlayController : public ViewerOverlayControllerBase
{
    friend int runViewerRegressionTests();
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
        std::optional<vc::objects::ObjectVersion> reference;
    };

protected:
    void clearOverlay(VolumeViewerBase* viewer) const override;
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    bool needsOverlayRebuild(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct Listing {
        std::map<std::string, std::vector<Run>> runs;
        bool complete = true;
        std::vector<std::pair<vc::objects::AttachmentIdentity, std::string>> remote;
    };
    // A plane cutting a surface viewer's surface.
    struct PlaneCut {
        std::string plane;
        vc3d::store::Region slab;
        bool operator==(const PlaneCut&) const = default;
    };
    struct View {
        // One request lifecycle and result set for every geometry source.
        vc3d::store::RegionRequests region;
        std::optional<vc3d::store::Region> displayed;
        QRectF displayedRect;
        std::vector<std::string> targets;
        std::map<std::string, std::vector<Run>> runs;
        std::uint64_t runsVersion{0};
        // Stored lines are a group of their own, drawn for this camera and
        // style: a pan only moves them.
        std::vector<std::uint64_t> draftRevisions;
        std::map<std::string, std::shared_ptr<const vc::objects::ResidentSurface>> displayedSurfaces;
        struct Drawn {
            std::uint64_t runsVersion{0}, styleVersion{0};
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
        // Cutting planes; plane viewers use an unnamed viewport cut.
        std::vector<PlaneCut> cuts;
    };
    void request(VolumeViewerBase* viewer, const SurfaceCatalog::RegionInput& input, const std::vector<PlaneCut>& cuts);
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
    void drawPlaneViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder);
    void drawSurfaceViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder);
    std::shared_ptr<vc::objects::ResidentSurface> residentSurface(QuadSurface& surface);

    CState* _state;
    std::vector<std::string> _surfaces;
    std::set<std::string> _highlighted;
    std::uint64_t _styleVersion{0};
    std::set<std::string> _fetching;
    std::map<std::string, QString> _fetchFailures;
    std::shared_ptr<std::atomic_bool> _fetchCancel = std::make_shared<std::atomic_bool>(false);
    std::unordered_map<VolumeViewerBase*, std::unique_ptr<View>> _views;
    std::map<std::string, std::pair<const QuadSurface*, std::shared_ptr<vc::objects::ResidentSurface>>> _residentSurfaces;
};
