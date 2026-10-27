#include "overlays/StoreSurfaceOverlayController.hpp"

#include "CState.hpp"
#include "SurfaceCatalog.hpp"
#include "SurfaceIntersectionStyle.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <QGraphicsView>
#include <QPainterPath>

namespace {
// Surfaces cross the slab's mid-plane; the depth only selects candidate tiles.
constexpr double kHalfDepthVx = 1.0;
// A cold query charges each crossed chunk's payload and hash check (~0.4 MB),
// so the store's default (64 MB) stopped a 2000 vx view of 16 surfaces; warm
// queries (verified chunks retained per connection) charge only what they read.
constexpr uint64_t kProcessingBudget = uint64_t(1) << 30;

struct Listing {
    std::vector<StoreSurfaceOverlayController::Run> runs;
    bool complete = true;
};
}

StoreSurfaceOverlayController::StoreSurfaceOverlayController(CState* state, SurfaceCatalog* catalog,
                                                             ViewerManager* manager, QObject* parent)
    : ViewerOverlayControllerBase("store-surfaces", parent), _state(state), _catalog(catalog)
{
    // Commits to drawn surfaces change their geometry; a new source changes everything.
    connect(_catalog, &SurfaceCatalog::surfacesCommitted, this, [this](const std::vector<std::string>& uuids) {
        if (std::any_of(uuids.begin(), uuids.end(), [this](const std::string& uuid) {
                return std::binary_search(_surfaces.begin(), _surfaces.end(), uuid);
            }))
            invalidate();
    });
    // The surface being edited is drawn by the viewer: redraw without it.
    connect(_state, &CState::surfaceChanged, this, [this](const std::string& name, const std::shared_ptr<Surface>&, bool edit) {
        if (name == "segmentation" && !edit) refreshAll();
    });
    bindToViewerManager(manager);
}

StoreSurfaceOverlayController::~StoreSurfaceOverlayController()
{
    for (auto& [viewer, view] : _views)
        if (view->region.cancelled) view->region.cancelled->store(true);
}

void StoreSurfaceOverlayController::setSurfaces(const std::vector<std::string>& uuids)
{
    auto sorted = uuids;
    std::sort(sorted.begin(), sorted.end());
    if (sorted == _surfaces) return;
    _surfaces = std::move(sorted);
    invalidate();
}

void StoreSurfaceOverlayController::setHighlighted(const std::vector<std::string>& uuids)
{
    std::set<std::string> highlighted(uuids.begin(), uuids.end());
    if (highlighted == _highlighted) return;
    _highlighted = std::move(highlighted);
    for (auto& [viewer, view] : _views) view->dirty = true;
    refreshAll();
}

void StoreSurfaceOverlayController::invalidate()
{
    for (auto& [viewer, view] : _views) {
        view->region.reset();
        view->runs.clear();
        view->displayed.reset();
        view->dirty = true;
    }
    refreshAll();
}

void StoreSurfaceOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    if (auto it = _views.find(viewer); it != _views.end()) {
        if (it->second->region.cancelled) it->second->region.cancelled->store(true);
        _views.erase(it);
    }
    ViewerOverlayControllerBase::detachViewer(viewer);
}

bool StoreSurfaceOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && !_surfaces.empty() && !_catalog->sources().empty() && _catalog->volume() &&
           dynamic_cast<PlaneSurface*>(viewer->currentSurface()) &&
           viewer->surfName() != "segmentation";
}

bool StoreSurfaceOverlayController::needsOverlayRebuild(VolumeViewerBase* viewer) const
{
    const auto found = _views.find(viewer);
    if (found == _views.end()) return true;
    const auto& v = *found->second;
    return v.dirty || v.displayed != planeViewRegion(viewer, kHalfDepthVx) || v.displayedRect != visibleSceneRect(viewer);
}

void StoreSurfaceOverlayController::request(VolumeViewerBase* viewer, const vc3d::store::Region& view)
{
    auto& v = _views[viewer];
    if (!v) v = std::make_unique<View>();
    vc3d::store::requestRegion<Listing>(this, v->region, view,
        [this, viewer]() -> View* {
            const auto found = _views.find(viewer);
            return found == _views.end() ? nullptr : found->second.get();
        },
        [session = _catalog->session(), sources = _catalog->sources(), volume = _catalog->volume()->id(),
         uuids = _surfaces](const vc3d::store::Region& query, const std::atomic_bool& cancelled) {
            vc::objects::RegionOptions options;
            options.uuids = uuids;
            options.processingByteBudget = kProcessingBudget;
            auto crossings = SurfaceCatalog::region(*session, sources, volume, vc3d::store::regionSlice(query), options,
                                                    [&] { return cancelled.load(); });
            Listing listing;
            listing.complete = crossings.complete;
            for (auto& run : crossings.runs) listing.runs.push_back({std::move(run.objectUuid), std::move(run.points)});
            return listing;
        },
        [this, viewer](View& v, Listing* listing) {
            v.dirty = true;
            if (listing) v.runs = std::move(listing->runs);
            refreshViewer(viewer);
        });
}

void StoreSurfaceOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    const auto current = planeViewRegion(viewer, kHalfDepthVx);
    if (!current) return;
    request(viewer, *current);
    auto& v = *_views.at(viewer);
    v.dirty = false;
    v.displayed = current;
    v.displayedRect = visibleSceneRect(viewer);
    const auto notice = !v.region.error.isEmpty() ? tr("Surfaces: %1").arg(v.region.error)
                        : v.region.loaded && !v.region.complete ? tr("Partial surface view · zoom in")
                                                                : QString{};
    if (!notice.isEmpty()) {
        OverlayStyle info;
        info.penColor = QColor(255, 210, 100);
        info.z = vc3d::kIntersectionZ;
        builder.addText(v.displayedRect.topLeft() + QPointF(8, 60), notice, QFont{}, info);
    }
    // The active surface is drawn by the viewer from its live (possibly edited) geometry.
    const auto active = _state->surface("segmentation");
    const std::string activeId = active ? active->id : std::string{};
    const float opacity = std::clamp(viewer->intersectionOpacity(), 0.0f, 1.0f);
    std::map<std::string, QPainterPath> paths;
    for (const auto& run : v.runs) {
        if (run.uuid == activeId || run.points.size() < 2) continue;
        auto& path = paths[run.uuid];
        path.moveTo(viewer->volumeToScene(cv::Vec3f(run.points[0][0], run.points[0][1], run.points[0][2])));
        for (size_t i = 1; i < run.points.size(); ++i)
            path.lineTo(viewer->volumeToScene(cv::Vec3f(run.points[i][0], run.points[i][1], run.points[i][2])));
    }
    for (const auto& [uuid, path] : paths) {
        const bool highlighted = _highlighted.contains(uuid);
        OverlayStyle style;
        style.penColor = highlighted ? vc3d::highlightedIntersectionColor() : vc3d::intersectionColor(uuid);
        style.penColor.setAlphaF(opacity);
        style.penWidth = viewer->intersectionThickness();
        style.z = highlighted ? vc3d::kHighlightedIntersectionZ : vc3d::kIntersectionZ;
        builder.addPainterPath(path, style);
    }
}
