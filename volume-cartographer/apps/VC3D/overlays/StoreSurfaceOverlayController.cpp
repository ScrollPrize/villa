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

vc::fibers::Point point(const cv::Vec3f& p)
{
    return {p[0], p[1], p[2]};
}

struct Listing {
    std::vector<std::pair<std::string, std::vector<std::array<double, 3>>>> runs;
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
    ++_revision;
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

std::optional<vc3d::store::Region> StoreSurfaceOverlayController::region(VolumeViewerBase* viewer) const
{
    if (!dynamic_cast<PlaneSurface*>(viewer->currentSurface())) return std::nullopt;
    const auto rect = visibleSceneRect(viewer);
    if (rect.isEmpty()) return std::nullopt;
    return vc3d::store::viewRegion(point(viewer->sceneToVolume(rect.topLeft())), point(viewer->sceneToVolume(rect.topRight())),
                                   point(viewer->sceneToVolume(rect.bottomLeft())), kHalfDepthVx);
}

bool StoreSurfaceOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && !_surfaces.empty() && _catalog->attachment() && _catalog->volume() &&
           dynamic_cast<PlaneSurface*>(viewer->currentSurface()) &&
           viewer->surfName() != "segmentation";
}

bool StoreSurfaceOverlayController::needsOverlayRebuild(VolumeViewerBase* viewer) const
{
    const auto found = _views.find(viewer);
    if (found == _views.end()) return true;
    const auto& v = *found->second;
    return v.dirty || v.displayed != region(viewer) || v.displayedRect != visibleSceneRect(viewer);
}

void StoreSurfaceOverlayController::request(VolumeViewerBase* viewer, const vc3d::store::Region& view)
{
    auto& ptr = _views[viewer];
    if (!ptr) ptr = std::make_unique<View>();
    auto& v = *ptr;
    const auto started = v.region.start(view);
    if (!started) return;
    const auto query = *started;
    const auto cancel = v.region.cancelled;
    const auto revision = _revision;
    vc3d::store::background<Listing>(this,
        [session = _catalog->session(), attachment = *_catalog->attachment(), volume = _catalog->volume()->id(),
         uuids = _surfaces, query, cancel]() {
            Listing listing;
            session->read(attachment, [&](const auto& db, const auto& txn, const auto& cancelled) {
                vc::objects::RegionOptions options;
                options.kind = "surface";
                options.volume = volume;
                options.uuids = uuids;
                auto result = db.geometryRegion(txn, vc3d::store::regionSlice(query), options,
                                                [&](const auto&) { return cancel->load() || cancelled.load(); });
                listing.complete = result.complete();
                for (auto& run : result.runs) listing.runs.emplace_back(std::move(run.objectUuid), std::move(run.points));
            });
            return listing;
        },
        [this, viewer, cancel, view, query, revision](vc3d::store::Outcome<Listing> result) {
            auto found = _views.find(viewer);
            if (found == _views.end()) return;
            auto& v = *found->second;
            using Finished = vc3d::store::RegionRequests::Finished;
            const auto finished = v.region.finish(cancel, revision == _revision, view, query, result.value.complete, result.error);
            if (finished == Finished::Stale) return;
            v.dirty = true;
            if (finished == Finished::Install && result.error.isEmpty()) {
                v.runs.clear();
                for (auto& [uuid, points] : result.value.runs) v.runs.push_back({std::move(uuid), std::move(points)});
            }
            refreshViewer(viewer);
        });
}

QColor StoreSurfaceOverlayController::colorFor(const std::string& uuid)
{
    const auto [found, inserted] = _colorIndex.try_emplace(uuid, _colorIndex.size());
    return QColor::fromRgba(vc3d::kIntersectionPalette[found->second % vc3d::kIntersectionPalette.size()]);
}

void StoreSurfaceOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    const auto current = region(viewer);
    if (!current) return;
    request(viewer, *current);
    auto& v = *_views.at(viewer);
    v.dirty = false;
    v.displayed = current;
    v.displayedRect = visibleSceneRect(viewer);
    if (!v.region.error.isEmpty()) {
        OverlayStyle info;
        info.penColor = QColor(255, 210, 100);
        info.z = vc3d::kIntersectionZ;
        builder.addText(v.displayedRect.topLeft() + QPointF(8, 60), tr("Surfaces: %1").arg(v.region.error), QFont{}, info);
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
        style.penColor = highlighted ? vc3d::highlightedIntersectionColor() : colorFor(uuid);
        style.penColor.setAlphaF(opacity);
        style.penWidth = viewer->intersectionThickness();
        style.z = highlighted ? vc3d::kHighlightedIntersectionZ : vc3d::kIntersectionZ;
        builder.addPainterPath(path, style);
    }
}
