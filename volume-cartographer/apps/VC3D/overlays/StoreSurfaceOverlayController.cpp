#include "overlays/StoreSurfaceOverlayController.hpp"

#include "CState.hpp"
#include "SurfaceCatalog.hpp"
#include "SurfaceIntersectionStyle.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

#include <QGraphicsView>
#include <QPainterPath>
#include <algorithm>
#include <cmath>
#include <utility>

namespace {
// Surfaces cross the slab's mid-plane; the depth only selects candidate tiles.
constexpr double kHalfDepthVx = 1.0;
// A plane cutting a surface viewer's surface: far beyond any volume.
constexpr double kPlaneExtentVx = 1e6;
// A cold query charges each crossed chunk's payload and hash check (~0.4 MB),
// so the store's default (64 MB) stopped a 2000 vx view of 16 surfaces; warm
// queries (verified chunks retained per connection) charge only what they read.
constexpr uint64_t kProcessingBudget = uint64_t(1) << 30;
// The planes a surface viewer can be cut by.
constexpr std::array<const char*, 3> kCutPlanes = {"seg xy", "seg xz", "seg yz"};

using Run = StoreSurfaceOverlayController::Run;

std::vector<Run> runsOf(std::vector<vc::objects::VisibleRun>&& visible)
{
    std::vector<Run> runs;
    runs.reserve(visible.size());
    for (auto& run : visible) runs.push_back({std::move(run.objectUuid), std::move(run.points), std::move(run.grid)});
    return runs;
}

vc::objects::RegionOptions surfaceOptions(const std::vector<std::string>& uuids)
{
    vc::objects::RegionOptions options;
    options.uuids = uuids;
    options.processingByteBudget = kProcessingBudget;
    return options;
}

cv::Vec3f vec(const std::array<double, 3>& p) { return {float(p[0]), float(p[1]), float(p[2])}; }

bool finite(const QPointF& p) { return std::isfinite(p.x()) && std::isfinite(p.y()); }
bool finite(const cv::Vec3f& v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }
}

StoreSurfaceOverlayController::StoreSurfaceOverlayController(CState* state, ViewerManager* manager, QObject* parent)
    : ViewerOverlayControllerBase("store-surfaces", parent), _state(state)
{
    connect(_state, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
        using Kind = vc::objects::StoreSessionEvent::Kind;
        if (event.kind == Kind::Hub) return;
        if (event.kind == Kind::Attached || event.kind == Kind::Detached) {
            invalidate();
        } else if (event.kind == Kind::Draft) {
            // Drafts are cut on every redraw; a draft starting or ending
            // moves its surface between drafted and stored targets there.
            for (auto& [viewer, view] : _views) view->dirty = true;
            refreshAll();
        } else if (event.touches({"surface"})) {
            refreshCommittedSurfaces();
        }
    });
    // volumeChanged also repeats for the same volume; only a new one changes the data.
    connect(_state, &CState::volumeChanged, this, [this](std::shared_ptr<Volume>, const std::string& volume) {
        if (std::exchange(_volume, volume) != volume) invalidate();
    });
    // Edits update their drafts before announcing the surface; intersect
    // targets and planes are named surfaces.
    connect(_state, &CState::surfaceChanged, this, [this](std::string, std::shared_ptr<Surface>, bool) {
        for (auto& [viewer, view] : _views) view->dirty = true;
        refreshAll();
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
    for (auto& [viewer, view] : _views) view->dirty = true;
    refreshAll();
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
        view->targets.clear();
        view->cuts.clear();
        view->planeRuns.clear();
        view->dirty = true;
        view->refreshPending = false;
    }
    refreshAll();
}

void StoreSurfaceOverlayController::refreshCommittedSurfaces()
{
    for (auto& [viewer, view] : _views) {
        view->dirty = true;
        view->cutsStale = true;
        if (view->region.busy) {
            // Fetching a placeholder commits its geometry. Cancelling here
            // would repeatedly interrupt the batch that is waiting for it.
            view->refreshPending = true;
        } else {
            view->region.refresh();
        }
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

std::string StoreSurfaceOverlayController::activeUuid() const
{
    const auto active = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    return active ? active->id : std::string{};
}

std::vector<std::string> StoreSurfaceOverlayController::targets(VolumeViewerBase* viewer) const
{
    std::vector<std::string> uuids = _surfaces;
    for (const auto& name : viewer->intersects())
        if (const auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name));
            quad && vc::objects::util::isUuid(quad->id))
            uuids.push_back(quad->id);
    std::sort(uuids.begin(), uuids.end());
    uuids.erase(std::unique(uuids.begin(), uuids.end()), uuids.end());
    return uuids;
}

std::vector<StoreSurfaceOverlayController::PlaneCut> StoreSurfaceOverlayController::planeCuts(VolumeViewerBase* viewer) const
{
    std::vector<PlaneCut> cuts;
    const auto& names = viewer->intersects();
    for (const char* name : kCutPlanes) {
        if (!names.count(name)) continue;
        const auto plane = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(name));
        if (!plane) continue;
        const cv::Vec3d o = plane->origin(), bx = plane->basisX(), by = plane->basisY();
        const auto corner = [&](double x, double y) {
            const cv::Vec3d p = o + bx * x + by * y;
            return vc::fibers::Point{p[0], p[1], p[2]};
        };
        const double l = kPlaneExtentVx;
        if (auto slab = vc3d::store::viewRegion(corner(-l, -l), corner(l, -l), corner(-l, l), kHalfDepthVx))
            cuts.push_back({name, *slab});
    }
    return cuts;
}

std::vector<Run> StoreSurfaceOverlayController::draftRuns(const std::vector<std::string>& uuids,
                                                          const vc3d::store::Region& slab) const
{
    const auto session = _state->objectStoreSession();
    if (uuids.empty() || !session || !_state->currentVolume()) return {};
    auto options = surfaceOptions(uuids);
    options.kind = "surface";
    options.volume = _state->currentVolumeId();
    std::vector<vc::objects::VisibleRun> visible;
    // No attachments: only drafts are read, all in memory.
    for (auto& region : session->geometryRegion({}, vc3d::store::regionSlice(slab), options))
        for (auto& run : region.result.runs) visible.push_back(std::move(run));
    return runsOf(std::move(visible));
}

bool StoreSurfaceOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!viewer || !_state->currentVolume() || !_state->objectStoreSession()) return false;
    if (viewer->asQObject()->property("vc_viewer_role").toString() == QStringLiteral("annotation")) return false;
    if (dynamic_cast<PlaneSurface*>(viewer->currentSurface())) return viewer->surfName() != "segmentation";
    return dynamic_cast<QuadSurface*>(viewer->currentSurface()) && viewer->planeIntersectionLinesVisible();
}

bool StoreSurfaceOverlayController::needsOverlayRebuild(VolumeViewerBase* viewer) const
{
    const auto found = _views.find(viewer);
    if (found == _views.end()) return true;
    const auto& v = *found->second;
    if (v.dirty || v.displayedRect != visibleSceneRect(viewer)) return true;
    // Drafts change while they are edited, without a store event: cut them on every redraw.
    const auto session = _state->objectStoreSession();
    if (dynamic_cast<PlaneSurface*>(viewer->currentSurface())) {
        const auto uuids = targets(viewer);
        return v.displayed != planeViewRegion(viewer, kHalfDepthVx) ||
               std::any_of(uuids.begin(), uuids.end(), [&](const auto& uuid) { return session->hasDraft(uuid); });
    }
    if (auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
        surface && vc::objects::util::isUuid(surface->id) && session->hasDraft(surface->id))
        return true;
    const auto cuts = planeCuts(viewer);
    return cuts.size() != v.cuts.size() ||
           !std::equal(cuts.begin(), cuts.end(), v.cuts.begin(),
                       [](const PlaneCut& a, const PlaneCut& b) { return a.plane == b.plane && a.slab == b.slab; });
}

void StoreSurfaceOverlayController::request(VolumeViewerBase* viewer, const vc3d::store::Region& view,
                                            const std::vector<std::string>& uuids)
{
    auto& v = *_views.at(viewer);
    vc3d::store::requestRegion<Listing>(this, v.region, view,
        [this, viewer]() -> View* {
            const auto found = _views.find(viewer);
            return found == _views.end() ? nullptr : found->second.get();
        },
        [session = _state->objectStoreSession(), sources = _state->surfaceSources(), volume = _state->currentVolumeId(),
         uuids](const vc3d::store::Region& query, const std::atomic_bool& cancelled) {
            auto crossings = SurfaceCatalog::region(*session, sources, volume,
                                                    vc3d::store::regionSlice(query), surfaceOptions(uuids),
                                                    [&] { return cancelled.load(); }, true);
            return Listing{runsOf(std::move(crossings.runs)), crossings.complete};
        },
        [this, viewer](View& v, Listing* listing) {
            v.dirty = true;
            if (listing) v.runs = std::move(listing->runs);
            // Commits made while it read: show this read, then read once more.
            if (std::exchange(v.refreshPending, false)) v.region.refresh();
            refreshViewer(viewer);
        });
}

void StoreSurfaceOverlayController::requestCuts(VolumeViewerBase* viewer, const std::string& uuid,
                                                const std::vector<PlaneCut>& cuts)
{
    auto& v = *_views.at(viewer);
    if (v.cutsBusy) {
        v.cutsStale = true;
        return;
    }
    v.cutsBusy = true;
    v.cutsStale = false;
    using Cut = std::map<std::string, std::vector<Run>>;
    vc3d::store::background<Cut>(this,
        [session = _state->objectStoreSession(), sources = _state->surfaceSources(), volume = _state->currentVolumeId(),
         uuid, cuts] {
            Cut result;
            for (const auto& cut : cuts) {
                auto crossings = SurfaceCatalog::region(*session, sources, volume,
                                                        vc3d::store::regionSlice(cut.slab), surfaceOptions({uuid}),
                                                        {}, true);
                result[cut.plane] = runsOf(std::move(crossings.runs));
            }
            return result;
        },
        [this, viewer](vc3d::store::Outcome<Cut> outcome) {
            const auto found = _views.find(viewer);
            if (found == _views.end()) return;
            auto& v = *found->second;
            v.cutsBusy = false;
            if (outcome.error.isEmpty()) v.planeRuns = std::move(outcome.value);
            v.dirty = true;
            refreshViewer(viewer);
        });
}

void StoreSurfaceOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    auto& v = _views[viewer];
    if (!v) v = std::make_unique<View>();
    v->dirty = false;
    v->displayedRect = visibleSceneRect(viewer);
    if (dynamic_cast<PlaneSurface*>(viewer->currentSurface())) drawPlaneViewer(viewer, *v, builder);
    else drawSurfaceViewer(viewer, *v, builder);
}

void StoreSurfaceOverlayController::drawPlaneViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder)
{
    const auto current = planeViewRegion(viewer, kHalfDepthVx);
    v.displayed = current;
    if (!current) return;
    const auto session = _state->objectStoreSession();
    std::vector<std::string> drafted, stored;
    for (const auto& uuid : targets(viewer)) (session->hasDraft(uuid) ? drafted : stored).push_back(uuid);
    if (v.targets != stored) {
        v.targets = stored;
        v.region.reset();
        v.runs.clear();
    }
    if (!stored.empty()) request(viewer, *current, stored);

    const auto notice = !v.region.error.isEmpty() ? tr("Surfaces: %1").arg(v.region.error)
                        : v.region.busy ? tr("Fetching surface intersections…")
                        : v.region.loaded && !v.region.complete ? tr("Partial surface view · zoom in")
                                                                : QString{};
    if (!notice.isEmpty()) {
        OverlayStyle info;
        info.penColor = QColor(255, 210, 100);
        info.z = vc3d::kIntersectionZ;
        builder.addText(v.displayedRect.topLeft() + QPointF(8, 60), notice, QFont{}, info);
    }

    auto runs = draftRuns(drafted, *current);
    for (const auto& run : v.runs)
        if (!std::binary_search(drafted.begin(), drafted.end(), run.uuid)) runs.push_back(run);

    const std::string activeId = activeUuid();
    const auto active = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    // Dotted copies of the active surface's line, displaced along its normal:
    // where the flattened viewer's offset lands, or with compositing the slab's
    // front and behind extents (the front marked with triangles pointing into a
    // volumetric slab).
    std::vector<float> offsets;
    bool volumetric = false;
    auto* manager = managerForViewer(viewer);
    if (auto* segViewer = manager ? manager->segmentationViewer() : nullptr; segViewer && segViewer != viewer) {
        const float offset = segViewer->normalOffset();
        if (segViewer->isCompositeEnabled()) {
            const auto& cs = segViewer->compositeRenderSettings();
            const float step = cs.reverseDirection ? -1.0f : 1.0f;
            volumetric = cs.params.method == "volumetric";
            offsets = {offset + float(std::max(0, cs.layersFront)) * step,
                       offset - float(std::max(0, cs.layersBehind)) * step};
        } else if (std::abs(offset) > 1e-3f) {
            offsets = {offset};
        }
    }

    const float opacity = std::clamp(viewer->intersectionOpacity(), 0.0f, 1.0f);
    const float thickness = viewer->intersectionThickness();
    std::map<std::string, QPainterPath> paths;
    QPainterPath dotted, markers;
    constexpr qreal kMarkerSpacing = 48.0;  // scene px between triangles
    constexpr qreal kMarkerSize = 8.0;      // scene px, base-to-apex
    qreal markerAccum = kMarkerSpacing;     // a marker near the start
    for (const auto& run : runs) {
        if (run.points.size() < 2) continue;
        auto& path = paths[run.uuid];
        path.moveTo(viewer->volumeToScene(vec(run.points[0])));
        for (size_t i = 1; i < run.points.size(); ++i) path.lineTo(viewer->volumeToScene(vec(run.points[i])));
        if (run.uuid != activeId || !active || offsets.empty() || run.grid.size() != run.points.size()) continue;
        std::vector<cv::Vec3f> normals;
        normals.reserve(run.grid.size());
        for (const auto& g : run.grid) normals.push_back(active->normal(active->gridToPtr({float(g[0]), float(g[1])})));
        for (size_t oi = 0; oi < offsets.size(); ++oi)
            for (size_t i = 0; i + 1 < run.points.size(); ++i) {
                const auto &n0 = normals[i], &n1 = normals[i + 1];
                if (!finite(n0) || !finite(n1)) continue;
                const cv::Vec3f w0 = vec(run.points[i]), w1 = vec(run.points[i + 1]);
                const QPointF a = viewer->volumeToScene(w0 + n0 * offsets[oi]);
                const QPointF b = viewer->volumeToScene(w1 + n1 * offsets[oi]);
                if (!finite(a) || !finite(b)) continue;
                dotted.moveTo(a);
                dotted.lineTo(b);
                // offsets[0] is the front bound: the volumetric camera side.
                if (!volumetric || oi != 0) continue;
                markerAccum += std::hypot(b.x() - a.x(), b.y() - a.y());
                if (markerAccum < kMarkerSpacing) continue;
                const cv::Vec3f mid = (w0 + w1) * 0.5f, normal = (n0 + n1) * 0.5f;
                const QPointF from = viewer->volumeToScene(mid + normal * offsets[0]);
                const QPointF toward = viewer->volumeToScene(mid + normal * offsets[1]);
                QPointF dir = toward - from;
                const qreal length = std::hypot(dir.x(), dir.y());
                if (!finite(from) || !finite(toward) || length < 1e-6) continue;
                markerAccum = 0.0;
                dir /= length;
                const QPointF perp(-dir.y(), dir.x());
                markers.moveTo(from + perp * (kMarkerSize * 0.5));
                markers.lineTo(from - perp * (kMarkerSize * 0.5));
                markers.lineTo(from + dir * kMarkerSize);
                markers.closeSubpath();
            }
    }
    for (const auto& [uuid, path] : paths) {
        const bool isActive = uuid == activeId;
        const bool highlighted = !isActive && _highlighted.contains(uuid);
        OverlayStyle style;
        if (isActive) {
            style.penColor = vc3d::activeIntersectionColor(viewer->surfName());
            style.penColor.setAlphaF(std::clamp(opacity * vc3d::kActiveIntersectionOpacityScale, 0.0f, 1.0f));
            style.penWidth = vc3d::activeIntersectionWidth(thickness);
            style.z = vc3d::kActiveIntersectionZ;
        } else {
            style.penColor = highlighted ? vc3d::highlightedIntersectionColor() : vc3d::intersectionColor(uuid);
            style.penColor.setAlphaF(opacity);
            style.penWidth = thickness;
            style.penCap = Qt::FlatCap;
            style.z = highlighted ? vc3d::kHighlightedIntersectionZ : vc3d::kIntersectionZ;
        }
        builder.addPainterPath(path, style);
        if (!isActive) continue;
        if (!dotted.isEmpty()) {
            auto dots = style;
            dots.penStyle = Qt::DotLine;
            dots.penCap = Qt::FlatCap;
            builder.addPainterPath(dotted, dots);
        }
        if (!markers.isEmpty()) {
            auto filled = style;
            filled.brushColor = style.penColor;
            filled.penColor = Qt::transparent;
            builder.addPainterPath(markers, filled);
        }
    }
}

void StoreSurfaceOverlayController::drawSurfaceViewer(VolumeViewerBase* viewer, View& v, OverlayBuilder& builder)
{
    auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (!surface || !vc::objects::util::isUuid(surface->id)) return;
    const auto cuts = planeCuts(viewer);
    const bool cutsChanged = cuts.size() != v.cuts.size() ||
        !std::equal(cuts.begin(), cuts.end(), v.cuts.begin(),
                    [](const PlaneCut& a, const PlaneCut& b) { return a.plane == b.plane && a.slab == b.slab; });
    v.cuts = cuts;
    if (_state->objectStoreSession()->hasDraft(surface->id)) {
        v.planeRuns.clear();
        for (const auto& cut : cuts) v.planeRuns[cut.plane] = draftRuns({surface->id}, cut.slab);
    } else if (cutsChanged || v.cutsStale) {
        requestCuts(viewer, surface->id, cuts);
    }

    const float opacity = std::clamp(viewer->intersectionOpacity() * vc3d::kActiveIntersectionOpacityScale, 0.0f, 1.0f);
    const float width = std::max(viewer->intersectionThickness(), vc3d::kActiveIntersectionMinWidthDelta);
    for (const auto& [plane, runs] : v.planeRuns) {
        QPainterPath path;
        for (const auto& run : runs) {
            if (run.uuid != surface->id || run.grid.size() < 2) continue;
            bool open = false;
            for (const auto& g : run.grid) {
                const cv::Vec3f loc = surface->loc(surface->gridToPtr({float(g[0]), float(g[1])}));
                const QPointF p = viewer->surfaceCoordsToScene(loc[0], loc[1]);
                if (!finite(p)) { open = false; continue; }
                if (open) path.lineTo(p); else path.moveTo(p);
                open = true;
            }
        }
        if (path.isEmpty()) continue;
        OverlayStyle style;
        style.penColor = vc3d::activeIntersectionColor(plane);
        style.penColor.setAlphaF(opacity);
        style.penWidth = width;
        style.z = vc3d::kActiveIntersectionZ;
        builder.addPainterPath(path, style);
    }
}
