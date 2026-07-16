#include "SpiralOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>

namespace {
QColor categoryColor(const QString& category)
{
    if (category == QStringLiteral("fibers")) return QColor(80, 220, 255);
    if (category == QStringLiteral("tracks")) return QColor(255, 190, 60);
    return QColor(220, 100, 255);
}
}

SpiralOverlayController::SpiralOverlayController(QObject* parent)
    : ViewerOverlayControllerBase("spiral_geometry", parent)
{
    _visible = {{QStringLiteral("fibers"), false}, {QStringLiteral("tracks"), false},
                {QStringLiteral("pcls"), false}};
}

void SpiralOverlayController::publishIndex(std::shared_ptr<const PolylineIndex> index, quint64 generation)
{
    if (generation < _indexGeneration) return;
    _index = std::move(index);
    _indexGeneration = generation;
    _cache.clear();
    rebuildChains();
    refreshAll();
}

void SpiralOverlayController::reset()
{
    _index.reset();
    _indexGeneration = 0;
    ++_requestGeneration;
    _cache.clear();
    _chains.clear();
    refreshAll();
}

void SpiralOverlayController::rebuildChains()
{
    _chains.clear();
    if (!_index) return;
    // Fibers and pcls are decimated control-point strips, small enough to
    // keep resident for the lifetime of the generation; only the dense track
    // curves go through the viewport-limited spatial query.
    for (const auto& polyline : _index->polylines()) {
        const QString category = QString::fromStdString(polyline.category);
        if (category == QStringLiteral("tracks") || polyline.points.empty()) continue;
        ChainEntry entry;
        entry.category = category;
        entry.points = &polyline.points;
        entry.lo = entry.hi = polyline.points.front();
        for (const cv::Vec3f& point : polyline.points) {
            for (int axis = 0; axis < 3; ++axis) {
                entry.lo[axis] = std::min(entry.lo[axis], point[axis]);
                entry.hi[axis] = std::max(entry.hi[axis], point[axis]);
            }
        }
        _chains.push_back(std::move(entry));
    }
}

void SpiralOverlayController::setCategoryVisible(const QString& category, bool visible)
{
    if (_visible.value(category) == visible) return;
    _visible[category] = visible;
    _cache.clear();
    refreshAll();
}

void SpiralOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    _cache.erase(viewer);
    ViewerOverlayControllerBase::detachViewer(viewer);
}

bool SpiralOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && _index && !_index->empty()
        && std::any_of(_visible.begin(), _visible.end(), [](bool value) { return value; });
}

void SpiralOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) return;
    const QRectF rect = visibleSceneRect(viewer);
    const bool surfacePane = dynamic_cast<QuadSurface*>(viewer->currentSurface()) != nullptr;
    const int samplesPerAxis = surfacePane ? 5 : 2;
    cv::Vec3f lo, hi;
    bool haveBounds = false;
    for (int row = 0; row < samplesPerAxis; ++row) {
        const qreal y = rect.top() + rect.height() * row / (samplesPerAxis - 1);
        for (int col = 0; col < samplesPerAxis; ++col) {
            const qreal x = rect.left() + rect.width() * col / (samplesPerAxis - 1);
            const cv::Vec3f point = sceneToVolume(viewer, QPointF(x, y));
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
                continue;
            if (!haveBounds) {
                lo = hi = point;
                haveBounds = true;
            } else {
                for (int axis = 0; axis < 3; ++axis) {
                    lo[axis] = std::min(lo[axis], point[axis]);
                    hi[axis] = std::max(hi[axis], point[axis]);
                }
            }
        }
    }
    if (!haveBounds) return;
    const float slab = surfacePane ? 10.0f : 4.0f;
    lo -= cv::Vec3f(slab, slab, slab); hi += cv::Vec3f(slab, slab, slab);
    const QString key = QStringLiteral("%1:%2:%3:%4:%5:%6:%7:%8")
        .arg(_indexGeneration).arg(lo[0], 0, 'f', 1).arg(lo[1], 0, 'f', 1).arg(lo[2], 0, 'f', 1)
        .arg(hi[0], 0, 'f', 1).arg(hi[1], 0, 'f', 1).arg(hi[2], 0, 'f', 1)
        .arg(QString::fromStdString(viewer->surfName()));
    auto& cache = _cache[viewer];
    if (_visible.value(QStringLiteral("tracks")) && cache.requestKey != key)
        schedule(viewer, key, lo, hi);
    // Fibers and pcls are ordered control-point chains: draw them like point
    // collections (dots + polylines joining only consecutive points), fading
    // out with distance to the pane's plane or surface. They render
    // synchronously from the resident chain list so panning never waits on
    // the async viewport query; the viewport box culls whole chains first and
    // individual points after.
    const VolumeBounds bounds{lo, hi};
    for (const auto& entry : _chains) {
        if (!_visible.value(entry.category)) continue;
        if (entry.hi[0] < lo[0] || entry.lo[0] > hi[0]
            || entry.hi[1] < lo[1] || entry.lo[1] > hi[1]
            || entry.hi[2] < lo[2] || entry.lo[2] > hi[2]) continue;
        PointChainStyle chainStyle;
        chainStyle.color = categoryColor(entry.category);
        chainStyle.lineOpacity = 0.75f;
        chainStyle.distanceTolerance = slab;
        renderPointChain(viewer, builder, *entry.points, chainStyle, bounds);
    }
    for (const auto& path : cache.paths) {
        if (!surfacePane) {
            builder.addPath(path);
            continue;
        }
        std::vector<QPointF> points;
        points.reserve(path.points.size());
        for (const cv::Vec3f& point : path.points) {
            const QPointF scenePoint = viewer->volumeToScene(point);
            if (!std::isfinite(scenePoint.x()) || !std::isfinite(scenePoint.y())) {
                points.clear();
                break;
            }
            points.push_back(scenePoint);
        }
        if (points.size() < 2) continue;
        OverlayStyle style;
        style.penColor = path.color;
        style.penColor.setAlphaF(std::clamp(path.opacity, 0.0, 1.0));
        style.penWidth = path.lineWidth;
        style.z = path.z;
        builder.addLineStrip(points, path.closed, style);
    }
}

void SpiralOverlayController::schedule(VolumeViewerBase* viewer, const QString& key,
                                       const cv::Vec3f& lo, const cv::Vec3f& hi)
{
    auto index = _index;
    const quint64 request = ++_requestGeneration;
    auto& cache = _cache[viewer];
    cache.requestKey = key;
    cache.requestGeneration = request;
    auto* watcher = new QFutureWatcher<std::vector<PathPrimitive>>(this);
    connect(watcher, &QFutureWatcher<std::vector<PathPrimitive>>::finished, this,
            [this, watcher, viewer, request]() {
        auto found = _cache.find(viewer);
        if (found != _cache.end() && found->second.requestGeneration == request) {
            found->second.paths = watcher->result();
            refreshViewer(viewer);
        }
        watcher->deleteLater();
    });
    // Only the dense track curves need the spatial query; fibers/pcls render
    // synchronously from the resident chain list.
    watcher->setFuture(QtConcurrent::run([index, lo, hi]() {
        std::vector<PathPrimitive> paths;
        constexpr std::size_t maximum = 12000;
        const auto segments = index->query(lo, hi, "tracks", maximum);
        for (const auto& segment : segments) {
            PathPrimitive path;
            path.points = {segment.first, segment.second};
            path.color = categoryColor(QStringLiteral("tracks"));
            path.lineWidth = 1.0;
            path.opacity = 0.9;
            path.z = 70.0;
            paths.push_back(std::move(path));
        }
        return paths;
    }));
}
