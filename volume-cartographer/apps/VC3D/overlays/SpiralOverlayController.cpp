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
    refreshAll();
}

void SpiralOverlayController::reset()
{
    _index.reset();
    _indexGeneration = 0;
    ++_requestGeneration;
    _cache.clear();
    refreshAll();
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
    if (cache.requestKey != key) schedule(viewer, key, lo, hi);
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
    const auto visible = _visible;
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
    watcher->setFuture(QtConcurrent::run([index, visible, lo, hi]() {
        std::vector<PathPrimitive> paths;
        constexpr std::size_t maximum = 12000;
        for (auto it = visible.begin(); it != visible.end() && paths.size() < maximum; ++it) {
            if (!it.value()) continue;
            const auto segments = index->query(lo, hi, it.key().toStdString(), maximum - paths.size());
            for (const auto& segment : segments) {
                PathPrimitive path;
                path.points = {segment.first, segment.second};
                path.color = categoryColor(it.key());
                path.lineWidth = it.key() == QStringLiteral("tracks") ? 1.0 : 2.0;
                path.opacity = 0.9;
                path.z = 70.0;
                paths.push_back(std::move(path));
            }
        }
        return paths;
    }));
}
