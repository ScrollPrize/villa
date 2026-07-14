#include "SpiralOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"

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
    // First-version Surface-pane projection is deliberately deferred; indexed
    // input geometry is rendered only in the axis plane panes.
    if (!dynamic_cast<PlaneSurface*>(viewer->currentSurface())) return;
    const QRectF rect = visibleSceneRect(viewer);
    const cv::Vec3f corners[] = {
        sceneToVolume(viewer, rect.topLeft()), sceneToVolume(viewer, rect.topRight()),
        sceneToVolume(viewer, rect.bottomLeft()), sceneToVolume(viewer, rect.bottomRight())};
    cv::Vec3f lo = corners[0], hi = corners[0];
    for (const auto& point : corners) for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::min(lo[axis], point[axis]); hi[axis] = std::max(hi[axis], point[axis]);
    }
    constexpr float slab = 4.0f;
    lo -= cv::Vec3f(slab, slab, slab); hi += cv::Vec3f(slab, slab, slab);
    const QString key = QStringLiteral("%1:%2:%3:%4:%5:%6:%7:%8")
        .arg(_indexGeneration).arg(lo[0], 0, 'f', 1).arg(lo[1], 0, 'f', 1).arg(lo[2], 0, 'f', 1)
        .arg(hi[0], 0, 'f', 1).arg(hi[1], 0, 'f', 1).arg(hi[2], 0, 'f', 1)
        .arg(QString::fromStdString(viewer->surfName()));
    auto& cache = _cache[viewer];
    if (cache.requestKey != key) schedule(viewer, key, lo, hi);
    for (const auto& path : cache.paths) builder.addPath(path);
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
