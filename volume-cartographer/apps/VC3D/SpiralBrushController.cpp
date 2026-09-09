#include "SpiralBrushController.hpp"

#include "SpiralBrushCursorWidget.hpp"
#include "SurfaceOverlayColors.hpp"
#include "VCSettings.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
// cv::boundingRect moved from imgproc into the geometry module in OpenCV 5;
// OpenCvCompat pulls that header in on 5 and is a no-op on 4.
#include "vc/core/util/OpenCvCompat.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QDateTime>
#include <QEvent>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLineF>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPathStroker>
#include <QRandomGenerator>
#include <QSettings>
#include <QWheelEvent>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace {
constexpr int kMinimumDiameter = 4;
constexpr int kMaximumDiameter = 256;
constexpr qreal kPaintOpacity = 0.45;
constexpr float kFreehandPolylineSpacingVoxels = 10.0f;
constexpr float kAnchoredPolylineSpacingVoxels = 30.0f;
constexpr float kPointCollectionSpacingVoxels = 10.0f;
constexpr qreal kPolylineWidth = 3.0;
constexpr qreal kControlPointRadius = 3.5;
constexpr qreal kEditablePclHitRadius = 8.0;
constexpr float kPolylineProjectionToleranceVoxels = 100.0f;

bool validPoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f && std::isfinite(point[0])
        && std::isfinite(point[1]) && std::isfinite(point[2]);
}

QColor collectionColor(const QJsonObject& collection)
{
    const QJsonArray color = collection.value(QStringLiteral("color")).toArray();
    if (color.size() != 3) return QColor(50, 255, 215);
    QColor result;
    result.setRgbF(std::clamp(color[0].toDouble(), 0.0, 1.0),
                   std::clamp(color[1].toDouble(), 0.0, 1.0),
                   std::clamp(color[2].toDouble(), 0.0, 1.0));
    return result;
}

std::optional<std::vector<cv::Vec2f>> exactPointCollectionSurfacePositions(
    const vc3d::spiral::EditablePclDraft* draft,
    const std::shared_ptr<QuadSurface>& source, const Surface* current)
{
    if (!draft || !source || source.get() != current) return std::nullopt;
    std::vector<cv::Vec2f> result;
    result.reserve(draft->points.size());
    for (const auto& point : draft->points) {
        if (!point.previewSurfacePosition
            || !std::isfinite(point.previewSurfacePosition->x())
            || !std::isfinite(point.previewSurfacePosition->y()))
            return std::nullopt;
        result.emplace_back(
            static_cast<float>(point.previewSurfacePosition->x()),
            static_cast<float>(point.previewSurfacePosition->y()));
    }
    return result;
}

QTransform surfaceToSceneTransform(const VolumeViewerBase* viewer)
{
    if (!viewer) return {};
    const QPointF origin = viewer->surfaceCoordsToScene(0.0f, 0.0f);
    const QPointF xStep = viewer->surfaceCoordsToScene(1.0f, 0.0f) - origin;
    const QPointF yStep = viewer->surfaceCoordsToScene(0.0f, 1.0f) - origin;
    return {xStep.x(), xStep.y(), yStep.x(), yStep.y(),
            origin.x(), origin.y()};
}
}

SpiralBrushController::SpiralBrushController(QObject* parent)
    : ViewerOverlayControllerBase("spiral_brush", parent)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _diameterPx = std::clamp(
        settings.value(QStringLiteral("spiral/brush_diameter_px"), 32).toInt(),
        kMinimumDiameter, kMaximumDiameter);
}

void SpiralBrushController::setPaintSurface(const std::shared_ptr<QuadSurface>& surface)
{
    if (_paintSurface == surface) return;
    clearEditablePclHover();
    if (_pointPlacement.surfaceChanged(_activePolyline >= 0)
        == SpiralPointPlacementMode::Transition::ClearInteractionPreserveDraft) {
        finishPointCollection(false);
        updateCursorWidget();
    }
    _paintSurface = surface;
    invalidateEditablePclHitIndex();
    clearPointChainProjectionCache();
    refreshAll();
}

void SpiralBrushController::bindFlattenedViewer(VolumeViewerBase* viewer)
{
    if (_viewer && _viewer != viewer)
        _viewer->setLocalCursorCrosshairSuppressed(false);
    if (_viewport) _viewport->removeEventFilter(this);
    if (_viewObject) _viewObject->removeEventFilter(this);
    _viewer = viewer;
    invalidateEditablePclHitIndex();
    auto* view = viewer ? viewer->graphicsView() : nullptr;
    _viewObject = view;
    _viewport = view ? view->viewport() : nullptr;
    if (_viewport) {
        _viewport->installEventFilter(this);
        if (auto* widget = qobject_cast<QWidget*>(_viewport)) widget->setMouseTracking(true);
    }
    if (_cursorWidget) _cursorWidget->deleteLater();
    _cursorWidget = nullptr;
    if (auto* viewportWidget = qobject_cast<QWidget*>(_viewport)) {
        _cursorWidget = new SpiralBrushCursorWidget(viewportWidget);
        _cursorWidget->setGeometry(viewportWidget->rect());
        _cursorWidget->show();
        _cursorWidget->raise();
    }
    if (_viewObject) _viewObject->installEventFilter(this);
    if (view) view->setRenderHint(QPainter::Antialiasing, true);
    updateCursorWidget();
}

void SpiralBrushController::resetSession()
{
    _gestures.clear();
    _polylines.clear();
    _sameWindingSources.clear();
    _sameWindingSourceIndexById.clear();
    _editableSameWindingCollectionIds.clear();
    _suppressedSameWindingCollectionIds.clear();
    _visiblePointCollectionIds.clear();
    clearPointChainProjectionCache();
    _usedColors.clear();
    _sampledColor.reset();
    _pointPlacement.deactivate();
    _cursorInside = false;
    updateCursorWidget();
    _dragMode = DragMode::None;
    _activeGesture = -1;
    _activePolyline = -1;
    _nextPolylineSequence = 1;
    _polylineBlocked = false;
    _vHeld = false;
    _vClickConsumed = false;
    _hoveredEditablePcl.reset();
    invalidateEditablePclHitIndex();
    _pclLeftClickConsumed = false;
    emit suppressedSameWindingCollectionIdsChanged({});
    refreshAll();
    emit paintStateChanged();
}

bool SpiralBrushController::hasUnfinalizedPaint() const
{
    return std::any_of(_gestures.begin(), _gestures.end(), [](const Gesture& gesture) {
        return gesture.state == GestureState::Painted && !gesture.shape.isEmpty();
    });
}

bool SpiralBrushController::hasUnfinalizedPolylines() const
{
    return std::any_of(_polylines.begin(), _polylines.end(), [this](const PolylineGesture& line) {
        return line.state == GestureState::Painted
            && (line.kind != PolylineGesture::Kind::PointCollection
                ? line.volumePoints.size() >= 2
                : pointCollectionHasChanges(line));
    });
}

bool SpiralBrushController::hasReadyDrafts() const
{
    return std::any_of(_gestures.begin(), _gestures.end(), [](const Gesture& gesture) {
        return gesture.state == GestureState::Ready && !gesture.shape.isEmpty();
    }) || std::any_of(_polylines.begin(), _polylines.end(), [this](const PolylineGesture& line) {
        return line.state == GestureState::Ready
            && (line.kind != PolylineGesture::Kind::PointCollection
                ? line.volumePoints.size() >= 2
                : pointCollectionHasChanges(line));
    });
}

void SpiralBrushController::markDraftsReady()
{
    deactivatePointPlacement();
    if (_dragMode != DragMode::None) return;
    for (auto& gesture : _gestures) {
        if (gesture.state == GestureState::Painted && !gesture.shape.isEmpty())
            gesture.state = GestureState::Ready;
    }
    for (auto& line : _polylines) {
        const bool usable = line.kind != PolylineGesture::Kind::PointCollection
            ? line.volumePoints.size() >= 2 : pointCollectionHasChanges(line);
        if (line.state == GestureState::Painted && usable)
            line.state = GestureState::Ready;
    }
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::setVisiblePointCollectionIds(const QSet<QString>& ids)
{
    if (_visiblePointCollectionIds == ids) return;
    _visiblePointCollectionIds = ids;
    refreshAll();
}

void SpiralBrushController::setSameWindingSource(
    const QJsonDocument& document, double sourceToPreviewScale,
    const QString& sourceRevision, bool editable)
{
    clearEditablePclHover();
    _sameWindingSources = vc3d::spiral::importEditablePcls(
        document, sourceToPreviewScale, sourceRevision, editable);
    _sameWindingSourceIndexById.clear();
    _editableSameWindingCollectionIds.clear();
    for (std::size_t index = 0; index < _sameWindingSources.size(); ++index) {
        const auto& source = _sameWindingSources[index];
        bool ok = false;
        const qulonglong collectionId = source.collectionId.toULongLong(&ok);
        if (!ok) continue;
        _sameWindingSourceIndexById[collectionId] = index;
        if (source.editable)
            _editableSameWindingCollectionIds.insert(collectionId);
    }
    invalidateEditablePclHitIndex();
    refreshAll();
}

void SpiralBrushController::setSameWindingSourceVisible(bool visible)
{
    if (_sameWindingSourceVisible == visible) return;
    _sameWindingSourceVisible = visible;
    if (!visible) clearEditablePclHover();
    refreshAll();
}

void SpiralBrushController::setSameWindingHitOverlay(
    PointsOverlayController* overlay)
{
    _sameWindingHitOverlay = overlay;
    clearEditablePclHover();
}

void SpiralBrushController::setPointViewTolerance(double tolerance)
{
    const float clamped = static_cast<float>(
        std::clamp(tolerance, 0.0, 10000.0));
    if (std::abs(_pointViewToleranceVoxels - clamped) < 0.001f) return;
    _pointViewToleranceVoxels = clamped;
    invalidateEditablePclHitIndex();
    refreshAll();
}

void SpiralBrushController::replacementConflict(const QString& id, bool discardDraft)
{
    for (auto line = _polylines.begin(); line != _polylines.end(); ++line) {
        if (line->id != id || !line->pclEdit) continue;
        const QString target = line->pclEdit->collectionId;
        if (discardDraft) {
            const int index = static_cast<int>(std::distance(_polylines.begin(), line));
            if (_activePolyline == index) _activePolyline = -1;
            _polylines.erase(line);
        } else {
            line->id.clear();
            line->state = GestureState::Ready;
            line->pclEdit->submissionBlocked = true;
        }
        updateSuppressedSameWindingIds();
        invalidateEditablePclHitIndex();
        refreshAll();
        emit paintStateChanged();
        return;
    }
}

const std::vector<cv::Vec3f>& SpiralBrushController::pointCollectionPositions(
    const PolylineGesture& line) const
{
    if (!line.pclEdit) return line.volumePoints;
    return line.pclEdit->projectionPositions();
}

bool SpiralBrushController::pointCollectionHasChanges(const PolylineGesture& line) const
{
    if (!line.pclEdit) return false;
    if (!line.pclEdit->collectionId.isEmpty())
        return line.pclEdit->dirty
            && (line.pclEdit->deleted || line.pclEdit->points.size() >= 2);
    return line.pclEdit->points.size() >= 2;
}

void SpiralBrushController::updateSuppressedSameWindingIds()
{
    QSet<QString> suppressed;
    for (const auto& line : _polylines) {
        if (line.pclEdit && !line.pclEdit->collectionId.isEmpty()
            && line.pclEdit->dirty)
            suppressed.insert(line.pclEdit->collectionId);
    }
    if (suppressed == _suppressedSameWindingCollectionIds) return;
    _suppressedSameWindingCollectionIds = std::move(suppressed);
    emit suppressedSameWindingCollectionIdsChanged(
        _suppressedSameWindingCollectionIds);
}

QColor SpiralBrushController::nextColor()
{
    if (_sampledColor) {
        const QColor color = *_sampledColor;
        _sampledColor.reset();
        return color;
    }
    std::array<int, 12> choices{};
    int count = 0;
    for (int index = 0; index < static_cast<int>(choices.size()); ++index) {
        if (!_usedColors.contains(vc3d::surfaceOverlayColor(index).rgb())) choices[count++] = index;
    }
    QColor color;
    if (count > 0) {
        color = vc3d::surfaceOverlayColor(
            choices[QRandomGenerator::global()->bounded(count)]);
    } else {
        do {
            color = QColor::fromHsv(QRandomGenerator::global()->bounded(360),
                                    150 + QRandomGenerator::global()->bounded(90),
                                    210 + QRandomGenerator::global()->bounded(46));
        } while (_usedColors.contains(color.rgb()));
    }
    _usedColors.insert(color.rgb());
    return color;
}

QPainterPath SpiralBrushController::deviceDisk(const QPointF& center) const
{
    QPainterPath path;
    const qreal radius = _diameterPx * 0.5;
    path.addEllipse(center, radius, radius);
    return path;
}

QPainterPath SpiralBrushController::deviceSweep(const QPointF& from, const QPointF& to) const
{
    if (QLineF(from, to).length() < 0.01) return deviceDisk(to);
    QPainterPath centerline(from);
    centerline.lineTo(to);
    QPainterPathStroker stroker;
    stroker.setWidth(_diameterPx);
    stroker.setCapStyle(Qt::RoundCap);
    stroker.setJoinStyle(Qt::RoundJoin);
    return stroker.createStroke(centerline);
}

QPainterPath SpiralBrushController::deviceToSurface(const QPainterPath& path) const
{
    auto* view = _viewer ? _viewer->graphicsView() : nullptr;
    if (!view || !_viewer) return {};
    bool viewportTransformValid = false;
    const QTransform viewportToScene =
        view->viewportTransform().inverted(&viewportTransformValid);
    bool surfaceTransformValid = false;
    const QTransform sceneToSurface =
        surfaceToSceneTransform(_viewer).inverted(&surfaceTransformValid);
    if (!viewportTransformValid || !surfaceTransformValid) return {};
    return sceneToSurface.map(viewportToScene.map(path));
}

std::optional<QPointF> SpiralBrushController::scenePointToSurface(
    const QPointF& point) const
{
    if (!_viewer) return std::nullopt;
    bool valid = false;
    const QTransform transform = surfaceToSceneTransform(_viewer).inverted(&valid);
    if (!valid) return std::nullopt;
    const QPointF surface = transform.map(point);
    if (!std::isfinite(surface.x()) || !std::isfinite(surface.y()))
        return std::nullopt;
    return surface;
}

std::optional<QPointF> SpiralBrushController::devicePointToSurface(
    const QPointF& point) const
{
    auto* view = _viewer ? _viewer->graphicsView() : nullptr;
    if (!view) return std::nullopt;
    bool valid = false;
    const QTransform viewportToScene = view->viewportTransform().inverted(&valid);
    if (!valid) return std::nullopt;
    return scenePointToSurface(viewportToScene.map(point));
}

QPainterPath SpiralBrushController::surfaceToScene(const QPainterPath& path) const
{
    if (!_viewer) return {};
    return surfaceToSceneTransform(_viewer).map(path);
}

void SpiralBrushController::beginPaint(const QPointF& devicePos)
{
    if (!_viewer || _viewer->surfName() != "segmentation") return;
    auto* sourceRaw = dynamic_cast<QuadSurface*>(_viewer->currentSurface());
    if (!sourceRaw || !_paintSurface || _paintSurface.get() != sourceRaw) return;
    std::shared_ptr<QuadSurface> source = _paintSurface;
    const cv::Vec2d gridOrigin = sourceRaw->gridToSurface({0.0, 0.0});
    const cv::Vec2d gridColumn = sourceRaw->gridToSurface({1.0, 0.0});
    const cv::Vec2d gridRow = sourceRaw->gridToSurface({0.0, 1.0});
    if (!std::isfinite(gridOrigin[0]) || !std::isfinite(gridOrigin[1])) return;
    Gesture gesture;
    gesture.color = nextColor();
    gesture.source = std::move(source);
    gesture.gridOrigin = QPointF(gridOrigin[0], gridOrigin[1]);
    gesture.columnStep =
        QPointF(gridColumn[0] - gridOrigin[0], gridColumn[1] - gridOrigin[1]);
    gesture.rowStep =
        QPointF(gridRow[0] - gridOrigin[0], gridRow[1] - gridOrigin[1]);
    gesture.shape = deviceToSurface(deviceDisk(devicePos));
    _gestures.push_back(std::move(gesture));
    _activeGesture = static_cast<int>(_gestures.size()) - 1;
    _lastDevicePos = devicePos;
    _dragMode = DragMode::Paint;
    refreshViewer(_viewer);
    emit paintStateChanged();
}

std::optional<std::pair<QPointF, cv::Vec3f>>
SpiralBrushController::pointOnSurface(
    const QPointF& devicePos, const std::shared_ptr<QuadSurface>& source) const
{
    if (!source || source.get() != (_viewer ? _viewer->currentSurface() : nullptr))
        return std::nullopt;
    const auto surfacePoint = devicePointToSurface(devicePos);
    if (!surfacePoint) return std::nullopt;
    const auto volume = volumePointOnSurface(*surfacePoint, source);
    if (!volume) return std::nullopt;
    return std::make_pair(*surfacePoint, *volume);
}

std::optional<cv::Vec3f> SpiralBrushController::volumePointOnSurface(
    const QPointF& surfacePos, const std::shared_ptr<QuadSurface>& source) const
{
    if (!source) return std::nullopt;
    const auto sample = source->sampleAtSurface(
        {static_cast<double>(surfacePos.x()), static_cast<double>(surfacePos.y())});
    return sample ? std::optional<cv::Vec3f>{sample.volume} : std::nullopt;
}

bool SpiralBrushController::surfaceSegmentValid(
    const QPointF& from, const QPointF& to,
    const std::shared_ptr<QuadSurface>& source) const
{
    if (!source || !volumePointOnSurface(from, source)
        || !volumePointOnSurface(to, source))
        return false;
    const cv::Vec2d a = source->surfaceToGrid({from.x(), from.y()});
    const cv::Vec2d b = source->surfaceToGrid({to.x(), to.y()});
    std::vector<double> crossings{0.0, 1.0};
    auto appendCrossings = [&](double start, double end) {
        if (start == end) return;
        const double low = std::min(start, end);
        const double high = std::max(start, end);
        for (double boundary = std::floor(low) + 1.0; boundary < high; boundary += 1.0)
            crossings.push_back((boundary - start) / (end - start));
    };
    appendCrossings(a[0], b[0]);
    appendCrossings(a[1], b[1]);
    std::sort(crossings.begin(), crossings.end());
    crossings.erase(std::unique(crossings.begin(), crossings.end(),
                                [](double l, double r) {
                                    return std::abs(l - r) < 1e-12;
                                }),
                    crossings.end());
    for (std::size_t index = 1; index < crossings.size(); ++index) {
        const double t = (crossings[index - 1] + crossings[index]) * 0.5;
        const QPointF midpoint = from * (1.0 - t) + to * t;
        if (!volumePointOnSurface(midpoint, source)) return false;
    }
    return true;
}

bool SpiralBrushController::appendPolylinePoint(const QPointF& devicePos)
{
    if (_activePolyline < 0 || _activePolyline >= static_cast<int>(_polylines.size()))
        return false;
    auto& line = _polylines[static_cast<std::size_t>(_activePolyline)];
    const auto sample = pointOnSurface(devicePos, line.source);
    if (!sample) return false;
    if (!line.surfacePoints.empty()
        && !surfaceSegmentValid(line.surfacePoints.back(), sample->first, line.source))
        return false;
    if (!line.volumePoints.empty()) {
        const cv::Vec3f delta = sample->second - line.volumePoints.back();
        if (delta.dot(delta) < 1e-4f) return true;
    }
    line.surfacePoints.push_back(sample->first);
    line.volumePoints.push_back(sample->second);
    clearPointChainProjectionCache();
    return true;
}

void SpiralBrushController::beginPolyline(const QPointF& devicePos)
{
    if (!_viewer || _viewer->surfName() != "segmentation") return;
    auto* sourceRaw = dynamic_cast<QuadSurface*>(_viewer->currentSurface());
    if (!sourceRaw || !_paintSurface || _paintSurface.get() != sourceRaw) return;
    PolylineGesture line;
    line.kind = PolylineGesture::Kind::Freehand;
    line.color = nextColor();
    line.source = _paintSurface;
    line.creationTime = QDateTime::currentMSecsSinceEpoch();
    line.sequence = _nextPolylineSequence++;
    _polylines.push_back(std::move(line));
    _activePolyline = static_cast<int>(_polylines.size()) - 1;
    _polylineBlocked = false;
    if (!appendPolylinePoint(devicePos)) {
        _polylines.pop_back();
        _activePolyline = -1;
        return;
    }
    _lastDevicePos = devicePos;
    _dragMode = DragMode::Polyline;
    refreshViewer(_viewer);
    emit paintStateChanged();
}

void SpiralBrushController::resamplePolyline(PolylineGesture& line)
{
    if (line.volumePoints.size() < 2 || line.surfacePoints.size() != line.volumePoints.size()) {
        line.volumePoints.clear();
        line.surfacePoints.clear();
        return;
    }
    std::vector<float> cumulative(line.volumePoints.size(), 0.0f);
    for (std::size_t index = 1; index < line.volumePoints.size(); ++index) {
        const cv::Vec3f delta = line.volumePoints[index] - line.volumePoints[index - 1];
        cumulative[index] = cumulative[index - 1] + std::sqrt(delta.dot(delta));
    }
    const float total = cumulative.back();
    if (total < 1e-3f) {
        line.volumePoints.clear();
        line.surfacePoints.clear();
        return;
    }

    std::vector<float> targets{0.0f};
    for (float distance = kFreehandPolylineSpacingVoxels; distance < total;
         distance += kFreehandPolylineSpacingVoxels)
        targets.push_back(distance);
    if (total - targets.back() > 1e-3f) targets.push_back(total);

    std::vector<cv::Vec3f> volumePoints;
    std::vector<QPointF> surfacePoints;
    volumePoints.reserve(targets.size());
    surfacePoints.reserve(targets.size());
    std::size_t segment = 1;
    for (float target : targets) {
        while (segment + 1 < cumulative.size() && cumulative[segment] < target) ++segment;
        const float startDistance = cumulative[segment - 1];
        const float segmentLength = cumulative[segment] - startDistance;
        const float fraction = segmentLength > 1e-6f
            ? std::clamp((target - startDistance) / segmentLength, 0.0f, 1.0f) : 0.0f;
        volumePoints.push_back(line.volumePoints[segment - 1] * (1.0f - fraction)
                               + line.volumePoints[segment] * fraction);
        surfacePoints.push_back(line.surfacePoints[segment - 1] * (1.0 - fraction)
                                + line.surfacePoints[segment] * fraction);
    }
    line.volumePoints = std::move(volumePoints);
    line.surfacePoints = std::move(surfacePoints);
    line.anchors.clear();
    line.anchors.reserve(line.volumePoints.size());
    for (std::size_t index = 0; index < line.volumePoints.size(); ++index) {
        line.anchors.push_back({line.surfacePoints[index], line.volumePoints[index]});
    }
    clearPointChainProjectionCache();
}

bool SpiralBrushController::rebuildAnchoredPolyline(PolylineGesture& line)
{
    const auto result = vc3d::spiral::buildPointChain(
        line.anchors,
        [this, source = line.source](const QPointF& surface) {
            return volumePointOnSurface(surface, source);
        },
        kAnchoredPolylineSpacingVoxels,
        [this, source = line.source](const QPointF& from, const QPointF& to) {
            return surfaceSegmentValid(from, to, source);
        });
    if (result.error != vc3d::spiral::PointChainBuildError::None) return false;
    line.surfacePoints.clear();
    line.volumePoints.clear();
    line.surfacePoints.reserve(result.samples.size());
    line.volumePoints.reserve(result.samples.size());
    for (const auto& sample : result.samples) {
        line.surfacePoints.push_back(sample.surface);
        line.volumePoints.push_back(sample.volume);
    }
    clearPointChainProjectionCache();
    return true;
}

void SpiralBrushController::appendAnchoredPoint(const QPointF& devicePos)
{
    if (!_viewer || _viewer->surfName() != "segmentation"
        || _dragMode != DragMode::None)
        return;
    auto* sourceRaw = dynamic_cast<QuadSurface*>(_viewer->currentSurface());
    if (!sourceRaw || !_paintSurface || _paintSurface.get() != sourceRaw) return;

    if (_activePolyline < 0) {
        PolylineGesture line;
        line.kind = PolylineGesture::Kind::Anchored;
        line.color = nextColor();
        line.source = _paintSurface;
        line.creationTime = QDateTime::currentMSecsSinceEpoch();
        line.sequence = _nextPolylineSequence++;
        _polylines.push_back(std::move(line));
        _activePolyline = static_cast<int>(_polylines.size()) - 1;
    }
    if (_activePolyline >= static_cast<int>(_polylines.size())
        || _polylines[static_cast<std::size_t>(_activePolyline)].kind
            != PolylineGesture::Kind::Anchored)
        return;
    auto& line = _polylines[static_cast<std::size_t>(_activePolyline)];
    const auto sample = pointOnSurface(devicePos, line.source);
    if (!sample) {
        emit pointPlacementRejected(tr("Point must lie on valid Spiral surface data"));
        return;
    }

    line.anchors.push_back({sample->first, sample->second});
    const auto result = vc3d::spiral::buildPointChain(
        line.anchors,
        [this, source = line.source](const QPointF& surface) {
            return volumePointOnSurface(surface, source);
        },
        kAnchoredPolylineSpacingVoxels,
        [this, source = line.source](const QPointF& from, const QPointF& to) {
            return surfaceSegmentValid(from, to, source);
        });
    if (result.error != vc3d::spiral::PointChainBuildError::None) {
        line.anchors.pop_back();
        const QString reason =
            result.error == vc3d::spiral::PointChainBuildError::SelfIntersection
            ? tr("Point rejected: the ordered curve would intersect itself")
            : result.error == vc3d::spiral::PointChainBuildError::DegenerateSpan
            ? tr("Point rejected: it does not advance along the curve")
            : tr("Point rejected: the curve would leave valid Spiral surface data");
        emit pointPlacementRejected(reason);
        return;
    }
    line.surfacePoints.clear();
    line.volumePoints.clear();
    line.surfacePoints.reserve(result.samples.size());
    line.volumePoints.reserve(result.samples.size());
    for (const auto& point : result.samples) {
        line.surfacePoints.push_back(point.surface);
        line.volumePoints.push_back(point.volume);
    }
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
    emit paintStateChanged();
}

void SpiralBrushController::finishAnchoredPolyline()
{
    if (_activePolyline >= 0 && _activePolyline < static_cast<int>(_polylines.size())) {
        const auto index = static_cast<std::size_t>(_activePolyline);
        if (_polylines[index].kind == PolylineGesture::Kind::Anchored) {
            if (_polylines[index].anchors.size() < 2)
                _polylines.erase(_polylines.begin() + _activePolyline);
            _activePolyline = -1;
        }
    }
    _vClickConsumed = false;
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
    emit paintStateChanged();
}

void SpiralBrushController::appendPointCollectionPoint(const QPointF& devicePos)
{
    if (!_viewer || _viewer->surfName() != "segmentation"
        || _dragMode != DragMode::None)
        return;
    auto* sourceRaw = dynamic_cast<QuadSurface*>(_viewer->currentSurface());
    if (!sourceRaw || !_paintSurface || _paintSurface.get() != sourceRaw) return;

    const auto sample = pointOnSurface(devicePos, _paintSurface);
    if (!sample) {
        emit pointPlacementRejected(tr("Point must lie on valid Spiral surface data"));
        return;
    }
    if (_activePolyline < 0) {
        PolylineGesture collection;
        collection.kind = PolylineGesture::Kind::PointCollection;
        collection.color = nextColor();
        collection.source = _paintSurface;
        collection.creationTime = QDateTime::currentMSecsSinceEpoch();
        collection.sequence = _nextPolylineSequence++;
        collection.pclEdit.emplace();
        collection.pclEdit->sourceCollection = QJsonObject{
            {QStringLiteral("name"),
             QStringLiteral("same_winding_%1").arg(
                 collection.sequence, 4, 10, QLatin1Char('0'))},
            {QStringLiteral("metadata"),
             QJsonObject{{QStringLiteral("winding_is_absolute"), false}}},
            {QStringLiteral("color"),
             QJsonArray{collection.color.redF(), collection.color.greenF(),
                        collection.color.blueF()}},
        };
        collection.pclEdit->topLevel = QJsonObject{
            {QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")}};
        _polylines.push_back(std::move(collection));
        _activePolyline = static_cast<int>(_polylines.size()) - 1;
    }
    if (_activePolyline >= static_cast<int>(_polylines.size())
        || _polylines[static_cast<std::size_t>(_activePolyline)].kind
            != PolylineGesture::Kind::PointCollection)
        return;

    auto& collection = _polylines[static_cast<std::size_t>(_activePolyline)];
    const std::vector<cv::Vec3f>& positions = pointCollectionPositions(collection);
    if (!vc3d::spiral::meetsMinimumVolumeSpacing(
            sample->second, positions, kPointCollectionSpacingVoxels)) {
        emit pointPlacementRejected(
            tr("Point rejected: it must be at least 10 voxels from every other point"));
        return;
    }
    collection.pclEdit->appendPreviewPoint(sample->second, sample->first);
    invalidateEditablePclHitIndex();
    updateSuppressedSameWindingIds();
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
    emit paintStateChanged();
}

void SpiralBrushController::selectEditablePcl(std::size_t sourceIndex)
{
    if (sourceIndex >= _sameWindingSources.size()) return;
    const auto& source = _sameWindingSources[sourceIndex];
    if (!source.editable) return;
    for (std::size_t index = 0; index < _polylines.size(); ++index) {
        auto& line = _polylines[index];
        if (line.pclEdit
            && line.pclEdit->collectionId == source.collectionId) {
            line.pclEdit->setDeleted(false);
            _activePolyline = static_cast<int>(index);
            invalidateEditablePclHitIndex();
            updateCursorWidget();
            refreshAll();
            emit paintStateChanged();
            return;
        }
    }
    PolylineGesture line;
    line.kind = PolylineGesture::Kind::PointCollection;
    line.color = collectionColor(source.sourceCollection);
    line.creationTime = QDateTime::currentMSecsSinceEpoch();
    line.sequence = _nextPolylineSequence++;
    line.pclEdit = source;
    _polylines.push_back(std::move(line));
    _activePolyline = static_cast<int>(_polylines.size()) - 1;
    invalidateEditablePclHitIndex();
    updateCursorWidget();
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::selectEditablePcl(const EditablePclHit& hit)
{
    if (hit.polylineIndex >= 0
        && hit.polylineIndex < static_cast<int>(_polylines.size())) {
        const auto& line = _polylines[static_cast<std::size_t>(hit.polylineIndex)];
        if (line.kind != PolylineGesture::Kind::PointCollection || !line.pclEdit
            || line.pclEdit->deleted)
            return;
        _activePolyline = hit.polylineIndex;
        refreshAll();
        emit paintStateChanged();
        return;
    }
    if (hit.sourceIndex) selectEditablePcl(*hit.sourceIndex);
}

void SpiralBrushController::reverseActivePcl()
{
    if (_activePolyline < 0
        || _activePolyline >= static_cast<int>(_polylines.size())) return;
    auto& active = _polylines[static_cast<std::size_t>(_activePolyline)];
    if (active.kind != PolylineGesture::Kind::PointCollection || !active.pclEdit
        || active.pclEdit->deleted || active.pclEdit->points.size() < 2)
        return;
    active.pclEdit->reverse();
    if (active.state == GestureState::Ready)
        active.state = GestureState::Painted;
    clearEditablePclHover();
    invalidateEditablePclHitIndex();
    updateSuppressedSameWindingIds();
    clearPointChainProjectionCache();
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::confirmDeleteActivePcl()
{
    if (_activePolyline < 0
        || _activePolyline >= static_cast<int>(_polylines.size())
        || !_viewer || !_viewer->graphicsView()) return;
    const auto index = static_cast<std::size_t>(_activePolyline);
    auto& active = _polylines[index];
    if (active.kind != PolylineGesture::Kind::PointCollection || !active.pclEdit)
        return;

    const QString collectionId = active.pclEdit->collectionId;
    const QString name = active.pclEdit->sourceCollection
                             .value(QStringLiteral("name")).toString();
    QString identity;
    if (collectionId.isEmpty()) {
        identity = name.isEmpty() ? tr("new same-winding collection") : name;
    } else if (name.isEmpty()) {
        identity = tr("same-winding collection %1").arg(collectionId);
    } else {
        identity = tr("same-winding collection %1 (%2)").arg(collectionId, name);
    }
    const auto answer = QMessageBox::question(
        _viewer->graphicsView(), tr("Delete same-winding PCL"),
        tr("Delete %1?").arg(identity), QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);
    if (answer != QMessageBox::Yes) return;

    _pointPlacement.deactivate();
    _pclLeftClickConsumed = false;
    clearEditablePclHover();
    if (collectionId.isEmpty()) {
        _polylines.erase(_polylines.begin() + _activePolyline);
    } else {
        active.pclEdit->setDeleted(true);
        if (active.state == GestureState::Ready)
            active.state = GestureState::Painted;
    }
    _activePolyline = -1;
    invalidateEditablePclHitIndex();
    updateCursorWidget();
    updateSuppressedSameWindingIds();
    clearPointChainProjectionCache();
    refreshAll();
    emit paintStateChanged();
}

std::optional<SpiralBrushController::EditablePclHit>
SpiralBrushController::editablePclHitAt(const QPointF& devicePos)
{
    if (!_viewer || !_viewer->graphicsView()) return std::nullopt;
    std::optional<EditablePclHit> best;
    auto consider = [&best, &devicePos](EditablePclHit hit) {
        const QPointF delta = hit.devicePosition - devicePos;
        const qreal distance = delta.x() * delta.x() + delta.y() * delta.y();
        if (!best) {
            best = std::move(hit);
            return;
        }
        const QPointF bestDelta = best->devicePosition - devicePos;
        const qreal bestDistance = bestDelta.x() * bestDelta.x()
            + bestDelta.y() * bestDelta.y();
        if (distance < bestDistance
            || (distance == bestDistance
                && std::tie(hit.stableCollectionOrder, hit.stablePointOrder)
                    < std::tie(best->stableCollectionOrder,
                               best->stablePointOrder))) {
            best = std::move(hit);
        }
    };

    if (_sameWindingHitOverlay && _sameWindingSourceVisible) {
        const auto sourceHit = _sameWindingHitOverlay->displayPointHitAt(
            _viewer, devicePos, kEditablePclHitRadius,
            _editableSameWindingCollectionIds);
        if (sourceHit) {
            const auto source = _sameWindingSourceIndexById.find(
                sourceHit->ref.collectionId);
            if (source != _sameWindingSourceIndexById.end()) {
                consider({source->second, -1,
                          static_cast<std::size_t>(sourceHit->ref.pointId),
                          sourceHit->scenePosition, sourceHit->devicePosition,
                          sourceHit->color, true,
                          static_cast<std::uint64_t>(source->second),
                          sourceHit->ref.pointId});
            }
        }
    }
    if (auto draftHit = draftEditablePclHitAt(devicePos))
        consider(std::move(*draftHit));
    return best;
}

void SpiralBrushController::invalidateEditablePclHitIndex()
{
    _editablePclHitIndex.valid = false;
    _editablePclHitIndex.projectionPositions.clear();
    clearPointChainProjectionCache();
    ++_editablePclHitContentRevision;
    if (_editablePclHitContentRevision == 0) {
        _editablePclHitContentRevision = 1;
    }
}

void SpiralBrushController::rebuildEditablePclHitIndex()
{
    auto& state = _editablePclHitIndex;
    state.records.clear();
    state.index.clear();
    state.valid = true;
    state.contentRevision = _editablePclHitContentRevision;
    if (!_viewer || !_viewer->graphicsView()) return;

    auto* view = _viewer->graphicsView();
    state.projectionContext = _viewer->surfaceProjectionContext();
    state.viewportTransform = view->viewportTransform();
    state.viewportSize = view->viewport() ? view->viewport()->size() : QSize{};
    const QRectF visibleRect = visibleSceneRect(_viewer);
    bool clearedTransientProjectionEntries = false;

    for (std::size_t lineIndex = 0; lineIndex < _polylines.size(); ++lineIndex) {
        const auto& line = _polylines[lineIndex];
        if (line.kind != PolylineGesture::Kind::PointCollection || !line.pclEdit
            || line.pclEdit->deleted
            || line.state == GestureState::Finalizing
            || line.state == GestureState::Finalized
            || (!line.pclEdit->collectionId.isEmpty() && !line.pclEdit->dirty))
            continue;

        std::uint64_t collectionOrder = _sameWindingSources.size()
            + static_cast<std::uint64_t>(std::max(line.sequence, 0));
        if (!line.pclEdit->collectionId.isEmpty()) {
            bool ok = false;
            const qulonglong collectionId =
                line.pclEdit->collectionId.toULongLong(&ok);
            const auto source = ok
                ? _sameWindingSourceIndexById.find(collectionId)
                : _sameWindingSourceIndexById.end();
            if (source != _sameWindingSourceIndexById.end())
                collectionOrder = source->second;
        }

        std::vector<QPointF> scenePositions;
        std::vector<std::size_t> pointIndices;
        const auto exact = exactPointCollectionSurfacePositions(
            &*line.pclEdit, line.source, _viewer->currentSurface());
        if (exact) {
            scenePositions.reserve(exact->size());
            pointIndices.reserve(exact->size());
            for (std::size_t pointIndex = 0; pointIndex < exact->size(); ++pointIndex) {
                const QPointF scenePosition = _viewer->surfaceCoordsToScene(
                    (*exact)[pointIndex][0], (*exact)[pointIndex][1]);
                if (!visibleRect.contains(scenePosition)) continue;
                scenePositions.push_back(scenePosition);
                pointIndices.push_back(pointIndex);
            }
        } else {
            auto [positions, inserted] =
                state.projectionPositions.try_emplace(line.sequence);
            if (inserted) {
                positions->second = pointCollectionPositions(line);
                // collectPrimitives() also assembles temporary PCL vectors.
                // Drop their pointer-keyed entries once before installing the
                // retained vectors, so allocator reuse cannot look like a hit.
                if (!clearedTransientProjectionEntries) {
                    clearPointChainProjectionCache();
                    clearedTransientProjectionEntries = true;
                }
            }
            std::vector<float> opacities;
            const FilteredPoints projected = projectedPointChain(
                _viewer, positions->second, _pointViewToleranceVoxels,
                &opacities);
            scenePositions.reserve(projected.scenePoints.size());
            pointIndices.reserve(projected.scenePoints.size());
            for (std::size_t index = 0;
                 index < projected.scenePoints.size(); ++index) {
                if ((index < opacities.size() && opacities[index] <= 0.0f)
                    || !visibleRect.contains(projected.scenePoints[index]))
                    continue;
                scenePositions.push_back(projected.scenePoints[index]);
                pointIndices.push_back(projected.sourceIndices.empty()
                                           ? index
                                           : projected.sourceIndices[index]);
            }
        }

        state.records.reserve(state.records.size() + scenePositions.size());
        state.index.reserve(state.records.size() + scenePositions.size());
        for (std::size_t index = 0; index < scenePositions.size(); ++index) {
            const std::size_t pointIndex = pointIndices.empty()
                ? index : pointIndices[index];
            const QPointF devicePosition =
                state.viewportTransform.map(scenePositions[index]);
            const std::size_t recordIndex = state.records.size();
            state.records.push_back({
                std::nullopt, static_cast<int>(lineIndex), pointIndex,
                scenePositions[index], devicePosition, line.color, false,
                collectionOrder, static_cast<std::uint64_t>(pointIndex)});
            state.index.insert({devicePosition, collectionOrder,
                                static_cast<std::uint64_t>(pointIndex),
                                recordIndex});
        }
    }
}

std::optional<SpiralBrushController::EditablePclHit>
SpiralBrushController::draftEditablePclHitAt(const QPointF& devicePos)
{
    if (!_viewer || !_viewer->graphicsView()) return std::nullopt;
    auto* view = _viewer->graphicsView();
    const SurfaceProjectionContext context = _viewer->surfaceProjectionContext();
    const QSize viewportSize = view->viewport()
        ? view->viewport()->size() : QSize{};
    if (!_editablePclHitIndex.valid
        || _editablePclHitIndex.contentRevision != _editablePclHitContentRevision
        || !(_editablePclHitIndex.projectionContext == context)
        || _editablePclHitIndex.viewportTransform != view->viewportTransform()
        || _editablePclHitIndex.viewportSize != viewportSize) {
        rebuildEditablePclHitIndex();
    }
    const auto hit = _editablePclHitIndex.index.closest(
        devicePos, kEditablePclHitRadius);
    return hit && *hit < _editablePclHitIndex.records.size()
        ? std::optional<EditablePclHit>(_editablePclHitIndex.records[*hit])
        : std::nullopt;
}

void SpiralBrushController::updateEditablePclHover(const QPointF& devicePos)
{
    if (_pointPlacement.active() || _dragMode != DragMode::None) {
        clearEditablePclHover();
        return;
    }
    const auto hit = editablePclHitAt(devicePos);
    const bool unchanged = hit && _hoveredEditablePcl
        && hit->sourceIndex == _hoveredEditablePcl->sourceIndex
        && hit->polylineIndex == _hoveredEditablePcl->polylineIndex
        && hit->pointIndex == _hoveredEditablePcl->pointIndex
        && hit->scenePosition == _hoveredEditablePcl->scenePosition
        && hit->devicePosition == _hoveredEditablePcl->devicePosition
        && hit->sourceMarker == _hoveredEditablePcl->sourceMarker;
    if (unchanged || (!hit && !_hoveredEditablePcl)) return;
    _hoveredEditablePcl = hit;
    updateCursorWidget();
}

void SpiralBrushController::clearEditablePclHover()
{
    if (!_hoveredEditablePcl) return;
    _hoveredEditablePcl.reset();
    updateCursorWidget();
}

void SpiralBrushController::finishPointCollection(
    bool removeIncompleteNewCollection)
{
    bool pclRemoved = false;
    if (_activePolyline >= 0 && _activePolyline < static_cast<int>(_polylines.size())) {
        const auto index = static_cast<std::size_t>(_activePolyline);
        if (_polylines[index].kind == PolylineGesture::Kind::PointCollection) {
            const auto& line = _polylines[index];
            if (removeIncompleteNewCollection && line.pclEdit
                && line.pclEdit->isIncompleteNewCollection()) {
                _polylines.erase(_polylines.begin() + _activePolyline);
                pclRemoved = true;
            }
            _activePolyline = -1;
        }
    }
    _pclLeftClickConsumed = false;
    clearEditablePclHover();
    if (pclRemoved) invalidateEditablePclHitIndex();
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
    emit paintStateChanged();
}

void SpiralBrushController::deactivatePointPlacement()
{
    if (!_pointPlacement.deactivate() && _activePolyline < 0) return;
    finishPointCollection();
    updateCursorWidget();
}

void SpiralBrushController::beginErase(const QPointF& devicePos)
{
    if (!_viewer || _viewer->surfName() != "segmentation"
        || !dynamic_cast<QuadSurface*>(_viewer->currentSurface())) return;
    _lastDevicePos = devicePos;
    _dragMode = DragMode::Erase;
    eraseWith(deviceDisk(devicePos));
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
}

void SpiralBrushController::extendDrag(const QPointF& devicePos)
{
    if (_dragMode == DragMode::Paint && _activeGesture >= 0
        && _activeGesture < static_cast<int>(_gestures.size())) {
        const QPainterPath addition = deviceToSurface(deviceSweep(_lastDevicePos, devicePos));
        auto& gesture = _gestures[static_cast<std::size_t>(_activeGesture)];
        gesture.shape = gesture.shape.united(addition);
    } else if (_dragMode == DragMode::Erase) {
        eraseWith(deviceSweep(_lastDevicePos, devicePos));
    } else if (_dragMode == DragMode::Polyline && !_polylineBlocked) {
        if (!appendPolylinePoint(devicePos)) _polylineBlocked = true;
    }
    _lastDevicePos = devicePos;
    refreshViewer(_viewer);
}

void SpiralBrushController::finishDrag(const QPointF& devicePos)
{
    if (_dragMode != DragMode::None) extendDrag(devicePos);
    if (_dragMode == DragMode::Polyline && _activePolyline >= 0
        && _activePolyline < static_cast<int>(_polylines.size())) {
        auto& line = _polylines[static_cast<std::size_t>(_activePolyline)];
        resamplePolyline(line);
        if (line.volumePoints.size() < 2)
            _polylines.erase(_polylines.begin() + _activePolyline);
    }
    _dragMode = DragMode::None;
    _activeGesture = -1;
    _activePolyline = -1;
    _polylineBlocked = false;
    _gestures.erase(std::remove_if(_gestures.begin(), _gestures.end(), [](const Gesture& gesture) {
        return gesture.state == GestureState::Painted && gesture.shape.isEmpty();
    }), _gestures.end());
    emit paintStateChanged();
}

void SpiralBrushController::eraseWith(const QPainterPath& deviceShape)
{
    Surface* current = _viewer ? _viewer->currentSurface() : nullptr;
    const QPainterPath surfaceShape = deviceToSurface(deviceShape);
    for (auto& gesture : _gestures) {
        if ((gesture.state != GestureState::Painted
             && gesture.state != GestureState::Ready)
            || gesture.source.get() != current) continue;
        gesture.shape = gesture.shape.subtracted(surfaceShape);
    }

    auto* view = _viewer ? _viewer->graphicsView() : nullptr;
    bool pointChainsChanged = false;
    bool pclPointsChanged = false;
    if (view) {
        for (auto line = _polylines.begin(); line != _polylines.end();) {
            if ((line->state != GestureState::Painted
                 && line->state != GestureState::Ready)) {
                ++line;
                continue;
            }
            if (line->kind == PolylineGesture::Kind::PointCollection
                && line->pclEdit) {
                const auto& positions = pointCollectionPositions(*line);
                std::vector<bool> touched(positions.size(), false);
                const auto surfacePositions =
                    exactPointCollectionSurfacePositions(
                        &*line->pclEdit, line->source, current);
                if (surfacePositions) {
                    for (std::size_t index = 0;
                         index < surfacePositions->size(); ++index) {
                        const QPointF scenePoint = _viewer->surfaceCoordsToScene(
                            (*surfacePositions)[index][0],
                            (*surfacePositions)[index][1]);
                        const QPointF devicePoint =
                            view->viewportTransform().map(scenePoint);
                        if (deviceShape.contains(devicePoint))
                            touched[index] = true;
                    }
                } else {
                    const FilteredPoints projected = projectPointChainForHitTest(
                        _viewer, positions, _pointViewToleranceVoxels);
                    for (std::size_t index = 0;
                         index < projected.scenePoints.size(); ++index) {
                        const std::size_t sourceIndex = projected.sourceIndices[index];
                        const QPointF devicePoint = view->viewportTransform().map(
                            projected.scenePoints[index]);
                        if (sourceIndex < touched.size()
                            && deviceShape.contains(devicePoint))
                            touched[sourceIndex] = true;
                    }
                }
                for (std::size_t index = touched.size(); index-- > 0;) {
                    if (touched[index]) line->pclEdit->erase(index);
                }
                pointChainsChanged = pointChainsChanged
                    || std::any_of(touched.begin(), touched.end(), [](bool value) {
                           return value;
                       });
                pclPointsChanged = pclPointsChanged
                    || std::any_of(touched.begin(), touched.end(), [](bool value) {
                           return value;
                       });
                ++line;
                continue;
            }
            if (line->anchors.empty()) {
                ++line;
                continue;
            }
            std::vector<cv::Vec3f> anchorVolumes;
            anchorVolumes.reserve(line->anchors.size());
            for (const auto& anchor : line->anchors) anchorVolumes.push_back(anchor.volume);
            std::vector<bool> touched(line->anchors.size(), false);
            if (line->source.get() == current) {
                for (std::size_t index = 0; index < line->anchors.size(); ++index) {
                    const QPointF scenePoint = _viewer->surfaceCoordsToScene(
                        static_cast<float>(line->anchors[index].surface.x()),
                        static_cast<float>(line->anchors[index].surface.y()));
                    const QPointF devicePoint =
                        view->viewportTransform().map(scenePoint);
                    if (deviceShape.contains(devicePoint)) touched[index] = true;
                }
            } else {
                const FilteredPoints projected = projectPointChainForHitTest(
                    _viewer, anchorVolumes, kPolylineProjectionToleranceVoxels);
                for (std::size_t index = 0; index < projected.scenePoints.size(); ++index) {
                    const std::size_t sourceIndex = projected.sourceIndices[index];
                    const QPointF devicePoint =
                        view->viewportTransform().map(projected.scenePoints[index]);
                    if (sourceIndex < touched.size() && deviceShape.contains(devicePoint))
                        touched[sourceIndex] = true;
                }
            }
            const auto decision = vc3d::spiral::classifyAnchorErase(touched);
            if (decision.action == vc3d::spiral::AnchorEraseAction::None) {
                ++line;
                continue;
            }
            if (decision.action == vc3d::spiral::AnchorEraseAction::DeleteChain) {
                const int lineIndex = static_cast<int>(
                    std::distance(_polylines.begin(), line));
                if (_activePolyline == lineIndex)
                    _activePolyline = -1;
                else if (_activePolyline > lineIndex)
                    --_activePolyline;
                line = _polylines.erase(line);
                pointChainsChanged = true;
                continue;
            }

            line->anchors.erase(
                line->anchors.end() - static_cast<std::ptrdiff_t>(decision.removeSuffix),
                line->anchors.end());
            line->anchors.erase(
                line->anchors.begin(),
                line->anchors.begin() + static_cast<std::ptrdiff_t>(decision.removePrefix));
            if (line->kind == PolylineGesture::Kind::Anchored) {
                if (!rebuildAnchoredPolyline(*line)) {
                    const int lineIndex = static_cast<int>(
                        std::distance(_polylines.begin(), line));
                    if (_activePolyline == lineIndex)
                        _activePolyline = -1;
                    else if (_activePolyline > lineIndex)
                        --_activePolyline;
                    line = _polylines.erase(line);
                    pointChainsChanged = true;
                    continue;
                }
            } else {
                line->surfacePoints.clear();
                line->volumePoints.clear();
                line->surfacePoints.reserve(line->anchors.size());
                line->volumePoints.reserve(line->anchors.size());
                for (const auto& anchor : line->anchors) {
                    line->surfacePoints.push_back(anchor.surface);
                    line->volumePoints.push_back(anchor.volume);
                }
                clearPointChainProjectionCache();
            }
            pointChainsChanged = true;
            ++line;
        }
    }
    updateSuppressedSameWindingIds();
    if (pclPointsChanged) invalidateEditablePclHitIndex();
    if (pointChainsChanged) clearPointChainProjectionCache();
    emit paintStateChanged();
}

void SpiralBrushController::updateCursor(const QPointF& devicePos)
{
    _cursorDevicePos = devicePos;
    _cursorInside = true;
    updateCursorWidget();
}

void SpiralBrushController::updateCursorWidget()
{
    if (_viewer)
        _viewer->setLocalCursorCrosshairSuppressed(_pointPlacement.active());
    if (!_cursorWidget) return;
    qreal hoverRadiusX = 0.0;
    qreal hoverRadiusY = 0.0;
    qreal hoverPenWidth = 0.0;
    if (_hoveredEditablePcl && _viewer && _viewer->graphicsView()) {
        const QTransform transform = _viewer->graphicsView()->viewportTransform();
        const QPointF scenePosition = _hoveredEditablePcl->scenePosition;
        const QPointF devicePosition = transform.map(scenePosition);
        const qreal baseRadius = _hoveredEditablePcl->sourceMarker ? 5.0 : 3.5;
        const qreal sceneRadius = vc3d::spiral::editablePclPointRadius(
            baseRadius, true);
        hoverRadiusX = QLineF(
            devicePosition,
            transform.map(scenePosition + QPointF(sceneRadius, 0.0))).length();
        hoverRadiusY = QLineF(
            devicePosition,
            transform.map(scenePosition + QPointF(0.0, sceneRadius))).length();
        const qreal scaleX = QLineF(
            devicePosition,
            transform.map(scenePosition + QPointF(1.0, 0.0))).length();
        const qreal scaleY = QLineF(
            devicePosition,
            transform.map(scenePosition + QPointF(0.0, 1.0))).length();
        const qreal scenePenWidth = _hoveredEditablePcl->sourceMarker ? 1.5 : 1.0;
        hoverPenWidth = scenePenWidth * (scaleX + scaleY) * 0.5;
    }
    _cursorWidget->setEditablePclHover(
        _hoveredEditablePcl
            ? std::optional<QPointF>(_hoveredEditablePcl->devicePosition)
            : std::nullopt,
        _hoveredEditablePcl ? _hoveredEditablePcl->color : QColor{},
        _hoveredEditablePcl && _hoveredEditablePcl->sourceMarker,
        hoverRadiusX, hoverRadiusY, hoverPenWidth);
    const bool pointPlacementVisible = _cursorInside && _pointPlacement.active();
    _cursorWidget->setCursorState(
        _cursorDevicePos, _diameterPx,
        _cursorInside && !pointPlacementVisible && (_shiftHeld || _controlHeld),
        pointPlacementVisible);
}

void SpiralBrushController::sampleColor(const QPointF& scenePos)
{
    Surface* current = _viewer ? _viewer->currentSurface() : nullptr;
    const auto surfacePos = scenePointToSurface(scenePos);
    if (!surfacePos) return;
    for (auto it = _gestures.rbegin(); it != _gestures.rend(); ++it) {
        if (it->state == GestureState::Painted && it->source.get() == current
            && it->shape.contains(*surfacePos)) {
            _sampledColor = it->color;
            return;
        }
    }
}

bool SpiralBrushController::eventFilter(QObject* watched, QEvent* event)
{
    if ((watched != _viewport && watched != _viewObject) || !_viewer || !event) return false;
    if (event->type() == QEvent::KeyPress || event->type() == QEvent::KeyRelease) {
        auto* key = static_cast<QKeyEvent*>(event);
        if (key->key() == Qt::Key_Q || key->key() == Qt::Key_Escape
            || key->key() == Qt::Key_F || key->key() == Qt::Key_Delete) {
            if (_vHeld) return true;
            const bool hasActivePcl = _activePolyline >= 0
                && _activePolyline < static_cast<int>(_polylines.size())
                && _polylines[static_cast<std::size_t>(_activePolyline)].pclEdit
                    .has_value();
            const auto result = _pointPlacement.handleEvent(*event, hasActivePcl);
            if (result.transition
                == SpiralPointPlacementMode::Transition::ClearInteraction) {
                finishPointCollection();
            } else if (result.transition
                       == SpiralPointPlacementMode::Transition::ReverseActive) {
                reverseActivePcl();
            } else if (result.transition
                       == SpiralPointPlacementMode::Transition::DeleteActive) {
                confirmDeleteActivePcl();
            }
            if (result.transition != SpiralPointPlacementMode::Transition::None) {
                clearEditablePclHover();
                updateCursorWidget();
            }
            return result.handled;
        }
        if (key->key() == Qt::Key_G && !key->isAutoRepeat()) {
            _gHeld = event->type() == QEvent::KeyPress;
            return true;
        }
        if (key->key() == Qt::Key_Shift && !key->isAutoRepeat()) {
            _shiftHeld = event->type() == QEvent::KeyPress;
            updateCursorWidget();
            return false;
        }
        if (key->key() == Qt::Key_Control && !key->isAutoRepeat()) {
            _controlHeld = event->type() == QEvent::KeyPress;
            updateCursorWidget();
            return false;
        }
        if (key->key() == Qt::Key_V && !key->isAutoRepeat()) {
            if (event->type() == QEvent::KeyPress) {
                if (!_pointPlacement.active()) _vHeld = true;
            } else {
                _vHeld = false;
                finishAnchoredPolyline();
            }
            return true;
        }
    }
    if (watched == _viewport && event->type() == QEvent::Leave) {
        _cursorInside = false;
        _gHeld = false;
        _shiftHeld = false;
        _controlHeld = false;
        if (_vHeld) finishAnchoredPolyline();
        _vHeld = false;
        _pclLeftClickConsumed = false;
        clearEditablePclHover();
        updateCursorWidget();
        return false;
    }
    if (event->type() == QEvent::WindowDeactivate) {
        _gHeld = false;
        _shiftHeld = false;
        _controlHeld = false;
        if (_vHeld) finishAnchoredPolyline();
        _vHeld = false;
        _pclLeftClickConsumed = false;
        clearEditablePclHover();
        updateCursorWidget();
        return false;
    }
    if (watched == _viewport && event->type() == QEvent::Resize) {
        clearEditablePclHover();
        if (_cursorWidget) {
            if (auto* viewportWidget = qobject_cast<QWidget*>(_viewport))
                _cursorWidget->setGeometry(viewportWidget->rect());
            _cursorWidget->raise();
        }
        return false;
    }
    if (event->type() == QEvent::Wheel) {
        clearEditablePclHover();
        auto* wheel = static_cast<QWheelEvent*>(event);
        if (wheel->modifiers() == Qt::ControlModifier) {
            _controlHeld = true;
            const int steps = wheel->angleDelta().y() / 120;
            if (steps != 0) {
                _diameterPx = std::clamp(_diameterPx + steps * 2,
                                         kMinimumDiameter, kMaximumDiameter);
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(QStringLiteral("spiral/brush_diameter_px"), _diameterPx);
                const QPointF devicePos = watched == _viewport
                    ? wheel->position()
                    : QPointF(qobject_cast<QWidget*>(_viewport)->mapFromGlobal(
                          wheel->globalPosition().toPoint()));
                updateCursor(devicePos);
                emit brushDiameterChanged(_diameterPx);
            }
            return true;
        }
    }
    if (event->type() == QEvent::MouseMove) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        const QPointF devicePos = watched == _viewport
            ? mouse->position()
            : QPointF(qobject_cast<QWidget*>(_viewport)->mapFromGlobal(
                  mouse->globalPosition().toPoint()));
        _shiftHeld = mouse->modifiers().testFlag(Qt::ShiftModifier);
        _controlHeld = mouse->modifiers().testFlag(Qt::ControlModifier);
        updateCursor(devicePos);
        if (mouse->buttons() == Qt::NoButton)
            updateEditablePclHover(devicePos);
        else
            clearEditablePclHover();
        const bool paintDragging = _dragMode == DragMode::Paint
            && mouse->buttons().testFlag(Qt::LeftButton);
        const bool rightDragging = (_dragMode == DragMode::Polyline || _dragMode == DragMode::Erase)
            && mouse->buttons().testFlag(Qt::RightButton);
        if (paintDragging || rightDragging) {
            extendDrag(devicePos);
            return true;
        }
        if ((_vHeld || _pointPlacement.active())
            && mouse->buttons().testFlag(Qt::LeftButton)) return true;
        return false;
    }
    if (event->type() == QEvent::MouseButtonPress) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        const QPointF devicePos = watched == _viewport
            ? mouse->position()
            : QPointF(qobject_cast<QWidget*>(_viewport)->mapFromGlobal(
                  mouse->globalPosition().toPoint()));
        if (_gHeld && mouse->button() == Qt::LeftButton) {
            sampleColor(_viewer->graphicsView()->mapToScene(devicePos.toPoint()));
            return true;
        }
        if (_vHeld && mouse->button() == Qt::LeftButton) {
            appendAnchoredPoint(devicePos);
            _vClickConsumed = true;
            return true;
        }
        if (_pointPlacement.active() && mouse->button() == Qt::LeftButton) {
            appendPointCollectionPoint(devicePos);
            _pclLeftClickConsumed = true;
            return true;
        }
        if (mouse->button() == Qt::LeftButton
            && mouse->modifiers() == Qt::NoModifier) {
            if (const auto hit = editablePclHitAt(devicePos)) {
                selectEditablePcl(*hit);
                _pclLeftClickConsumed = true;
                return true;
            }
            // Empty-space clicks retain the active collection and continue to
            // the flattened viewer's ordinary interaction path.
        }
        if (mouse->button() == Qt::LeftButton && mouse->modifiers() == Qt::ShiftModifier) {
            _shiftHeld = true;
            updateCursor(devicePos);
            beginPaint(devicePos);
            return _dragMode == DragMode::Paint;
        }
        if (mouse->button() == Qt::RightButton && mouse->modifiers() == Qt::ShiftModifier) {
            _shiftHeld = true;
            updateCursor(devicePos);
            beginPolyline(devicePos);
            return _dragMode == DragMode::Polyline;
        }
        if (mouse->button() == Qt::RightButton &&
            mouse->modifiers() == (Qt::ControlModifier | Qt::ShiftModifier)) {
            beginErase(devicePos);
            return _dragMode == DragMode::Erase;
        }
    }
    if (event->type() == QEvent::MouseButtonRelease) {
        auto* mouse = static_cast<QMouseEvent*>(event);
        const QPointF devicePos = watched == _viewport
            ? mouse->position()
            : QPointF(qobject_cast<QWidget*>(_viewport)->mapFromGlobal(
                  mouse->globalPosition().toPoint()));
        const bool matchingRelease =
            (mouse->button() == Qt::LeftButton && _dragMode == DragMode::Paint)
            || (mouse->button() == Qt::RightButton
                && (_dragMode == DragMode::Polyline || _dragMode == DragMode::Erase));
        if (matchingRelease) {
            finishDrag(devicePos);
            updateCursor(devicePos);
            return true;
        }
        if (mouse->button() == Qt::LeftButton && _vClickConsumed) {
            _vClickConsumed = false;
            updateCursor(devicePos);
            return true;
        }
        if (mouse->button() == Qt::LeftButton && _pclLeftClickConsumed) {
            _pclLeftClickConsumed = false;
            updateCursor(devicePos);
            return true;
        }
    }
    return false;
}

bool SpiralBrushController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!viewer || viewer != _viewer || viewer->surfName() != "segmentation") return false;
    Surface* current = viewer->currentSurface();
    const bool hasPaint = std::any_of(
        _gestures.begin(), _gestures.end(), [current](const Gesture& gesture) {
            return gesture.source.get() == current && !gesture.shape.isEmpty();
        });
    return hasPaint || std::any_of(
        _polylines.begin(), _polylines.end(), [this](const PolylineGesture& line) {
            const bool visible = line.state != GestureState::Finalized
                || _visiblePointCollectionIds.contains(line.id);
            const bool hasPclPoints = line.pclEdit && !line.pclEdit->points.empty();
            return visible && (hasPclPoints || !line.volumePoints.empty()
                || (line.kind == PolylineGesture::Kind::Anchored && !line.anchors.empty()));
        });
}

void SpiralBrushController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) return;
    Surface* current = viewer->currentSurface();
    for (const auto& gesture : _gestures) {
        if (gesture.source.get() != current || gesture.shape.isEmpty()) continue;
        OverlayStyle style;
        style.penColor = Qt::transparent;
        style.brushColor = gesture.color;
        style.brushColor.setAlphaF(kPaintOpacity);
        style.z = 118.0;
        builder.addPainterPath(surfaceToScene(gesture.shape), style);
    }
    for (std::size_t lineIndex = 0; lineIndex < _polylines.size(); ++lineIndex) {
        const auto& line = _polylines[lineIndex];
        const bool visible = line.state != GestureState::Finalized
            || _visiblePointCollectionIds.contains(line.id);
        const auto& renderPositions = line.kind == PolylineGesture::Kind::PointCollection
            ? pointCollectionPositions(line) : line.volumePoints;
        if (!visible || renderPositions.empty()) continue;
        // Volume points remain the canonical line. renderPointChain projects
        // them through the current preview's indexed surface generation, so a
        // fitted replacement surface cannot strand the overlay on stale grid
        // coordinates.
        PointChainStyle style;
        style.color = line.color;
        style.pointBorderColor = line.color;
        style.pointRadius = kControlPointRadius;
        style.pointPenWidth = 1.0;
        style.lineWidth = kPolylineWidth;
        style.lineOpacity = 1.0f;
        style.pointZ = 120.0;
        style.lineZ = 119.0;
        style.distanceTolerance = line.kind == PolylineGesture::Kind::PointCollection
            ? _pointViewToleranceVoxels
            : kPolylineProjectionToleranceVoxels;
        if (line.kind == PolylineGesture::Kind::PointCollection)
            style.drawLines = false;
        const auto exactPclSurfacePositions =
            line.kind == PolylineGesture::Kind::PointCollection
            ? exactPointCollectionSurfacePositions(
                  line.pclEdit ? &*line.pclEdit : nullptr, line.source, current)
            : std::nullopt;
        const bool sameSurface = exactPclSurfacePositions.has_value()
            || (line.kind != PolylineGesture::Kind::PointCollection
                && line.source.get() == current
                && line.surfacePoints.size() == line.volumePoints.size());
        FilteredPoints labelPoints;
        std::vector<float> labelOpacities;
        if (sameSurface) {
            std::vector<cv::Vec2f> surfacePoints = exactPclSurfacePositions
                ? *exactPclSurfacePositions : std::vector<cv::Vec2f>{};
            if (!exactPclSurfacePositions) {
                surfacePoints.reserve(line.surfacePoints.size());
                for (const QPointF& point : line.surfacePoints) {
                    surfacePoints.emplace_back(
                        static_cast<float>(point.x()),
                        static_cast<float>(point.y()));
                }
            }
            if (style.drawLines && surfacePoints.size() >= 2) {
                OverlayStyle lineStyle;
                lineStyle.penColor = style.color;
                lineStyle.penColor.setAlphaF(style.lineOpacity);
                lineStyle.penWidth = style.lineWidth;
                lineStyle.z = style.lineZ;
                builder.addSurfaceLineStrip(surfacePoints, false, lineStyle);
            }
            if (style.drawPoints) {
                OverlayStyle pointStyle;
                pointStyle.penColor = style.pointBorderColor;
                pointStyle.penWidth = style.pointPenWidth;
                pointStyle.brushColor = style.color;
                pointStyle.z = style.pointZ;
                for (std::size_t index = 0; index < surfacePoints.size(); ++index) {
                    const cv::Vec2f& point = surfacePoints[index];
                    builder.addSurfacePoint(point, style.pointRadius, pointStyle);
                    if (line.kind == PolylineGesture::Kind::PointCollection
                        && static_cast<int>(lineIndex) == _activePolyline) {
                        labelPoints.scenePoints.push_back(
                            viewer->surfaceCoordsToScene(point[0], point[1]));
                        labelPoints.sourceIndices.push_back(index);
                        labelOpacities.push_back(1.0f);
                    }
                }
            }
        } else {
            const bool sourceReplacementNotChanged = line.pclEdit
                && !line.pclEdit->collectionId.isEmpty() && !line.pclEdit->dirty;
            const bool activePointCollection =
                line.kind == PolylineGesture::Kind::PointCollection
                && static_cast<int>(lineIndex) == _activePolyline;
            if (!sourceReplacementNotChanged) {
                renderPointChain(
                    viewer, builder, renderPositions, style, std::nullopt,
                    activePointCollection ? &labelPoints : nullptr,
                    activePointCollection ? &labelOpacities : nullptr);
            } else if (activePointCollection) {
                labelPoints = projectPointChainForHitTest(
                    viewer, renderPositions, _pointViewToleranceVoxels);
            }
        }
        if (line.kind == PolylineGesture::Kind::PointCollection
            && static_cast<int>(lineIndex) == _activePolyline) {
            OverlayStyle labelStyle;
            labelStyle.penColor = Qt::white;
            labelStyle.z = style.pointZ + 1.0;
            for (std::size_t index = 0;
                 index < labelPoints.scenePoints.size(); ++index) {
                if (!labelOpacities.empty()
                    && (index >= labelOpacities.size()
                        || labelOpacities[index] <= 0.0f))
                    continue;
                const std::size_t sourceIndex = labelPoints.sourceIndices.empty()
                    ? index : labelPoints.sourceIndices[index];
                builder.addText(
                    labelPoints.scenePoints[index]
                        + QPointF(kControlPointRadius + 2.0,
                                  -kControlPointRadius - 2.0),
                    QString::number(sourceIndex), QFont(), labelStyle);
            }
        }
        if (line.kind == PolylineGesture::Kind::Anchored && !line.anchors.empty()) {
            std::vector<cv::Vec3f> anchorPoints;
            anchorPoints.reserve(line.anchors.size());
            for (const auto& anchor : line.anchors) anchorPoints.push_back(anchor.volume);
            PointChainStyle anchorStyle = style;
            anchorStyle.pointBorderColor = QColor(255, 255, 255, 240);
            anchorStyle.pointRadius = kControlPointRadius + 2.5;
            anchorStyle.pointPenWidth = 1.5;
            anchorStyle.pointZ = 121.0;
            anchorStyle.drawLines = false;
            if (line.source.get() == current) {
                OverlayStyle pointStyle;
                pointStyle.penColor = anchorStyle.pointBorderColor;
                pointStyle.penWidth = anchorStyle.pointPenWidth;
                pointStyle.brushColor = anchorStyle.color;
                pointStyle.z = anchorStyle.pointZ;
                for (const auto& anchor : line.anchors) {
                    builder.addSurfacePoint(
                        cv::Vec2f(static_cast<float>(anchor.surface.x()),
                                  static_cast<float>(anchor.surface.y())),
                        anchorStyle.pointRadius, pointStyle);
                }
            } else {
                renderPointChain(viewer, builder, anchorPoints, anchorStyle);
            }
        }
    }
}

SpiralBrushController::PreparedPatch SpiralBrushController::makePatch(Gesture& gesture) const
{
    PreparedPatch result;
    if (!gesture.source || gesture.shape.isEmpty()) return result;
    const auto* points = gesture.source->rawPointsPtr();
    if (!points || points->empty()) return result;
    const qreal det = gesture.columnStep.x() * gesture.rowStep.y()
                    - gesture.columnStep.y() * gesture.rowStep.x();
    if (std::abs(det) < 1e-12) return result;
    auto sceneToGrid = [&](const QPointF& scene) {
        const QPointF delta = scene - gesture.gridOrigin;
        const qreal col = (delta.x() * gesture.rowStep.y()
                         - delta.y() * gesture.rowStep.x()) / det;
        const qreal row = (gesture.columnStep.x() * delta.y()
                         - gesture.columnStep.y() * delta.x()) / det;
        return QPointF(col, row);
    };
    const QRectF bounds = gesture.shape.boundingRect();
    const std::array<QPointF, 4> corners{{bounds.topLeft(), bounds.topRight(),
                                         bounds.bottomLeft(), bounds.bottomRight()}};
    qreal minCol = std::numeric_limits<qreal>::max();
    qreal maxCol = std::numeric_limits<qreal>::lowest();
    qreal minRow = std::numeric_limits<qreal>::max();
    qreal maxRow = std::numeric_limits<qreal>::lowest();
    for (const QPointF& corner : corners) {
        const QPointF grid = sceneToGrid(corner);
        minCol = std::min(minCol, grid.x()); maxCol = std::max(maxCol, grid.x());
        minRow = std::min(minRow, grid.y()); maxRow = std::max(maxRow, grid.y());
    }
    const int col0 = std::clamp(static_cast<int>(std::floor(minCol)) - 1, 0, points->cols - 1);
    const int col1 = std::clamp(static_cast<int>(std::ceil(maxCol)) + 1, 0, points->cols - 1);
    const int row0 = std::clamp(static_cast<int>(std::floor(minRow)) - 1, 0, points->rows - 1);
    const int row1 = std::clamp(static_cast<int>(std::ceil(maxRow)) + 1, 0, points->rows - 1);
    if (col1 <= col0 || row1 <= row0) return result;

    cv::Mat1b selected(row1 - row0 + 1, col1 - col0 + 1, uchar{0});
    for (int row = row0; row <= row1; ++row) {
        for (int col = col0; col <= col1; ++col) {
            if (!validPoint((*points)(row, col))) continue;
            const QPointF scene = gesture.gridOrigin
                + gesture.columnStep * col + gesture.rowStep * row;
            if (gesture.shape.contains(scene)) selected(row - row0, col - col0) = 1;
        }
    }
    cv::Mat1b retained(selected.rows, selected.cols, uchar{0});
    for (int row = 0; row + 1 < selected.rows; ++row) {
        for (int col = 0; col + 1 < selected.cols; ++col) {
            if (!selected(row, col) || !selected(row, col + 1)
                || !selected(row + 1, col) || !selected(row + 1, col + 1)) continue;
            retained(row, col) = retained(row, col + 1) = 1;
            retained(row + 1, col) = retained(row + 1, col + 1) = 1;
        }
    }
    std::vector<cv::Point> kept;
    cv::findNonZero(retained, kept);
    if (kept.empty()) return result;
    const cv::Rect crop = cv::boundingRect(kept);
    auto output = std::make_unique<cv::Mat_<cv::Vec3f>>(
        crop.height, crop.width, cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int row = 0; row < crop.height; ++row) {
        for (int col = 0; col < crop.width; ++col) {
            const int localRow = crop.y + row;
            const int localCol = crop.x + col;
            if (retained(localRow, localCol))
                (*output)(row, col) = (*points)(row0 + localRow, col0 + localCol);
        }
    }
    const QString stamp = QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    const QString suffix = QString::number(QRandomGenerator::global()->generate(), 16).rightJustified(8, '0');
    gesture.id = QStringLiteral("brush_%1_%2").arg(stamp, suffix);
    auto patch = std::make_shared<QuadSurface>(output.release(), gesture.source->scale());
    patch->id = gesture.id.toStdString();
    // Painted boundaries already encode the user's exact selection. Unlike
    // hand-authored input patches, they must not receive the fitter's generic
    // invalid-edge erosion when incorporated now or after dataset commit.
    patch->meta["spiral_patch_erode_cells"] = 0;
    result.id = gesture.id;
    result.color = gesture.color;
    result.surface = std::move(patch);
    return result;
}

std::vector<SpiralBrushController::PreparedPatch>
SpiralBrushController::preparePatches(QStringList& warnings)
{
    std::vector<PreparedPatch> patches;
    if (_dragMode != DragMode::None) {
        warnings.push_back(tr("Release the mouse button before finalizing brush paint"));
        return patches;
    }
    for (auto& gesture : _gestures) {
        if (gesture.state != GestureState::Ready || gesture.shape.isEmpty()) continue;
        PreparedPatch patch = makePatch(gesture);
        if (!patch.surface) {
            warnings.push_back(tr("A painted area was too small to contain a complete quad"));
            continue;
        }
        gesture.state = GestureState::Finalizing;
        patches.push_back(std::move(patch));
    }
    refreshAll();
    emit paintStateChanged();
    return patches;
}

std::vector<SpiralBrushController::PreparedPointCollections>
SpiralBrushController::preparePointCollections(QStringList& warnings)
{
    std::vector<PreparedPointCollections> results;
    if (_dragMode != DragMode::None) {
        warnings.push_back(tr("Release the mouse button before finalizing control-point lines"));
        return results;
    }
    for (auto& line : _polylines) {
        if (line.kind != PolylineGesture::Kind::PointCollection
            || line.state != GestureState::Ready || !line.pclEdit
            || line.pclEdit->collectionId.isEmpty() || !line.pclEdit->dirty)
            continue;
        if (line.pclEdit->submissionBlocked) {
            warnings.push_back(
                tr("Change to same-winding collection %1 is based on a stale source; "
                   "reload it before submitting")
                    .arg(line.pclEdit->collectionId));
            continue;
        }
        if (!line.pclEdit->deleted && line.pclEdit->points.size() < 2) {
            warnings.push_back(
                tr("Same-winding collection %1 needs at least two points")
                    .arg(line.pclEdit->collectionId));
            continue;
        }
        const QString stamp = QDateTime::currentDateTimeUtc().toString(
            QStringLiteral("yyyyMMdd_HHmmss_zzz"));
        const QString suffix = QString::number(
            QRandomGenerator::global()->generate(), 16).rightJustified(8, '0');
        PreparedPointCollections result;
        const QString operation = line.pclEdit->deleted
            ? QStringLiteral("delete_collection")
            : QStringLiteral("replace_collection");
        result.id = QStringLiteral("same_winding_%1_%2_%3_%4")
            .arg(line.pclEdit->deleted ? QStringLiteral("delete")
                                       : QStringLiteral("replace"))
            .arg(line.pclEdit->collectionId, stamp, suffix);
        result.role = QStringLiteral("same_winding");
        result.operation = operation;
        result.targetCollectionId = line.pclEdit->collectionId;
        result.baseSourceRevision = line.pclEdit->sourceRevision;
        result.document = line.pclEdit->replacementDocument();
        line.id = result.id;
        line.state = GestureState::Finalizing;
        results.push_back(std::move(result));
    }
    const auto prepareKinds = [this, &results](
                                  std::initializer_list<PolylineGesture::Kind> kinds,
                                  const QString& role, const QString& idPrefix,
                                  const QString& namePrefix) {
        const auto includesKind = [kinds](PolylineGesture::Kind kind) {
            return std::find(kinds.begin(), kinds.end(), kind) != kinds.end();
        };
        QJsonObject collections;
        int collectionId = 0;
        for (auto& line : _polylines) {
            if (!includesKind(line.kind) || line.state != GestureState::Ready
                || (line.kind == PolylineGesture::Kind::PointCollection
                    ? !pointCollectionHasChanges(line)
                    : line.volumePoints.size() < 2))
                continue;
            if (line.kind == PolylineGesture::Kind::PointCollection
                && line.pclEdit && !line.pclEdit->collectionId.isEmpty())
                continue;
            if (line.kind == PolylineGesture::Kind::PointCollection
                && line.pclEdit) {
                auto draft = *line.pclEdit;
                draft.collectionId = QString::number(collectionId);
                const QJsonObject serialized = draft.replacementDocument().object()
                    .value(QStringLiteral("collections")).toObject()
                    .value(draft.collectionId).toObject();
                collections[QString::number(collectionId++)] = serialized;
                continue;
            }
            QJsonObject points;
            for (int index = 0; index < static_cast<int>(line.volumePoints.size()); ++index) {
                const cv::Vec3f& point = line.volumePoints[static_cast<std::size_t>(index)];
                points[QString::number(index)] = QJsonObject{
                    {QStringLiteral("p"), QJsonArray{point[0], point[1], point[2]}},
                    {QStringLiteral("wind_a"), QJsonValue::Null},
                    {QStringLiteral("creation_time"), line.creationTime + index},
                };
            }
            collections[QString::number(collectionId++)] = QJsonObject{
                {QStringLiteral("name"),
                 QStringLiteral("%1_%2").arg(namePrefix).arg(
                     line.sequence, 4, 10, QLatin1Char('0'))},
                {QStringLiteral("points"), points},
                {QStringLiteral("metadata"),
                 QJsonObject{{QStringLiteral("winding_is_absolute"), false}}},
                {QStringLiteral("color"),
                 QJsonArray{line.color.redF(), line.color.greenF(), line.color.blueF()}},
            };
        }
        if (collections.isEmpty()) return;
        const QString stamp = QDateTime::currentDateTimeUtc().toString(
            QStringLiteral("yyyyMMdd_HHmmss_zzz"));
        const QString suffix = QString::number(QRandomGenerator::global()->generate(), 16)
                                   .rightJustified(8, '0');
        PreparedPointCollections result;
        result.id = QStringLiteral("%1_%2_%3").arg(idPrefix, stamp, suffix);
        result.role = role;
        result.document = QJsonDocument(QJsonObject{
            {QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")},
            {QStringLiteral("collections"), collections},
        });
        for (auto& line : _polylines) {
            if (includesKind(line.kind) && line.state == GestureState::Ready
                && (line.kind == PolylineGesture::Kind::PointCollection
                    ? pointCollectionHasChanges(line)
                    : line.volumePoints.size() >= 2)
                && !(line.kind == PolylineGesture::Kind::PointCollection
                     && line.pclEdit && !line.pclEdit->collectionId.isEmpty())) {
                line.id = result.id;
                line.state = GestureState::Finalizing;
            }
        }
        results.push_back(std::move(result));
    };
    prepareKinds({PolylineGesture::Kind::Freehand, PolylineGesture::Kind::Anchored},
                 QStringLiteral("drawn_control_points"),
                 QStringLiteral("drawn_control_points"), QStringLiteral("drawn_line"));
    prepareKinds({PolylineGesture::Kind::PointCollection}, QStringLiteral("same_winding"),
                 QStringLiteral("same_winding_points"), QStringLiteral("same_winding"));
    invalidateEditablePclHitIndex();
    refreshAll();
    emit paintStateChanged();
    return results;
}

void SpiralBrushController::finalizationSucceeded(const QString& id)
{
    _gestures.erase(std::remove_if(_gestures.begin(), _gestures.end(), [&](const Gesture& gesture) {
        return gesture.id == id;
    }), _gestures.end());
    for (auto& line : _polylines) {
        if (line.id == id && line.state == GestureState::Finalizing)
            line.state = GestureState::Finalized;
    }
    invalidateEditablePclHitIndex();
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::finalizationFailed(const QString& id)
{
    for (auto& gesture : _gestures) {
        if (gesture.id == id) {
            gesture.id.clear();
            gesture.state = GestureState::Ready;
        }
    }
    for (auto& line : _polylines) {
        if (line.id == id && line.state == GestureState::Finalizing) {
            line.id.clear();
            line.state = GestureState::Ready;
        }
    }
    invalidateEditablePclHitIndex();
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::discardUnfinalized()
{
    _pointPlacement.deactivate();
    _pclLeftClickConsumed = false;
    _hoveredEditablePcl.reset();
    _activeGesture = -1;
    _activePolyline = -1;
    _dragMode = DragMode::None;
    _polylineBlocked = false;
    _gestures.clear();
    _polylines.clear();
    _visiblePointCollectionIds.clear();
    _suppressedSameWindingCollectionIds.clear();
    clearPointChainProjectionCache();
    invalidateEditablePclHitIndex();
    _sampledColor.reset();
    updateCursorWidget();
    emit suppressedSameWindingCollectionIdsChanged({});
    refreshAll();
    emit paintStateChanged();
}
