#include "SpiralBrushController.hpp"

#include "SurfaceOverlayColors.hpp"
#include "VCSettings.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QDateTime>
#include <QEvent>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
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
constexpr float kPolylineSpacingVoxels = 30.0f;
constexpr qreal kPolylineWidth = 10.0;
constexpr qreal kControlPointRadius = 3.5;
constexpr float kPolylineProjectionToleranceVoxels = 100.0f;

bool validPoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f && std::isfinite(point[0])
        && std::isfinite(point[1]) && std::isfinite(point[2]);
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

class SpiralBrushCursorWidget final : public QWidget
{
public:
    explicit SpiralBrushCursorWidget(QWidget* parent) : QWidget(parent)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
        setAttribute(Qt::WA_TranslucentBackground);
    }

    void setBrushCursor(const QPointF& position, int diameter, bool visible)
    {
        _position = position;
        _diameter = diameter;
        _visible = visible;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        if (!_visible) return;
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        QPen pen(QColor(255, 255, 255, 220));
        pen.setWidthF(1.5);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        const qreal radius = _diameter * 0.5;
        painter.drawEllipse(_position, radius, radius);
    }

private:
    QPointF _position;
    int _diameter = 32;
    bool _visible = false;
};

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
    _paintSurface = surface;
    clearPointChainProjectionCache();
    refreshAll();
}

void SpiralBrushController::bindFlattenedViewer(VolumeViewerBase* viewer)
{
    if (_viewport) _viewport->removeEventFilter(this);
    if (_viewObject) _viewObject->removeEventFilter(this);
    _viewer = viewer;
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
}

void SpiralBrushController::resetSession()
{
    _gestures.clear();
    _polylines.clear();
    _visiblePointCollectionIds.clear();
    clearPointChainProjectionCache();
    _usedColors.clear();
    _sampledColor.reset();
    _cursorInside = false;
    updateCursorWidget();
    _dragMode = DragMode::None;
    _activeGesture = -1;
    _activePolyline = -1;
    _nextPolylineSequence = 1;
    _polylineBlocked = false;
    _vHeld = false;
    _vClickConsumed = false;
    _qHeld = false;
    _qClickConsumed = false;
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
    return std::any_of(_polylines.begin(), _polylines.end(), [](const PolylineGesture& line) {
        return line.state == GestureState::Painted && line.volumePoints.size() >= 2;
    });
}

void SpiralBrushController::setVisiblePointCollectionIds(const QSet<QString>& ids)
{
    if (_visiblePointCollectionIds == ids) return;
    _visiblePointCollectionIds = ids;
    refreshAll();
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
    const cv::Vec2f scale = sourceRaw->scale();
    if (!std::isfinite(scale[0]) || !std::isfinite(scale[1])
        || std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) return;
    const cv::Vec3f center = sourceRaw->center();
    Gesture gesture;
    gesture.color = nextColor();
    gesture.source = std::move(source);
    gesture.gridOrigin = QPointF(-center[0], -center[1]);
    gesture.columnStep = QPointF(1.0 / scale[0], 0.0);
    gesture.rowStep = QPointF(0.0, 1.0 / scale[1]);
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
    auto* view = _viewer ? _viewer->graphicsView() : nullptr;
    if (!view || !source || source.get() != _viewer->currentSurface()) return std::nullopt;
    bool transformValid = false;
    const QTransform viewportToScene = view->viewportTransform().inverted(&transformValid);
    if (!transformValid) return std::nullopt;
    const QPointF scenePos = viewportToScene.map(devicePos);
    const cv::Vec2f surface = _viewer->sceneToSurfaceCoords(scenePos);
    if (!std::isfinite(surface[0]) || !std::isfinite(surface[1])) return std::nullopt;
    const QPointF surfacePoint(surface[0], surface[1]);
    const auto volume = volumePointOnSurface(surfacePoint, source);
    if (!volume) return std::nullopt;
    return std::make_pair(surfacePoint, *volume);
}

std::optional<cv::Vec3f> SpiralBrushController::volumePointOnSurface(
    const QPointF& surfacePos, const std::shared_ptr<QuadSurface>& source) const
{
    if (!source) return std::nullopt;
    const auto* points = source->rawPointsPtr();
    const cv::Vec2f scale = source->scale();
    const cv::Vec3f center = source->center();
    if (!points || points->empty() || !std::isfinite(scale[0]) || !std::isfinite(scale[1])
        || std::abs(scale[0]) < 1e-6f || std::abs(scale[1]) < 1e-6f) return std::nullopt;
    const float col = (static_cast<float>(surfacePos.x()) + center[0]) * scale[0];
    const float row = (static_cast<float>(surfacePos.y()) + center[1]) * scale[1];
    const int col0 = static_cast<int>(std::floor(col));
    const int row0 = static_cast<int>(std::floor(row));
    if (col0 < 0 || row0 < 0 || col0 + 1 >= points->cols || row0 + 1 >= points->rows)
        return std::nullopt;
    const cv::Vec3f& p00 = (*points)(row0, col0);
    const cv::Vec3f& p01 = (*points)(row0, col0 + 1);
    const cv::Vec3f& p10 = (*points)(row0 + 1, col0);
    const cv::Vec3f& p11 = (*points)(row0 + 1, col0 + 1);
    if (!validPoint(p00) || !validPoint(p01) || !validPoint(p10) || !validPoint(p11))
        return std::nullopt;
    const float colFraction = col - static_cast<float>(col0);
    const float rowFraction = row - static_cast<float>(row0);
    const cv::Vec3f top = p00 * (1.0f - colFraction) + p01 * colFraction;
    const cv::Vec3f bottom = p10 * (1.0f - colFraction) + p11 * colFraction;
    const cv::Vec3f point = top * (1.0f - rowFraction) + bottom * rowFraction;
    if (!validPoint(point)) return std::nullopt;
    return point;
}

bool SpiralBrushController::appendPolylinePoint(const QPointF& devicePos)
{
    if (_activePolyline < 0 || _activePolyline >= static_cast<int>(_polylines.size()))
        return false;
    auto& line = _polylines[static_cast<std::size_t>(_activePolyline)];
    const auto sample = pointOnSurface(devicePos, line.source);
    if (!sample) return false;
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
    for (float distance = kPolylineSpacingVoxels; distance < total; distance += kPolylineSpacingVoxels)
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
        kPolylineSpacingVoxels);
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
        kPolylineSpacingVoxels);
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
        _polylines.push_back(std::move(collection));
        _activePolyline = static_cast<int>(_polylines.size()) - 1;
    }
    if (_activePolyline >= static_cast<int>(_polylines.size())
        || _polylines[static_cast<std::size_t>(_activePolyline)].kind
            != PolylineGesture::Kind::PointCollection)
        return;

    auto& collection = _polylines[static_cast<std::size_t>(_activePolyline)];
    collection.anchors.push_back({sample->first, sample->second});
    collection.surfacePoints.push_back(sample->first);
    collection.volumePoints.push_back(sample->second);
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
    emit paintStateChanged();
}

void SpiralBrushController::finishPointCollection()
{
    if (_activePolyline >= 0 && _activePolyline < static_cast<int>(_polylines.size())) {
        const auto index = static_cast<std::size_t>(_activePolyline);
        if (_polylines[index].kind == PolylineGesture::Kind::PointCollection) {
            if (_polylines[index].volumePoints.size() < 2)
                _polylines.erase(_polylines.begin() + _activePolyline);
            _activePolyline = -1;
        }
    }
    _qClickConsumed = false;
    clearPointChainProjectionCache();
    refreshViewer(_viewer);
    emit paintStateChanged();
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
        if (gesture.state != GestureState::Painted || gesture.source.get() != current) continue;
        gesture.shape = gesture.shape.subtracted(surfaceShape);
    }

    auto* view = _viewer ? _viewer->graphicsView() : nullptr;
    bool pointChainsChanged = false;
    if (view) {
        for (auto line = _polylines.begin(); line != _polylines.end();) {
            if (line->state != GestureState::Painted || line->anchors.empty()) {
                ++line;
                continue;
            }
            std::vector<cv::Vec3f> anchorVolumes;
            anchorVolumes.reserve(line->anchors.size());
            for (const auto& anchor : line->anchors) anchorVolumes.push_back(anchor.volume);
            const FilteredPoints projected = projectPointChainForHitTest(
                _viewer, anchorVolumes, kPolylineProjectionToleranceVoxels);
            std::vector<bool> touched(line->anchors.size(), false);
            for (std::size_t index = 0; index < projected.scenePoints.size(); ++index) {
                const std::size_t sourceIndex = projected.sourceIndices[index];
                const QPointF devicePoint =
                    view->viewportTransform().map(projected.scenePoints[index]);
                if (sourceIndex < touched.size() && deviceShape.contains(devicePoint))
                    touched[sourceIndex] = true;
            }
            const auto decision = vc3d::spiral::classifyAnchorErase(touched);
            if (line->kind == PolylineGesture::Kind::PointCollection) {
                for (std::size_t index = touched.size(); index-- > 0;) {
                    if (!touched[index]) continue;
                    line->anchors.erase(line->anchors.begin()
                        + static_cast<std::ptrdiff_t>(index));
                    line->surfacePoints.erase(line->surfacePoints.begin()
                        + static_cast<std::ptrdiff_t>(index));
                    line->volumePoints.erase(line->volumePoints.begin()
                        + static_cast<std::ptrdiff_t>(index));
                }
                if (line->volumePoints.size() < 2) {
                    const int lineIndex = static_cast<int>(
                        std::distance(_polylines.begin(), line));
                    if (_activePolyline == lineIndex)
                        _activePolyline = -1;
                    else if (_activePolyline > lineIndex)
                        --_activePolyline;
                    line = _polylines.erase(line);
                } else {
                    ++line;
                }
                pointChainsChanged = true;
                continue;
            }
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
    if (_cursorWidget)
        _cursorWidget->setBrushCursor(_cursorDevicePos, _diameterPx,
                                      _cursorInside && (_shiftHeld || _controlHeld));
}

void SpiralBrushController::sampleColor(const QPointF& scenePos)
{
    Surface* current = _viewer ? _viewer->currentSurface() : nullptr;
    const cv::Vec2f surface = _viewer ? _viewer->sceneToSurfaceCoords(scenePos) : cv::Vec2f{};
    const QPointF surfacePos(surface[0], surface[1]);
    for (auto it = _gestures.rbegin(); it != _gestures.rend(); ++it) {
        if (it->state == GestureState::Painted && it->source.get() == current
            && it->shape.contains(surfacePos)) {
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
                if (!_qHeld) _vHeld = true;
            } else {
                _vHeld = false;
                finishAnchoredPolyline();
            }
            return true;
        }
        if (key->key() == Qt::Key_Q && !key->isAutoRepeat()) {
            if (event->type() == QEvent::KeyPress) {
                if (!_vHeld) _qHeld = true;
            } else {
                _qHeld = false;
                finishPointCollection();
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
        if (_qHeld) finishPointCollection();
        _qHeld = false;
        updateCursorWidget();
        return false;
    }
    if (event->type() == QEvent::WindowDeactivate) {
        _gHeld = false;
        _shiftHeld = false;
        _controlHeld = false;
        if (_vHeld) finishAnchoredPolyline();
        _vHeld = false;
        if (_qHeld) finishPointCollection();
        _qHeld = false;
        updateCursorWidget();
        return false;
    }
    if (watched == _viewport && event->type() == QEvent::Resize) {
        if (_cursorWidget) {
            if (auto* viewportWidget = qobject_cast<QWidget*>(_viewport))
                _cursorWidget->setGeometry(viewportWidget->rect());
            _cursorWidget->raise();
        }
        return false;
    }
    if (event->type() == QEvent::Wheel) {
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
        const bool paintDragging = _dragMode == DragMode::Paint
            && mouse->buttons().testFlag(Qt::LeftButton);
        const bool rightDragging = (_dragMode == DragMode::Polyline || _dragMode == DragMode::Erase)
            && mouse->buttons().testFlag(Qt::RightButton);
        if (paintDragging || rightDragging) {
            extendDrag(devicePos);
            return true;
        }
        if ((_vHeld || _qHeld) && mouse->buttons().testFlag(Qt::LeftButton)) return true;
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
        if (_qHeld && mouse->button() == Qt::LeftButton) {
            appendPointCollectionPoint(devicePos);
            _qClickConsumed = true;
            return true;
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
        if (mouse->button() == Qt::RightButton && mouse->modifiers() == Qt::ControlModifier) {
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
        if (mouse->button() == Qt::LeftButton && _qClickConsumed) {
            _qClickConsumed = false;
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
            return visible && (!line.volumePoints.empty()
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
    for (const auto& line : _polylines) {
        const bool visible = line.state != GestureState::Finalized
            || _visiblePointCollectionIds.contains(line.id);
        if (!visible || line.volumePoints.empty()) continue;
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
        style.distanceTolerance = kPolylineProjectionToleranceVoxels;
        if (line.kind == PolylineGesture::Kind::PointCollection)
            style.drawLines = false;
        renderPointChain(viewer, builder, line.volumePoints, style);
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
            renderPointChain(viewer, builder, anchorPoints, anchorStyle);
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
        if (gesture.state != GestureState::Painted || gesture.shape.isEmpty()) continue;
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
            if (!includesKind(line.kind) || line.state != GestureState::Painted
                || line.volumePoints.size() < 2)
                continue;
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
            if (includesKind(line.kind) && line.state == GestureState::Painted
                && line.volumePoints.size() >= 2) {
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
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::finalizationFailed(const QString& id)
{
    for (auto& gesture : _gestures) {
        if (gesture.id == id) {
            gesture.id.clear();
            gesture.state = GestureState::Painted;
        }
    }
    for (auto& line : _polylines) {
        if (line.id == id && line.state == GestureState::Finalizing) {
            line.id.clear();
            line.state = GestureState::Painted;
        }
    }
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::discardUnfinalized()
{
    _gestures.clear();
    _polylines.clear();
    _visiblePointCollectionIds.clear();
    clearPointChainProjectionCache();
    _sampledColor.reset();
    refreshAll();
    emit paintStateChanged();
}
