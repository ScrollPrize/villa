#include "SpiralBrushController.hpp"

#include "SurfaceOverlayColors.hpp"
#include "VCSettings.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QDateTime>
#include <QEvent>
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
#include <limits>

namespace {
constexpr int kMinimumDiameter = 4;
constexpr int kMaximumDiameter = 256;
constexpr qreal kPaintOpacity = 0.45;

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
    _usedColors.clear();
    _sampledColor.reset();
    _cursorInside = false;
    updateCursorWidget();
    _dragMode = DragMode::None;
    _activeGesture = -1;
    refreshAll();
    emit paintStateChanged();
}

bool SpiralBrushController::hasUnfinalizedPaint() const
{
    return std::any_of(_gestures.begin(), _gestures.end(), [](const Gesture& gesture) {
        return gesture.state == GestureState::Painted && !gesture.shape.isEmpty();
    });
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

void SpiralBrushController::beginErase(const QPointF& devicePos)
{
    if (!_viewer || _viewer->surfName() != "segmentation"
        || !dynamic_cast<QuadSurface*>(_viewer->currentSurface())) return;
    _lastDevicePos = devicePos;
    _dragMode = DragMode::Erase;
    eraseWith(deviceToSurface(deviceDisk(devicePos)));
}

void SpiralBrushController::extendDrag(const QPointF& devicePos)
{
    const QPainterPath addition = deviceToSurface(deviceSweep(_lastDevicePos, devicePos));
    if (_dragMode == DragMode::Paint && _activeGesture >= 0
        && _activeGesture < static_cast<int>(_gestures.size())) {
        auto& gesture = _gestures[static_cast<std::size_t>(_activeGesture)];
        gesture.shape = gesture.shape.united(addition);
    } else if (_dragMode == DragMode::Erase) {
        eraseWith(addition);
    }
    _lastDevicePos = devicePos;
    refreshViewer(_viewer);
}

void SpiralBrushController::finishDrag(const QPointF& devicePos)
{
    if (_dragMode != DragMode::None) extendDrag(devicePos);
    _dragMode = DragMode::None;
    _activeGesture = -1;
    _gestures.erase(std::remove_if(_gestures.begin(), _gestures.end(), [](const Gesture& gesture) {
        return gesture.state == GestureState::Painted && gesture.shape.isEmpty();
    }), _gestures.end());
    emit paintStateChanged();
}

void SpiralBrushController::eraseWith(const QPainterPath& surfaceShape)
{
    Surface* current = _viewer ? _viewer->currentSurface() : nullptr;
    for (auto& gesture : _gestures) {
        if (gesture.state != GestureState::Painted || gesture.source.get() != current) continue;
        gesture.shape = gesture.shape.subtracted(surfaceShape);
    }
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
                                      _cursorInside && _shiftHeld);
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
    }
    if (watched == _viewport && event->type() == QEvent::Leave) {
        _cursorInside = false;
        _gHeld = false;
        _shiftHeld = false;
        updateCursorWidget();
        return false;
    }
    if (event->type() == QEvent::WindowDeactivate) {
        _gHeld = false;
        _shiftHeld = false;
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
        updateCursor(devicePos);
        if (_dragMode != DragMode::None && (mouse->buttons() & Qt::RightButton)) {
            extendDrag(devicePos);
            return true;
        }
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
        if (mouse->button() == Qt::RightButton && mouse->modifiers() == Qt::ShiftModifier) {
            _shiftHeld = true;
            updateCursor(devicePos);
            beginPaint(devicePos);
            return _dragMode == DragMode::Paint;
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
        if (mouse->button() == Qt::RightButton && _dragMode != DragMode::None) {
            finishDrag(devicePos);
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
    return std::any_of(
        _gestures.begin(), _gestures.end(), [current](const Gesture& gesture) {
            return gesture.source.get() == current && !gesture.shape.isEmpty();
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

void SpiralBrushController::finalizationSucceeded(const QString& id)
{
    _gestures.erase(std::remove_if(_gestures.begin(), _gestures.end(), [&](const Gesture& gesture) {
        return gesture.id == id;
    }), _gestures.end());
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
    refreshAll();
    emit paintStateChanged();
}

void SpiralBrushController::discardUnfinalized()
{
    _gestures.clear();
    _sampledColor.reset();
    refreshAll();
    emit paintStateChanged();
}
