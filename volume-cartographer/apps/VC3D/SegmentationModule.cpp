#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "SegmentationWidget.hpp"
#include "ViewerManager.hpp"

#include "vc/core/util/Surface.hpp"

#include <QApplication>
#include <QGraphicsSimpleTextItem>
#include <QKeyEvent>
#include <QColor>
#include <QFont>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPointer>
#include <QTimer>
#include <QVariant>

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kMinRadius = 1.0f;
constexpr int kMaxRadiusSteps = 32;
constexpr int kStatusShort = 1500;
constexpr int kStatusMedium = 2000;
constexpr int kStatusLong = 5000;
}

SegmentationModule::SegmentationModule(SegmentationWidget* widget,
                                       SegmentationEditManager* editManager,
                                       SegmentationOverlayController* overlay,
                                       ViewerManager* viewerManager,
                                       CSurfaceCollection* surfaces,
                                       bool editingEnabled,
                                       int downsample,
                                       float radius,
                                       float sigma,
                                       QObject* parent)
    : QObject(parent)
    , _widget(widget)
    , _editManager(editManager)
    , _overlay(overlay)
    , _viewerManager(viewerManager)
    , _surfaces(surfaces)
    , _editingEnabled(editingEnabled)
    , _downsample(downsample)
    , _radius(static_cast<float>(std::clamp(static_cast<int>(std::lround(radius)), 1, kMaxRadiusSteps)))
    , _sigma(std::clamp(sigma, 0.10f, 2.0f))
{
    if (_overlay && _editManager) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
        _overlay->setDownsample(_downsample);
        _overlay->setRadius(_radius);
    }

    bindWidgetSignals();

    if (_widget) {
        _holeSearchRadius = _widget->holeSearchRadius();
        _holeSmoothIterations = _widget->holeSmoothIterations();
        _showHandlesAlways = _widget->handlesAlwaysVisible();
        _handleDisplayDistance = _widget->handleDisplayDistance();
        _influenceMode = _widget->influenceMode();
    }
    if (_editManager) {
        _editManager->setHoleSearchRadius(_holeSearchRadius);
        _editManager->setHoleSmoothIterations(_holeSmoothIterations);
        _editManager->setInfluenceMode(_influenceMode);
    }
    if (_overlay) {
        _overlay->setHandleVisibility(_showHandlesAlways, _handleDisplayDistance);
        _overlay->setCursorWorld(cv::Vec3f(0, 0, 0), false);
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationOverlay(_overlay);
        _viewerManager->setSegmentationModule(this);
    }
}

void SegmentationModule::bindWidgetSignals()
{
    if (!_widget) {
        return;
    }

    connect(_widget, &SegmentationWidget::editingModeChanged,
            this, &SegmentationModule::setEditingEnabled,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::downsampleChanged,
            this, &SegmentationModule::setDownsample,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::radiusChanged,
            this, &SegmentationModule::setRadius,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::sigmaChanged,
            this, &SegmentationModule::setSigma,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::influenceModeChanged,
            this, &SegmentationModule::setInfluenceMode,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::holeSearchRadiusChanged,
            this, &SegmentationModule::setHoleSearchRadius,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::holeSmoothIterationsChanged,
            this, &SegmentationModule::setHoleSmoothIterations,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::handlesAlwaysVisibleChanged,
            this, &SegmentationModule::setHandlesAlwaysVisible,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::handleDisplayDistanceChanged,
            this, &SegmentationModule::setHandleDisplayDistance,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::applyRequested,
            this, &SegmentationModule::applyEdits,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::resetRequested,
            this, &SegmentationModule::resetEdits,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::stopToolsRequested,
            this, &SegmentationModule::stopTools,
            Qt::UniqueConnection);
}

void SegmentationModule::bindViewerSignals(CVolumeViewer* viewer)
{
    if (!viewer || viewer->property("vc_segmentation_bound").toBool()) {
        return;
    }

    connect(viewer, &CVolumeViewer::sendMousePressVolume,
            this, [this, viewer](cv::Vec3f worldPos, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                handleMousePress(viewer, worldPos, normal, button, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
            this, [this, viewer](cv::Vec3f worldPos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                handleMouseMove(viewer, worldPos, buttons, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
            this, [this, viewer](cv::Vec3f worldPos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                handleMouseRelease(viewer, worldPos, button, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendSegmentationRadiusWheel,
            this, [this, viewer](int steps, QPointF scenePoint, cv::Vec3f worldPos) {
                handleRadiusWheel(viewer, steps, scenePoint, worldPos);
            });

    viewer->setProperty("vc_segmentation_bound", true);
    viewer->setSegmentationEditActive(_editingEnabled);
}

void SegmentationModule::attachViewer(CVolumeViewer* viewer)
{
    bindViewerSignals(viewer);
    updateViewerCursors();
}

void SegmentationModule::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;

    if (_overlay) {
        _overlay->setEditingEnabled(enabled);
        _overlay->setDownsample(_downsample);
        _overlay->setRadius(_radius);
        _overlay->refreshAll();
    }

    if (!enabled) {
        resetInteractionState();
    } else {
        if (_viewerManager) {
            _viewerManager->forEachViewer([](CVolumeViewer* v) {
                if (v) {
                    v->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }
        updateViewerCursors();
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationEditActive(enabled);
    }

    emit editingEnabledChanged(enabled);
}

void SegmentationModule::setDownsample(int value)
{
    if (value == _downsample) {
        return;
    }
    _downsample = value;
    if (_widget && _widget->downsample() != value) {
        _widget->setDownsample(value);
    }
    if (_editManager && _editManager->hasSession()) {
        _editManager->setDownsample(value);
        emitPendingChanges();
    }
    if (_overlay) {
        _overlay->setDownsample(value);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setRadius(float radius)
{
    const int snapped = std::clamp(static_cast<int>(std::lround(radius)), 1, kMaxRadiusSteps);
    const float snappedRadius = static_cast<float>(snapped);
    if (std::fabs(snappedRadius - _radius) < 1e-4f) {
        return;
    }
    _radius = snappedRadius;

    if (_widget && std::fabs(_widget->radius() - _radius) > 1e-4f) {
        _widget->setRadius(_radius);
    }

    if (_editManager && _editManager->hasSession()) {
        _editManager->setRadius(_radius);
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
    }

    if (_overlay) {
        _overlay->setRadius(_radius);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setSigma(float sigma)
{
    const float clamped = std::clamp(sigma, 0.10f, 2.0f);
    if (std::fabs(clamped - _sigma) < 1e-4f) {
        return;
    }
    _sigma = clamped;

    if (_widget && std::fabs(_widget->sigma() - _sigma) > 1e-4f) {
        _widget->setSigma(_sigma);
    }

    if (_editManager && _editManager->hasSession()) {
        _editManager->setSigma(_sigma);
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
    }
}

void SegmentationModule::setInfluenceMode(SegmentationInfluenceMode mode)
{
    if (_influenceMode == mode) {
        return;
    }
    _influenceMode = mode;
    if (_editManager) {
        _editManager->setInfluenceMode(mode);
        if (_editManager->hasSession()) {
            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            emitPendingChanges();
            refreshOverlay();
        }
    }
}

void SegmentationModule::setHoleSearchRadius(int radius)
{
    const int clamped = std::clamp(radius, 1, 64);
    if (clamped == _holeSearchRadius) {
        return;
    }
    _holeSearchRadius = clamped;
    if (_widget && _widget->holeSearchRadius() != clamped) {
        _widget->setHoleSearchRadius(clamped);
    }
    if (_editManager) {
        _editManager->setHoleSearchRadius(_holeSearchRadius);
    }
}

void SegmentationModule::setHoleSmoothIterations(int iterations)
{
    const int clamped = std::clamp(iterations, 1, 200);
    if (clamped == _holeSmoothIterations) {
        return;
    }
    _holeSmoothIterations = clamped;
    if (_widget && _widget->holeSmoothIterations() != clamped) {
        _widget->setHoleSmoothIterations(clamped);
    }
    if (_editManager) {
        _editManager->setHoleSmoothIterations(_holeSmoothIterations);
    }
}

void SegmentationModule::setHandlesAlwaysVisible(bool value)
{
    if (_showHandlesAlways == value) {
        return;
    }
    _showHandlesAlways = value;
    if (_widget && _widget->handlesAlwaysVisible() != value) {
        _widget->setHandlesAlwaysVisible(value);
    }
    if (_overlay) {
        _overlay->setHandleVisibility(_showHandlesAlways, _handleDisplayDistance);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setHandleDisplayDistance(float distance)
{
    const float clamped = std::max(1.0f, distance);
    if (std::fabs(clamped - _handleDisplayDistance) < 1e-4f) {
        return;
    }
    _handleDisplayDistance = clamped;
    if (_widget && std::fabs(_widget->handleDisplayDistance() - clamped) > 1e-4f) {
        _widget->setHandleDisplayDistance(clamped);
    }
    if (_overlay) {
        _overlay->setHandleVisibility(_showHandlesAlways, _handleDisplayDistance);
        _overlay->refreshAll();
    }
}

void SegmentationModule::applyEdits()
{
    emit statusMessageRequested(tr("Applying segmentation edits"), kStatusMedium);
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    _editManager->applyPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();

    if (auto* base = _editManager->baseSurface()) {
        try {
            base->saveOverwrite();
        } catch (const std::exception& e) {
            emit statusMessageRequested(tr("Failed to save segmentation: ") + e.what(), kStatusLong);
        }
    }

    resetInteractionState();
}

void SegmentationModule::resetEdits()
{
    emit statusMessageRequested(tr("Segmentation edits reset"), kStatusMedium);
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    _editManager->resetPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();

    resetInteractionState();
}

void SegmentationModule::stopTools()
{
    emit stopToolsRequested();
}

void SegmentationModule::updateViewerCursors()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachViewer([this](CVolumeViewer* viewer) {
        if (!viewer || !viewer->fGraphicsView) {
            return;
        }
        if (!_editingEnabled) {
            viewer->fGraphicsView->unsetCursor();
            return;
        }
        if (_drag.active && viewer == _drag.viewer) {
            return;
        }
        if (_pointAddMode) {
            viewer->fGraphicsView->setCursor(addCursor());
        } else {
            viewer->fGraphicsView->setCursor(Qt::ArrowCursor);
        }
    });
}

void SegmentationModule::setPointAddMode(bool enabled, bool silent)
{
    if (_pointAddMode == enabled) {
        return;
    }
    _pointAddMode = enabled;
    updateViewerCursors();
    if (!silent) {
        const auto message = enabled ? tr("Segmentation point-add mode enabled")
                                     : tr("Segmentation point-add mode disabled");
        emit statusMessageRequested(message, kStatusShort);
    }
}

void SegmentationModule::togglePointAddMode()
{
    setPointAddMode(!_pointAddMode);
}

bool SegmentationModule::handleKeyPress(QKeyEvent* event)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        return false;
    }

    bool handled = false;
    if (event->key() == Qt::Key_Shift && event->modifiers() == Qt::ShiftModifier && !event->isAutoRepeat()) {
        togglePointAddMode();
        handled = true;
    } else if (event->key() == Qt::Key_R && event->modifiers() == Qt::NoModifier) {
        std::optional<std::pair<int, int>> target;
        if (_drag.active) {
            target = std::make_pair(_drag.row, _drag.col);
        } else if (_hover.valid) {
            target = std::make_pair(_hover.row, _hover.col);
        } else if (_cursorValid) {
            if (auto* nearest = _editManager->findNearestHandle(_cursorWorld, -1.0f)) {
                target = std::make_pair(nearest->row, nearest->col);
            }
        }

        if (target) {
            if (auto pos = _editManager->handleWorldPosition(target->first, target->second)) {
                if (_surfaces) {
                    POI* poi = _surfaces->poi("focus");
                    if (!poi) {
                        poi = new POI;
                    }
                    poi->p = *pos;
                    poi->src = _surfaces->surface("segmentation");
                    _surfaces->setPOI("focus", poi);
                }

                if (!_hover.valid || _hover.row != target->first || _hover.col != target->second) {
                    _hover.set(target->first, target->second, *pos);
                }
                if (_overlay) {
                    _overlay->setKeyboardHandle(target);
                    _overlay->setHoverHandle(target);
                }
                emit focusPoiRequested(*pos, _editManager->baseSurface());
                handled = true;
            }
        }
    } else if ((event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) && event->modifiers() == Qt::NoModifier) {
        std::optional<std::pair<int, int>> target;
        if (_drag.active) {
            target = std::make_pair(_drag.row, _drag.col);
        } else if (_hover.valid) {
            target = std::make_pair(_hover.row, _hover.col);
        }
        if (target) {
            if (_editManager->removeHandle(target->first, target->second)) {
                _hover.clear();
                _drag.reset();
                if (_surfaces) {
                    _surfaces->setSurface("segmentation", _editManager->previewSurface());
                }
                emitPendingChanges();
                if (_overlay) {
                    _overlay->setActiveHandle(std::nullopt, false);
                    _overlay->setHoverHandle(std::nullopt, false);
                    _overlay->setKeyboardHandle(std::nullopt, false);
                    _overlay->refreshAll();
                }
                handled = true;
            }
        }
    }

    if (handled) {
        event->accept();
    }
    return handled;
}

bool SegmentationModule::beginEditingSession(QuadSurface* activeSurface)
{
    if (!_editingEnabled || !_editManager || !activeSurface) {
        return false;
    }

    if (!_editManager->beginSession(activeSurface, _downsample)) {
        return false;
    }

    _editManager->setRadius(_radius);
    _editManager->setSigma(_sigma);
    _editManager->setDownsample(_downsample);
    _editManager->setHoleSearchRadius(_holeSearchRadius);
    _editManager->setHoleSmoothIterations(_holeSmoothIterations);
    _editManager->setInfluenceMode(_influenceMode);

    if (_surfaces) {
        if (auto* preview = _editManager->previewSurface()) {
            _surfaces->setSurface("segmentation", preview);
        }
    }

    emitPendingChanges();
    refreshOverlay();
    updateViewerCursors();

    return true;
}

void SegmentationModule::endEditingSession()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    QuadSurface* base = _editManager->baseSurface();
    _editManager->endSession();
    if (_surfaces && base) {
        _surfaces->setSurface("segmentation", base);
    }

    emitPendingChanges();
    resetInteractionState();
    updateViewerCursors();
}

bool SegmentationModule::hasActiveSession() const
{
    return _editManager && _editManager->hasSession();
}

void SegmentationModule::refreshOverlay()
{
    if (_overlay) {
        _overlay->refreshAll();
    }
}

void SegmentationModule::emitPendingChanges()
{
    if (_widget && _editManager) {
        const bool pending = _editManager->hasPendingChanges();
        _widget->setPendingChanges(pending);
        emit pendingChangesChanged(pending);
    }
}

void SegmentationModule::resetInteractionState()
{
    _drag.reset();
    _hover.clear();
    _cursorValid = false;

    if (_overlay) {
        _overlay->setActiveHandle(std::nullopt, false);
        _overlay->setHoverHandle(std::nullopt, false);
        _overlay->setKeyboardHandle(std::nullopt, false);
        _overlay->setCursorWorld(cv::Vec3f(0, 0, 0), false);
        _overlay->refreshAll();
    }

    if (_pointAddMode) {
        setPointAddMode(false, true);
    }

    if (_viewerManager) {
        _viewerManager->forEachViewer([](CVolumeViewer* v) {
            if (!v) {
                return;
            }
            v->clearOverlayGroup("segmentation_radius_indicator");
            if (v->fGraphicsView) {
                v->fGraphicsView->unsetCursor();
            }
        });
    }
}

float SegmentationModule::gridStepWorld() const
{
    if (_editManager && _editManager->hasSession()) {
        if (auto* surface = _editManager->baseSurface()) {
            const cv::Vec2f scale = surface->scale();
            const float sx = std::fabs(scale[0]);
            const float sy = std::fabs(scale[1]);
            const float step = std::max(sx, sy);
            if (std::isfinite(step) && step > 1e-4f) {
                return step;
            }
        }
    }
    return 1.0f;
}

float SegmentationModule::radiusWorldExtent(float gridRadius) const
{
    const float step = gridStepWorld();
    const float cells = std::max(gridRadius, 1.0f);
    const float baseExtent = (cells + 0.5f) * step;
    const float minExtent = std::max(step, 3.0f);
    return std::max(baseExtent, minExtent);
}

void SegmentationModule::handleMousePress(CVolumeViewer* viewer,
                                          const cv::Vec3f& worldPos,
                                          const cv::Vec3f& normal,
                                          Qt::MouseButton button,
                                          Qt::KeyboardModifiers modifiers)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        return;
    }
    if (button != Qt::LeftButton) {
        return;
    }

    PlaneSurface* planeSurface = viewer ? dynamic_cast<PlaneSurface*>(viewer->currentSurface()) : nullptr;
    Qt::KeyboardModifiers effectiveModifiers = modifiers & ~Qt::ShiftModifier;

    if (effectiveModifiers.testFlag(Qt::ControlModifier)) {
        const float tolerance = radiusWorldExtent(_radius);
        if (auto* handle = _editManager->findNearestHandle(worldPos, tolerance)) {
            if (_editManager->removeHandle(handle->row, handle->col)) {
                _hover.clear();
                emitPendingChanges();
                if (_surfaces) {
                    _surfaces->setSurface("segmentation", _editManager->previewSurface());
                }
                if (_overlay) {
                    _overlay->setActiveHandle(std::nullopt, false);
                    _overlay->setHoverHandle(std::nullopt, false);
                    _overlay->setKeyboardHandle(std::nullopt, false);
                    _overlay->refreshAll();
                }
            }
        }
        return;
    }
    if (!viewer) {
        return;
    }

    const float pickTolerance = radiusWorldExtent(_radius);
    const float addTolerance = _pointAddMode ? -1.0f : pickTolerance;
    const float addPlaneTolerance = planeSurface ? (_pointAddMode ? -1.0f : pickTolerance) : 0.0f;

    auto* handle = _pointAddMode ? nullptr : _editManager->findNearestHandle(worldPos, pickTolerance);
    if (!handle) {
        if (auto added = _editManager->addHandleAtWorld(worldPos, addTolerance, planeSurface, addPlaneTolerance, _pointAddMode)) {
            cv::Vec3f handleWorld = worldPos;
            if (auto world = _editManager->handleWorldPosition(added->first, added->second)) {
                handleWorld = *world;
            }
            _hover.set(added->first, added->second, handleWorld);
            emitPendingChanges();
            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            if (_overlay) {
                _overlay->setActiveHandle(*added);
                _overlay->setHoverHandle(*added);
                _overlay->setKeyboardHandle(*added, false);
                _overlay->refreshAll();
            }
        }
        return;
    }

    _drag.active = true;
    _drag.row = handle->row;
    _drag.col = handle->col;
    _drag.viewer = viewer;
    _drag.startWorld = handle->currentWorld;
    _drag.moved = false;

    _hover.clear();
    if (_overlay) {
        _overlay->setHoverHandle(std::nullopt);
        _overlay->setActiveHandle(std::make_pair(handle->row, handle->col));
    }

    if (viewer->fGraphicsView) {
        viewer->fGraphicsView->setCursor(Qt::ClosedHandCursor);
    }
}

void SegmentationModule::handleMouseMove(CVolumeViewer* viewer,
                                         const cv::Vec3f& worldPos,
                                         Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers /*modifiers*/)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        return;
    }

    _cursorWorld = worldPos;
    _cursorValid = true;
    if (_overlay) {
        _overlay->setCursorWorld(worldPos, true);
        if (!_showHandlesAlways) {
            _overlay->refreshViewer(viewer);
        }
    }

    if (_drag.active && viewer == _drag.viewer) {
        if (!(buttons & Qt::LeftButton)) {
            return;
        }

        const bool moved = _editManager->updateHandleWorldPosition(_drag.row, _drag.col, worldPos);
        if (!moved) {
            emit statusMessageRequested(tr("Handle move failed; see terminal for details"), kStatusMedium);
            return;
        }
        _drag.moved = true;
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
        if (auto world = _editManager->handleWorldPosition(_drag.row, _drag.col)) {
            _hover.set(_drag.row, _drag.col, *world);
        }
        if (_overlay) {
            _overlay->setActiveHandle(std::make_pair(_drag.row, _drag.col), false);
            _overlay->setHoverHandle(std::nullopt, false);
            _overlay->refreshAll();
        }
        return;
    }

    auto* handle = _editManager->findNearestHandle(worldPos, -1.0f);
    if (handle) {
        const bool changed = !_hover.valid || _hover.row != handle->row || _hover.col != handle->col;
        _hover.set(handle->row, handle->col, handle->currentWorld);
        if (_overlay) {
            _overlay->setHoverHandle(std::make_pair(handle->row, handle->col));
            if (changed) {
                _overlay->refreshViewer(viewer);
            }
        }
    } else if (_hover.valid) {
        _hover.clear();
        if (_overlay) {
            _overlay->setHoverHandle(std::nullopt);
            _overlay->refreshViewer(viewer);
        }
    }
}

void SegmentationModule::handleMouseRelease(CVolumeViewer* viewer,
                                            const cv::Vec3f& worldPos,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers /*modifiers*/)
{
    if (!_drag.active || viewer != _drag.viewer) {
        return;
    }
    if (button != Qt::LeftButton) {
        return;
    }

    if (viewer && viewer->fGraphicsView) {
        viewer->fGraphicsView->setCursor(Qt::ArrowCursor);
    }

    if (_pointAddMode && _editManager && _editManager->hasSession() && !_drag.moved) {
        const bool moved = _editManager->updateHandleWorldPosition(_drag.row, _drag.col, worldPos);
        if (!moved) {
            emit statusMessageRequested(tr("Handle move failed; see terminal for details"), kStatusMedium);
        }
        if (auto world = _editManager->handleWorldPosition(_drag.row, _drag.col)) {
            _hover.set(_drag.row, _drag.col, *world);
        }
    }

    if (_editManager && _editManager->hasSession()) {
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
    }

    emitPendingChanges();

    if (_overlay) {
        _overlay->setActiveHandle(std::nullopt, false);
        if (_hover.valid) {
            _overlay->setHoverHandle(std::make_pair(_hover.row, _hover.col));
        } else {
            _overlay->setHoverHandle(std::nullopt);
        }
    }

    _drag.reset();
    updateViewerCursors();
}

void SegmentationModule::handleRadiusWheel(CVolumeViewer* viewer,
                                           int steps,
                                           const QPointF& scenePoint,
                                           const cv::Vec3f& /*worldPos*/)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        return;
    }
    if (steps == 0) {
        return;
    }

    int deltaSteps = steps > 0 ? 1 : -1;
    if (QApplication::keyboardModifiers().testFlag(Qt::ControlModifier)) {
        deltaSteps *= 2;
    }

    const int currentSteps = std::clamp(static_cast<int>(std::lround(_radius)), 1, kMaxRadiusSteps);
    const int newSteps = std::clamp(currentSteps + deltaSteps, 1, kMaxRadiusSteps);
    if (newSteps == currentSteps) {
        return;
    }

    _radius = static_cast<float>(newSteps);
    if (_widget && std::fabs(_widget->radius() - _radius) > 1e-4f) {
        _widget->setRadius(_radius);
    }
    _editManager->setRadius(_radius);
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();

    if (_overlay) {
        _overlay->setRadius(_radius);
        _overlay->refreshAll();
    }

    if (viewer) {
        showRadiusIndicator(viewer, scenePoint, _radius);
    }
}

void SegmentationModule::showRadiusIndicator(CVolumeViewer* viewer,
                                             const QPointF& scenePoint,
                                             float radius)
{
    if (!viewer) {
        return;
    }

    viewer->clearOverlayGroup("segmentation_radius_indicator");

    const int steps = std::clamp(static_cast<int>(std::lround(radius)), 1, kMaxRadiusSteps);
    const QString label = steps == 1 ? tr("1 step") : tr("%1 steps").arg(steps);
    auto* textItem = new QGraphicsSimpleTextItem(label);
    QFont font = textItem->font();
    font.setPointSizeF(11.0);
    textItem->setFont(font);
    textItem->setBrush(QColor(255, 255, 255));
    textItem->setPen(QPen(Qt::black, 0.8));
    textItem->setZValue(150.0);

    const QPointF offset(12.0, -12.0);
    textItem->setPos(scenePoint + offset);

    viewer->setOverlayGroup("segmentation_radius_indicator", {textItem});

    QPointer<CVolumeViewer> guard(viewer);
    QTimer::singleShot(800, this, [guard]() {
        if (guard) {
            guard->clearOverlayGroup("segmentation_radius_indicator");
        }
    });
}

const QCursor& SegmentationModule::addCursor()
{
    static bool initialized = false;
    static QCursor cursor;
    if (!initialized) {
        QPixmap pixmap(32, 32);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing);
        QPen pen(Qt::white, 2);
        painter.setPen(pen);
        painter.drawEllipse(QPointF(16, 16), 12, 12);
        painter.drawLine(QPointF(16, 6), QPointF(16, 26));
        painter.drawLine(QPointF(6, 16), QPointF(26, 16));
        painter.end();
        cursor = QCursor(pixmap, 16, 16);
        initialized = true;
    }
    return cursor;
}
