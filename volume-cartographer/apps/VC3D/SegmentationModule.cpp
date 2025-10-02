#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"

#include <QCursor>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLoggingCategory>
#include <QPointer>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <optional>
#include <limits>
#include <unordered_set>
#include <utility>

Q_LOGGING_CATEGORY(lcSegModule, "vc.segmentation.module")

namespace
{
constexpr int kStatusShort = 1500;
constexpr int kStatusMedium = 2000;
constexpr int kStatusLong = 5000;
constexpr int kMaxUndoStates = 5;
constexpr float kMaxBrushSampleSpacing = 0.5f;

float averageScale(const cv::Vec2f& scale)
{
    const float sx = std::abs(scale[0]);
    const float sy = std::abs(scale[1]);
    const float avg = 0.5f * (sx + sy);
    return (avg > 1e-4f) ? avg : 1.0f;
}
}

void SegmentationModule::DragState::reset()
{
    active = false;
    row = 0;
    col = 0;
    startWorld = cv::Vec3f(0.0f, 0.0f, 0.0f);
    lastWorld = cv::Vec3f(0.0f, 0.0f, 0.0f);
    viewer = nullptr;
    moved = false;
}

void SegmentationModule::HoverState::set(int r, int c, const cv::Vec3f& w, CVolumeViewer* v)
{
    valid = true;
    row = r;
    col = c;
    world = w;
    viewer = v;
}

void SegmentationModule::HoverState::clear()
{
    valid = false;
    viewer = nullptr;
}

SegmentationModule::SegmentationModule(SegmentationWidget* widget,
                                       SegmentationEditManager* editManager,
                                       SegmentationOverlayController* overlay,
                                       ViewerManager* viewerManager,
                                       CSurfaceCollection* surfaces,
                                       VCCollection* pointCollection,
                                       bool editingEnabled,
                                       float radiusSteps,
                                       float sigmaSteps,
                                       QObject* parent)
    : QObject(parent)
    , _widget(widget)
    , _editManager(editManager)
    , _overlay(overlay)
    , _viewerManager(viewerManager)
    , _surfaces(surfaces)
    , _pointCollection(pointCollection)
    , _editingEnabled(editingEnabled)
    , _radiusSteps(radiusSteps)
    , _sigmaSteps(sigmaSteps)
    , _growthMethod(_widget ? _widget->growthMethod() : SegmentationGrowthMethod::Tracer)
    , _growthSteps(_widget ? _widget->growthSteps() : 5)
{
    if (_widget) {
        _pushPullStepMultiplier = std::clamp(_widget->pushPullStep(), 0.05f, 10.0f);
    }

    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
    }

    if (_editManager) {
        _editManager->setRadius(_radiusSteps);
        _editManager->setSigma(_sigmaSteps);
    }

    _pushPullTimer = new QTimer(this);
    _pushPullTimer->setInterval(30);
    connect(_pushPullTimer, &QTimer::timeout,
            this, &SegmentationModule::onPushPullTick);

    bindWidgetSignals();

    if (_viewerManager) {
        _viewerManager->setSegmentationModule(this);
    }

    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationModule::onSurfaceCollectionChanged);
    }

    if (_pointCollection) {
        const auto& collections = _pointCollection->getAllCollections();
        for (const auto& entry : collections) {
            _pendingCorrectionIds.push_back(entry.first);
        }

        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t id) {
            const uint64_t sentinel = std::numeric_limits<uint64_t>::max();
            if (id == sentinel) {
                _pendingCorrectionIds.clear();
                _managedCorrectionIds.clear();
                _activeCorrectionId = 0;
                setCorrectionsAnnotateMode(false, false);
            } else {
                auto eraseIt = std::remove(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), id);
                if (eraseIt != _pendingCorrectionIds.end()) {
                    _pendingCorrectionIds.erase(eraseIt, _pendingCorrectionIds.end());
                    _managedCorrectionIds.erase(id);
                    if (_activeCorrectionId == id) {
                        _activeCorrectionId = 0;
                        setCorrectionsAnnotateMode(false, false);
                    }
                }
            }
            updateCorrectionsWidget();
        });

        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t id) {
            if (id == std::numeric_limits<uint64_t>::max()) {
                return;
            }
            if (std::find(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), id) != _pendingCorrectionIds.end()) {
                updateCorrectionsWidget();
            }
        });
    }

    updateCorrectionsWidget();

    if (_widget) {
        if (auto range = _widget->correctionsZRange()) {
            onCorrectionsZRangeChanged(true, range->first, range->second);
        } else {
            onCorrectionsZRangeChanged(false, 0, 0);
        }
    }
}

void SegmentationModule::setRotationHandleHitTester(std::function<bool(CVolumeViewer*, const cv::Vec3f&)> tester)
{
    _rotationHandleHitTester = std::move(tester);
}

void SegmentationModule::bindWidgetSignals()
{
    if (!_widget) {
        return;
    }

    connect(_widget, &SegmentationWidget::editingModeChanged,
            this, &SegmentationModule::setEditingEnabled);
    connect(_widget, &SegmentationWidget::radiusChanged,
            this, &SegmentationModule::setRadius);
    connect(_widget, &SegmentationWidget::sigmaChanged,
            this, &SegmentationModule::setSigma);
    connect(_widget, &SegmentationWidget::applyRequested,
            this, &SegmentationModule::applyEdits);
    connect(_widget, &SegmentationWidget::resetRequested,
            this, &SegmentationModule::resetEdits);
    connect(_widget, &SegmentationWidget::stopToolsRequested,
            this, &SegmentationModule::stopTools);
    connect(_widget, &SegmentationWidget::growSurfaceRequested,
            this, &SegmentationModule::handleGrowSurfaceRequested);
    connect(_widget, &SegmentationWidget::growthMethodChanged,
            this, [this](SegmentationGrowthMethod method) {
                _growthMethod = method;
            });
    connect(_widget, &SegmentationWidget::pushPullStepChanged,
            this, &SegmentationModule::setPushPullStepMultiplier);
    connect(_widget, &SegmentationWidget::correctionsCreateRequested,
            this, &SegmentationModule::onCorrectionsCreateRequested);
    connect(_widget, &SegmentationWidget::correctionsCollectionSelected,
            this, &SegmentationModule::onCorrectionsCollectionSelected);
    connect(_widget, &SegmentationWidget::correctionsAnnotateToggled,
            this, &SegmentationModule::onCorrectionsAnnotateToggled);
    connect(_widget, &SegmentationWidget::correctionsZRangeChanged,
            this, &SegmentationModule::onCorrectionsZRangeChanged);

    _widget->setEraseBrushActive(false);
}

void SegmentationModule::bindViewerSignals(CVolumeViewer* viewer)
{
    if (!viewer || viewer->property("vc_segmentation_bound").toBool()) {
        return;
    }

    connect(viewer, &CVolumeViewer::sendMousePressVolume,
            this, [this, viewer](const cv::Vec3f& worldPos,
                                 const cv::Vec3f& normal,
                                 Qt::MouseButton button,
                                 Qt::KeyboardModifiers modifiers) {
                handleMousePress(viewer, worldPos, normal, button, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
            this, [this, viewer](const cv::Vec3f& worldPos,
                                 Qt::MouseButtons buttons,
                                 Qt::KeyboardModifiers modifiers) {
                handleMouseMove(viewer, worldPos, buttons, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
            this, [this, viewer](const cv::Vec3f& worldPos,
                                 Qt::MouseButton button,
                                 Qt::KeyboardModifiers modifiers) {
                handleMouseRelease(viewer, worldPos, button, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendSegmentationRadiusWheel,
            this, [this, viewer](int steps, const QPointF& scenePoint, const cv::Vec3f& worldPos) {
                handleWheel(viewer, steps, scenePoint, worldPos);
            });

    viewer->setProperty("vc_segmentation_bound", true);
    viewer->setSegmentationEditActive(_editingEnabled);
    _attachedViewers.insert(viewer);
}

void SegmentationModule::attachViewer(CVolumeViewer* viewer)
{
    bindViewerSignals(viewer);
    updateViewerCursors();
}

void SegmentationModule::updateViewerCursors()
{
    for (auto* viewer : std::as_const(_attachedViewers)) {
        if (!viewer) {
            continue;
        }
        viewer->setSegmentationEditActive(_editingEnabled);
    }
}

void SegmentationModule::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;

    if (_overlay) {
        _overlay->setEditingEnabled(enabled);
    }
    updateViewerCursors();
    if (!enabled) {
        stopAllPushPull();
        setCorrectionsAnnotateMode(false, false);
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
        clearUndoStack();
    }
    updateCorrectionsWidget();
    emit editingEnabledChanged(enabled);
}

void SegmentationModule::setRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (std::fabs(sanitized - _radiusSteps) < 1e-4f) {
        return;
    }
    _radiusSteps = sanitized;
    if (_editManager) {
        _editManager->setRadius(_radiusSteps);
    }
    if (_overlay) {
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
        _overlay->refreshAll();
    }
    if (_widget) {
        _widget->setRadius(_radiusSteps);
    }
}

void SegmentationModule::setSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (std::fabs(sanitized - _sigmaSteps) < 1e-4f) {
        return;
    }
    _sigmaSteps = sanitized;
    if (_editManager) {
        _editManager->setSigma(_sigmaSteps);
    }
    if (_overlay) {
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
        _overlay->refreshAll();
    }
    if (_widget) {
        _widget->setSigma(_sigmaSteps);
    }
}

void SegmentationModule::setPushPullStepMultiplier(float multiplier)
{
    const float sanitized = std::clamp(multiplier, 0.05f, 10.0f);
    if (std::fabs(sanitized - _pushPullStepMultiplier) < 1e-4f) {
        return;
    }
    _pushPullStepMultiplier = sanitized;
    if (_widget) {
        _widget->setPushPullStep(_pushPullStepMultiplier);
    }
}

void SegmentationModule::applyEdits()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    const bool hadPendingChanges = _editManager->hasPendingChanges();
    if (hadPendingChanges) {
        if (!captureUndoSnapshot()) {
            qCWarning(lcSegModule) << "Failed to capture undo snapshot before applying edits.";
        }
    }
    clearInvalidationBrush();
    _editManager->applyPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();
    if (hadPendingChanges) {
        emit statusMessageRequested(tr("Applied segmentation edits."), kStatusShort);
    }
}

void SegmentationModule::resetEdits()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    const bool hadPendingChanges = _editManager->hasPendingChanges();
    if (hadPendingChanges) {
        if (!captureUndoSnapshot()) {
            qCWarning(lcSegModule) << "Failed to capture undo snapshot before resetting edits.";
        }
    }
    cancelDrag();
    clearInvalidationBrush();
    _editManager->resetPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    refreshOverlay();
    emitPendingChanges();
    if (hadPendingChanges) {
        emit statusMessageRequested(tr("Reset pending segmentation edits."), kStatusShort);
    }
}

void SegmentationModule::stopTools()
{
    cancelDrag();
    emit stopToolsRequested();
}

bool SegmentationModule::beginEditingSession(QuadSurface* surface)
{
    if (!_editManager || !surface) {
        return false;
    }

    stopAllPushPull();
    clearUndoStack();
    clearInvalidationBrush();
    setInvalidationBrushActive(false);
    if (!_editManager->beginSession(surface)) {
        qCWarning(lcSegModule) << "Failed to begin segmentation editing session";
        return false;
    }

    _editManager->setRadius(_radiusSteps);
    _editManager->setSigma(_sigmaSteps);

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    if (_overlay) {
        _overlay->setEditingEnabled(_editingEnabled);
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
        refreshOverlay();
    }

    emitPendingChanges();
    return true;
}

void SegmentationModule::endEditingSession()
{
    stopAllPushPull();
    clearUndoStack();
    cancelDrag();
    clearInvalidationBrush();
    setInvalidationBrushActive(false);
    if (_overlay) {
        _overlay->setActiveVertex(std::nullopt);
        _overlay->setTouchedVertices({});
        _overlay->refreshAll();
    }
    QuadSurface* baseSurface = _editManager ? _editManager->baseSurface() : nullptr;
    QuadSurface* previewSurface = _editManager ? _editManager->previewSurface() : nullptr;

    if (_surfaces && previewSurface) {
        Surface* currentSurface = _surfaces->surface("segmentation");
        if (currentSurface == previewSurface) {
            const bool previousGuard = _ignoreSegSurfaceChange;
            _ignoreSegSurfaceChange = true;
            _surfaces->setSurface("segmentation", baseSurface);
            _ignoreSegSurfaceChange = previousGuard;
        }
    }

    if (_editManager) {
        _editManager->endSession();
    }
}

void SegmentationModule::onSurfaceCollectionChanged(std::string name, Surface* surface)
{
    if (name != "segmentation" || !_editingEnabled || _ignoreSegSurfaceChange) {
        return;
    }

    if (!_editManager) {
        setEditingEnabled(false);
        return;
    }

    QuadSurface* previewSurface = _editManager->previewSurface();
    QuadSurface* baseSurface = _editManager->baseSurface();

    if (surface == previewSurface || surface == baseSurface) {
        return;
    }

    qCInfo(lcSegModule) << "Segmentation surface changed externally; disabling editing.";
    emit statusMessageRequested(tr("Segmentation editing disabled because the surface changed."),
                                kStatusMedium);
    setEditingEnabled(false);
}

bool SegmentationModule::captureUndoSnapshot()
{
    if (_suppressUndoCapture) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    const auto& previewPoints = _editManager->previewPoints();
    if (previewPoints.empty()) {
        return false;
    }

    UndoState state;
    state.points = previewPoints.clone();
    if (state.points.empty()) {
        return false;
    }

    if (_undoStack.size() >= static_cast<std::size_t>(kMaxUndoStates)) {
        _undoStack.pop_front();
    }
    _undoStack.push_back(std::move(state));
    return true;
}

void SegmentationModule::discardLastUndoSnapshot()
{
    if (!_undoStack.empty()) {
        _undoStack.pop_back();
    }
}

bool SegmentationModule::restoreUndoSnapshot()
{
    if (_undoStack.empty()) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    UndoState state = std::move(_undoStack.back());
    _undoStack.pop_back();
    if (state.points.empty()) {
        return false;
    }

    _suppressUndoCapture = true;
    bool applied = _editManager->setPreviewPoints(state.points, false);
    if (applied) {
        _editManager->applyPreview();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        clearInvalidationBrush();
        refreshOverlay();
        emitPendingChanges();
        _pushPullUndoCaptured = false;
    } else {
        _undoStack.push_back(std::move(state));
    }
    _suppressUndoCapture = false;

    return applied;
}

void SegmentationModule::clearUndoStack()
{
    _undoStack.clear();
    _pushPullUndoCaptured = false;
}

bool SegmentationModule::hasActiveSession() const
{
    return _editManager && _editManager->hasSession();
}

QuadSurface* SegmentationModule::activeBaseSurface() const
{
    return _editManager ? _editManager->baseSurface() : nullptr;
}

void SegmentationModule::refreshSessionFromSurface(QuadSurface* surface)
{
    if (!_editManager || !surface) {
        return;
    }
    if (_editManager->baseSurface() != surface) {
        return;
    }
    cancelDrag();
    _editManager->clearInvalidatedEdits();
    _editManager->refreshFromBaseSurface();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    refreshOverlay();
    emitPendingChanges();
}

bool SegmentationModule::handleKeyPress(QKeyEvent* event)
{
    if (!event) {
        return false;
    }

    if (!event->isAutoRepeat()) {
        const bool undoRequested = (event->matches(QKeySequence::Undo) == QKeySequence::ExactMatch) ||
                                   (event->key() == Qt::Key_Z && event->modifiers().testFlag(Qt::ControlModifier));
        if (undoRequested) {
            if (restoreUndoSnapshot()) {
                emit statusMessageRequested(tr("Undid last segmentation change."), kStatusShort);
                event->accept();
                return true;
            }
            return false;
        }
    }

    if (event->key() == Qt::Key_Shift && !event->isAutoRepeat()) {
        setInvalidationBrushActive(true);
        event->accept();
        return true;
    }

    if (event->key() == Qt::Key_E && !event->isAutoRepeat()) {
        const Qt::KeyboardModifiers mods = event->modifiers();
        if (mods == Qt::NoModifier || mods == Qt::ShiftModifier) {
            if (applyInvalidationBrush()) {
                event->accept();
                return true;
            }
        }
    }

    if (event->key() == Qt::Key_Escape) {
        if (_drag.active) {
            cancelDrag();
            return true;
        }
    }

    if ((event->key() == Qt::Key_F || event->key() == Qt::Key_G) && event->modifiers() == Qt::NoModifier) {
        if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
            return false;
        }

        const int direction = (event->key() == Qt::Key_G) ? 1 : -1;
        if (startPushPull(direction)) {
            event->accept();
            return true;
        }
        return false;
    }

    if (event->key() == Qt::Key_T && event->modifiers() == Qt::NoModifier) {
        onCorrectionsCreateRequested();
        event->accept();
        return true;
    }

    if (event->modifiers() == Qt::NoModifier && !event->isAutoRepeat()) {
        // Directional growth shortcuts (1-5).
        SegmentationGrowthDirection shortcutDirection{SegmentationGrowthDirection::All};
        bool matchedShortcut = true;
        switch (event->key()) {
        case Qt::Key_1:
            shortcutDirection = SegmentationGrowthDirection::Left;
            break;
        case Qt::Key_2:
            shortcutDirection = SegmentationGrowthDirection::Up;
            break;
        case Qt::Key_3:
            shortcutDirection = SegmentationGrowthDirection::Down;
            break;
        case Qt::Key_4:
            shortcutDirection = SegmentationGrowthDirection::Right;
            break;
        case Qt::Key_5:
            shortcutDirection = SegmentationGrowthDirection::All;
            break;
        default:
            matchedShortcut = false;
            break;
        }

        if (matchedShortcut) {
            _pendingShortcutDirections = std::vector<SegmentationGrowthDirection>{shortcutDirection};
            const int steps = _widget ? std::max(1, _widget->growthSteps()) : std::max(1, _growthSteps);
            const SegmentationGrowthMethod method = _widget ? _widget->growthMethod() : _growthMethod;
            handleGrowSurfaceRequested(method, shortcutDirection, steps);
            event->accept();
            return true;
        }
    }

    return false;
}

std::optional<std::vector<SegmentationGrowthDirection>> SegmentationModule::takeShortcutDirectionOverride()
{
    if (!_pendingShortcutDirections) {
        return std::nullopt;
    }
    auto result = std::move(*_pendingShortcutDirections);
    _pendingShortcutDirections.reset();
    return result;
}

bool SegmentationModule::handleKeyRelease(QKeyEvent* event)
{
    if (!event) {
        return false;
    }

    if (event->key() == Qt::Key_Shift && !event->isAutoRepeat()) {
        setInvalidationBrushActive(false);
        event->accept();
        return true;
    }

    if ((event->key() == Qt::Key_F || event->key() == Qt::Key_G) && event->modifiers() == Qt::NoModifier) {
        const int direction = (event->key() == Qt::Key_G) ? 1 : -1;
        stopPushPull(direction);
        event->accept();
        return true;
    }

    return false;
}

void SegmentationModule::markNextEditsFromGrowth()
{
    if (_editManager) {
        _editManager->markNextEditsAsGrowth();
    }
}

void SegmentationModule::setGrowthInProgress(bool running)
{
    _growthInProgress = running;
    if (_widget) {
        _widget->setGrowthInProgress(running);
    }
    if (running) {
        setCorrectionsAnnotateMode(false, false);
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
    }
    updateCorrectionsWidget();
}

SegmentationCorrectionsPayload SegmentationModule::buildCorrectionsPayload() const
{
    SegmentationCorrectionsPayload payload;
    if (!_pointCollection) {
        return payload;
    }

    const auto& collections = _pointCollection->getAllCollections();
    for (uint64_t id : _pendingCorrectionIds) {
        auto it = collections.find(id);
        if (it == collections.end()) {
            continue;
        }

        SegmentationCorrectionsPayload::Collection entry;
        entry.id = it->second.id;
        entry.name = it->second.name;
        entry.metadata = it->second.metadata;
        entry.color = it->second.color;

        std::vector<ColPoint> points;
        points.reserve(it->second.points.size());
        for (const auto& pair : it->second.points) {
            points.push_back(pair.second);
        }
        std::sort(points.begin(), points.end(), [](const ColPoint& a, const ColPoint& b) {
            return a.id < b.id;
        });
        if (points.empty()) {
            continue;
        }
        entry.points = std::move(points);
        payload.collections.push_back(std::move(entry));
    }

    return payload;
}

void SegmentationModule::clearPendingCorrections()
{
    setCorrectionsAnnotateMode(false, false);

    if (_pointCollection) {
        for (uint64_t id : _pendingCorrectionIds) {
            if (_managedCorrectionIds.count(id) > 0) {
                _pointCollection->clearCollection(id);
            }
        }
    }

    _pendingCorrectionIds.clear();
    _managedCorrectionIds.clear();
    _activeCorrectionId = 0;
    updateCorrectionsWidget();
}

std::optional<std::pair<int, int>> SegmentationModule::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

void SegmentationModule::emitPendingChanges()
{
    if (!_widget || !_editManager) {
        return;
    }
    const bool pending = _editManager->hasPendingChanges();
    _widget->setPendingChanges(pending);
    emit pendingChangesChanged(pending);
}

void SegmentationModule::refreshOverlay()
{
    if (!_overlay || !_editManager) {
        return;
    }

    std::optional<SegmentationOverlayController::VertexMarker> activeMarker;
    if (_drag.active) {
        if (auto world = _editManager->vertexWorldPosition(_drag.row, _drag.col)) {
            activeMarker = SegmentationOverlayController::VertexMarker{
                .row = _drag.row,
                .col = _drag.col,
                .world = *world,
                .isActive = true,
                .isGrowth = false
            };
        }
    } else if (_hover.valid) {
        activeMarker = SegmentationOverlayController::VertexMarker{
            .row = _hover.row,
            .col = _hover.col,
            .world = _hover.world,
            .isActive = false,
            .isGrowth = false
        };
    }

    std::vector<SegmentationOverlayController::VertexMarker> neighbours;
    if (_editManager && _drag.active) {
        const auto touched = _editManager->recentTouched();
        neighbours.reserve(touched.size());
        for (const auto& key : touched) {
            if (key.row == _drag.row && key.col == _drag.col) {
                continue;
            }
            if (auto world = _editManager->vertexWorldPosition(key.row, key.col)) {
                neighbours.push_back({key.row, key.col, *world, false, false});
            }
        }
    }

    std::vector<cv::Vec3f> maskPoints;
    maskPoints.reserve(_paintOverlayPoints.size() + _currentPaintStroke.size());
    maskPoints.insert(maskPoints.end(), _paintOverlayPoints.begin(), _paintOverlayPoints.end());
    maskPoints.insert(maskPoints.end(), _currentPaintStroke.begin(), _currentPaintStroke.end());

    const bool maskVisible = !maskPoints.empty();
    const float brushPixelRadius = std::clamp(_radiusSteps * 1.5f, 3.0f, 18.0f);
    const float brushOpacity = _invalidationBrushActive ? 0.6f : 0.45f;

    _overlay->setActiveVertex(activeMarker);
    _overlay->setTouchedVertices(neighbours);
    if (maskVisible) {
        _overlay->setMaskOverlay(maskPoints, true, brushPixelRadius, brushOpacity);
    } else {
        _overlay->setMaskOverlay({}, false, 0.0f, 0.0f);
    }
    _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
    _overlay->refreshAll();
}

void SegmentationModule::refreshMaskOverlay()
{
    if (_overlay) {
        _overlay->setMaskOverlay({}, false, 0.0f, 0.0f);
    }
}

void SegmentationModule::updateCorrectionsWidget()
{
    if (!_widget) {
        return;
    }

    pruneMissingCorrections();

    const bool correctionsAvailable = (_pointCollection != nullptr) && !_growthInProgress;
    QVector<QPair<uint64_t, QString>> entries;
    if (_pointCollection) {
        const auto& collections = _pointCollection->getAllCollections();
        entries.reserve(static_cast<int>(_pendingCorrectionIds.size()));
        for (uint64_t id : _pendingCorrectionIds) {
            auto it = collections.find(id);
            if (it != collections.end()) {
                entries.append({id, QString::fromStdString(it->second.name)});
            }
        }
    }

    std::optional<uint64_t> active;
    if (_activeCorrectionId != 0) {
        active = _activeCorrectionId;
    }

    _widget->setCorrectionCollections(entries, active);
    _widget->setCorrectionsEnabled(correctionsAvailable);
    _widget->setCorrectionsAnnotateChecked(_correctionsAnnotateMode && correctionsAvailable);
}

void SegmentationModule::setCorrectionsAnnotateMode(bool enabled, bool userInitiated)
{
    if (!_pointCollection || _growthInProgress || !_editingEnabled) {
        enabled = false;
    }

    if (enabled && _activeCorrectionId == 0) {
        if (!createCorrectionCollection(false)) {
            enabled = false;
        }
    }

    if (_correctionsAnnotateMode == enabled) {
        updateCorrectionsWidget();
        return;
    }

    _correctionsAnnotateMode = enabled;
    if (_widget) {
        _widget->setCorrectionsAnnotateChecked(enabled);
    }

    if (enabled) {
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
    }

    if (userInitiated) {
        const QString message = enabled ? tr("Correction annotation enabled")
                                        : tr("Correction annotation disabled");
        emit statusMessageRequested(message, kStatusShort);
    }

    updateCorrectionsWidget();
}

void SegmentationModule::setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated)
{
    if (!_pointCollection) {
        return;
    }

    if (collectionId == 0) {
        _activeCorrectionId = 0;
        setCorrectionsAnnotateMode(false, false);
        updateCorrectionsWidget();
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    if (collections.find(collectionId) == collections.end()) {
        pruneMissingCorrections();
        emit statusMessageRequested(tr("Selected correction set no longer exists."), kStatusShort);
        updateCorrectionsWidget();
        return;
    }

    if (std::find(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), collectionId) == _pendingCorrectionIds.end()) {
        _pendingCorrectionIds.push_back(collectionId);
    }

    _activeCorrectionId = collectionId;

    if (userInitiated) {
        emit statusMessageRequested(tr("Active correction set changed."), kStatusShort);
    }

    updateCorrectionsWidget();
}

uint64_t SegmentationModule::createCorrectionCollection(bool announce)
{
    if (!_pointCollection) {
        return 0;
    }

    const std::string newName = _pointCollection->generateNewCollectionName("correction");
    const uint64_t newId = _pointCollection->addCollection(newName);
    if (newId == 0) {
        return 0;
    }

    if (std::find(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), newId) == _pendingCorrectionIds.end()) {
        _pendingCorrectionIds.push_back(newId);
    }
    _managedCorrectionIds.insert(newId);
    _activeCorrectionId = newId;

    if (announce) {
        emit statusMessageRequested(tr("Created correction set '%1'.").arg(QString::fromStdString(newName)), kStatusShort);
    }

    updateCorrectionsWidget();
    return newId;
}

void SegmentationModule::handleCorrectionPointAdded(const cv::Vec3f& worldPos)
{
    if (!_pointCollection || _activeCorrectionId == 0) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    auto it = collections.find(_activeCorrectionId);
    if (it == collections.end()) {
        pruneMissingCorrections();
        updateCorrectionsWidget();
        return;
    }

    _pointCollection->addPoint(it->second.name, worldPos);
}

void SegmentationModule::handleCorrectionPointRemove(const cv::Vec3f& worldPos)
{
    if (!_pointCollection || _activeCorrectionId == 0) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    auto it = collections.find(_activeCorrectionId);
    if (it == collections.end()) {
        pruneMissingCorrections();
        updateCorrectionsWidget();
        return;
    }

    const auto& points = it->second.points;
    if (points.empty()) {
        return;
    }

    uint64_t closestId = 0;
    float closestDistance = std::numeric_limits<float>::max();
    for (const auto& entry : points) {
        const float dist = cv::norm(entry.second.p - worldPos);
        if (dist < closestDistance) {
            closestDistance = dist;
            closestId = entry.second.id;
        }
    }

    if (closestId != 0) {
        _pointCollection->removePoint(closestId);
    }
}

void SegmentationModule::pruneMissingCorrections()
{
    if (!_pointCollection) {
        _pendingCorrectionIds.clear();
        _managedCorrectionIds.clear();
        _activeCorrectionId = 0;
        _correctionsAnnotateMode = false;
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    auto endIt = std::remove_if(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), [&](uint64_t id) {
        const bool missing = collections.find(id) == collections.end();
        if (missing) {
            _managedCorrectionIds.erase(id);
            if (_activeCorrectionId == id) {
                _activeCorrectionId = 0;
                _correctionsAnnotateMode = false;
            }
        }
        return missing;
    });
    _pendingCorrectionIds.erase(endIt, _pendingCorrectionIds.end());

    if (_activeCorrectionId != 0 && collections.find(_activeCorrectionId) == collections.end()) {
        _activeCorrectionId = 0;
        _correctionsAnnotateMode = false;
    }
}

void SegmentationModule::onCorrectionsCreateRequested()
{
    if (createCorrectionCollection(true) != 0) {
        setCorrectionsAnnotateMode(true, false);
    }
}

void SegmentationModule::onCorrectionsCollectionSelected(uint64_t id)
{
    setActiveCorrectionCollection(id, true);
}

void SegmentationModule::onCorrectionsAnnotateToggled(bool enabled)
{
    setCorrectionsAnnotateMode(enabled, true);
}

void SegmentationModule::onCorrectionsZRangeChanged(bool enabled, int zMin, int zMax)
{
    _correctionsZRangeEnabled = enabled;
    if (zMin > zMax) {
        std::swap(zMin, zMax);
    }
    _correctionsZMin = zMin;
    _correctionsZMax = zMax;
    if (enabled) {
        _correctionsRange = std::make_pair(_correctionsZMin, _correctionsZMax);
    } else {
        _correctionsRange.reset();
    }
}

void SegmentationModule::handleGrowSurfaceRequested(SegmentationGrowthMethod method,
                                                    SegmentationGrowthDirection direction,
                                                    int steps)
{
    qCInfo(lcSegModule) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << steps;

    if (_growthInProgress) {
        emit statusMessageRequested(tr("Surface growth already in progress"), kStatusMedium);
        return;
    }
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        emit statusMessageRequested(tr("Enable segmentation editing before growing surfaces"), kStatusMedium);
        return;
    }

    // Ensure any pending invalidation brush strokes are committed before growth.
    applyInvalidationBrush();

    _growthMethod = method;
    _growthSteps = std::max(1, steps);
    markNextEditsFromGrowth();
    emit growSurfaceRequested(method, direction, _growthSteps);
}

void SegmentationModule::setInvalidationBrushActive(bool active)
{
    const bool shouldEnable = active && _editingEnabled && !_growthInProgress && !_correctionsAnnotateMode &&
                              _editManager && _editManager->hasSession();
    if (_invalidationBrushActive == shouldEnable) {
        if (_widget) {
            _widget->setEraseBrushActive(shouldEnable);
        }
        return;
    }

    _invalidationBrushActive = shouldEnable;
    if (!_invalidationBrushActive) {
        _hasLastPaintSample = false;
    }

    if (_widget) {
        _widget->setEraseBrushActive(_invalidationBrushActive);
    }

    refreshOverlay();
}

void SegmentationModule::clearInvalidationBrush()
{
    _paintStrokeActive = false;
    _currentPaintStroke.clear();
    _pendingPaintStrokes.clear();
    _paintOverlayPoints.clear();
    _hasLastPaintSample = false;

    if (!_invalidationBrushActive && _widget) {
        _widget->setEraseBrushActive(false);
    }

    refreshOverlay();
}

void SegmentationModule::startPaintStroke(const cv::Vec3f& worldPos)
{
    _paintStrokeActive = true;
    _currentPaintStroke.clear();
    _currentPaintStroke.push_back(worldPos);
    _paintOverlayPoints.push_back(worldPos);
    _lastPaintSample = worldPos;
    _hasLastPaintSample = true;
    refreshOverlay();
}

void SegmentationModule::extendPaintStroke(const cv::Vec3f& worldPos, bool forceSample)
{
    if (!_paintStrokeActive) {
        return;
    }

    const float baseSpacing = std::max(gridStepWorld() * 0.3f, 0.25f);
    const float spacing = std::min(baseSpacing, kMaxBrushSampleSpacing);
    const float spacingSq = spacing * spacing;

    if (_hasLastPaintSample) {
        const cv::Vec3f delta = worldPos - _lastPaintSample;
        const float distanceSq = delta.dot(delta);
        if (!forceSample && distanceSq < spacingSq) {
            return;
        }

        const float distance = std::sqrt(distanceSq);
        if (distance > spacing) {
            const cv::Vec3f direction = delta / distance;
            float travelled = spacing;
            while (travelled < distance) {
                const cv::Vec3f intermediate = _lastPaintSample + direction * travelled;
                _currentPaintStroke.push_back(intermediate);
                _paintOverlayPoints.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _currentPaintStroke.push_back(worldPos);
    _paintOverlayPoints.push_back(worldPos);
    _lastPaintSample = worldPos;
    _hasLastPaintSample = true;
    refreshOverlay();
}

void SegmentationModule::finishPaintStroke()
{
    if (!_paintStrokeActive) {
        return;
    }

    _paintStrokeActive = false;
    if (!_currentPaintStroke.empty()) {
        _pendingPaintStrokes.push_back(_currentPaintStroke);
    }
    _currentPaintStroke.clear();
    _hasLastPaintSample = false;
    refreshOverlay();
}

bool SegmentationModule::applyInvalidationBrush()
{
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (_paintStrokeActive) {
        finishPaintStroke();
    }

    if (_pendingPaintStrokes.empty()) {
        return false;
    }

    using GridKey = SegmentationEditManager::GridKey;
    using GridKeyHash = SegmentationEditManager::GridKeyHash;

    std::unordered_set<GridKey, GridKeyHash> targets;
    std::size_t estimate = 0;
    for (const auto& stroke : _pendingPaintStrokes) {
        estimate += stroke.size();
    }
    targets.reserve(estimate);

    for (const auto& stroke : _pendingPaintStrokes) {
        for (const auto& world : stroke) {
            if (auto grid = _editManager->worldToGridIndex(world)) {
                targets.insert(GridKey{grid->first, grid->second});
            }
        }
    }

    if (targets.empty()) {
        clearInvalidationBrush();
        return false;
    }

    bool snapshotCaptured = captureUndoSnapshot();
    const float brushRadiusSteps = std::max(_radiusSteps, 0.5f);
    bool anyChanged = false;
    for (const auto& key : targets) {
        if (_editManager->markInvalidRegion(key.row, key.col, brushRadiusSteps)) {
            anyChanged = true;
        }
    }

    clearInvalidationBrush();

    if (!anyChanged) {
        if (snapshotCaptured) {
            discardLastUndoSnapshot();
        }
        return false;
    }

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    emitPendingChanges();
    emit statusMessageRequested(tr("Invalidated %1 brush target(s).").arg(static_cast<int>(targets.size())),
                                kStatusMedium);
    return true;
}

void SegmentationModule::handleMousePress(CVolumeViewer* viewer,
                                          const cv::Vec3f& worldPos,
                                          const cv::Vec3f& /*surfaceNormal*/,
                                          Qt::MouseButton button,
                                          Qt::KeyboardModifiers modifiers)
{
    if (!_editingEnabled) {
        return;
    }

    const bool isLeftButton = (button == Qt::LeftButton);
    if (isLeftButton && isNearRotationHandle(viewer, worldPos)) {
        return;
    }

    if (_correctionsAnnotateMode) {
        if (!isLeftButton) {
            return;
        }
        if (modifiers.testFlag(Qt::ControlModifier)) {
            handleCorrectionPointRemove(worldPos);
        } else {
            handleCorrectionPointAdded(worldPos);
        }
        updateCorrectionsWidget();
        return;
    }

    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    if (!isLeftButton) {
        return;
    }

    if (modifiers.testFlag(Qt::ControlModifier) || modifiers.testFlag(Qt::AltModifier)) {
        return;
    }

    if (_invalidationBrushActive) {
        stopAllPushPull();
        startPaintStroke(worldPos);
        return;
    }

    stopAllPushPull();
    auto gridIndex = _editManager->worldToGridIndex(worldPos);
    if (!gridIndex) {
        return;
    }

    if (!_editManager->beginActiveDrag(*gridIndex)) {
        return;
    }

    beginDrag(gridIndex->first, gridIndex->second, viewer, worldPos);
    refreshOverlay();
}

void SegmentationModule::handleMouseMove(CVolumeViewer* viewer,
                                         const cv::Vec3f& worldPos,
                                         Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);

    if (_paintStrokeActive) {
        if (buttons.testFlag(Qt::LeftButton)) {
            extendPaintStroke(worldPos);
        } else {
            finishPaintStroke();
        }
        return;
    }

    if (_drag.active) {
        updateDrag(worldPos);
        return;
    }

    if (_correctionsAnnotateMode) {
        return;
    }

    if (!buttons.testFlag(Qt::LeftButton)) {
        updateHover(viewer, worldPos);
    }
}

void SegmentationModule::handleMouseRelease(CVolumeViewer* /*viewer*/,
                                            const cv::Vec3f& worldPos,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers /*modifiers*/)
{
    if (_paintStrokeActive && button == Qt::LeftButton) {
        extendPaintStroke(worldPos, true);
        finishPaintStroke();
        return;
    }

    if (!_drag.active || button != Qt::LeftButton) {
        if (_correctionsAnnotateMode && button == Qt::LeftButton) {
            return;
        }
        return;
    }

    updateDrag(worldPos);
    finishDrag();
}

void SegmentationModule::handleWheel(CVolumeViewer* viewer,
                                     int deltaSteps,
                                     const QPointF& /*scenePos*/,
                                     const cv::Vec3f& worldPos)
{
    if (!_editingEnabled) {
        return;
    }
    const float step = deltaSteps * 0.25f;
    setRadius(_radiusSteps + step);
    updateHover(viewer, worldPos);
    emit statusMessageRequested(tr("Gaussian radius: %1 steps").arg(_radiusSteps, 0, 'f', 2), kStatusShort);
}

bool SegmentationModule::isSegmentationViewer(const CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    return viewer->surfName() == "segmentation";
}

float SegmentationModule::gridStepWorld() const
{
    if (!_editManager || !_editManager->hasSession()) {
        return 1.0f;
    }
    const auto* surface = _editManager->previewSurface();
    if (!surface) {
        return 1.0f;
    }
    return averageScale(surface->scale());
}

void SegmentationModule::beginDrag(int row, int col, CVolumeViewer* viewer, const cv::Vec3f& worldPos)
{
    _drag.active = true;
    _drag.row = row;
    _drag.col = col;
    _drag.startWorld = worldPos;
    _drag.lastWorld = worldPos;
    _drag.viewer = viewer;
    _drag.moved = false;
}

void SegmentationModule::updateDrag(const cv::Vec3f& worldPos)
{
    if (!_drag.active || !_editManager) {
        return;
    }

    bool snapshotCaptured = false;
    if (!_drag.moved) {
        snapshotCaptured = captureUndoSnapshot();
    }

    if (!_editManager->updateActiveDrag(worldPos)) {
        if (!_drag.moved && snapshotCaptured) {
            discardLastUndoSnapshot();
        }
        return;
    }

    _drag.lastWorld = worldPos;
    _drag.moved = true;

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    refreshOverlay();
    emitPendingChanges();
}

void SegmentationModule::finishDrag()
{
    if (!_drag.active || !_editManager) {
        return;
    }

    const bool moved = _drag.moved;

    _editManager->commitActiveDrag();
    _drag.reset();

    if (moved) {
        _editManager->applyPreview();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
    }

    refreshOverlay();
    emitPendingChanges();
}

void SegmentationModule::cancelDrag()
{
    if (!_drag.active || !_editManager) {
        return;
    }

    _editManager->cancelActiveDrag();
    _drag.reset();
    refreshOverlay();
    emitPendingChanges();
}

bool SegmentationModule::isNearRotationHandle(CVolumeViewer* viewer, const cv::Vec3f& worldPos) const
{
    if (!_rotationHandleHitTester || !viewer) {
        return false;
    }
    return _rotationHandleHitTester(viewer, worldPos);
}

void SegmentationModule::updateHover(CVolumeViewer* viewer, const cv::Vec3f& worldPos)
{
    if (!_editManager || !_editManager->hasSession()) {
        _hover.clear();
        if (_overlay) {
            _overlay->setActiveVertex(std::nullopt);
            _overlay->refreshViewer(viewer);
        }
        return;
    }

    auto gridIndex = _editManager->worldToGridIndex(worldPos);
    if (!gridIndex) {
        _hover.clear();
        if (_overlay) {
            _overlay->setActiveVertex(std::nullopt);
            _overlay->refreshViewer(viewer);
        }
        return;
    }

    if (auto world = _editManager->vertexWorldPosition(gridIndex->first, gridIndex->second)) {
        _hover.set(gridIndex->first, gridIndex->second, *world, viewer);
        if (_overlay) {
            _overlay->setActiveVertex(SegmentationOverlayController::VertexMarker{
                .row = _hover.row,
                .col = _hover.col,
                .world = *world,
                .isActive = false,
                .isGrowth = false
            });
            _overlay->refreshViewer(viewer);
        }
    }
}

bool SegmentationModule::startPushPull(int direction)
{
    if (direction == 0) {
        return false;
    }

    if (_pushPull.active && _pushPull.direction == direction) {
        if (_pushPullTimer && !_pushPullTimer->isActive()) {
            _pushPullTimer->start();
        }
        return true;
    }

    if (!_hover.valid || !_hover.viewer || !isSegmentationViewer(_hover.viewer)) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    _pushPull.active = true;
    _pushPull.direction = direction;
    _pushPullUndoCaptured = false;

    if (_pushPullTimer && !_pushPullTimer->isActive()) {
        _pushPullTimer->start();
    }

    if (!applyPushPullStep()) {
        stopAllPushPull();
        return false;
    }

    return true;
}

void SegmentationModule::stopPushPull(int direction)
{
    if (!_pushPull.active) {
        return;
    }
    if (direction != 0 && direction != _pushPull.direction) {
        return;
    }
    stopAllPushPull();
}

void SegmentationModule::stopAllPushPull()
{
    _pushPull.active = false;
    _pushPull.direction = 0;
    if (_pushPullTimer && _pushPullTimer->isActive()) {
        _pushPullTimer->stop();
    }
    _pushPullUndoCaptured = false;
}

bool SegmentationModule::applyPushPullStep()
{
    if (!_pushPull.active || !_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (!_hover.valid || !_hover.viewer || !isSegmentationViewer(_hover.viewer)) {
        return false;
    }

    const int row = _hover.row;
    const int col = _hover.col;

    bool snapshotCapturedThisStep = false;
    if (!_pushPullUndoCaptured) {
        snapshotCapturedThisStep = captureUndoSnapshot();
        if (snapshotCapturedThisStep) {
            _pushPullUndoCaptured = true;
        }
    }

    if (!_editManager->beginActiveDrag({row, col})) {
        if (snapshotCapturedThisStep) {
            discardLastUndoSnapshot();
            _pushPullUndoCaptured = false;
        }
        return false;
    }

    auto centerWorldOpt = _editManager->vertexWorldPosition(row, col);
    if (!centerWorldOpt) {
        _editManager->cancelActiveDrag();
        if (snapshotCapturedThisStep) {
            discardLastUndoSnapshot();
            _pushPullUndoCaptured = false;
        }
        return false;
    }
    const cv::Vec3f centerWorld = *centerWorldOpt;

    QuadSurface* baseSurface = _editManager->baseSurface();
    if (!baseSurface) {
        _editManager->cancelActiveDrag();
        return false;
    }

    cv::Vec3f ptr = baseSurface->pointer();
    baseSurface->pointTo(ptr, centerWorld, std::numeric_limits<float>::max(), 400);
    cv::Vec3f normal = baseSurface->normal(ptr);
    if (std::isnan(normal[0]) || std::isnan(normal[1]) || std::isnan(normal[2])) {
        _editManager->cancelActiveDrag();
        return false;
    }

    const float norm = cv::norm(normal);
    if (norm <= 1e-4f) {
        _editManager->cancelActiveDrag();
        return false;
    }
    normal /= norm;

    const float stepWorld = gridStepWorld() * _pushPullStepMultiplier;
    if (stepWorld <= 0.0f) {
        _editManager->cancelActiveDrag();
        return false;
    }

    const cv::Vec3f targetWorld = centerWorld + normal * (static_cast<float>(_pushPull.direction) * stepWorld);
    if (!_editManager->updateActiveDrag(targetWorld)) {
        _editManager->cancelActiveDrag();
        if (snapshotCapturedThisStep) {
            discardLastUndoSnapshot();
            _pushPullUndoCaptured = false;
        }
        return false;
    }

    _editManager->commitActiveDrag();
    _editManager->applyPreview();

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    refreshOverlay();
    emitPendingChanges();
    return true;
}

void SegmentationModule::onPushPullTick()
{
    if (!applyPushPullStep()) {
        stopAllPushPull();
    }
}
