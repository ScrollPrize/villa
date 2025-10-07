#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "SegmentationBrushTool.hpp"
#include "SegmentationLineTool.hpp"
#include "SegmentationPushPullTool.hpp"
#include "SegmentationCorrections.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/ui/VCCollection.hpp"

#include <QLoggingCategory>
#include <QPointer>
#include <QString>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <optional>
#include <limits>
#include <exception>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>


Q_LOGGING_CATEGORY(lcSegModule, "vc.segmentation.module")

namespace
{
float averageScale(const cv::Vec2f& scale)
{
    const float sx = std::abs(scale[0]);
    const float sy = std::abs(scale[1]);
    const float avg = 0.5f * (sx + sy);
    return (avg > 1e-4f) ? avg : 1.0f;
}

void ensureSurfaceMetaObject(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    if (surface->meta && surface->meta->is_object()) {
        return;
    }
    if (surface->meta) {
        delete surface->meta;
    }
    surface->meta = new nlohmann::json(nlohmann::json::object());
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

SegmentationModule::~SegmentationModule() = default;

SegmentationModule::SegmentationModule(SegmentationWidget* widget,
                                       SegmentationEditManager* editManager,
                                       SegmentationOverlayController* overlay,
                                       ViewerManager* viewerManager,
                                       CSurfaceCollection* surfaces,
                                       VCCollection* pointCollection,
                                       bool editingEnabled,
                                       QObject* parent)
    : QObject(parent)
    , _widget(widget)
    , _editManager(editManager)
    , _overlay(overlay)
    , _viewerManager(viewerManager)
    , _surfaces(surfaces)
    , _pointCollection(pointCollection)
    , _editingEnabled(editingEnabled)
    , _growthMethod(_widget ? _widget->growthMethod() : SegmentationGrowthMethod::Tracer)
    , _growthSteps(_widget ? _widget->growthSteps() : 5)
{
    float initialPushPullStep = 4.0f;
    AlphaPushPullConfig initialAlphaConfig{};

    if (_widget) {
        _dragRadiusSteps = _widget->dragRadius();
        _dragSigmaSteps = _widget->dragSigma();
        _lineRadiusSteps = _widget->lineRadius();
        _lineSigmaSteps = _widget->lineSigma();
        _pushPullRadiusSteps = _widget->pushPullRadius();
        _pushPullSigmaSteps = _widget->pushPullSigma();
        initialPushPullStep = std::clamp(_widget->pushPullStep(), 0.05f, 10.0f);
        _smoothStrength = std::clamp(_widget->smoothingStrength(), 0.0f, 1.0f);
        _smoothIterations = std::clamp(_widget->smoothingIterations(), 1, 25);
        initialAlphaConfig = SegmentationPushPullTool::sanitizeConfig(_widget->alphaPushPullConfig());
    }

    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
    }

    _brushTool = std::make_unique<SegmentationBrushTool>(*this, _editManager, _widget, _surfaces);
    _lineTool = std::make_unique<SegmentationLineTool>(*this, _editManager, _surfaces, _smoothStrength, _smoothIterations);
    _pushPullTool = std::make_unique<SegmentationPushPullTool>(*this, _editManager, _widget, _overlay, _surfaces);
    _pushPullTool->setStepMultiplier(initialPushPullStep);
    _pushPullTool->setAlphaConfig(initialAlphaConfig);

    _corrections = std::make_unique<segmentation::CorrectionsState>(*this, _widget, _pointCollection);

    useFalloff(FalloffTool::Drag);

    bindWidgetSignals();

    if (_viewerManager) {
        _viewerManager->setSegmentationModule(this);
    }

    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationModule::onSurfaceCollectionChanged);
    }

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t id) {
            if (_corrections) {
                _corrections->onCollectionRemoved(id);
                updateCorrectionsWidget();
            }
        });

        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t id) {
            if (_corrections) {
                _corrections->onCollectionChanged(id);
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

    ensureAutosaveTimer();
    updateAutosaveState();
}

void SegmentationModule::setRotationHandleHitTester(std::function<bool(CVolumeViewer*, const cv::Vec3f&)> tester)
{
    _rotationHandleHitTester = std::move(tester);
}

SegmentationModule::HoverInfo SegmentationModule::hoverInfo() const
{
    HoverInfo info;
    if (_hover.valid) {
        info.valid = true;
        info.row = _hover.row;
        info.col = _hover.col;
        info.world = _hover.world;
        info.viewer = _hover.viewer;
    }
    return info;
}

void SegmentationModule::bindWidgetSignals()
{
    if (!_widget) {
        return;
    }

    connect(_widget, &SegmentationWidget::editingModeChanged,
            this, &SegmentationModule::setEditingEnabled);
    connect(_widget, &SegmentationWidget::dragRadiusChanged,
            this, &SegmentationModule::setDragRadius);
    connect(_widget, &SegmentationWidget::dragSigmaChanged,
            this, &SegmentationModule::setDragSigma);
    connect(_widget, &SegmentationWidget::lineRadiusChanged,
            this, &SegmentationModule::setLineRadius);
    connect(_widget, &SegmentationWidget::lineSigmaChanged,
            this, &SegmentationModule::setLineSigma);
    connect(_widget, &SegmentationWidget::pushPullRadiusChanged,
            this, &SegmentationModule::setPushPullRadius);
    connect(_widget, &SegmentationWidget::pushPullSigmaChanged,
            this, &SegmentationModule::setPushPullSigma);
    connect(_widget, &SegmentationWidget::alphaPushPullConfigChanged,
            this, [this]() {
                if (_widget) {
                    setAlphaPushPullConfig(_widget->alphaPushPullConfig());
                }
            });
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
    connect(_widget, &SegmentationWidget::smoothingStrengthChanged,
            this, &SegmentationModule::setSmoothingStrength);
    connect(_widget, &SegmentationWidget::smoothingIterationsChanged,
            this, &SegmentationModule::setSmoothingIterations);
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
        deactivateInvalidationBrush();
        clearLineDragStroke();
        _lineDrawKeyActive = false;
        clearUndoStack();
        if (_pendingAutosave) {
            performAutosave();
        }
    }
    updateCorrectionsWidget();
    refreshOverlay();
    emit editingEnabledChanged(enabled);
    updateAutosaveState();
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false);
    }
    emitPendingChanges();
    markAutosaveNeeded(true);
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
    clearLineDragStroke();
    _editManager->resetPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false);
    }
    refreshOverlay();
    emitPendingChanges();
    if (hadPendingChanges) {
        emit statusMessageRequested(tr("Reset pending segmentation edits."), kStatusShort);
    }
}

void SegmentationModule::stopTools()
{
    _lineDrawKeyActive = false;
    clearLineDragStroke();
    cancelDrag();
    emit stopToolsRequested();
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

void SegmentationModule::markNextEditsFromGrowth()
{
    if (_editManager) {
        _editManager->markNextEditsAsGrowth();
    }
}

void SegmentationModule::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    if (_widget) {
        _widget->setGrowthInProgress(running);
    }
    if (_corrections) {
        _corrections->setGrowthInProgress(running);
    }
    if (running) {
        setCorrectionsAnnotateMode(false, false);
        deactivateInvalidationBrush();
        clearLineDragStroke();
        _lineDrawKeyActive = false;
    }
    updateCorrectionsWidget();
    emit growthInProgressChanged(_growthInProgress);
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
    if (!_overlay) {
        return;
    }

    SegmentationOverlayController::State state;
    state.gaussianRadiusSteps = falloffRadius(_activeFalloff);
    state.gaussianSigmaSteps = falloffSigma(_activeFalloff);
    state.gridStepWorld = gridStepWorld();

    const auto toFalloffMode = [](FalloffTool tool) {
        using Mode = SegmentationOverlayController::State::FalloffMode;
        switch (tool) {
        case FalloffTool::Drag:
            return Mode::Drag;
        case FalloffTool::Line:
            return Mode::Line;
        case FalloffTool::PushPull:
            return Mode::PushPull;
        }
        return Mode::Drag;
    };
    state.falloff = toFalloffMode(_activeFalloff);

    const bool hasSession = _editManager && _editManager->hasSession();
    if (!hasSession) {
        _overlay->applyState(state);
        return;
    }

    if (_drag.active) {
        if (auto world = _editManager->vertexWorldPosition(_drag.row, _drag.col)) {
            state.activeMarker = SegmentationOverlayController::VertexMarker{
                .row = _drag.row,
                .col = _drag.col,
                .world = *world,
                .isActive = true,
                .isGrowth = false
            };
        }
    } else if (_hover.valid) {
        state.activeMarker = SegmentationOverlayController::VertexMarker{
            .row = _hover.row,
            .col = _hover.col,
            .world = _hover.world,
            .isActive = false,
            .isGrowth = false
        };
    }

    if (_drag.active) {
        const auto touched = _editManager->recentTouched();
        state.neighbours.reserve(touched.size());
        for (const auto& key : touched) {
            if (key.row == _drag.row && key.col == _drag.col) {
                continue;
            }
            if (auto world = _editManager->vertexWorldPosition(key.row, key.col)) {
                state.neighbours.push_back({key.row, key.col, *world, false, false});
            }
        }
    }

    std::vector<cv::Vec3f> maskPoints;
    std::size_t maskReserve = 0;
    const bool brushHasOverlay = _brushTool &&
                                 (!_brushTool->overlayPoints().empty() ||
                                  !_brushTool->currentStrokePoints().empty());
    if (_brushTool) {
        maskReserve += _brushTool->overlayPoints().size();
        maskReserve += _brushTool->currentStrokePoints().size();
    }
    if (_lineTool) {
        maskReserve += _lineTool->overlayPoints().size();
    }
    maskPoints.reserve(maskReserve);
    if (_brushTool) {
        const auto& overlayPts = _brushTool->overlayPoints();
        maskPoints.insert(maskPoints.end(), overlayPts.begin(), overlayPts.end());
        const auto& strokePts = _brushTool->currentStrokePoints();
        maskPoints.insert(maskPoints.end(), strokePts.begin(), strokePts.end());
    }
    if (_lineTool) {
        const auto& linePts = _lineTool->overlayPoints();
        maskPoints.insert(maskPoints.end(), linePts.begin(), linePts.end());
    }

    const bool hasLineStroke = _lineTool && !_lineTool->overlayPoints().empty();
    const bool lineStrokeActive = _lineTool && _lineTool->strokeActive();
    const bool brushActive = _brushTool && _brushTool->brushActive();
    const bool brushStrokeActive = _brushTool && _brushTool->strokeActive();
    const bool pushPullActive = _pushPullTool && _pushPullTool->isActive();

    state.maskPoints = std::move(maskPoints);
    state.maskVisible = !state.maskPoints.empty();
    state.hasLineStroke = hasLineStroke;
    state.lineStrokeActive = lineStrokeActive;
    state.brushActive = brushActive;
    state.brushStrokeActive = brushStrokeActive;
    state.pushPullActive = pushPullActive;

    FalloffTool overlayTool = _activeFalloff;
    if (hasLineStroke) {
        overlayTool = FalloffTool::Line;
    } else if (brushHasOverlay || brushStrokeActive || brushActive) {
        overlayTool = FalloffTool::Drag;
    } else if (pushPullActive) {
        overlayTool = FalloffTool::PushPull;
    }

    state.displayRadiusSteps = falloffRadius(overlayTool);

    _overlay->applyState(state);
}



void SegmentationModule::updateCorrectionsWidget()
{
    if (_corrections) {
        _corrections->refreshWidget();
    }
}

void SegmentationModule::setCorrectionsAnnotateMode(bool enabled, bool userInitiated)
{
    if (!_corrections) {
        return;
    }

    const bool wasActive = _corrections->annotateMode();
    const bool isActive = _corrections->setAnnotateMode(enabled, userInitiated, _editingEnabled);
    if (isActive && !wasActive) {
        deactivateInvalidationBrush();
    }
}

void SegmentationModule::setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated)
{
    if (_corrections) {
        _corrections->setActiveCollection(collectionId, userInitiated);
    }
}

uint64_t SegmentationModule::createCorrectionCollection(bool announce)
{
    return _corrections ? _corrections->createCollection(announce) : 0;
}

void SegmentationModule::handleCorrectionPointAdded(const cv::Vec3f& worldPos)
{
    if (_corrections) {
        _corrections->handlePointAdded(worldPos);
    }
}

void SegmentationModule::handleCorrectionPointRemove(const cv::Vec3f& worldPos)
{
    if (_corrections) {
        _corrections->handlePointRemoved(worldPos);
    }
}

void SegmentationModule::pruneMissingCorrections()
{
    if (_corrections) {
        _corrections->pruneMissing();
        _corrections->refreshWidget();
    }
}

void SegmentationModule::onCorrectionsCreateRequested()
{
    if (!_corrections) {
        return;
    }

    const bool wasActive = _corrections->annotateMode();
    const uint64_t created = _corrections->createCollection(true);
    if (created != 0) {
        const bool nowActive = _corrections->setAnnotateMode(true, false, _editingEnabled);
        if (nowActive && !wasActive) {
            deactivateInvalidationBrush();
        }
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
    if (_corrections) {
        _corrections->onZRangeChanged(enabled, zMin, zMax);
    }
}

void SegmentationModule::clearPendingCorrections()
{
    if (_corrections) {
        _corrections->clearAll(_editingEnabled);
    }
}

std::optional<std::pair<int, int>> SegmentationModule::correctionsZRange() const
{
    return _corrections ? _corrections->zRange() : std::nullopt;
}

SegmentationCorrectionsPayload SegmentationModule::buildCorrectionsPayload() const
{
    return _corrections ? _corrections->buildPayload() : SegmentationCorrectionsPayload{};
}
void SegmentationModule::handleGrowSurfaceRequested(SegmentationGrowthMethod method,
                                                    SegmentationGrowthDirection direction,
                                                    int steps,
                                                    bool inpaintOnly)
{
    qCInfo(lcSegModule) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << steps
                        << "inpaintOnly" << inpaintOnly;

    if (_growthInProgress) {
        emit statusMessageRequested(tr("Surface growth already in progress"), kStatusMedium);
        return;
    }
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        emit statusMessageRequested(tr("Enable segmentation editing before growing surfaces"), kStatusMedium);
        return;
    }

    // Ensure any pending invalidation brush strokes are committed before growth.
    if (_brushTool) {
        _brushTool->applyPending(_dragRadiusSteps);
    }

    if (!inpaintOnly) {
        _growthMethod = method;
        _growthSteps = std::max(1, steps);
    }
    markNextEditsFromGrowth();
    const int sanitizedSteps = inpaintOnly ? std::max(0, steps) : std::max(1, steps);
    emit growSurfaceRequested(method, direction, sanitizedSteps, inpaintOnly);
}

void SegmentationModule::setInvalidationBrushActive(bool active)
{
    if (!_brushTool) {
        return;
    }

    const bool canUseBrush = _editingEnabled && !_growthInProgress &&
                             !(_corrections && _corrections->annotateMode()) &&
                             _editManager && _editManager->hasSession();
    const bool shouldEnable = active && canUseBrush;

    if (!shouldEnable) {
        if (_brushTool->brushActive()) {
            _brushTool->setActive(false);
        }
        // Only discard pending strokes when brush use is no longer possible.
        if (!canUseBrush) {
            _brushTool->clear();
        }
        return;
    }

    if (!_brushTool->brushActive()) {
        _brushTool->setActive(true);
    }
}

void SegmentationModule::clearInvalidationBrush()
{
    if (_brushTool) {
        _brushTool->clear();
    }
}

void SegmentationModule::deactivateInvalidationBrush()
{
    if (!_brushTool) {
        return;
    }
    if (_brushTool->brushActive()) {
        _brushTool->setActive(false);
    }
    _brushTool->clear();
}

void SegmentationModule::clearLineDragStroke()
{
    if (_lineTool) {
        _lineTool->clear();
    }
    if (!_lineDrawKeyActive && _activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Drag);
    }
}

bool SegmentationModule::isSegmentationViewer(const CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    const std::string& name = viewer->surfName();
    return name.rfind("seg", 0) == 0 || name == "xy plane";
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false);
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

    if (moved && _smoothStrength > 0.0f && _smoothIterations > 0) {
        _editManager->smoothRecentTouched(_smoothStrength, _smoothIterations);
    }

    _editManager->commitActiveDrag();
    _drag.reset();

    if (moved) {
        _editManager->applyPreview();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false);
        }
        markAutosaveNeeded();
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
    bool hoverChanged = false;

    if (!_editManager || !_editManager->hasSession()) {
        if (_hover.valid) {
            _hover.clear();
            hoverChanged = true;
        }
    } else {
        auto gridIndex = _editManager->worldToGridIndex(worldPos);
        if (!gridIndex) {
            if (_hover.valid) {
                _hover.clear();
                hoverChanged = true;
            }
        } else if (auto world = _editManager->vertexWorldPosition(gridIndex->first, gridIndex->second)) {
            const bool rowChanged = !_hover.valid || _hover.row != gridIndex->first;
            const bool colChanged = !_hover.valid || _hover.col != gridIndex->second;
            const bool worldChanged = !_hover.valid || cv::norm(_hover.world - *world) >= 1e-4f;
            const bool viewerChanged = !_hover.valid || _hover.viewer != viewer;
            if (rowChanged || colChanged || worldChanged || viewerChanged) {
                _hover.set(gridIndex->first, gridIndex->second, *world, viewer);
                hoverChanged = true;
            }
        } else if (_hover.valid) {
            _hover.clear();
            hoverChanged = true;
        }
    }

    if (hoverChanged) {
        refreshOverlay();
    }
}

bool SegmentationModule::startPushPull(int direction, std::optional<bool> alphaOverride)
{
    return _pushPullTool ? _pushPullTool->start(direction, alphaOverride) : false;
}

void SegmentationModule::stopPushPull(int direction)
{
    if (_pushPullTool) {
        _pushPullTool->stop(direction);
    }
}

void SegmentationModule::stopAllPushPull()
{
    if (_pushPullTool) {
        _pushPullTool->stopAll();
    }
}

bool SegmentationModule::applyPushPullStep()
{
    return _pushPullTool ? _pushPullTool->applyStep() : false;
}

void SegmentationModule::markAutosaveNeeded(bool immediate)
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    _pendingAutosave = true;
    _autosaveNotifiedFailure = false;

    ensureAutosaveTimer();
    if (_editingEnabled && _autosaveTimer && !_autosaveTimer->isActive()) {
        _autosaveTimer->start();
    }

    if (immediate) {
        performAutosave();
    }
}

void SegmentationModule::performAutosave()
{
    if (!_pendingAutosave) {
        return;
    }
    if (!_editManager) {
        return;
    }
    QuadSurface* surface = _editManager->baseSurface();
    if (!surface) {
        return;
    }
    if (surface->path.empty() || surface->id.empty()) {
        if (!_autosaveNotifiedFailure) {
            qCWarning(lcSegModule) << "Skipping autosave: segmentation surface lacks path or id.";
            emit statusMessageRequested(tr("Cannot autosave segmentation: surface is missing file metadata."),
                                        kStatusMedium);
            _autosaveNotifiedFailure = true;
        }
        return;
    }

    ensureSurfaceMetaObject(surface);

    try {
        surface->saveOverwrite();
        _pendingAutosave = false;
        _autosaveNotifiedFailure = false;
    } catch (const std::exception& ex) {
        qCWarning(lcSegModule) << "Autosave failed:" << ex.what();
        if (!_autosaveNotifiedFailure) {
            emit statusMessageRequested(tr("Failed to autosave segmentation: %1")
                                            .arg(QString::fromUtf8(ex.what())),
                                        kStatusLong);
            _autosaveNotifiedFailure = true;
        }
    }
}

void SegmentationModule::ensureAutosaveTimer()
{
    if (_autosaveTimer) {
        return;
    }
    _autosaveTimer = new QTimer(this);
    _autosaveTimer->setInterval(kAutosaveIntervalMs);
    _autosaveTimer->setSingleShot(false);
    connect(_autosaveTimer, &QTimer::timeout, this, [this]() {
        performAutosave();
    });
}

void SegmentationModule::updateAutosaveState()
{
    ensureAutosaveTimer();
    if (!_autosaveTimer) {
        return;
    }

    const bool shouldRun = _editingEnabled && _editManager && _editManager->hasSession();
    if (shouldRun) {
        if (!_autosaveTimer->isActive()) {
            _autosaveTimer->start();
        }
    } else if (_autosaveTimer->isActive()) {
        _autosaveTimer->stop();
    }
}
