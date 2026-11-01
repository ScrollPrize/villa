#include "SegmentationModule.hpp"
#include "SurfaceStoreEdit.hpp"

#include "../OpenDataManifest.hpp"

#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "CState.hpp"
#include "tools/SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "tools/SegmentationPushPullTool.hpp"
#include "tools/SurfaceMaskBrushTool.hpp"
#include "tools/ThinPlateSpline3d.hpp"
#include "growth/SegmentationCorrections.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QApplication>
#include <QDebug>
#include <QCursor>
#include <QFileDialog>
#include <QLoggingCategory>
#include <QMessageBox>
#include <QPointF>
#include <QPushButton>
#include <QPointer>
#include <QString>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <tuple>
#include <limits>
#include <exception>
#include <set>
#include <utility>
#include <vector>

#include "utils/Json.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"


Q_LOGGING_CATEGORY(lcSegModule, "vc.segmentation.module")

namespace
{
constexpr std::size_t kMaxInteractiveNeighborMarkers = 512;

float averageScale(const cv::Vec2f& scale)
{
    const float sx = std::abs(scale[0]);
    const float sy = std::abs(scale[1]);
    const float avg = 0.5f * (sx + sy);
    return (avg > 1e-4f) ? avg : 1.0f;
}

std::optional<std::pair<int, int>> segmentationSceneToGrid(VolumeViewerBase* viewer,
                                                           const QPointF& scenePos,
                                                           int rows,
                                                           int cols)
{
    if (!viewer || rows <= 0 || cols <= 0) {
        return std::nullopt;
    }

    const cv::Vec2f surfaceCoords = viewer->sceneToSurfaceCoords(scenePos);
    int col = 0;
    int row = 0;
    if (auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface())) {
        const cv::Vec2f surfScale = quad->scale();
        const cv::Vec3f center = quad->center();
        if (std::abs(surfScale[0]) < 1e-6f || std::abs(surfScale[1]) < 1e-6f) {
            return std::nullopt;
        }
        col = static_cast<int>(std::lround((surfaceCoords[0] + center[0]) * surfScale[0]));
        row = static_cast<int>(std::lround((surfaceCoords[1] + center[1]) * surfScale[1]));
    } else {
        col = static_cast<int>(std::lround(surfaceCoords[0]));
        row = static_cast<int>(std::lround(surfaceCoords[1]));
    }

    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        return std::nullopt;
    }
    return std::make_pair(row, col);
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

void SegmentationModule::HoverState::set(int r, int c, const cv::Vec3f& w, VolumeViewerBase* v)
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

SegmentationModule::~SegmentationModule()
{
    // Cancel any in-progress background save on shutdown
    _autosaveState.endSession();
    if (_autosaveState.saveInProgress() && _saveFuture.isRunning()) {
        _saveFuture.cancel();
    }
}

SegmentationModule::SegmentationModule(SegmentationWidget* widget,
                                       SegmentationEditManager* editManager,
                                       SegmentationOverlayController* overlay,
                                       ViewerManager* viewerManager,
                                       CState* state,
                                       VCCollection* pointCollection,
                                       bool editingEnabled,
                                       QObject* parent)
    : QObject(parent)
    , _widget(widget)
    , _editManager(editManager)
    , _overlay(overlay)
    , _viewerManager(viewerManager)
    , _state(state)
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
        _pushPullRadiusSteps = _widget->pushPullRadius();
        _pushPullSigmaSteps = _widget->pushPullSigma();
        initialPushPullStep = std::clamp(_widget->pushPullStep(), 0.05f, 100.0f);
        if (_editManager) {
            _editManager->setEditScale(std::clamp(_widget->editScale(), 0.1f, 10.0f));
        }
        _smoothStrength = std::clamp(_widget->smoothingStrength(), 0.0f, 1.0f);
        _smoothIterations = std::clamp(_widget->smoothingIterations(), 1, 25);
        initialAlphaConfig = SegmentationPushPullTool::sanitizeConfig(_widget->alphaPushPullConfig());
        _hoverPreviewEnabled = _widget->showHoverMarker();
        _maskBrushRadius = std::max(1.0f, _widget->maskBrushRadius());
        _drawMaskEnabled = _widget->drawMaskEnabled();
    }

    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
    }

    _pushPullTool = std::make_unique<SegmentationPushPullTool>(*this, _editManager, _widget, _state);
    _pushPullTool->setStepMultiplier(initialPushPullStep);
    _pushPullTool->setAlphaConfig(initialAlphaConfig);

    _surfaceMaskTool = std::make_unique<SurfaceMaskBrushTool>(*this);

    _corrections = std::make_unique<segmentation::CorrectionsState>(*this, _widget, _pointCollection);
    _manualAddTool = std::make_unique<ManualAddTool>();

    useFalloff(FalloffTool::Drag);

    bindWidgetSignals();

    if (_viewerManager) {
        _viewerManager->setSegmentationModule(this);
    }

    if (_state) {
        connect(_state, &CState::surfaceChanged,
                this, &SegmentationModule::onSurfaceCollectionChanged);
        connect(_state, &CState::pointCollectionsSaveRequested, this, &SegmentationModule::saveCorrections);
    }

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t id) {
            if (_corrections) {
                _corrections->onCollectionRemoved(id);
                updateCorrectionsWidget();
            }
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t id) {
            if (_corrections) {
                _corrections->onCollectionChanged(id);
            }
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>&) {
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::pointAdded, this, [this](const ColPoint&) {
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::pointChanged, this, [this](const ColPoint&) {
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::pointRemoved, this, [this](vc::PointRef) {
            scheduleCorrectionsAutoSave();
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

void SegmentationModule::setRotationHandleHitTester(std::function<bool(VolumeViewerBase*, const cv::Vec3f&)> tester)
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

void SegmentationModule::setHoverPreviewEnabled(bool enabled)
{
    if (_hoverPreviewEnabled == enabled) {
        return;
    }
    _hoverPreviewEnabled = enabled;
    if (!enabled && _hover.valid) {
        _hover.clear();
    }
    refreshOverlay();
}

bool SegmentationModule::ensureHoverTarget()
{
    if (!_editManager || !_editManager->hasSession()) {
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: no active editing session.";
        return false;
    }

    // Push/pull calls this from a timer. Reuse the hover resolved by mouse-move
    // events when possible; polling the widget under the cursor and doing a
    // patch-index lookup every tick makes held push/pull noticeably choppy.
    if (_hoverPreviewEnabled && _hover.valid && _hoverPointer.valid &&
        _hover.viewer == _hoverPointer.viewer) {
        return true;
    }

    recoverHoverPointerFromCursor();

    if (_hover.valid && _hoverPointer.valid && _hover.viewer != _hoverPointer.viewer) {
        _hover.clear();
    }

    if (!_hoverPointer.valid) {
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: no cursor sample for a segmentation viewer.";
        return false;
    }
    VolumeViewerBase* viewer = _hoverPointer.viewer;
    if (!viewer) {
        _hoverPointer.valid = false;
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: cursor sample viewer no longer exists.";
        return false;
    }
    if (!isSegmentationViewer(viewer)) {
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: viewer is not a segmentation viewer:"
                               << QString::fromStdString(viewer->surfName());
        _hoverPointer.valid = false;
        return false;
    }
    auto gridIndex = _editManager->worldToGridIndex(_hoverPointer.world);
    if (!gridIndex) {
        _hover.clear();
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: surface patch index lookup failed.";
        return false;
    }
    auto world = _editManager->vertexWorldPosition(gridIndex->first, gridIndex->second);
    if (!world) {
        _hover.clear();
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: resolved grid vertex is invalid"
                               << "row" << gridIndex->first << "col" << gridIndex->second;
        return false;
    }
    _hover.set(gridIndex->first, gridIndex->second, *world, viewer);
    return true;
}

void SegmentationModule::bindWidgetSignals()
{
    if (!_widget) {
        return;
    }

    connect(_widget, &SegmentationWidget::editingModeChanged,
            this, &SegmentationModule::setEditingEnabled);
    connect(_widget, &SegmentationWidget::annotateToggled,
            this, &SegmentationModule::setAnnotateMode);
    connect(_widget, &SegmentationWidget::drawMaskChanged,
            this, &SegmentationModule::setDrawMaskEnabled);
    connect(_widget, &SegmentationWidget::dragRadiusChanged,
            this, &SegmentationModule::setDragRadius);
    connect(_widget, &SegmentationWidget::dragSigmaChanged,
            this, &SegmentationModule::setDragSigma);
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
    connect(_widget, &SegmentationWidget::editScaleChanged,
            this, &SegmentationModule::setEditScale);
    connect(_widget, &SegmentationWidget::smoothingStrengthChanged,
            this, &SegmentationModule::setSmoothingStrength);
    connect(_widget, &SegmentationWidget::smoothingIterationsChanged,
            this, &SegmentationModule::setSmoothingIterations);
    connect(_widget, &SegmentationWidget::hoverMarkerToggled,
            this, &SegmentationModule::setHoverPreviewEnabled);
    connect(_widget, &SegmentationWidget::correctionsCreateRequested,
            this, &SegmentationModule::onCorrectionsCreateRequested);
    connect(_widget, &SegmentationWidget::correctionsCollectionSelected,
            this, &SegmentationModule::onCorrectionsCollectionSelected);
    connect(_widget, &SegmentationWidget::correctionsCollectionSelectionCleared,
            this, &SegmentationModule::clearActiveCorrectionCollection);
    connect(_widget, &SegmentationWidget::correctionsZRangeChanged,
            this, &SegmentationModule::onCorrectionsZRangeChanged);
    connect(_widget, &SegmentationWidget::maskBrushRadiusChanged,
            this, &SegmentationModule::setMaskBrushRadius);
    connect(_widget, &SegmentationWidget::manualAddConfigChanged, this, [this]() {
        if (_manualAddTool && _widget) {
            _manualAddTool->setConfig(_widget->manualAddConfig());
            if (_manualAddMode) {
                refreshOverlay();
            }
        }
    });
    connect(_widget, &SegmentationWidget::manualAddClearPendingRequested,
            this, &SegmentationModule::clearManualAddPending);
    connect(_widget, &SegmentationWidget::manualAddRecomputeRequested,
            this, &SegmentationModule::recomputeManualAdd);
    connect(_widget, &SegmentationWidget::manualAddApplyExitRequested,
            this, [this]() { finishManualAdd(true); });
    connect(_widget, &SegmentationWidget::manualAddCancelRequested,
            this, [this]() { finishManualAdd(false); });
}

void SegmentationModule::bindViewerSignals(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }
    QObject* viewerObject = viewer->asQObject();
    if (!viewerObject || viewerObject->property("vc_segmentation_bound").toBool()) {
        return;
    }

    if (auto* chunkedViewer = qobject_cast<CChunkedVolumeViewer*>(viewerObject)) {
        connect(chunkedViewer, &CChunkedVolumeViewer::sendMousePressVolume,
                this, [this, viewer](const cv::Vec3f& worldPos,
                                     const cv::Vec3f& normal,
                                     Qt::MouseButton button,
                                     Qt::KeyboardModifiers modifiers,
                                     const QPointF& scenePos) {
                    handleMousePress(viewer, worldPos, normal, button, modifiers, scenePos);
                });
        connect(chunkedViewer, &CChunkedVolumeViewer::sendMouseMoveVolume,
                this, [this, viewer](const cv::Vec3f& worldPos,
                                     Qt::MouseButtons buttons,
                                     Qt::KeyboardModifiers modifiers,
                                     const QPointF& scenePos) {
                    handleMouseMove(viewer, worldPos, buttons, modifiers, scenePos);
                });
        connect(chunkedViewer, &CChunkedVolumeViewer::sendMouseReleaseVolume,
                this, [this, viewer](const cv::Vec3f& worldPos,
                                     Qt::MouseButton button,
                                     Qt::KeyboardModifiers modifiers,
                                     const QPointF& scenePos) {
                    handleMouseRelease(viewer, worldPos, button, modifiers, scenePos);
                });
        connect(chunkedViewer, &CChunkedVolumeViewer::sendMouseDoubleClickVolume,
                this, [this, viewer](const cv::Vec3f& worldPos,
                                     Qt::MouseButton button,
                                     Qt::KeyboardModifiers modifiers) {
                    handleMouseDoubleClick(viewer, worldPos, button, modifiers);
                });
        connect(chunkedViewer, &CChunkedVolumeViewer::sendSegmentationRadiusWheel,
                this, [this, viewer](int steps, const QPointF& scenePoint, const cv::Vec3f& worldPos) {
                    handleWheel(viewer, steps, scenePoint, worldPos);
                });
    } else {
        return;
    }

    viewerObject->setProperty("vc_segmentation_bound", true);
    _attachedViewers.insert(viewer);
    connect(viewerObject, &QObject::destroyed, this, [this, viewer]() {
        detachViewer(viewer);
    });
}

void SegmentationModule::attachViewer(VolumeViewerBase* viewer)
{
    bindViewerSignals(viewer);
}

void SegmentationModule::detachViewer(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }
    _attachedViewers.remove(viewer);
    if (_hover.viewer == viewer) {
        _hover.clear();
    }
    if (_drag.viewer == viewer) {
        _drag.viewer = nullptr;
    }
    if (_correctionDrag.viewer == viewer) {
        _correctionDrag.viewer = nullptr;
    }
    if (_hoverPointer.viewer == viewer) {
        _hoverPointer.valid = false;
        _hoverPointer.viewer = nullptr;
    }
}

void SegmentationModule::publishEditDraft(const std::optional<cv::Rect>& vertices)
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    if (vertices) {
        _editManager->publishDraft(*vertices);
    } else {
        _editManager->publishWholeDraft();
    }
}

void SegmentationModule::setIgnoreSegSurfaceChange(bool ignore)
{
    _ignoreSegSurfaceChange = ignore;
}

void SegmentationModule::setAnnotateMode(bool enabled)
{
    if (_annotateMode == enabled) {
        return;
    }
    _annotateMode = enabled;
    emit annotateModeChanged(enabled);
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
    if (_surfaceMaskTool) {
        _surfaceMaskTool->setActive(enabled && hasActiveSession() && _drawMaskEnabled);
    }
    if (!enabled) {
        resetManualAddState(true);
        stopAllPushPull();
        clearUndoStack();
        _hoverPointer.valid = false;
        _hoverPointer.viewer = nullptr;
        if (_autosaveState.pending() && _editManager && _editManager->hasSession()) {
            performAutosave();
        } else if (!_editManager || !_editManager->hasSession()) {
            _autosaveState.clearDeferred();
            _pendingAutosaveVertexUpdates.clear();
        }
    }
    updateCorrectionsWidget();
    refreshOverlay();
    emit editingEnabledChanged(enabled);
    updateAutosaveState();
}

void SegmentationModule::onActiveSegmentChanged(QuadSurface* newSurface)
{
    qCInfo(lcSegModule) << "Active segment changed";

    resetManualAddState(true);

    if (_surfaceMaskTool) {
        _surfaceMaskTool->setSurface(newSurface);
        _surfaceMaskTool->setActive(_editingEnabled && hasActiveSession() && _drawMaskEnabled);
    }

    // Save corrections for old segment and load for new segment
    if (_pointCollection) {
        saveCorrections();
    }
    // The previous surface's pending saves are done: its store edit ends here.
    if (_storeEdit && (!newSurface || newSurface->id != _storeEdit->uuid())) {
        _storeEdit.reset();
        _editLease = {};
    }
    if (_pointCollection) {
        loadCorrections(newSurface);
    }

    refreshOverlay();
}

void SegmentationModule::setDrawMaskEnabled(bool enabled)
{
    if (_drawMaskEnabled == enabled) {
        return;
    }

    _drawMaskEnabled = enabled;
    _shiftDrawMaskActive = false;

    if (_surfaceMaskTool) {
        _surfaceMaskTool->setActive(_editingEnabled && hasActiveSession() && _drawMaskEnabled);

        QuadSurface* surface = nullptr;
        std::shared_ptr<Surface> surfaceHolder;
        if (_state) {
            surfaceHolder = _state->surface("segmentation");
            surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
        }
        if (!surface && _editManager && _editManager->hasSession()) {
            surface = _editManager->baseSurface().get();
        }
        _surfaceMaskTool->setSurface(surface);
    }

    if (enabled) {
        stopAllPushPull();
        cancelDrag();
    }

    refreshOverlay();
}

void SegmentationModule::setMaskBrushRadius(float radius)
{
    _maskBrushRadius = std::max(1.0f, radius);
    refreshOverlay();
}

void SegmentationModule::applyEdits()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    const bool hadPendingChanges = _editManager->hasPendingChanges();
    const auto autosaveEdits = hadPendingChanges
        ? _editManager->editedVertices()
        : std::vector<SegmentationEditManager::VertexEdit>{};

    // Capture delta for undo before applyPreview() clears edited vertices
    if (hadPendingChanges) {
        (void)captureUndoDelta();
    }

    _editManager->applyPreview();
    if (_state) {
        auto preview = _editManager->previewSurface();
        _state->setSurface("segmentation", preview, false, true);
    }
    emitPendingChanges();
    queueAutosaveVertexUpdates(autosaveEdits);
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
    cancelDrag();
    _editManager->resetPreview();
    if (_state) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
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
    cancelCorrectionDrag();
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
        setAnnotateMode(false);
    } else if (_editingEnabled && _hoverPreviewEnabled && !_hover.valid) {
        recoverHoverPointerFromCursor();
        ensureHoverTarget();
    }
    updateCorrectionsWidget();
    refreshOverlay();
    emit growthInProgressChanged(_growthInProgress);
}

void SegmentationModule::emitPendingChanges()
{
    if (!_widget || !_editManager) {
        return;
    }
    const bool pending = _editManager->hasPendingChanges();
    _widget->setPendingChanges(pending);
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

    const bool hasSession = _editManager && _editManager->hasSession();

    // Add correction drag state (before hasSession check - corrections work without full editing session)
    if (_correctionDrag.active) {
        state.correctionDragActive = true;
        state.correctionDragStart = _correctionDrag.startWorld;
        state.correctionDragCurrent = _correctionDrag.currentWorld;
    }

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
    } else if (_hover.valid && _hoverPreviewEnabled) {
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
        const std::size_t stride = touched.size() > kMaxInteractiveNeighborMarkers
            ? (touched.size() + kMaxInteractiveNeighborMarkers - 1) / kMaxInteractiveNeighborMarkers
            : 1;
        state.neighbours.reserve(std::min(touched.size(), kMaxInteractiveNeighborMarkers));
        for (std::size_t i = 0; i < touched.size(); i += stride) {
            const auto& key = touched[i];
            if (key.row == _drag.row && key.col == _drag.col) {
                continue;
            }
            if (auto world = _editManager->vertexWorldPosition(key.row, key.col)) {
                state.neighbours.push_back({key.row, key.col, *world, false, false});
            }
        }
    }

    if (_surfaceMaskTool) {
        state.surfaceMaskPoints = _surfaceMaskTool->overlaySurfacePoints();
    }

    const bool surfaceMaskActive = _surfaceMaskTool && _surfaceMaskTool->active();
    const bool surfaceMaskStrokeActive = _surfaceMaskTool && _surfaceMaskTool->strokeActive();
    const bool pushPullActive = _pushPullTool && _pushPullTool->isActive();

    state.pushPullActive = pushPullActive;

    const FalloffTool overlayTool = pushPullActive ? FalloffTool::PushPull : _activeFalloff;

    state.displayRadiusSteps = falloffRadius(overlayTool);
    if (surfaceMaskActive || surfaceMaskStrokeActive) {
        state.displayRadiusSteps = _maskBrushRadius;
    }

    if (_manualAddMode && _manualAddTool) {
        state.manualAddActive = true;
        state.manualAddTintOpacity = _manualAddTool->config().tintOpacity;
        state.manualAddRevision = _manualAddTool->revision();
        const auto& preview = _manualAddTool->previewPoints();
        auto convertLine = [&](const ManualAddTool::GridPolyline& source) {
            SegmentationOverlayController::State::ManualAddLine line;
            line.committed = source.committed;
            line.surfacePoints.reserve(source.vertices.size());
            line.worldPoints.reserve(source.vertices.size());
            for (const auto& p : source.vertices) {
                const int row = p.y;
                const int col = p.x;
                line.surfacePoints.push_back(QPointF(col, row));
                if (row >= 0 && row < preview.rows && col >= 0 && col < preview.cols &&
                    vc::geometry::validSurfacePoint(preview(row, col))) {
                    line.worldPoints.push_back(preview(row, col));
                }
            }
            return line;
        };
        for (const auto& line : _manualAddTool->hoverPolylines()) {
            state.manualAddHoverLines.push_back(convertLine(line));
        }
        for (const auto& line : _manualAddTool->committedPolylines()) {
            state.manualAddCommittedLines.push_back(convertLine(line));
        }
        if (auto hover = _manualAddTool->hoverVertex()) {
            state.manualAddHoverVertex = QPointF(hover->col, hover->row);
            state.manualAddHoverCrossFill =
                _manualAddTool->config().linePreviewMode == ManualAddTool::LinePreviewMode::CrossFill;
        }
        std::vector<SegmentationEditManager::GridKey> hoverFillVertices = _manualAddTool->hoverFillVertices();
        std::sort(hoverFillVertices.begin(), hoverFillVertices.end(), [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.row, lhs.col) < std::tie(rhs.row, rhs.col);
        });
        for (const auto& key : hoverFillVertices) {
            if (!state.manualAddHoverFillSpans.empty()) {
                auto& span = state.manualAddHoverFillSpans.back();
                if (span.row == key.row && span.colLast + 1 == key.col) {
                    span.colLast = key.col;
                    continue;
                }
            }
            state.manualAddHoverFillSpans.push_back(
                SegmentationOverlayController::State::ManualAddFillSpan{key.row, key.col, key.col});
        }
        for (const auto& key : _manualAddTool->fillVertices()) {
            if (key.row >= 0 && key.row < preview.rows && key.col >= 0 && key.col < preview.cols &&
                vc::geometry::validSurfacePoint(preview(key.row, key.col))) {
                state.manualAddPreviewVertices.push_back(preview(key.row, key.col));
            }
        }
        std::set<std::pair<int, int>> previewCells;
        for (const auto& key : _manualAddTool->fillVertices()) {
            for (int row = key.row - 1; row <= key.row; ++row) {
                for (int col = key.col - 1; col <= key.col; ++col) {
                    if (row >= 0 && row + 1 < preview.rows && col >= 0 && col + 1 < preview.cols) {
                        previewCells.insert({row, col});
                    }
                }
            }
        }
        state.manualAddPreviewQuads.reserve(previewCells.size());
        for (const auto& [row, col] : previewCells) {
            const std::array<cv::Vec3f, 4> quad{
                preview(row, col),
                preview(row, col + 1),
                preview(row + 1, col + 1),
                preview(row + 1, col)
            };
            if (std::all_of(quad.begin(), quad.end(), [](const cv::Vec3f& point) {
                    return vc::geometry::validSurfacePoint(point);
                })) {
                state.manualAddPreviewQuads.push_back(quad);
            }
        }
        for (const auto& constraint : _manualAddTool->userPlaneConstraints()) {
            state.manualAddPlaneConstraints.push_back(constraint.world);
        }
    }

    _overlay->applyState(state);
}



void SegmentationModule::updateCorrectionsWidget()
{
    if (_corrections) {
        _corrections->refreshWidget();
    }
}

void SegmentationModule::setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated)
{
    if (_corrections) {
        _corrections->setActiveCollection(collectionId, userInitiated);
    }
}

void SegmentationModule::clearActiveCorrectionCollection()
{
    if (_corrections) {
        _corrections->clearActiveCollection();
    }
}

uint64_t SegmentationModule::createCorrectionCollection(bool announce)
{
    return _corrections ? _corrections->createCollection(announce) : 0;
}

void SegmentationModule::handleCorrectionPointAdded(
    const cv::Vec3f& worldPos, std::optional<uint64_t> collectionId)
{
    if (!_corrections) return;

    // If a specific collection is requested, switch to it
    if (collectionId) {
        _corrections->setActiveCollection(*collectionId, false);
    }

    // Auto-create collection on first annotation
    if (!_corrections->hasActiveCollection()) {
        createCorrectionCollection(false);
    }

    // Look up winding depth index from d.tif channel → store in winding_annotation
    float wind_a = NAN;
    QuadSurface* surface = activeBaseSurface();
    std::shared_ptr<Surface> surfaceHolder;

    // Fallback: get surface from state if no editing session
    if (!surface && _state) {
        surfaceHolder = _state->surface("segmentation");
        surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
    }

    if (surface) {
        // Try edit manager first (has preview points if editing)
        std::optional<std::pair<int, int>> gridIdx;
        if (_editManager && _editManager->hasSession()) {
            gridIdx = _editManager->worldToGridIndex(worldPos);
        }

        // Otherwise where the surface (stored, or a draft) lies at the point.
        if (!gridIdx && _viewerManager) {
            const auto* pts = surface->rawPointsPtr();
            if (pts && pts->rows > 0 && pts->cols > 0) {
                const float tol = std::max(surface->scale()[0] * 64.0f, 512.0f);
                if (const auto hit = _viewerManager->locate(*surface, worldPos, tol)) {
                    const cv::Vec2f grid = surface->ptrToGrid(hit->ptr);
                    const int col = std::clamp(static_cast<int>(std::round(grid[0])), 0, pts->cols - 1);
                    const int row = std::clamp(static_cast<int>(std::round(grid[1])), 0, pts->rows - 1);
                    gridIdx = {row, col};
                }
            }
        }

        if (gridIdx) {
            wind_a = lookupDepthIndex(surface, gridIdx->first, gridIdx->second);
        }
    }
    qCInfo(lcSegModule) << "handleCorrectionPointAdded: wind_a=" << wind_a;
    _corrections->handlePointAdded(worldPos, wind_a);
}

void SegmentationModule::handleCorrectionPointRemove(const cv::Vec3f& worldPos)
{
    if (_corrections) {
        _corrections->handlePointRemoved(worldPos);
    }
}

void SegmentationModule::beginCorrectionDrag(int row, int col, VolumeViewerBase* viewer, const cv::Vec3f& worldPos)
{
    _correctionDrag.active = true;
    _correctionDrag.anchorRow = row;
    _correctionDrag.anchorCol = col;
    _correctionDrag.startWorld = worldPos;
    _correctionDrag.currentWorld = worldPos;
    _correctionDrag.viewer = viewer;
    _correctionDrag.moved = false;

    qCInfo(lcSegModule) << "Correction drag started at grid" << row << col << "world" << worldPos[0] << worldPos[1] << worldPos[2];
    emit statusMessageRequested(tr("Drag to correction target position..."), kStatusShort);
    refreshOverlay();
}

void SegmentationModule::updateCorrectionDrag(const cv::Vec3f& worldPos)
{
    if (!_correctionDrag.active) {
        return;
    }

    const cv::Vec3f delta = worldPos - _correctionDrag.startWorld;
    const float distance = cv::norm(delta);
    if (distance > 1.0f) {
        _correctionDrag.moved = true;
    }
    _correctionDrag.currentWorld = worldPos;

    // TODO: Add visual feedback (line from start to current)
    refreshOverlay();
}

void SegmentationModule::finishCorrectionDrag()
{
    if (!_correctionDrag.active) {
        return;
    }

    const bool didMove = _correctionDrag.moved;
    const cv::Vec3f targetWorld = _correctionDrag.currentWorld;
    const int anchorRow = _correctionDrag.anchorRow;
    const int anchorCol = _correctionDrag.anchorCol;

    _correctionDrag.reset();

    if (!didMove) {
        // User clicked without dragging - fall back to old behavior (add single point)
        handleCorrectionPointAdded(targetWorld);
        updateCorrectionsWidget();
        return;
    }

    // Create correction with anchor2d
    if (!_corrections || !_pointCollection) {
        emit statusMessageRequested(tr("No correction collection available"), kStatusMedium);
        return;
    }

    // Ensure we have an active collection
    auto collectionId = _corrections->activeCollection();
    if (!collectionId) {
        collectionId = _corrections->createCollection(true);
        if (!collectionId || *collectionId == 0) {
            emit statusMessageRequested(tr("Failed to create correction collection"), kStatusMedium);
            return;
        }
    }

    // Set anchor2d on the collection (the grid location where user started dragging)
    cv::Vec2f anchor2d(static_cast<float>(anchorCol), static_cast<float>(anchorRow));
    _pointCollection->setCollectionAnchor2d(*collectionId, anchor2d);

    // Look up winding depth index from d.tif at the anchor position → store in winding_annotation
    float wind_a = lookupDepthIndex(activeBaseSurface(), anchorRow, anchorCol);

    // Add the correction point (3D world target)
    _corrections->handlePointAdded(targetWorld, wind_a);

    qCInfo(lcSegModule) << "Correction drag completed: anchor2d" << anchorCol << anchorRow
                        << "target" << targetWorld[0] << targetWorld[1] << targetWorld[2]
                        << "wind_a" << wind_a;

    updateCorrectionsWidget();

    // Immediately trigger the solver with corrections
    emit statusMessageRequested(tr("Applying correction..."), kStatusShort);
    handleGrowSurfaceRequested(SegmentationGrowthMethod::Corrections,
                               SegmentationGrowthDirection::All,
                               0,
                               false);
}

void SegmentationModule::cancelCorrectionDrag()
{
    if (_correctionDrag.active) {
        _correctionDrag.reset();
        refreshOverlay();
        emit statusMessageRequested(tr("Correction drag cancelled"), kStatusShort);
    }
}

SegmentationModule::NearestPointResult SegmentationModule::findNearestPoint(const cv::Vec3f& worldPos, float maxDist)
{
    NearestPointResult result;
    if (!_pointCollection) {
        return result;
    }

    const auto& collections = _pointCollection->getAllCollections();
    for (const auto& [colId, col] : collections) {
        for (const auto& [ptId, pt] : col.points) {
            const float dist = static_cast<float>(cv::norm(pt.p - worldPos));
            if (dist < result.distance && dist <= maxDist) {
                result.point = vc::PointRef{colId, ptId};
                result.distance = dist;
            }
        }
    }
    return result;
}

void SegmentationModule::setSelectedAnnotationCollection(uint64_t collectionId)
{
    _selectedAnnotationCollectionId = collectionId;
}

void SegmentationModule::clearSelectedAnnotationCollection()
{
    _selectedAnnotationCollectionId.reset();
}

void SegmentationModule::beginPointMoveDrag(uint64_t pointId, uint64_t collectionId,
                                            VolumeViewerBase* viewer, const cv::Vec3f& worldPos)
{
    _pointMoveDrag.active = true;
    _pointMoveDrag.pointId = pointId;
    _pointMoveDrag.collectionId = collectionId;
    _pointMoveDrag.startWorld = worldPos;
    _pointMoveDrag.currentWorld = worldPos;
    _pointMoveDrag.viewer = viewer;
    _pointMoveDrag.moved = false;
}

void SegmentationModule::updatePointMoveDrag(const cv::Vec3f& worldPos)
{
    if (!_pointMoveDrag.active) {
        return;
    }

    const cv::Vec3f delta = worldPos - _pointMoveDrag.startWorld;
    const float distance = cv::norm(delta);
    if (distance > 1.0f) {
        _pointMoveDrag.moved = true;
    }
    _pointMoveDrag.currentWorld = worldPos;
}

void SegmentationModule::finishPointMoveDrag()
{
    if (!_pointMoveDrag.active) {
        return;
    }

    const bool didMove = _pointMoveDrag.moved;
    const uint64_t pointId = _pointMoveDrag.pointId;
    const uint64_t collectionId = _pointMoveDrag.collectionId;
    const cv::Vec3f targetWorld = _pointMoveDrag.currentWorld;

    _pointMoveDrag.reset();

    if (didMove) {
        // Update point position
        if (_pointCollection) {
            auto ptOpt = _pointCollection->getPoint({collectionId, pointId});
            if (ptOpt) {
                ColPoint updated = *ptOpt;
                updated.p = targetWorld;
                _pointCollection->updatePoint(updated);
                qCInfo(lcSegModule) << "Moved point" << pointId << "to"
                                    << targetWorld[0] << targetWorld[1] << targetWorld[2];
            }
        }
        updateCorrectionsWidget();
    } else {
        // Click without drag: select the point
        emit annotationPointSelected({collectionId, pointId});
        emit annotationCollectionSelected(collectionId);
    }
}

void SegmentationModule::onCorrectionsCreateRequested()
{
    if (!_corrections) {
        return;
    }

    _corrections->createCollection(true);
}

void SegmentationModule::onCorrectionsCollectionSelected(uint64_t id)
{
    setActiveCorrectionCollection(id, true);
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
        _corrections->clearAll();
    }
}

std::optional<std::pair<int, int>> SegmentationModule::correctionsZRange() const
{
    return _corrections ? _corrections->zRange() : std::nullopt;
}

SegmentationCorrectionsPayload SegmentationModule::buildCorrectionsPayload(bool onlyActiveCollection) const
{
    if (!_corrections) {
        return SegmentationCorrectionsPayload{};
    }

    return _corrections->buildPayload(onlyActiveCollection);
}
void SegmentationModule::handleGrowSurfaceRequested(SegmentationGrowthMethod method,
                                                    SegmentationGrowthDirection direction,
                                                    int steps,
                                                    bool inpaintOnly)
{
    if (method == SegmentationGrowthMethod::ManualAdd) {
        emit statusMessageRequested(tr("Manual Add is interactive; use Shift+E and the Manual Add panel."), kStatusMedium);
        return;
    }
    if (_manualAddMode) {
        emit statusMessageRequested(tr("Apply or cancel Manual Add before running growth."), kStatusMedium);
        return;
    }

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    int sanitizedSteps = allowZeroSteps ? std::max(0, steps) : std::max(1, steps);
    const bool usingCorrections = !inpaintOnly &&
                                  method == SegmentationGrowthMethod::Corrections &&
                                  _corrections && _corrections->hasCorrections();
    if (usingCorrections) {
        sanitizedSteps = 0;
    }

    qCInfo(lcSegModule) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << sanitizedSteps
                        << "inpaintOnly" << inpaintOnly;

    if (_growthInProgress) {
        emit statusMessageRequested(tr("Surface growth already in progress"), kStatusMedium);
        return;
    }
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        emit statusMessageRequested(tr("Enable segmentation editing before growing surfaces"), kStatusMedium);
        return;
    }

    if (!inpaintOnly) {
        _growthMethod = method;
        if (method == SegmentationGrowthMethod::Corrections) {
            _growthSteps = std::max(0, steps);
        } else {
            _growthSteps = std::max(1, steps);
        }
    }
    markNextEditsFromGrowth();
    emit growSurfaceRequested(method, direction, sanitizedSteps, inpaintOnly);
}

cv::Mat SegmentationModule::takePendingManualAddTracerMask()
{
    cv::Mat mask = _pendingManualAddTracerMask;
    _pendingManualAddTracerMask.release();
    return mask;
}

bool SegmentationModule::applyManualAddTracerPreview(QuadSurface* surface)
{
    if (!_manualAddMode || !_manualAddTool || !_editManager || !surface) {
        return false;
    }

    const auto& snapshot = _manualAddTool->entrySnapshotPoints();
    const auto& fill = _manualAddTool->fillVertices();
    const auto* tracedPoints = surface->rawPointsPtr();
    if (snapshot.empty() || fill.empty() || !tracedPoints || tracedPoints->empty()) {
        return false;
    }

    int offsetCol = 0;
    int offsetRow = 0;
    if (!surface->meta.is_null() && surface->meta.is_object() && surface->meta.contains("grid_offset")) {
        const auto& offset = surface->meta["grid_offset"];
        if (offset.is_array() && offset.size() >= 2 && offset[0].is_number() && offset[1].is_number()) {
            offsetCol = offset[0].get_int();
            offsetRow = offset[1].get_int();
        }
    }

    struct TracerFillPoint
    {
        SegmentationEditManager::GridKey key;
        cv::Vec3f world;
    };

    auto toTpsSample = [](const SegmentationEditManager::GridKey& key, const cv::Vec3f& world) {
        return ThinPlateSpline3d::Sample{cv::Point2d(key.col, key.row),
                                         cv::Vec3d(world[0], world[1], world[2])};
    };

    auto appendChanged = [](std::vector<SegmentationEditManager::GridKey>& target,
                            std::set<std::pair<int, int>>& seen,
                            const SegmentationEditManager::GridKey& key) {
        if (seen.insert({key.row, key.col}).second) {
            target.push_back(key);
        }
    };

    cv::Mat_<cv::Vec3f> rawPreview = snapshot.clone();
    std::vector<SegmentationEditManager::GridKey> rawChanged;
    std::set<std::pair<int, int>> rawChangedSeen;
    std::vector<TracerFillPoint> tracedFillPoints;
    rawChanged.reserve(fill.size());
    tracedFillPoints.reserve(fill.size());

    for (const auto& key : fill) {
        const int srcRow = key.row + offsetRow;
        const int srcCol = key.col + offsetCol;
        if (key.row < 0 || key.row >= rawPreview.rows || key.col < 0 || key.col >= rawPreview.cols ||
            srcRow < 0 || srcRow >= tracedPoints->rows || srcCol < 0 || srcCol >= tracedPoints->cols) {
            continue;
        }

        const cv::Vec3f& point = (*tracedPoints)(srcRow, srcCol);
        if (!vc::geometry::validSurfacePoint(point)) {
            continue;
        }
        rawPreview(key.row, key.col) = point;
        appendChanged(rawChanged, rawChangedSeen, key);
        tracedFillPoints.push_back(TracerFillPoint{key, point});
    }

    if (rawChanged.empty()) {
        return false;
    }

    cv::Mat_<cv::Vec3f> preview = rawPreview;
    std::vector<SegmentationEditManager::GridKey> changed = rawChanged;

    const auto config = _manualAddTool->config();
    if (config.allowBoundarySmoothing) {
        std::vector<ThinPlateSpline3d::Sample> boundarySamples;
        std::vector<SegmentationEditManager::GridKey> boundaryKeys;
        boundarySamples.reserve(_manualAddTool->borderSampleVertices().size());
        boundaryKeys.reserve(_manualAddTool->borderSampleVertices().size());
        for (const auto& key : _manualAddTool->borderSampleVertices()) {
            if (key.row < 0 || key.row >= snapshot.rows || key.col < 0 || key.col >= snapshot.cols) {
                continue;
            }
            const cv::Vec3f& point = snapshot(key.row, key.col);
            if (!vc::geometry::validSurfacePoint(point)) {
                continue;
            }
            boundarySamples.push_back(toTpsSample(key, point));
            boundaryKeys.push_back(key);
        }

        std::vector<ThinPlateSpline3d::Sample> fillSamples;
        fillSamples.reserve(tracedFillPoints.size());
        for (const auto& point : tracedFillPoints) {
            fillSamples.push_back(toTpsSample(point.key, point.world));
        }

        const int sampleCap = std::max(3, config.sampleCap);
        std::vector<ThinPlateSpline3d::Sample> tpsSamples;
        tpsSamples.reserve(static_cast<std::size_t>(sampleCap));

        auto appendSampleSubset = [](const std::vector<ThinPlateSpline3d::Sample>& source,
                                     int limit,
                                     std::vector<ThinPlateSpline3d::Sample>& target) {
            if (limit <= 0 || source.empty()) {
                return;
            }
            if (static_cast<int>(source.size()) <= limit) {
                target.insert(target.end(), source.begin(), source.end());
                return;
            }
            for (int i = 0; i < limit; ++i) {
                const std::size_t index = static_cast<std::size_t>(i) * source.size() / static_cast<std::size_t>(limit);
                target.push_back(source[std::min(index, source.size() - 1)]);
            }
        };

        const int totalSamples = static_cast<int>(boundarySamples.size() + fillSamples.size());
        if (totalSamples <= sampleCap) {
            appendSampleSubset(boundarySamples, static_cast<int>(boundarySamples.size()), tpsSamples);
            appendSampleSubset(fillSamples, static_cast<int>(fillSamples.size()), tpsSamples);
        } else {
            int boundaryLimit = std::min<int>(boundarySamples.size(), sampleCap);
            int fillLimit = std::min<int>(fillSamples.size(), sampleCap - boundaryLimit);
            const int desiredFill = std::min<int>(fillSamples.size(), std::max(1, sampleCap / 3));
            if (fillLimit < desiredFill && boundaryLimit > 0) {
                const int shift = std::min(boundaryLimit, desiredFill - fillLimit);
                boundaryLimit -= shift;
                fillLimit += shift;
            }
            appendSampleSubset(boundarySamples, boundaryLimit, tpsSamples);
            appendSampleSubset(fillSamples, fillLimit, tpsSamples);
        }

        ThinPlateSpline3d tps;
        if (tps.fit(tpsSamples, config.regularization)) {
            cv::Mat_<cv::Vec3f> smoothedPreview = snapshot.clone();
            std::vector<SegmentationEditManager::GridKey> smoothedChanged;
            std::set<std::pair<int, int>> smoothedChangedSeen;
            smoothedChanged.reserve(tracedFillPoints.size() + boundaryKeys.size());
            bool smoothingOk = true;

            for (const auto& point : tracedFillPoints) {
                auto predicted = tps.evaluate(cv::Point2d(point.key.col, point.key.row));
                if (!predicted || !vc::geometry::validSurfacePoint(*predicted)) {
                    smoothingOk = false;
                    break;
                }
                smoothedPreview(point.key.row, point.key.col) = *predicted;
                appendChanged(smoothedChanged, smoothedChangedSeen, point.key);
            }

            if (smoothingOk) {
                for (const auto& key : boundaryKeys) {
                    auto predicted = tps.evaluate(cv::Point2d(key.col, key.row));
                    if (!predicted || !vc::geometry::validSurfacePoint(*predicted)) {
                        smoothingOk = false;
                        break;
                    }
                    smoothedPreview(key.row, key.col) = *predicted;
                    appendChanged(smoothedChanged, smoothedChangedSeen, key);
                }
            }

            if (smoothingOk && !smoothedChanged.empty()) {
                preview = std::move(smoothedPreview);
                changed = std::move(smoothedChanged);
            } else {
                qCInfo(lcSegModule) << "Manual Add tracer TPS smoothing failed; using raw tracer preview.";
            }
        } else {
            qCInfo(lcSegModule) << "Manual Add tracer TPS smoothing could not fit; using raw tracer preview.";
        }
    }

    std::optional<cv::Rect> bounds;
    if (!_editManager->setPreviewPointsOnly(preview, changed, true, &bounds)) {
        return false;
    }
    if (_state && _editManager->previewSurface()) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }
    emitPendingChanges();
    refreshOverlay();
    return true;
}

bool SegmentationModule::beginManualAdd()
{
    if (_manualAddMode) {
        return true;
    }
    if (!_widget || !_editManager) {
        return false;
    }
    if (!_editingEnabled) {
        setEditingEnabled(true);
        _widget->setEditingEnabled(true);
    }
    if (!_editManager->hasSession()) {
        emit statusMessageRequested(tr("Open a segmentation editing session before Manual Add."), kStatusMedium);
        return false;
    }
    if (_editManager->hasPendingChanges()) {
        emit statusMessageRequested(tr("Apply or cancel pending edits before Manual Add."), kStatusMedium);
        return false;
    }
    if (_autosaveState.saveInProgress() || _autosaveState.pending()) {
        emit statusMessageRequested(tr("Wait for the current save before Manual Add."), kStatusMedium);
        return false;
    }

    stopAllPushPull();
    cancelDrag();
    cancelCorrectionDrag();
    setAnnotateMode(false);
    _previousGrowthMethodBeforeManualAdd = _widget->growthMethod();
    if (_previousGrowthMethodBeforeManualAdd == SegmentationGrowthMethod::ManualAdd) {
        _previousGrowthMethodBeforeManualAdd = _widget->lastNonManualGrowthMethod();
    }
    _widget->setGrowthMethod(SegmentationGrowthMethod::ManualAdd);
    if (!_manualAddTool) {
        _manualAddTool = std::make_unique<ManualAddTool>();
    }
    if (!_manualAddTool->begin(_editManager->previewPoints(), _widget->manualAddConfig())) {
        emit statusMessageRequested(tr("Manual Add could not read the active surface."), kStatusMedium);
        return false;
    }
    _manualAddMode = true;
    _manualAddHoverThrottle.invalidate();
    _widget->setManualAddActive(true);
    refreshOverlay();
    emit statusMessageRequested(tr("Manual Add enabled."), kStatusShort);
    return true;
}

bool SegmentationModule::finishManualAdd(bool apply)
{
    if (!_manualAddMode || !_manualAddTool) {
        return false;
    }

    if (apply) {
        if (_manualAddTool->config().interpolationMode == ManualAddTool::InterpolationMode::TracerRestrictedToFill) {
            if (!_editManager || !_editManager->hasPendingChanges()) {
                emit statusMessageRequested(tr("Manual Add tracer preview is not ready yet."), kStatusMedium);
                return false;
            }
        } else if (_editManager) {
            std::optional<cv::Rect> bounds;
            if (!_editManager->setPreviewPointsOnly(_manualAddTool->previewPoints(),
                                                    _manualAddTool->changedVertices(),
                                                    true,
                                                    &bounds)) {
                emit statusMessageRequested(tr("Manual Add final preview update failed."), kStatusMedium);
                return false;
            }
            if (_state && _editManager->previewSurface()) {
                _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
            }
            emitPendingChanges();
            refreshOverlay();
        }
        if (_editManager && _editManager->hasPendingChanges()) {
            applyEdits();
        }
        emit statusMessageRequested(tr("Manual Add applied and saved."), kStatusMedium);
    } else {
        if (_editManager) {
            _editManager->restorePreviewSnapshot(_manualAddTool->entrySnapshotPoints());
            if (_state && _editManager->previewSurface()) {
                _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
            }
        }
        emitPendingChanges();
        emit statusMessageRequested(tr("Manual Add canceled."), kStatusShort);
    }

    _manualAddMode = false;
    _manualAddHoverThrottle.invalidate();
    _manualAddTool->clear();
    if (_widget) {
        _widget->setManualAddActive(false);
        const auto fallback = _previousGrowthMethodBeforeManualAdd == SegmentationGrowthMethod::ManualAdd
            ? SegmentationGrowthMethod::Tracer
            : _previousGrowthMethodBeforeManualAdd;
        _widget->setGrowthMethod(fallback);
    }
    refreshOverlay();
    return true;
}

bool SegmentationModule::setManualAddModeActive(bool active, bool apply)
{
    if (active) {
        return beginManualAdd();
    }
    return finishManualAdd(apply);
}

bool SegmentationModule::undoManualAddConstraint()
{
    return undoManualAddPlaneConstraint();
}

bool SegmentationModule::setCorrectionPointMode(bool active, QString* errorMessage)
{
    if (!active) {
        _correctionDragKeyActive = false;
        return true;
    }
    // Mirror the G-key preconditions (SegmentationModule_Input.cpp handleKeyPress).
    if (!_editingEnabled) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("editing_disabled");
        }
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("no_session");
        }
        return false;
    }
    if (_growthInProgress) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("growth_in_progress");
        }
        return false;
    }
    _correctionDragKeyActive = true;
    return true;
}

void SegmentationModule::resetManualAddState(bool restorePreview)
{
    _pendingManualAddTracerMask.release();

    if (_manualAddMode && _manualAddTool && restorePreview && _editManager) {
        _editManager->restorePreviewSnapshot(_manualAddTool->entrySnapshotPoints());
        if (_state && _editManager->previewSurface()) {
            _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
        }
        emitPendingChanges();
    }

    if (_manualAddTool) {
        _manualAddTool->clear();
    }

    const bool wasManualAddMode = _manualAddMode;
    _manualAddMode = false;
    _manualAddHoverThrottle.invalidate();

    if (_widget) {
        _widget->setManualAddActive(false);
        if (_widget->growthMethod() == SegmentationGrowthMethod::ManualAdd) {
            const auto fallback = _previousGrowthMethodBeforeManualAdd == SegmentationGrowthMethod::ManualAdd
                ? SegmentationGrowthMethod::Tracer
                : _previousGrowthMethodBeforeManualAdd;
            _widget->setGrowthMethod(fallback);
        }
    }

    if (wasManualAddMode) {
        refreshOverlay();
    }
}

bool SegmentationModule::recomputeManualAdd()
{
    if (!_manualAddMode || !_manualAddTool || !_editManager) {
        return false;
    }
    std::string status;
    if (!_manualAddTool->recompute(&status)) {
        emit statusMessageRequested(QString::fromStdString(status), kStatusMedium);
        return false;
    }

    if (_manualAddTool->config().interpolationMode == ManualAddTool::InterpolationMode::TracerRestrictedToFill) {
        if (_growthInProgress) {
            emit statusMessageRequested(tr("Surface growth already in progress"), kStatusMedium);
            return false;
        }

        const auto& snapshot = _manualAddTool->entrySnapshotPoints();
        const auto& fill = _manualAddTool->fillVertices();
        if (snapshot.empty() || fill.empty()) {
            emit statusMessageRequested(tr("Manual Add tracer fill is empty."), kStatusMedium);
            return false;
        }

        cv::Mat_<uint8_t> mask(snapshot.rows, snapshot.cols, static_cast<uint8_t>(0));
        for (const auto& key : fill) {
            if (key.row >= 0 && key.row < mask.rows && key.col >= 0 && key.col < mask.cols) {
                mask(key.row, key.col) = 255;
            }
        }
        _pendingManualAddTracerMask = mask;

        emit statusMessageRequested(tr("Running Manual Add tracer inside filled area..."), kStatusMedium);
        markNextEditsFromGrowth();
        emit growSurfaceRequested(SegmentationGrowthMethod::Tracer,
                                  SegmentationGrowthDirection::Fill,
                                  std::max(1, static_cast<int>(fill.size())),
                                  false);
        refreshOverlay();
        return true;
    }

    std::optional<cv::Rect> bounds;
    if (!_editManager->setPreviewPointsOnly(_manualAddTool->previewPoints(),
                                            _manualAddTool->changedVertices(),
                                            true,
                                            &bounds)) {
        emit statusMessageRequested(tr("Manual Add preview update failed."), kStatusMedium);
        return false;
    }
    if (_state && _editManager->previewSurface()) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }
    emitPendingChanges();
    refreshOverlay();
    emit statusMessageRequested(QString::fromStdString(status), kStatusShort);
    return true;
}

bool SegmentationModule::clearManualAddPending()
{
    if (!_manualAddMode || !_manualAddTool || !_editManager) {
        return false;
    }
    if (!_manualAddTool->clearPending(_widget ? _widget->manualAddConfig() : ManualAddTool::Config{})) {
        return false;
    }
    _editManager->restorePreviewSnapshot(_manualAddTool->entrySnapshotPoints());
    if (_state && _editManager->previewSurface()) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }
    emitPendingChanges();
    refreshOverlay();
    emit statusMessageRequested(_manualAddTool->initialFillCommitted()
                                    ? tr("Manual Add pending geometry cleared. Press Shift+E to save, then enable Manual Add again for another fill.")
                                    : tr("Manual Add pending geometry cleared."),
                                kStatusShort);
    return true;
}

bool SegmentationModule::undoManualAddPlaneConstraint()
{
    if (!_manualAddMode || !_manualAddTool || !_editManager) {
        return false;
    }

    std::string status;
    if (!_manualAddTool->removeLastPlaneConstraint(&status)) {
        emit statusMessageRequested(QString::fromStdString(status), kStatusShort);
        return false;
    }

    std::optional<cv::Rect> bounds;
    if (!_editManager->setPreviewPointsOnly(_manualAddTool->previewPoints(),
                                            _manualAddTool->changedVertices(),
                                            true,
                                            &bounds)) {
        emit statusMessageRequested(tr("Manual Add preview update failed."), kStatusMedium);
        return false;
    }
    if (_state && _editManager->previewSurface()) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }
    emitPendingChanges();
    refreshOverlay();
    emit statusMessageRequested(QString::fromStdString(status), kStatusShort);
    return true;
}

bool SegmentationModule::handleManualAddMousePress(VolumeViewerBase* viewer,
                                                   const cv::Vec3f& worldPos,
                                                   Qt::MouseButton button,
                                                   Qt::KeyboardModifiers modifiers,
                                                   const QPointF& scenePos)
{
    if (!_manualAddMode || !_manualAddTool || !viewer) {
        return false;
    }
    if (viewer->surfName() == "segmentation" && button == Qt::LeftButton && modifiers == Qt::NoModifier) {
        const auto grid = segmentationSceneToGrid(viewer,
                                                  scenePos,
                                                  _manualAddTool->entrySnapshotPoints().rows,
                                                  _manualAddTool->entrySnapshotPoints().cols);
        if (grid) {
            _manualAddTool->updateHover(grid->first, grid->second);
        }
        _manualAddHoverThrottle.restart();

        std::string status;
        if (!_manualAddTool->commitHover(&status)) {
            emit statusMessageRequested(QString::fromStdString(status), kStatusMedium);
            return true;
        }
        return recomputeManualAdd();
    }

    auto* planeSurf = dynamic_cast<PlaneSurface*>(viewer->currentSurface());
    if (planeSurf && modifiers.testFlag(Qt::ShiftModifier) &&
        (button == Qt::LeftButton || button == Qt::RightButton)) {
        if (button == Qt::RightButton) {
            std::string status;
            _manualAddTool->removePlaneConstraintNear(worldPos, _manualAddTool->config().planeConstraintRadius, &status);
            emit statusMessageRequested(QString::fromStdString(status), kStatusShort);
            return recomputeManualAdd();
        }

        const auto& preview = _manualAddTool->previewPoints();
        const cv::Vec2f clickPlane = viewer->sceneToSurfaceCoords(scenePos);
        const double radiusSq = _manualAddTool->config().planeConstraintRadius *
                                _manualAddTool->config().planeConstraintRadius;
        double bestDistSq = radiusSq;
        std::optional<SegmentationEditManager::GridKey> best;
        for (const auto& key : _manualAddTool->fillVertices()) {
            if (key.row < 0 || key.row >= preview.rows || key.col < 0 || key.col >= preview.cols) {
                continue;
            }
            const cv::Vec3f projected = planeSurf->project(preview(key.row, key.col), 1.0f, 1.0f);
            const double dx = static_cast<double>(projected[0]) - clickPlane[0];
            const double dy = static_cast<double>(projected[1]) - clickPlane[1];
            const double distSq = dx * dx + dy * dy;
            if (distSq <= bestDistSq) {
                bestDistSq = distSq;
                best = key;
            }
        }
        std::string status;
        if (!best || !_manualAddTool->addOrReplacePlaneConstraint(best->row, best->col, worldPos, &status)) {
            emit statusMessageRequested(QString::fromStdString(status), kStatusMedium);
            return true;
        }
        return recomputeManualAdd();
    }
    Q_UNUSED(scenePos);
    return true;
}

bool SegmentationModule::handleManualAddMouseMove(VolumeViewerBase* viewer,
                                                  Qt::MouseButtons buttons,
                                                  const QPointF& scenePos)
{
    if (!_manualAddMode || !_manualAddTool || !viewer || buttons != Qt::NoButton) {
        return false;
    }
    if (viewer->surfName() != "segmentation") {
        return true;
    }
    const auto grid = segmentationSceneToGrid(viewer,
                                              scenePos,
                                              _manualAddTool->entrySnapshotPoints().rows,
                                              _manualAddTool->entrySnapshotPoints().cols);
    bool gridIsInvalid = false;
    if (grid) {
        const auto& entry = _manualAddTool->entrySnapshotPoints();
        gridIsInvalid = grid->first >= 0 && grid->first < entry.rows &&
                        grid->second >= 0 && grid->second < entry.cols &&
                        !vc::geometry::validSurfacePoint(entry(grid->first, grid->second));
    }
    const int throttleMs = _manualAddTool->config().previewThrottleMs;
    if (gridIsInvalid && throttleMs > 0 &&
        _manualAddTool->config().linePreviewMode != ManualAddTool::LinePreviewMode::CrossFill &&
        _manualAddHoverThrottle.isValid() && _manualAddHoverThrottle.elapsed() < throttleMs) {
        return true;
    }

    const bool changed = grid ? _manualAddTool->updateHover(grid->first, grid->second)
                              : _manualAddTool->updateHover(-1, -1);
    _manualAddHoverThrottle.restart();
    if (changed) {
        refreshOverlay();
    }
    return true;
}

bool SegmentationModule::isSegmentationViewer(const VolumeViewerBase* viewer) const
{
    if (!viewer) {
        return false;
    }
    const std::string& name = viewer->surfName();
    return name.rfind("seg", 0) == 0 || name == "xy plane";
}

float SegmentationModule::gridStepWorld() const
{
    float result = 1.0f;
    const QuadSurface* surface = nullptr;

    if (!_editManager || !_editManager->hasSession()) {
        // Without an active session, use the base surface's scale if there is one
        if (_editManager && _editManager->baseSurface()) {
            surface = _editManager->baseSurface().get();
            result = averageScale(surface->scale());
        }
    } else {
        surface = _editManager->previewSurface().get();
        if (!surface) {
            surface = _editManager->baseSurface().get();
        }
        if (surface) {
            result = averageScale(surface->scale());
        }
    }

    return result;
}

void SegmentationModule::beginDrag(int row, int col, VolumeViewerBase* viewer, const cv::Vec3f& worldPos)
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

    if (!_editManager->updateActiveDrag(worldPos)) {
        return;
    }

    _drag.lastWorld = worldPos;
    _drag.moved = true;

    if (_drag.viewer)
        refreshEditedRegion(_drag.viewer);

    refreshOverlay();
    emitPendingChanges();
}

void SegmentationModule::refreshEditedRegion(VolumeViewerBase* editViewer)
{
    if (!_editManager)
        return;
    std::optional<cv::Rect> changedCells;
    if (const auto touched = _editManager->recentTouchedBounds()) {
        changedCells = cv::Rect(touched->x - 1, touched->y - 1,
                                touched->width + 2, touched->height + 2);
    }
    const auto refresh = [&](VolumeViewerBase* viewer) {
        if (changedCells)
            viewer->invalidateVisRegion("segmentation", *changedCells);
        else
            viewer->invalidateVis();
        viewer->requestRender("segmentation edit refresh");
    };
    if (editViewer)
        refresh(editViewer);
    // An edit made on a slice's intersection also moves the flattened view's
    // geometry. Left out here, that view learns of the edit only from the
    // final surface update, which drops its whole tile cache: a black flash.
    if (_viewerManager) {
        _viewerManager->forEachBaseViewer([&](VolumeViewerBase* viewer) {
            if (viewer && viewer != editViewer && viewer->surfName() == "segmentation")
                refresh(viewer);
        });
    }
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
        const auto editedVerts = _editManager->editedVertices();

        // Capture delta for undo before applyPreview() clears edited vertices
        (void)captureUndoDelta();

        _editManager->applyPreview();
        if (_state) {
            _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
        }
        queueAutosaveVertexUpdates(editedVerts);
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

bool SegmentationModule::isNearRotationHandle(VolumeViewerBase* viewer, const cv::Vec3f& worldPos) const
{
    if (!_rotationHandleHitTester || !viewer) {
        return false;
    }
    return _rotationHandleHitTester(viewer, worldPos);
}

bool SegmentationModule::recoverHoverPointerFromCursor()
{
    const QPoint globalCursor = QCursor::pos();

    VolumeViewerBase* targetViewer = nullptr;
    if (QWidget* widget = QApplication::widgetAt(globalCursor)) {
        for (QWidget* current = widget; current && !targetViewer; current = current->parentWidget()) {
            for (auto* viewer : std::as_const(_attachedViewers)) {
                if (!viewer || !isSegmentationViewer(viewer)) {
                    continue;
                }
                if (current == qobject_cast<QWidget*>(viewer->asQObject())) {
                    targetViewer = viewer;
                    break;
                }
            }
        }
    }

    if (!targetViewer) {
        for (auto* viewer : std::as_const(_attachedViewers)) {
            if (!viewer || !isSegmentationViewer(viewer)) {
                continue;
            }
            auto* graphicsView = viewer->graphicsView();
            if (!graphicsView || !graphicsView->viewport()) {
                continue;
            }
            const QPoint viewportCursor = graphicsView->viewport()->mapFromGlobal(globalCursor);
            if (graphicsView->viewport()->rect().contains(viewportCursor)) {
                targetViewer = viewer;
                break;
            }
        }
    }

    if (targetViewer) {
        auto* graphicsView = targetViewer->graphicsView();
        if (!graphicsView || !graphicsView->viewport()) {
            return false;
        }
        const QPoint viewportCursor = graphicsView->viewport()->mapFromGlobal(globalCursor);
        if (!graphicsView->viewport()->rect().contains(viewportCursor)) {
            return false;
        }
        const QPointF scenePos = graphicsView->mapToScene(viewportCursor);
        recordPointerSample(targetViewer, targetViewer->sceneToVolume(scenePos));
        return _hoverPointer.valid;
    }

    return false;
}

void SegmentationModule::recordPointerSample(VolumeViewerBase* viewer, const cv::Vec3f& worldPos)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        _hoverPointer.valid = false;
        _hoverPointer.viewer = nullptr;
        return;
    }
    if (!viewer || !isSegmentationViewer(viewer)) {
        _hoverPointer.valid = false;
        _hoverPointer.viewer = nullptr;
        return;
    }

    // Detect viewer change and reset stale cached state
    if (_hoverPointer.valid && _hoverPointer.viewer != viewer) {
        _editManager->resetPointerSeed();   // Reset pointTo() seed
        _hover.clear();                     // Invalidate stale hover
    }

    _hoverPointer.valid = true;
    _hoverPointer.viewer = viewer;
    _hoverPointer.world = worldPos;
}

void SegmentationModule::updateHover(VolumeViewerBase* viewer,
                                     const cv::Vec3f& worldPos,
                                     const QPointF& scenePos)
{
    VolumeViewerBase* targetViewer = viewer;
    cv::Vec3f targetWorld = worldPos;
    bool hoverChanged = false;

    const auto clearHover = [&]() {
        if (_hover.valid) {
            _hover.clear();
            hoverChanged = true;
        }
    };

    const auto setHover = [&](int row, int col, const cv::Vec3f& world) {
        const bool rowChanged = !_hover.valid || _hover.row != row;
        const bool colChanged = !_hover.valid || _hover.col != col;
        const bool worldChanged = !_hover.valid || cv::norm(_hover.world - world) >= 1e-4f;
        const bool viewerChanged = !_hover.valid || _hover.viewer != targetViewer;
        if (rowChanged || colChanged || worldChanged || viewerChanged) {
            _hover.set(row, col, world, targetViewer);
            hoverChanged = true;
        }
    };

    if (_growthInProgress) {
    } else if (!_editingEnabled || !_hoverPreviewEnabled) {
        clearHover();
    } else if (!_editManager || !_editManager->hasSession()) {
        clearHover();
    } else {
        if (!targetViewer || !isSegmentationViewer(targetViewer)) {
            clearHover();
        } else if (targetViewer->surfName() == "segmentation") {
            const cv::Mat_<cv::Vec3f>& points = _editManager->previewPoints();
            if (points.empty()) {
                clearHover();
            } else {
                const auto grid = segmentationSceneToGrid(targetViewer, scenePos, points.rows, points.cols);
                if (grid) {
                    if (auto world = _editManager->vertexWorldPosition(grid->first, grid->second)) {
                        setHover(grid->first, grid->second, *world);
                    } else {
                        clearHover();
                    }
                } else {
                    clearHover();
                }
            }
        } else {
            float surfaceDistance = 0.0f;
            if (auto gridIndex = _editManager->worldToGridIndex(targetWorld,
                                                                &surfaceDistance,
                                                                SegmentationEditManager::GridSearchResolution::High,
                                                                false)) {
                const float maxHoverDistance = std::max(gridStepWorld(), 1.0f) * 100.0f;
                if (surfaceDistance <= maxHoverDistance) {
                    if (auto world = _editManager->vertexWorldPosition(gridIndex->first, gridIndex->second)) {
                        setHover(gridIndex->first, gridIndex->second, *world);
                    } else {
                        clearHover();
                    }
                } else {
                    clearHover();
                }
            } else {
                clearHover();
            }
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

// Programmatic push/pull entry points use distinct names to avoid overload
// ambiguity in connect().
bool SegmentationModule::startPushPullMode(int direction, std::optional<bool> alphaOverride)
{
    return startPushPull(direction, alphaOverride);
}

void SegmentationModule::stopPushPullAll()
{
    stopAllPushPull();
}

void SegmentationModule::flushAutosave()
{
    // No-op when nothing is pending; marks dirty (follow-up save) when one is in flight.
    markAutosaveNeeded(true);
}

SegmentationModule::AutosaveStatus SegmentationModule::autosaveStatus() const
{
    AutosaveStatus status;
    status.pending = _autosaveState.pending();
    status.saveInProgress = _autosaveState.saveInProgress();
    status.dirtyAfterSave = _autosaveState.dirtyAfterSave();
    return status;
}

void SegmentationModule::markAutosaveNeeded(bool immediate)
{
    _autosaveState.markPending();

    ensureAutosaveTimer();
    if (_autosaveTimer && !_autosaveTimer->isActive()) {
        _autosaveTimer->start();
    }

    if (immediate) {
        performAutosave();
    }
}

void SegmentationModule::queueAutosaveVertexUpdates(
    const std::vector<SegmentationEditManager::VertexEdit>& edits)
{
    if (edits.empty()) {
        return;
    }

    _pendingAutosaveVertexUpdates.reserve(_pendingAutosaveVertexUpdates.size() + edits.size());
    for (const auto& edit : edits) {
        _pendingAutosaveVertexUpdates.push_back(AutosaveVertexUpdate{
            edit.row,
            edit.col,
            edit.currentWorld
        });
    }
}

void SegmentationModule::performAutosave()
{
    if (!_autosaveState.pending()) {
        return;
    }

    // If a save is already running, mark dirty so we re-save when it finishes
    if (_autosaveState.markDirtyIfSaving()) {
        return;
    }

    // The single autosave publishes the surface's edits (vertices and
    // channels such as the mask) to the store. Resolve the live surface from the edit session
    // if there is one, else from the active "segmentation" surface — a save
    // requested without a session (e.g. after growth) must still land.
    std::shared_ptr<QuadSurface> surfacePtr;
    if (_editManager && _editManager->hasSession()) {
        surfacePtr = _editManager->baseSurface();
    } else if (_state) {
        surfacePtr = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    }
    if (!surfacePtr) {
        _autosaveState.clearDeferred();
        _pendingAutosaveVertexUpdates.clear();
        return;
    }
    const auto storeEdit = storeEditFor(surfacePtr.get());
    if (!storeEdit) {
        if (!_autosaveState.failureNotified()) {
            qCWarning(lcSegModule) << "Skipping autosave: the segmentation surface is not from the object store.";
            emit statusMessageRequested(tr("Cannot autosave segmentation: the surface is not from the object store."),
                                        kStatusMedium);
            _autosaveState.setFailureNotified(true);
        }
        return;
    }

    if (_pendingAutosaveVertexUpdates.empty() && _editManager && _editManager->hasSession()
        && _editManager->hasPendingChanges()) {
        queueAutosaveVertexUpdates(_editManager->editedVertices());
    }

    auto vertexUpdates = std::move(_pendingAutosaveVertexUpdates);
    _pendingAutosaveVertexUpdates.clear();

    // Vertex edits patch the previous save's copy and publish the rectangle they
    // touch. Other saves (channels, masks, growth) copy the surface and publish
    // its loaded channels and every changed vertex.
    auto snapshot = _saveSnapshot;
    if (snapshot && (snapshot->id != surfacePtr->id ||
                     snapshot->rawPointsPtr()->size() != surfacePtr->rawPointsPtr()->size())) {
        snapshot.reset();
    }
    std::vector<std::string> channels;
    cv::Rect rect;
    if (vertexUpdates.empty() || !snapshot) {
        snapshot = std::make_shared<QuadSurface>(surfacePtr->rawPoints().clone(), surfacePtr->scale());
        for (const auto& name : surfacePtr->channelNames()) {
            if (!surfacePtr->channelLoaded(name)) {
                continue;
            }
            snapshot->setChannel(name, surfacePtr->channel(name, SURF_CHANNEL_NORESIZE).clone());
            if (vertexUpdates.empty()) {
                channels.push_back(name);
            }
        }
        snapshot->id = surfacePtr->id;
        snapshot->meta = surfacePtr->meta;
    }
    if (vertexUpdates.empty()) {
        rect = cv::Rect(0, 0, snapshot->rawPointsPtr()->cols, snapshot->rawPointsPtr()->rows);
    } else {
        int minRow = INT_MAX, minCol = INT_MAX, maxRow = -1, maxCol = -1;
        for (const auto& update : vertexUpdates) {
            minRow = std::min(minRow, update.row); maxRow = std::max(maxRow, update.row);
            minCol = std::min(minCol, update.col); maxCol = std::max(maxCol, update.col);
        }
        rect = cv::Rect(minCol, minRow, maxCol - minCol + 1, maxRow - minRow + 1);
    }

    const auto autosaveTicket = _autosaveState.startSave();

    emit statusMessageRequested(tr("Saving..."), kStatusShort);

    _saveFuture = QtConcurrent::run([snapshot, storeEdit, rect, channels,
                                      vertexUpdates = std::move(vertexUpdates)]() mutable -> std::shared_ptr<QuadSurface> {
        auto* points = snapshot->rawPointsPtr();
        if (!points || points->empty()) {
            throw std::runtime_error("Autosave snapshot has no point data");
        }
        for (const auto& update : vertexUpdates) {
            if (update.row < 0 || update.row >= points->rows ||
                update.col < 0 || update.col >= points->cols) {
                continue;
            }
            (*points)(update.row, update.col) = update.world;
        }
        if (!vertexUpdates.empty()) {
            snapshot->invalidateCache();
        }
        storeEdit->save(*snapshot, rect, channels);
        return snapshot;
    });
    auto saveFuture = _saveFuture;

    // Poll for completion via a single-shot timer to avoid blocking
    auto* pollTimer = new QTimer(this);
    pollTimer->setInterval(50);
    pollTimer->setSingleShot(false);
    connect(pollTimer, &QTimer::timeout, this, [this, pollTimer, saveFuture, autosaveTicket]() mutable {
        if (!saveFuture.isFinished()) {
            return;
        }
        pollTimer->stop();
        pollTimer->deleteLater();

        std::shared_ptr<QuadSurface> savedSnapshot;
        QString failureMessage;
        try {
            saveFuture.waitForFinished();
            savedSnapshot = saveFuture.result();
        } catch (const std::exception& ex) {
            failureMessage = QString::fromUtf8(ex.what());
        } catch (...) {
            failureMessage = tr("unknown error");
        }

        const bool canRetry = _editManager && _editManager->hasSession();
        const auto completion = failureMessage.isEmpty()
            ? _autosaveState.completeSuccess(autosaveTicket)
            : _autosaveState.completeFailure(autosaveTicket, canRetry);
        if (completion == segmentation::AutosaveState::Completion::Stale) {
            // A stale ticket still emits terminal completion for observers.
            updateAutosaveState();
            emit autosaveCompleted(failureMessage.isEmpty());
            return;
        }

        if (failureMessage.isEmpty()) {
            _saveSnapshot = savedSnapshot;
            emit statusMessageRequested(tr("Saved"), kStatusShort);
            emit autosaveCompleted(true);
        } else {
            qCWarning(lcSegModule) << "Autosave failed:" << failureMessage;
            if (!_autosaveState.failureNotified()) {
                emit statusMessageRequested(tr("Failed to autosave segmentation: %1")
                                                .arg(failureMessage),
                                            kStatusLong);
                _autosaveState.setFailureNotified(true);
            }
            emit autosaveCompleted(false);
        }

        // If another save was requested while we were saving, start it now
        if (_autosaveState.consumeDirtyAfterSave()) {
            performAutosave();
        }
    });
    pollTimer->start();
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

    // Keep the single autosave timer running while editing a segment OR while
    // any save is still pending (one can be requested with editing disabled).
    const bool shouldRun = (_editingEnabled && _editManager && _editManager->hasSession())
                        || _autosaveState.pending();
    if (shouldRun) {
        if (!_autosaveTimer->isActive()) {
            _autosaveTimer->start();
        }
    } else if (_autosaveTimer->isActive()) {
        _autosaveTimer->stop();
    }
}

void SegmentationModule::scheduleCorrectionsAutoSave()
{
    // Only auto-save corrections of a surface from the object store
    if (_correctionsSurfaceId.empty() || !_pointCollection) {
        return;
    }

    if (!_correctionsSaveTimer) {
        _correctionsSaveTimer = new QTimer(this);
        _correctionsSaveTimer->setSingleShot(true);
        connect(_correctionsSaveTimer, &QTimer::timeout, this, &SegmentationModule::performCorrectionsAutoSave);
    }

    _correctionsSaveTimer->start(kCorrectionsSaveDelayMs);
}

void SegmentationModule::performCorrectionsAutoSave()
{
    saveCorrections();
}

std::shared_ptr<SurfaceStoreEdit> SegmentationModule::storeEditFor(QuadSurface* surface)
{
    if (!surface || surface->id.empty()) {
        return nullptr;
    }
    const auto origin = _storePinResolver ? _storePinResolver(surface) : std::nullopt;
    // Saves continue from the edit's own revisions until the surface is reopened.
    if (_storeEdit && _storeEdit->uuid() == surface->id && (!origin || origin->pin == _storeEdit->origin())) {
        return _storeEdit;
    }
    if (!origin || !_state) {
        return nullptr;
    }
    try {
        _storeEdit = std::make_shared<SurfaceStoreEdit>(_state->objectStoreSession(), origin->pin, origin->view);
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("Cannot edit surface: %1").arg(QString::fromUtf8(error.what())), kStatusLong);
        return nullptr;
    }
    return _storeEdit;
}

bool SegmentationModule::saveSurface(QuadSurface* surface, bool vertices, const std::vector<std::string>& channels)
{
    const auto storeEdit = surface ? storeEditFor(surface) : nullptr;
    if (!storeEdit) {
        return false;
    }
    std::vector<std::string> loaded;
    for (const auto& name : channels.empty() ? surface->channelNames() : channels) {
        if (surface->channelLoaded(name)) {
            loaded.push_back(name);
        }
    }
    try {
        if (_saveFuture.isRunning()) {
            _saveFuture.waitForFinished();
        }
        const auto* points = surface->rawPointsPtr();
        storeEdit->save(*surface, vertices && points ? cv::Rect(0, 0, points->cols, points->rows) : cv::Rect(), loaded);
        return true;
    } catch (const std::exception& error) {
        qCWarning(lcSegModule) << "Saving surface channels failed:" << error.what();
        emit statusMessageRequested(tr("Failed to save surface: %1").arg(QString::fromUtf8(error.what())), kStatusLong);
        return false;
    }
}

std::optional<vc::objects::PinnedObject> SegmentationModule::storePin(QuadSurface* surface)
{
    const auto storeEdit = storeEditFor(surface);
    return storeEdit ? std::optional(storeEdit->pin()) : std::nullopt;
}
