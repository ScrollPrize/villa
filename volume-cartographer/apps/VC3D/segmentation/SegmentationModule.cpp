#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "SegmentationBrushTool.hpp"
#include "SegmentationLineTool.hpp"
#include "SegmentationPushPullTool.hpp"
#include "ApprovalMaskBrushTool.hpp"
#include "SegmentationCorrections.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/ui/VCCollection.hpp"

#include <QDebug>
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

#include "vc/core/util/QuadSurface.hpp"


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
    surface->meta = std::make_unique<nlohmann::json>(nlohmann::json::object());
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
        _hoverPreviewEnabled = _widget->showHoverMarker();
    }

    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
        if (_widget) {
            _overlay->setApprovalMaskOpacity(_widget->approvalMaskOpacity());
        }
    }

    _brushTool = std::make_unique<SegmentationBrushTool>(*this, _editManager, _widget, _surfaces);
    _lineTool = std::make_unique<SegmentationLineTool>(*this, _editManager, _surfaces, _smoothStrength, _smoothIterations);
    _pushPullTool = std::make_unique<SegmentationPushPullTool>(*this, _editManager, _widget, _overlay, _surfaces);
    _pushPullTool->setStepMultiplier(initialPushPullStep);
    _pushPullTool->setAlphaConfig(initialAlphaConfig);

    _approvalTool = std::make_unique<ApprovalMaskBrushTool>(*this, _editManager, _widget);

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

void SegmentationModule::setHoverPreviewEnabled(bool enabled)
{
    if (_hoverPreviewEnabled == enabled) {
        return;
    }
    _hoverPreviewEnabled = enabled;
    resetHoverLookupDetail();
    if (!enabled && _hover.valid) {
        _hover.clear();
    }
    refreshOverlay();
}

bool SegmentationModule::ensureHoverTarget()
{
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }
    if (_hoverPreviewEnabled && _hover.valid) {
        return true;
    }
    if (!_hoverPointer.valid) {
        return false;
    }
    CVolumeViewer* viewer = _hoverPointer.viewer.data();
    if (!viewer) {
        _hoverPointer.valid = false;
        return false;
    }
    auto gridIndex = _editManager->worldToGridIndex(_hoverPointer.world);
    if (!gridIndex) {
        _hover.clear();
        return false;
    }
    auto world = _editManager->vertexWorldPosition(gridIndex->first, gridIndex->second);
    if (!world) {
        _hover.clear();
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
    connect(_widget, &SegmentationWidget::hoverMarkerToggled,
            this, &SegmentationModule::setHoverPreviewEnabled);
    connect(_widget, &SegmentationWidget::correctionsCreateRequested,
            this, &SegmentationModule::onCorrectionsCreateRequested);
    connect(_widget, &SegmentationWidget::correctionsCollectionSelected,
            this, &SegmentationModule::onCorrectionsCollectionSelected);
    connect(_widget, &SegmentationWidget::correctionsAnnotateToggled,
            this, &SegmentationModule::onCorrectionsAnnotateToggled);
    connect(_widget, &SegmentationWidget::correctionsZRangeChanged,
            this, &SegmentationModule::onCorrectionsZRangeChanged);
    connect(_widget, &SegmentationWidget::showApprovalMaskChanged,
            this, &SegmentationModule::setShowApprovalMask);
    connect(_widget, &SegmentationWidget::editApprovedMaskChanged,
            this, &SegmentationModule::setEditApprovedMask);
    connect(_widget, &SegmentationWidget::editUnapprovedMaskChanged,
            this, &SegmentationModule::setEditUnapprovedMask);
    connect(_widget, &SegmentationWidget::approvalBrushRadiusChanged,
            this, &SegmentationModule::setApprovalMaskBrushRadius);
    connect(_widget, &SegmentationWidget::approvalBrushDepthChanged,
            this, &SegmentationModule::setApprovalBrushDepth);
    connect(_widget, &SegmentationWidget::approvalMaskOpacityChanged,
            _overlay, &SegmentationOverlayController::setApprovalMaskOpacity);
    connect(_widget, &SegmentationWidget::approvalStrokesUndoRequested,
            this, &SegmentationModule::undoApprovalStroke);

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
        resetHoverLookupDetail();
        _hoverPointer.valid = false;
        _hoverPointer.viewer = nullptr;
        if (_pendingAutosave) {
            performAutosave();
        }
    }
    updateCorrectionsWidget();
    refreshOverlay();
    emit editingEnabledChanged(enabled);
    updateAutosaveState();
}

void SegmentationModule::setShowApprovalMask(bool enabled)
{
    if (_showApprovalMask == enabled) {
        return;
    }

    _showApprovalMask = enabled;
    qCInfo(lcSegModule) << "=== Show Approval Mask:" << (enabled ? "ENABLED" : "DISABLED") << "===";

    if (_showApprovalMask) {
        // Showing approval mask - load it for display
        QuadSurface* surface = nullptr;
        std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
        if (_editManager && _editManager->hasSession()) {
            qCInfo(lcSegModule) << "  Loading approval mask (has active session)";
            surface = _editManager->baseSurface().get();
        } else if (_surfaces) {
            qCInfo(lcSegModule) << "  Loading approval mask (from surfaces collection)";
            surfaceHolder = _surfaces->surface("segmentation");
            surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
        }

        if (surface && _overlay) {
            _overlay->loadApprovalMaskImage(surface);
            qCInfo(lcSegModule) << "  Loaded approval mask into QImage";
        }
    }

    refreshOverlay();
}

void SegmentationModule::onActiveSegmentChanged(QuadSurface* newSurface)
{
    qCInfo(lcSegModule) << "Active segment changed";

    // Turn off any approval mask editing when switching segments
    if (isEditingApprovalMask()) {
        qCInfo(lcSegModule) << "  Turning off approval mask editing";
        if (_editApprovedMask) {
            setEditApprovedMask(false);
            if (_widget) {
                _widget->setEditApprovedMask(false);
            }
        }
        if (_editUnapprovedMask) {
            setEditUnapprovedMask(false);
            if (_widget) {
                _widget->setEditUnapprovedMask(false);
            }
        }
    }

    // Sync show approval mask state from widget (handles restored settings case)
    if (_widget && _widget->showApprovalMask() != _showApprovalMask) {
        qCInfo(lcSegModule) << "  Syncing showApprovalMask from widget:" << _widget->showApprovalMask();
        _showApprovalMask = _widget->showApprovalMask();
    }

    // Check if new surface has an approval mask
    bool hasApprovalMask = false;
    if (newSurface) {
        cv::Mat approvalChannel = newSurface->channel("approval", SURF_CHANNEL_NORESIZE);
        hasApprovalMask = !approvalChannel.empty();
        qCInfo(lcSegModule) << "  New surface has approval mask:" << hasApprovalMask;
    }

    if (_showApprovalMask) {
        if (hasApprovalMask && newSurface && _overlay) {
            // Load the new surface's approval mask
            qCInfo(lcSegModule) << "  Loading approval mask for new surface";
            _overlay->loadApprovalMaskImage(newSurface);
        } else {
            // No approval mask on new surface - turn off show mode
            qCInfo(lcSegModule) << "  No approval mask on new surface, turning off show mode";
            _showApprovalMask = false;
            if (_widget) {
                _widget->setShowApprovalMask(false);
            }
            if (_overlay) {
                _overlay->loadApprovalMaskImage(nullptr);  // Clear the mask
            }
        }
    }

    refreshOverlay();
}

void SegmentationModule::setEditApprovedMask(bool enabled)
{
    if (_editApprovedMask == enabled) {
        return;
    }

    // If enabling, ensure unapproved mode is off (mutual exclusion)
    if (enabled && _editUnapprovedMask) {
        setEditUnapprovedMask(false);
    }

    const bool wasEditing = isEditingApprovalMask();
    _editApprovedMask = enabled;
    qCInfo(lcSegModule) << "=== Edit Approved Mask:" << (enabled ? "ENABLED" : "DISABLED") << "===";

    if (_editApprovedMask) {
        // Entering approval mask editing mode (approve)
        qCInfo(lcSegModule) << "  Activating approval brush tool (approve mode)";
        if (_approvalTool) {
            _approvalTool->setActive(true);
            _approvalTool->setPaintMode(ApprovalMaskBrushTool::PaintMode::Approve);

            // Set surface on approval tool - prefer surface from collection since it has
            // the most up-to-date approval mask (preserved after tracer growth)
            QuadSurface* surface = nullptr;
            std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
            if (_surfaces) {
                surfaceHolder = _surfaces->surface("segmentation");
                surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
            }
            if (!surface && _editManager && _editManager->hasSession()) {
                surface = _editManager->baseSurface().get();
            }

            if (surface) {
                _approvalTool->setSurface(surface);
                // Reload approval mask image to ensure dimensions match current surface
                if (_overlay) {
                    _overlay->loadApprovalMaskImage(surface);
                }
            }
        }

        // Deactivate regular editing tools
        deactivateInvalidationBrush();
        clearLineDragStroke();
        stopAllPushPull();
    } else if (!isEditingApprovalMask()) {
        // Exiting all approval mask editing - save to disk
        qCInfo(lcSegModule) << "  Deactivating approval brush tool and saving";
        if (_approvalTool) {
            _approvalTool->setActive(false);
        }

        // Save changes to disk when exiting edit mode
        if (wasEditing) {
            saveApprovalMaskToDisk();
        }
    }

    refreshOverlay();
}

void SegmentationModule::setEditUnapprovedMask(bool enabled)
{
    if (_editUnapprovedMask == enabled) {
        return;
    }

    // If enabling, ensure approved mode is off (mutual exclusion)
    if (enabled && _editApprovedMask) {
        setEditApprovedMask(false);
    }

    const bool wasEditing = isEditingApprovalMask();
    _editUnapprovedMask = enabled;
    qCInfo(lcSegModule) << "=== Edit Unapproved Mask:" << (enabled ? "ENABLED" : "DISABLED") << "===";

    if (_editUnapprovedMask) {
        // Entering approval mask editing mode (unapprove)
        qCInfo(lcSegModule) << "  Activating approval brush tool (unapprove mode)";
        if (_approvalTool) {
            _approvalTool->setActive(true);
            _approvalTool->setPaintMode(ApprovalMaskBrushTool::PaintMode::Unapprove);

            // Set surface on approval tool - prefer surface from collection since it has
            // the most up-to-date approval mask (preserved after tracer growth)
            QuadSurface* surface = nullptr;
            std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
            if (_surfaces) {
                surfaceHolder = _surfaces->surface("segmentation");
                surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
            }
            if (!surface && _editManager && _editManager->hasSession()) {
                surface = _editManager->baseSurface().get();
            }

            if (surface) {
                _approvalTool->setSurface(surface);
                // Reload approval mask image to ensure dimensions match current surface
                if (_overlay) {
                    _overlay->loadApprovalMaskImage(surface);
                }
            }
        }

        // Deactivate regular editing tools
        deactivateInvalidationBrush();
        clearLineDragStroke();
        stopAllPushPull();
    } else if (!isEditingApprovalMask()) {
        // Exiting all approval mask editing - save to disk
        qCInfo(lcSegModule) << "  Deactivating approval brush tool and saving";
        if (_approvalTool) {
            _approvalTool->setActive(false);
        }

        // Save changes to disk when exiting edit mode
        if (wasEditing) {
            saveApprovalMaskToDisk();
        }
    }

    refreshOverlay();
}

void SegmentationModule::saveApprovalMaskToDisk()
{
    qCInfo(lcSegModule) << "Saving approval mask to disk...";

    QuadSurface* surface = nullptr;
    std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
    if (_editManager && _editManager->hasSession()) {
        surface = _editManager->baseSurface().get();
    } else if (_surfaces) {
        surfaceHolder = _surfaces->surface("segmentation");
        surface = dynamic_cast<QuadSurface*>(surfaceHolder.get());
    }

    if (_overlay && surface) {
        _overlay->saveApprovalMaskToSurface(surface);
        emit statusMessageRequested(tr("Saved approval mask."), kStatusShort);
        qCInfo(lcSegModule) << "  Approval mask saved to disk";

        // Emit signal so CWindow can mark this segment as recently edited
        // (to prevent inotify from triggering unwanted removals/reloads)
        if (!surface->id.empty()) {
            emit approvalMaskSaved(surface->id);
        }
    }
}

void SegmentationModule::setApprovalMaskBrushRadius(float radiusSteps)
{
    _approvalMaskBrushRadius = std::max(1.0f, radiusSteps);
}

void SegmentationModule::setApprovalBrushDepth(float depth)
{
    _approvalBrushDepth = std::clamp(depth, 1.0f, 500.0f);
}

void SegmentationModule::undoApprovalStroke()
{
    qCInfo(lcSegModule) << "Undoing last approval stroke...";
    if (!_overlay) {
        qCWarning(lcSegModule) << "  No overlay controller available";
        return;
    }

    if (!_overlay->canUndoApprovalMaskPaint()) {
        qCInfo(lcSegModule) << "  Nothing to undo";
        emit statusMessageRequested(tr("Nothing to undo."), kStatusShort);
        return;
    }

    if (_overlay->undoLastApprovalMaskPaint()) {
        refreshOverlay();
        emit statusMessageRequested(tr("Undid last approval stroke."), kStatusShort);
        qCInfo(lcSegModule) << "  Approval stroke undone";
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
        auto preview = _editManager->previewSurface();
        _surfaces->setSurface("segmentation", preview, false, true);
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
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

    // Get surface for approval mask - from edit session if available, otherwise from surfaces collection
    QuadSurface* approvalSurface = nullptr;
    std::shared_ptr<Surface> approvalSurfaceHolder;  // Keep surface alive during this scope
    if (hasSession && _editManager) {
        approvalSurface = _editManager->baseSurface().get();
    } else if (_surfaces) {
        approvalSurfaceHolder = _surfaces->surface("segmentation");
        approvalSurface = dynamic_cast<QuadSurface*>(approvalSurfaceHolder.get());
    }

    // Set approval mask state even without editing session (for view-only mode)
    // Show the mask when _showApprovalMask is true
    if (_showApprovalMask && approvalSurface) {
        state.approvalMaskMode = true;
        state.surface = approvalSurface;
    }

    // Populate brush/stroke info when editing is enabled
    if (isEditingApprovalMask() && approvalSurface) {
        state.approvalMaskMode = true;  // Must be true to render brush
        state.approvalBrushRadius = _approvalMaskBrushRadius;
        state.approvalBrushDepth = _approvalBrushDepth;
        state.surface = approvalSurface;
        if (_approvalTool) {
            state.approvalStrokeActive = _approvalTool->strokeActive();
            state.approvalStrokeSegments = _approvalTool->overlayStrokeSegments();
            state.approvalCurrentStroke = _approvalTool->overlayPoints();
            state.paintingApproval = (_approvalTool->paintMode() == ApprovalMaskBrushTool::PaintMode::Approve);
            state.approvalHoverWorld = _approvalTool->hoverWorldPos();
            state.approvalHoverScenePos = _approvalTool->hoverScenePos();
            state.approvalHoverViewerScale = _approvalTool->hoverViewerScale();
            state.approvalHoverPlaneNormal = _approvalTool->hoverPlaneNormal();
            if (_approvalTool->strokeActive() && _approvalTool->effectivePaintRadius() > 0.0f) {
                state.approvalEffectiveRadius = _approvalTool->effectivePaintRadius();
            } else {
                state.approvalEffectiveRadius = _approvalTool->hoverEffectiveRadius();
            }
        }
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

    // Ensure any pending invalidation brush strokes are committed before growth.
    if (_brushTool) {
        _brushTool->applyPending(_dragRadiusSteps);
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
    float result = 1.0f;
    const QuadSurface* surface = nullptr;

    if (!_editManager || !_editManager->hasSession()) {
        // For approval mask mode, try to get base surface scale even without active session
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
            _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
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

SegmentationEditManager::GridSearchResolution SegmentationModule::hoverLookupDetail(const cv::Vec3f& worldPos)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        resetHoverLookupDetail();
        return SegmentationEditManager::GridSearchResolution::High;
    }

    if (!_hoverLookup.initialized) {
        _hoverLookup.initialized = true;
        _hoverLookup.lastWorld = worldPos;
        _hoverLookup.smoothedWorldUnitsPerSecond = 0.0f;
        _hoverLookup.timer.start();
        return SegmentationEditManager::GridSearchResolution::High;
    }

    const qint64 elapsedNs = _hoverLookup.timer.nsecsElapsed();
    _hoverLookup.timer.restart();
    double dtSec = static_cast<double>(elapsedNs) / 1e9;
    if (dtSec <= 1e-4) {
        dtSec = 1e-4;
    }

    const cv::Vec3f delta = worldPos - _hoverLookup.lastWorld;
    _hoverLookup.lastWorld = worldPos;

    const float distance = cv::norm(delta);
    const float instantaneousSpeed = distance / static_cast<float>(dtSec);

    constexpr float kSmoothing = 0.2f;
    if (_hoverLookup.smoothedWorldUnitsPerSecond <= 0.0f) {
        _hoverLookup.smoothedWorldUnitsPerSecond = instantaneousSpeed;
    } else {
        _hoverLookup.smoothedWorldUnitsPerSecond =
            _hoverLookup.smoothedWorldUnitsPerSecond * (1.0f - kSmoothing) +
            instantaneousSpeed * kSmoothing;
    }

    constexpr float kMediumThreshold = 4.0f;
    constexpr float kLowThreshold = 12.0f;

    if (_hoverLookup.smoothedWorldUnitsPerSecond >= kLowThreshold) {
        return SegmentationEditManager::GridSearchResolution::Low;
    }
    if (_hoverLookup.smoothedWorldUnitsPerSecond >= kMediumThreshold) {
        return SegmentationEditManager::GridSearchResolution::Medium;
    }
    return SegmentationEditManager::GridSearchResolution::High;
}

void SegmentationModule::resetHoverLookupDetail()
{
    if (_hoverLookup.timer.isValid()) {
        _hoverLookup.timer.invalidate();
    }
    _hoverLookup.initialized = false;
    _hoverLookup.smoothedWorldUnitsPerSecond = 0.0f;
    _hoverLookup.lastWorld = cv::Vec3f(0.0f, 0.0f, 0.0f);
}

void SegmentationModule::recordPointerSample(CVolumeViewer* viewer, const cv::Vec3f& worldPos)
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

    _hoverPointer.valid = true;
    _hoverPointer.viewer = viewer;
    _hoverPointer.world = worldPos;
}

void SegmentationModule::updateHover(CVolumeViewer* viewer, const cv::Vec3f& worldPos)
{
    bool hoverChanged = false;

    if (!_hoverPreviewEnabled) {
        resetHoverLookupDetail();
        if (_hover.valid) {
            _hover.clear();
            hoverChanged = true;
        }
    } else if (!_editManager || !_editManager->hasSession()) {
        resetHoverLookupDetail();
        if (_hover.valid) {
            _hover.clear();
            hoverChanged = true;
        }
    } else {
        const auto detail = hoverLookupDetail(worldPos);
        if (detail != SegmentationEditManager::GridSearchResolution::High) {
            if (_hover.valid) {
                _hover.clear();
                hoverChanged = true;
            }
        } else {
            auto gridIndex = _editManager->worldToGridIndex(worldPos, nullptr, detail);
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
    QuadSurface* surface = _editManager->baseSurface().get();
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
