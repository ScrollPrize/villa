#include "SegmentationModule.hpp"

#include "adaptive/CAdaptiveVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CState.hpp"
#include "tools/SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "tools/SegmentationLineTool.hpp"
#include "tools/SegmentationPushPullTool.hpp"
#include "tools/ApprovalMaskBrushTool.hpp"
#include "tools/CellReoptimizationTool.hpp"
#include "growth/SegmentationCorrections.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/ui/VCCollection.hpp"

#include <QApplication>
#include <QDebug>
#include <QCursor>
#include <QLoggingCategory>
#include <QPointF>
#include <QPointer>
#include <QString>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
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
    if (!surface->meta.is_null() && surface->meta.is_object()) {
        return;
    }
    surface->meta = utils::Json::object();
}

std::optional<std::pair<int, int>> segmentationSceneToGrid(CTiledVolumeViewer* viewer,
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

void SegmentationModule::HoverState::set(int r, int c, const cv::Vec3f& w, CTiledVolumeViewer* v)
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
    if (_saveInProgress && _saveFuture.isRunning()) {
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
        _lineRadiusSteps = _widget->lineRadius();
        _lineSigmaSteps = _widget->lineSigma();
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
        _autoApprovalEnabled = _widget->autoApprovalEnabled();
        _autoApprovalRadius = _widget->autoApprovalRadius();
        _autoApprovalThreshold = _widget->autoApprovalThreshold();
        _autoApprovalMaxDistance = _widget->autoApprovalMaxDistance();
    }

    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
        if (_widget) {
            _overlay->setApprovalMaskOpacity(_widget->approvalMaskOpacity());
        }
    }

    _lineTool = std::make_unique<SegmentationLineTool>(*this, _editManager, _state, _smoothStrength, _smoothIterations);
    _pushPullTool = std::make_unique<SegmentationPushPullTool>(*this, _editManager, _widget, _overlay, _state);
    _pushPullTool->setStepMultiplier(initialPushPullStep);
    _pushPullTool->setAlphaConfig(initialAlphaConfig);

    _approvalTool = std::make_unique<ApprovalMaskBrushTool>(*this, _editManager, _widget);

    _cellReoptTool = std::make_unique<CellReoptimizationTool>(*this, _editManager, _overlay, _pointCollection, this);
    connect(_cellReoptTool.get(), &CellReoptimizationTool::statusMessage,
            this, [this](const QString& msg, int timeout) {
                emit statusMessageRequested(msg, timeout);
            });
    connect(_cellReoptTool.get(), &CellReoptimizationTool::collectionCreated,
            this, [this](uint64_t collectionId) {
                // Register the new collection with the corrections system so it will be used
                if (_corrections) {
                    _corrections->setActiveCollection(collectionId, false);
                }
            });

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
    }

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t id) {
            if (_corrections) {
                _corrections->onCollectionRemoved(id);
                updateCorrectionsWidget();
            }
            updateCellReoptCollections();
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t id) {
            if (_corrections) {
                _corrections->onCollectionChanged(id);
            }
            updateCellReoptCollections();
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>&) {
            updateCellReoptCollections();
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::pointAdded, this, [this](const ColPoint&) {
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::pointChanged, this, [this](const ColPoint&) {
            scheduleCorrectionsAutoSave();
        });

        connect(_pointCollection, &VCCollection::pointRemoved, this, [this](uint64_t) {
            scheduleCorrectionsAutoSave();
        });
    }

    updateCorrectionsWidget();
    updateCellReoptCollections();

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

void SegmentationModule::setRotationHandleHitTester(std::function<bool(CTiledVolumeViewer*, const cv::Vec3f&)> tester)
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
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: no active editing session.";
        return false;
    }

    // Push/pull calls this from a timer while the cursor can move between
    // mouse-move events delivered to the segmentation module. Refresh from the
    // viewer cursor first so the resolved grid target follows the mouse even
    // when hover preview is disabled.
    recoverHoverPointerFromCursor();

    if (_hover.valid && _hoverPointer.valid && _hover.viewer != _hoverPointer.viewer) {
        _hover.clear();
    }

    if (!_hoverPointer.valid) {
        qCWarning(lcSegModule) << "Cannot resolve segmentation hover target: no cursor sample for a segmentation viewer.";
        return false;
    }
    CTiledVolumeViewer* viewer = _hoverPointer.viewer.data();
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
    connect(_widget, &SegmentationWidget::autoApprovalEnabledChanged,
            this, &SegmentationModule::setAutoApprovalEnabled);
    connect(_widget, &SegmentationWidget::autoApprovalRadiusChanged,
            this, &SegmentationModule::setAutoApprovalRadius);
    connect(_widget, &SegmentationWidget::autoApprovalThresholdChanged,
            this, &SegmentationModule::setAutoApprovalThreshold);
    connect(_widget, &SegmentationWidget::autoApprovalMaxDistanceChanged,
            this, &SegmentationModule::setAutoApprovalMaxDistance);

    connect(_widget, &SegmentationWidget::approvalBrushRadiusChanged,
            this, &SegmentationModule::setApprovalMaskBrushRadius);
    connect(_widget, &SegmentationWidget::approvalBrushDepthChanged,
            this, &SegmentationModule::setApprovalBrushDepth);
    connect(_widget, &SegmentationWidget::approvalBrushColorChanged,
            this, &SegmentationModule::setApprovalBrushColor);
    connect(_widget, &SegmentationWidget::approvalMaskOpacityChanged,
            _overlay, &SegmentationOverlayController::setApprovalMaskOpacity);
    connect(_widget, &SegmentationWidget::approvalStrokesUndoRequested,
            this, &SegmentationModule::undoApprovalStroke);
    connect(_widget, &SegmentationWidget::cellReoptModeChanged,
            this, &SegmentationModule::setCellReoptimizationMode);
    connect(_widget, &SegmentationWidget::cellReoptMaxStepsChanged,
            this, &SegmentationModule::setCellReoptMaxSteps);
    connect(_widget, &SegmentationWidget::cellReoptMaxPointsChanged,
            this, &SegmentationModule::setCellReoptMaxPoints);
    connect(_widget, &SegmentationWidget::cellReoptMinSpacingChanged,
            this, &SegmentationModule::setCellReoptMinSpacing);
    connect(_widget, &SegmentationWidget::cellReoptPerimeterOffsetChanged,
            this, &SegmentationModule::setCellReoptPerimeterOffset);
    connect(_widget, &SegmentationWidget::cellReoptGrowthRequested,
            this, [this](uint64_t collectionId) {
                if (isEditingApprovalMask()) {
                    saveApprovalMaskToDisk();
                }
                // Cell reoptimization should not auto-approve the growth region
                _skipAutoApprovalOnGrowth = true;
                // Store the specific collection ID to use for this growth
                _cellReoptCollectionId = collectionId;
                emit growSurfaceRequested(SegmentationGrowthMethod::Corrections,
                                          SegmentationGrowthDirection::All,
                                          0, false);
            });
    connect(_widget, &SegmentationWidget::manualAddConfigChanged, this, [this]() {
        if (_manualAddTool && _widget) {
            _manualAddTool->setConfig(_widget->manualAddConfig());
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

void SegmentationModule::bindViewerSignals(CTiledVolumeViewer* viewer)
{
    if (!viewer || viewer->property("vc_segmentation_bound").toBool()) {
        return;
    }

    connect(viewer, &CTiledVolumeViewer::sendMousePressVolume,
            this, [this, viewer](const cv::Vec3f& worldPos,
                                 const cv::Vec3f& normal,
                                 Qt::MouseButton button,
                                 Qt::KeyboardModifiers modifiers,
                                 const QPointF& scenePos) {
                handleMousePress(viewer, worldPos, normal, button, modifiers, scenePos);
            });
    connect(viewer, &CTiledVolumeViewer::sendMouseMoveVolume,
            this, [this, viewer](const cv::Vec3f& worldPos,
                                 Qt::MouseButtons buttons,
                                 Qt::KeyboardModifiers modifiers,
                                 const QPointF& scenePos) {
                handleMouseMove(viewer, worldPos, buttons, modifiers, scenePos);
            });
    connect(viewer, &CTiledVolumeViewer::sendMouseReleaseVolume,
            this, [this, viewer](const cv::Vec3f& worldPos,
                                 Qt::MouseButton button,
                                 Qt::KeyboardModifiers modifiers,
                                 const QPointF& scenePos) {
                handleMouseRelease(viewer, worldPos, button, modifiers, scenePos);
            });
    connect(viewer, &CTiledVolumeViewer::sendSegmentationRadiusWheel,
            this, [this, viewer](int steps, const QPointF& scenePoint, const cv::Vec3f& worldPos) {
                handleWheel(viewer, steps, scenePoint, worldPos);
            });

    viewer->setProperty("vc_segmentation_bound", true);
    viewer->setSegmentationEditActive(_editingEnabled);
    _attachedViewers.insert(viewer);
}

void SegmentationModule::attachViewer(CTiledVolumeViewer* viewer)
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

void SegmentationModule::setIgnoreSegSurfaceChange(bool ignore)
{
    _ignoreSegSurfaceChange = ignore;
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
        } else if (_state) {
            qCInfo(lcSegModule) << "  Loading approval mask (from surfaces collection)";
            surfaceHolder = _state->surface("segmentation");
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

    // Flush any pending approval mask saves and clear images BEFORE turning off editing
    // loadApprovalMaskImage(nullptr) does both:
    // 1. Saves pending changes to _approvalSaveSurface (the previous segment)
    // 2. Clears the mask images so subsequent saveApprovalMaskToDisk() has nothing to save
    // This prevents the old mask from being incorrectly saved to the new segment
    if (_overlay) {
        _overlay->loadApprovalMaskImage(nullptr);
    }

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

    // Save corrections for old segment and load for new segment
    if (_pointCollection) {
        // Save pending corrections for the old segment
        if (_correctionsSaveTimer && _correctionsSaveTimer->isActive()) {
            _correctionsSaveTimer->stop();
        }
        if (!_correctionsSegmentPath.empty()) {
            qCInfo(lcSegModule) << "  Saving correction points for previous segment";
            _pointCollection->saveToSegmentPath(_correctionsSegmentPath);
        }

        // Load corrections for new segment
        if (newSurface && !newSurface->path.empty()) {
            qCInfo(lcSegModule) << "  Loading correction points for new segment:"
                                << QString::fromStdString(newSurface->path.string());
            _pointCollection->loadFromSegmentPath(newSurface->path);
            _correctionsSegmentPath = newSurface->path;
        } else {
            // No valid path - clear anchored collections and path
            qCInfo(lcSegModule) << "  No segment path, clearing anchored corrections";
            _pointCollection->loadFromSegmentPath({});  // Clears anchored collections
            _correctionsSegmentPath.clear();
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
            if (_state) {
                surfaceHolder = _state->surface("segmentation");
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
            if (_state) {
                surfaceHolder = _state->surface("segmentation");
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

void SegmentationModule::setAutoApprovalEnabled(bool enabled)
{
    _autoApprovalEnabled = enabled;
}

void SegmentationModule::setAutoApprovalRadius(float radius)
{
    _autoApprovalRadius = std::clamp(radius, 0.0f, 10.0f);
}

void SegmentationModule::setAutoApprovalThreshold(float threshold)
{
    _autoApprovalThreshold = std::clamp(threshold, 0.0f, 10.0f);
}

void SegmentationModule::setAutoApprovalMaxDistance(float distance)
{
    _autoApprovalMaxDistance = std::clamp(distance, 0.0f, 500.0f);
}

std::vector<std::pair<int, int>> SegmentationModule::filterVerticesForAutoApproval(
    const std::vector<SegmentationEditManager::VertexEdit>& edits,
    const std::optional<std::pair<int, int>>& dragCenter) const
{
    std::vector<std::pair<int, int>> result;
    result.reserve(edits.size());
    const float autoApprovalMaxDistanceSq = _autoApprovalMaxDistance > 0.0f ? _autoApprovalMaxDistance * _autoApprovalMaxDistance : 0.0f;

    for (const auto& edit : edits) {
        // Check threshold: skip if movement is below minimum
        if (_autoApprovalThreshold > 0.0f) {
            const cv::Vec3f delta = edit.currentWorld - edit.originalWorld;
            const float distance = cv::norm(delta);
            if (distance < _autoApprovalThreshold) {
                continue;
            }
        }

        // Check max distance from drag center: skip if too far
        if (_autoApprovalMaxDistance > 0.0f && dragCenter.has_value()) {
            const int dr = edit.row - dragCenter->first;
            const int dc = edit.col - dragCenter->second;
            const float gridDistSq = static_cast<float>(dr * dr + dc * dc);
            if (gridDistSq > _autoApprovalMaxDistance * _autoApprovalMaxDistance) {
                continue;
            }
        }

        result.emplace_back(edit.row, edit.col);
    }

    return result;
}

void SegmentationModule::performAutoApproval(const std::vector<std::pair<int, int>>& vertices)
{
    if (vertices.empty() || !_overlay || !_overlay->hasApprovalMaskData()) {
        return;
    }

    constexpr uint8_t kApproved = 255;
    constexpr bool kIsAutoApproval = true;
    const QColor brushColor = approvalBrushColor();
    _overlay->paintApprovalMaskDirect(vertices, _autoApprovalRadius, kApproved, brushColor, false, 0.0f, 0.0f, kIsAutoApproval);
    _overlay->scheduleDebouncedSave(_editManager->baseSurface().get());
    qCInfo(lcSegModule) << "Auto-approved" << vertices.size() << "vertices with radius" << _autoApprovalRadius;
}

void SegmentationModule::saveApprovalMaskToDisk()
{
    qCInfo(lcSegModule) << "Saving approval mask to disk...";

    QuadSurface* surface = nullptr;
    std::shared_ptr<Surface> surfaceHolder;  // Keep surface alive during this scope
    if (_editManager && _editManager->hasSession()) {
        surface = _editManager->baseSurface().get();
    } else if (_state) {
        surfaceHolder = _state->surface("segmentation");
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

void SegmentationModule::setApprovalBrushColor(const QColor& color)
{
    if (color.isValid()) {
        _approvalBrushColor = color;
    }
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

void SegmentationModule::setCellReoptimizationMode(bool enabled)
{
    if (_cellReoptMode == enabled) {
        return;
    }
    _cellReoptMode = enabled;

    // Disable conflicting modes when enabling cell reopt
    if (enabled) {
        setCorrectionsAnnotateMode(false, false);
        setEditApprovedMask(false);
        setEditUnapprovedMask(false);
    }
}

void SegmentationModule::setCellReoptMaxSteps(int steps)
{
    if (_cellReoptTool) {
        auto config = _cellReoptTool->config();
        config.maxFloodSteps = std::clamp(steps, 10, 10000);
        _cellReoptTool->setConfig(config);
    }
}

void SegmentationModule::setCellReoptMaxPoints(int points)
{
    if (_cellReoptTool) {
        auto config = _cellReoptTool->config();
        config.maxCorrectionPoints = std::clamp(points, 3, 200);
        _cellReoptTool->setConfig(config);
    }
}

void SegmentationModule::setCellReoptMinSpacing(float spacing)
{
    if (_cellReoptTool) {
        auto config = _cellReoptTool->config();
        config.minBoundarySpacing = std::clamp(spacing, 1.0f, 50.0f);
        _cellReoptTool->setConfig(config);
    }
}

void SegmentationModule::setCellReoptPerimeterOffset(float offset)
{
    if (_cellReoptTool) {
        auto config = _cellReoptTool->config();
        config.perimeterOffset = std::clamp(offset, -50.0f, 50.0f);
        _cellReoptTool->setConfig(config);
    }
}

void SegmentationModule::applyEdits()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    const bool hadPendingChanges = _editManager->hasPendingChanges();

    // Capture delta for undo before applyPreview() clears edited vertices
    if (hadPendingChanges) {
        (void)captureUndoDelta();
    }

    // Auto-approve edited regions if approval mask is active (you edited it, so it's reviewed)
    if (_autoApprovalEnabled && _overlay && _overlay->hasApprovalMaskData() && hadPendingChanges) {
        const auto editedVerts = _editManager->editedVertices();
        if (!editedVerts.empty()) {
            const auto filteredVerts = filterVerticesForAutoApproval(editedVerts, std::nullopt);
            performAutoApproval(filteredVerts);
        }
    }

    _editManager->applyPreview();
    if (_state) {
        auto preview = _editManager->previewSurface();
        _state->setSurface("segmentation", preview, false, true);
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
    cancelDrag();
    clearLineDragStroke();
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
    _lineDrawKeyActive = false;
    clearLineDragStroke();
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
        // Clear the cell reopt collection ID now that payload has been built
        _cellReoptCollectionId = 0;
        setCorrectionsAnnotateMode(false, false);
        clearLineDragStroke();
        _lineDrawKeyActive = false;
        resetHoverLookupDetail();
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
    } else if (_state) {
        approvalSurfaceHolder = _state->surface("segmentation");
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
            state.approvalHoverSurfacePos = _approvalTool->hoverSurfacePos();
            state.approvalHoverPlaneNormal = _approvalTool->hoverPlaneNormal();
            if (_approvalTool->strokeActive() && _approvalTool->effectivePaintRadius() > 0.0f) {
                state.approvalEffectiveRadius = _approvalTool->effectivePaintRadius();
            } else {
                state.approvalEffectiveRadius = _approvalTool->hoverEffectiveRadius();
            }
        }
    }

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
    if (_lineTool) {
        maskReserve += _lineTool->overlayPoints().size();
    }
    maskPoints.reserve(maskReserve);
    if (_lineTool) {
        const auto& linePts = _lineTool->overlayPoints();
        maskPoints.insert(maskPoints.end(), linePts.begin(), linePts.end());
    }

    const bool hasLineStroke = _lineTool && !_lineTool->overlayPoints().empty();
    const bool lineStrokeActive = _lineTool && _lineTool->strokeActive();
    const bool pushPullActive = _pushPullTool && _pushPullTool->isActive();

    state.maskPoints = std::move(maskPoints);
    state.maskVisible = !state.maskPoints.empty();
    state.hasLineStroke = hasLineStroke;
    state.lineStrokeActive = lineStrokeActive;
    state.brushActive = false;
    state.brushStrokeActive = false;
    state.pushPullActive = pushPullActive;

    FalloffTool overlayTool = _activeFalloff;
    if (hasLineStroke) {
        overlayTool = FalloffTool::Line;
    } else if (pushPullActive) {
        overlayTool = FalloffTool::PushPull;
    }

    state.displayRadiusSteps = falloffRadius(overlayTool);

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
                    !ManualAddTool::isInvalidPoint(preview(row, col))) {
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
        for (const auto& key : _manualAddTool->fillVertices()) {
            if (key.row >= 0 && key.row < preview.rows && key.col >= 0 && key.col < preview.cols &&
                !ManualAddTool::isInvalidPoint(preview(key.row, key.col))) {
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
                    return !ManualAddTool::isInvalidPoint(point);
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

void SegmentationModule::updateCellReoptCollections()
{
    if (!_widget || !_pointCollection) {
        return;
    }

    QVector<QPair<uint64_t, QString>> entries;
    const auto& collections = _pointCollection->getAllCollections();
    entries.reserve(static_cast<int>(collections.size()));
    for (const auto& [id, col] : collections) {
        entries.append({id, QString::fromStdString(col.name)});
    }

    // Sort by collection name for consistent ordering
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    _widget->setCellReoptCollections(entries);
}

void SegmentationModule::setCorrectionsAnnotateMode(bool enabled, bool userInitiated)
{
    if (!_corrections) {
        return;
    }

    const bool wasActive = _corrections->annotateMode();
    const bool isActive = _corrections->setAnnotateMode(enabled, userInitiated, _editingEnabled);
    if (isActive && !wasActive) {
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

void SegmentationModule::beginCorrectionDrag(int row, int col, CTiledVolumeViewer* viewer, const cv::Vec3f& worldPos)
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
    uint64_t collectionId = _corrections->activeCollection();
    if (collectionId == 0) {
        collectionId = _corrections->createCollection(true);
        if (collectionId == 0) {
            emit statusMessageRequested(tr("Failed to create correction collection"), kStatusMedium);
            return;
        }
    }

    // Set anchor2d on the collection (the grid location where user started dragging)
    cv::Vec2f anchor2d(static_cast<float>(anchorCol), static_cast<float>(anchorRow));
    _pointCollection->setCollectionAnchor2d(collectionId, anchor2d);

    // Add the correction point (3D world target)
    _corrections->handlePointAdded(targetWorld);

    qCInfo(lcSegModule) << "Correction drag completed: anchor2d" << anchorCol << anchorRow
                        << "target" << targetWorld[0] << targetWorld[1] << targetWorld[2];

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

SegmentationCorrectionsPayload SegmentationModule::buildCorrectionsPayload(bool onlyActiveCollection) const
{
    if (!_corrections) {
        return SegmentationCorrectionsPayload{};
    }

    // If a specific collection ID is set for cell reoptimization, build payload with only that
    if (_cellReoptCollectionId != 0) {
        return _corrections->buildPayloadForCollection(_cellReoptCollectionId);
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
    if (_saveInProgress || _pendingAutosave) {
        emit statusMessageRequested(tr("Wait for the current save before Manual Add."), kStatusMedium);
        return false;
    }

    stopAllPushPull();
    clearLineDragStroke();
    cancelDrag();
    cancelCorrectionDrag();
    setCorrectionsAnnotateMode(false, false);
    setEditApprovedMask(false);
    setEditUnapprovedMask(false);
    setCellReoptimizationMode(false);

    _previousGrowthMethodBeforeManualAdd = _widget->growthMethod();
    _widget->setGrowthMethod(SegmentationGrowthMethod::ManualAdd);
    if (!_manualAddTool) {
        _manualAddTool = std::make_unique<ManualAddTool>();
    }
    if (!_manualAddTool->begin(_editManager->previewPoints(), _widget->manualAddConfig())) {
        emit statusMessageRequested(tr("Manual Add could not read the active surface."), kStatusMedium);
        return false;
    }
    _manualAddMode = true;
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
        if (_widget) {
            _widget->setGrowthMethod(_previousGrowthMethodBeforeManualAdd);
        }
        emitPendingChanges();
        emit statusMessageRequested(tr("Manual Add canceled."), kStatusShort);
    }

    _manualAddMode = false;
    _manualAddTool->clear();
    if (_widget) {
        _widget->setManualAddActive(false);
    }
    refreshOverlay();
    return true;
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
    _manualAddTool->begin(_manualAddTool->entrySnapshotPoints(), _widget ? _widget->manualAddConfig() : ManualAddTool::Config{});
    _editManager->restorePreviewSnapshot(_manualAddTool->entrySnapshotPoints());
    if (_state && _editManager->previewSurface()) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }
    emitPendingChanges();
    refreshOverlay();
    emit statusMessageRequested(tr("Manual Add pending geometry cleared."), kStatusShort);
    return true;
}

bool SegmentationModule::handleManualAddMousePress(CTiledVolumeViewer* viewer,
                                                   const cv::Vec3f& worldPos,
                                                   Qt::MouseButton button,
                                                   Qt::KeyboardModifiers modifiers,
                                                   const QPointF& scenePos)
{
    if (!_manualAddMode || !_manualAddTool || !viewer) {
        return false;
    }
    if (viewer->surfName() == "segmentation" && button == Qt::LeftButton && modifiers == Qt::NoModifier) {
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

bool SegmentationModule::handleManualAddMouseMove(CTiledVolumeViewer* viewer,
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
    const bool changed = grid ? _manualAddTool->updateHover(grid->first, grid->second)
                              : _manualAddTool->updateHover(-1, -1);
    if (changed) {
        refreshOverlay();
    }
    return true;
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

bool SegmentationModule::isSegmentationViewer(const CTiledVolumeViewer* viewer) const
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

void SegmentationModule::beginDrag(int row, int col, CTiledVolumeViewer* viewer, const cv::Vec3f& worldPos)
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
        // Capture delta for undo before applyPreview() clears edited vertices
        (void)captureUndoDelta();

        // Auto-approve edited regions before applyPreview() clears them
        if (_autoApprovalEnabled && _overlay && _overlay->hasApprovalMaskData()) {
            const auto editedVerts = _editManager->editedVertices();
            if (!editedVerts.empty()) {
                // Get drag center from the active drag state
                const auto& activeDrag = _editManager->activeDrag();
                std::optional<std::pair<int, int>> dragCenter;
                if (activeDrag.center.row >= 0 && activeDrag.center.col >= 0) {
                    dragCenter = std::make_pair(activeDrag.center.row, activeDrag.center.col);
                }
                const auto filteredVerts = filterVerticesForAutoApproval(editedVerts, dragCenter);
                performAutoApproval(filteredVerts);
            }
        }

        _editManager->applyPreview();
        if (_state) {
            _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
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

bool SegmentationModule::isNearRotationHandle(CTiledVolumeViewer* viewer, const cv::Vec3f& worldPos) const
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

bool SegmentationModule::recoverHoverPointerFromCursor()
{
    const QPoint globalCursor = QCursor::pos();

    CTiledVolumeViewer* targetViewer = nullptr;
    if (QWidget* widget = QApplication::widgetAt(globalCursor)) {
        for (QWidget* current = widget; current && !targetViewer; current = current->parentWidget()) {
            for (auto* viewer : std::as_const(_attachedViewers)) {
                if (!viewer || !isSegmentationViewer(viewer)) {
                    continue;
                }
                if (current == viewer) {
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
            auto* graphicsView = viewer->fGraphicsView;
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
        auto* graphicsView = targetViewer->fGraphicsView;
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

void SegmentationModule::recordPointerSample(CTiledVolumeViewer* viewer, const cv::Vec3f& worldPos)
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
        resetHoverLookupDetail();           // Reset velocity tracking
        _editManager->resetPointerSeed();   // Reset pointTo() seed
        _hover.clear();                     // Invalidate stale hover
    }

    _hoverPointer.valid = true;
    _hoverPointer.viewer = viewer;
    _hoverPointer.world = worldPos;
}

void SegmentationModule::updateHover(CTiledVolumeViewer* viewer,
                                     const cv::Vec3f& worldPos,
                                     const QPointF& scenePos)
{
    CTiledVolumeViewer* targetViewer = viewer;
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
        resetHoverLookupDetail();
    } else if (!_editingEnabled || !_hoverPreviewEnabled) {
        resetHoverLookupDetail();
        clearHover();
    } else if (!_editManager || !_editManager->hasSession()) {
        resetHoverLookupDetail();
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

    // If a save is already running, mark dirty so we re-save when it finishes
    if (_saveInProgress) {
        _dirtyAfterSave = true;
        return;
    }

    auto surfacePtr = _editManager->baseSurface();
    if (!surfacePtr) {
        return;
    }
    if (surfacePtr->path.empty() || surfacePtr->id.empty()) {
        if (!_autosaveNotifiedFailure) {
            qCWarning(lcSegModule) << "Skipping autosave: segmentation surface lacks path or id.";
            emit statusMessageRequested(tr("Cannot autosave segmentation: surface is missing file metadata."),
                                        kStatusMedium);
            _autosaveNotifiedFailure = true;
        }
        return;
    }

    ensureSurfaceMetaObject(surfacePtr.get());

    // Snapshot the surface data on the main thread so the background write
    // does not race with ongoing edits.
    auto snapshot = std::make_shared<QuadSurface>(surfacePtr->rawPoints(), surfacePtr->scale());
    for (const auto& name : surfacePtr->channelNames()) {
        cv::Mat ch = surfacePtr->channel(name, SURF_CHANNEL_NORESIZE);
        if (!ch.empty()) {
            snapshot->setChannel(name, ch.clone());
        }
    }
    // Copy metadata needed by save()
    snapshot->path = surfacePtr->path;
    snapshot->id = surfacePtr->id;
    snapshot->meta = surfacePtr->meta;

    _pendingAutosave = false;
    _saveInProgress = true;
    _dirtyAfterSave = false;

    emit statusMessageRequested(tr("Saving..."), kStatusShort);

    _saveFuture = QtConcurrent::run([snapshot]() {
        snapshot->saveOverwrite();
    });

    // Poll for completion via a single-shot timer to avoid blocking
    auto* pollTimer = new QTimer(this);
    pollTimer->setInterval(50);
    pollTimer->setSingleShot(false);
    connect(pollTimer, &QTimer::timeout, this, [this, pollTimer]() {
        if (!_saveFuture.isFinished()) {
            return;
        }
        pollTimer->stop();
        pollTimer->deleteLater();

        _saveInProgress = false;

        // Check for exceptions from the future
        try {
            _saveFuture.waitForFinished();
            _autosaveNotifiedFailure = false;
            emit statusMessageRequested(tr("Saved"), kStatusShort);
        } catch (const std::exception& ex) {
            qCWarning(lcSegModule) << "Autosave failed:" << ex.what();
            if (!_autosaveNotifiedFailure) {
                emit statusMessageRequested(tr("Failed to autosave segmentation: %1")
                                                .arg(QString::fromUtf8(ex.what())),
                                            kStatusLong);
                _autosaveNotifiedFailure = true;
            }
            // Re-mark pending so the next timer tick retries
            _pendingAutosave = true;
        }

        // If another save was requested while we were saving, start it now
        if (_dirtyAfterSave) {
            _dirtyAfterSave = false;
            _pendingAutosave = true;
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

    const bool shouldRun = _editingEnabled && _editManager && _editManager->hasSession();
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
    // Only auto-save if we have a valid segment path
    if (_correctionsSegmentPath.empty() || !_pointCollection) {
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
    if (_correctionsSegmentPath.empty() || !_pointCollection) {
        return;
    }

    if (_pointCollection->saveToSegmentPath(_correctionsSegmentPath)) {
        qCDebug(lcSegModule) << "Auto-saved correction points to" << QString::fromStdString(_correctionsSegmentPath.string());
    } else {
        qCWarning(lcSegModule) << "Failed to auto-save correction points";
    }
}
