#include "SegmentationModule.hpp"

#include "adaptive/CAdaptiveVolumeViewer.hpp"
#include "tools/SegmentationBrushTool.hpp"
#include "tools/ApprovalMaskBrushTool.hpp"
#include "tools/SurfaceMaskBrushTool.hpp"
#include "tools/CellReoptimizationTool.hpp"
#include "growth/SegmentationCorrections.hpp"
#include "tools/SegmentationEditManager.hpp"
#include "tools/SegmentationLineTool.hpp"
#include "tools/SegmentationPushPullTool.hpp"
#include "SegmentationWidget.hpp"
#include "../overlays/SegmentationOverlayController.hpp"
#include "../Keybinds.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QKeyEvent>
#include <QKeySequence>
#include <QPointF>
#include <QString>
#include <QtGlobal>

#include <algorithm>

bool SegmentationModule::handleKeyPress(QKeyEvent* event)
{
    if (!event) {
        return false;
    }

    if (event->key() == vc3d::keybinds::keypress::ManualAddToggle.key &&
        event->modifiers() == vc3d::keybinds::keypress::ManualAddToggle.modifiers &&
        (!vc3d::keybinds::keypress::ManualAddToggle.requireNoAutoRepeat || !event->isAutoRepeat())) {
        if (_manualAddMode) {
            finishManualAdd(true);
        } else {
            beginManualAdd();
        }
        event->accept();
        return true;
    }

    if (_manualAddMode) {
        if (!event->isAutoRepeat()) {
            const bool undoRequested =
                event->matches(vc3d::keybinds::standard::Undo) ||
                (event->key() == vc3d::keybinds::keypress::SegmentationUndo.key &&
                 (event->modifiers() & vc3d::keybinds::keypress::SegmentationUndo.modifiers) ==
                     vc3d::keybinds::keypress::SegmentationUndo.modifiers);
            if (undoRequested) {
                if (undoManualAddPlaneConstraint()) {
                    event->accept();
                    return true;
                }
                return false;
            }
        }

        const bool manualAddLineModeCycle =
            !event->isAutoRepeat() &&
            event->key() == Qt::Key_Q &&
            event->modifiers() == Qt::ShiftModifier;
        if (manualAddLineModeCycle && _widget && _manualAddTool) {
            const ManualAddTool::LinePreviewMode mode = _widget->cycleManualAddLinePreviewMode();
            _manualAddTool->setConfig(_widget->manualAddConfig());
            if (auto hover = _manualAddTool->hoverVertex()) {
                _manualAddTool->updateHover(hover->row, hover->col);
            }
            refreshOverlay();
            QString label;
            switch (mode) {
            case ManualAddTool::LinePreviewMode::VerticalOnly:
                label = tr("Vertical only");
                break;
            case ManualAddTool::LinePreviewMode::HorizontalOnly:
                label = tr("Horizontal only");
                break;
            case ManualAddTool::LinePreviewMode::Cross:
                label = tr("Cross");
                break;
            case ManualAddTool::LinePreviewMode::CrossFill:
                label = tr("Cross-fill");
                break;
            }
            emit statusMessageRequested(tr("Manual Add yellow line: %1").arg(label), kStatusShort);
            event->accept();
            return true;
        }
        if (event->key() == vc3d::keybinds::keypress::CancelOperation.key) {
            finishManualAdd(false);
            event->accept();
            return true;
        }
        return false;
    }

    // B: Toggle edit approved mask (only if show approval mask is enabled)
    if (event->key() == vc3d::keybinds::keypress::ApprovalPaintToggle.key &&
        !event->isAutoRepeat() &&
        event->modifiers() == vc3d::keybinds::keypress::ApprovalPaintToggle.modifiers) {
        if (_showApprovalMask) {
            setEditApprovedMask(!_editApprovedMask);
            if (_widget) {
                _widget->setEditApprovedMask(_editApprovedMask);
            }
            emit statusMessageRequested(
                _editApprovedMask ? tr("Approval painting enabled.") : tr("Approval painting disabled."),
                kStatusShort);
            event->accept();
            return true;
        }
    }

    // N: Toggle edit unapproved mask (only if show approval mask is enabled)
    if (event->key() == vc3d::keybinds::keypress::UnapprovalPaintToggle.key &&
        !event->isAutoRepeat() &&
        event->modifiers() == vc3d::keybinds::keypress::UnapprovalPaintToggle.modifiers) {
        if (_showApprovalMask) {
            setEditUnapprovedMask(!_editUnapprovedMask);
            if (_widget) {
                _widget->setEditUnapprovedMask(_editUnapprovedMask);
            }
            emit statusMessageRequested(
                _editUnapprovedMask ? tr("Unapproval painting enabled.") : tr("Unapproval painting disabled."),
                kStatusShort);
            event->accept();
            return true;
        }
    }

    // Ctrl+B: Undo approval mask stroke (only when editing approval mask)
    if (event->key() == vc3d::keybinds::keypress::ApprovalUndo.key && !event->isAutoRepeat() &&
        event->modifiers() == vc3d::keybinds::keypress::ApprovalUndo.modifiers) {
        if (isEditingApprovalMask() && _overlay && _overlay->canUndoApprovalMaskPaint()) {
            undoApprovalStroke();
            event->accept();
            return true;
        }
    }

    if (!event->isAutoRepeat()) {
        const bool undoRequested =
            event->matches(vc3d::keybinds::standard::Undo) ||
            (event->key() == vc3d::keybinds::keypress::SegmentationUndo.key &&
             (event->modifiers() & vc3d::keybinds::keypress::SegmentationUndo.modifiers) ==
                 vc3d::keybinds::keypress::SegmentationUndo.modifiers);
        if (undoRequested) {
            if (restoreUndoSnapshot()) {
                emit statusMessageRequested(tr("Undid last segmentation change."), kStatusShort);
                event->accept();
                return true;
            }
            return false;
        }
    }

    if (event->key() == vc3d::keybinds::keypress::LineDrawHold.key && !event->isAutoRepeat()) {
        if (_editingEnabled && !_growthInProgress && _editManager && _editManager->hasSession()) {
            _lineDrawKeyActive = true;
            stopAllPushPull();
            clearLineDragStroke();
            cancelDrag();
            event->accept();
            return true;
        }
        _lineDrawKeyActive = false;
    }

    if (!event->isAutoRepeat() && event->key() == vc3d::keybinds::keypress::GrowSegmentation.key &&
        event->modifiers() == vc3d::keybinds::keypress::GrowSegmentation.modifiers) {
        if (!_editingEnabled || _growthInProgress || !_widget || !_widget->isEditingEnabled()) {
            return false;
        }

        SegmentationGrowthMethod method = _growthMethod;
        int steps = std::clamp(_growthSteps, 1, 1024);
        SegmentationGrowthDirection direction = SegmentationGrowthDirection::All;

        if (_widget) {
            method = _widget->growthMethod();
            steps = std::clamp(_widget->growthSteps(), 1, 1024);

            const auto allowed = _widget->allowedGrowthDirections();
            if (allowed.size() == 1) {
                direction = allowed.front();
            }
        }

        handleGrowSurfaceRequested(method, direction, steps, false);
        event->accept();
        return true;
    }

    if (event->key() == vc3d::keybinds::keypress::CancelOperation.key) {
        if (_drag.active) {
            cancelDrag();
            return true;
        }
        if (_correctionDrag.active) {
            cancelCorrectionDrag();
            return true;
        }
    }

    const bool pushPullKey =
        (event->key() == vc3d::keybinds::keypress::PushPullIn.key ||
         event->key() == vc3d::keybinds::keypress::PushPullOut.key);
    const Qt::KeyboardModifiers pushPullMods = event->modifiers();
    const bool controlActive = pushPullMods.testFlag(Qt::ControlModifier);
    const Qt::KeyboardModifiers disallowedMods = pushPullMods &
                                                 ~(Qt::ControlModifier | Qt::KeypadModifier);
    if (pushPullKey && disallowedMods == Qt::NoModifier) {
        if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
            return false;
        }

        const int direction = (event->key() == vc3d::keybinds::keypress::PushPullOut.key) ? 1 : -1;
        const std::optional<bool> alphaOverride = controlActive ? std::optional<bool>{true} : std::nullopt;
        if (startPushPull(direction, alphaOverride)) {
            event->accept();
            return true;
        }
        return false;
    }

    // Q and E keys to adjust push pull radius
    const bool radiusAdjustKey =
        (event->key() == vc3d::keybinds::keypress::PushPullRadiusDown.key ||
         event->key() == vc3d::keybinds::keypress::PushPullRadiusUp.key);
    if (radiusAdjustKey && event->modifiers() == Qt::NoModifier) {
        const float step = (event->key() == vc3d::keybinds::keypress::PushPullRadiusUp.key) ? 0.25f : -0.25f;
        const float newRadius = _pushPullRadiusSteps + step;
        setPushPullRadius(newRadius);
        event->accept();
        return true;
    }

    if (event->key() == vc3d::keybinds::keypress::EnableEditing.key &&
        event->modifiers() == vc3d::keybinds::keypress::EnableEditing.modifiers) {
        if (!_editingEnabled) {
            setEditingEnabled(true);
            if (_widget) {
                _widget->setEditingEnabled(true);
            }
        }
        event->accept();
        return true;
    }

    if (event->key() == vc3d::keybinds::keypress::ToggleAnnotation.key &&
        event->modifiers() == vc3d::keybinds::keypress::ToggleAnnotation.modifiers) {
        if (!_editingEnabled) {
            setEditingEnabled(true);
            if (_widget) {
                _widget->setEditingEnabled(true);
            }
            setCorrectionsAnnotateMode(true, true);
        } else {
            // Toggle correction point annotation mode
            bool currentMode = _corrections && _corrections->annotateMode();
            setCorrectionsAnnotateMode(!currentMode, true);
        }
        event->accept();
        return true;
    }

    const bool growthFillShortcut =
        event->modifiers() == vc3d::keybinds::keypress::GrowthFill.modifiers &&
        (event->key() == vc3d::keybinds::keypress::GrowthFill.key ||
         event->key() == Qt::Key_Percent);
    if (growthFillShortcut &&
        !event->isAutoRepeat()) {
        if (!_widget || !_widget->growthKeybindsEnabled()) {
            return false;
        }

        const SegmentationGrowthMethod method = _widget->growthMethod();
        if (method != SegmentationGrowthMethod::PatchTracer) {
            return false;
        }

        _pendingShortcutDirections = std::vector<SegmentationGrowthDirection>{SegmentationGrowthDirection::All};
        const int steps = std::max(1, _widget->growthSteps());
        handleGrowSurfaceRequested(method, SegmentationGrowthDirection::Fill, steps, false);
        event->accept();
        return true;
    }

    if (event->modifiers() == Qt::NoModifier && !event->isAutoRepeat()) {
        if (!_widget || !_widget->growthKeybindsEnabled()) {
            return false;
        }

        if (event->key() == vc3d::keybinds::keypress::GrowthStepAll.key) {
            const SegmentationGrowthMethod method = _widget ? _widget->growthMethod() : _growthMethod;
            handleGrowSurfaceRequested(method, SegmentationGrowthDirection::All, 1, false);
            event->accept();
            return true;
        }

        SegmentationGrowthDirection shortcutDirection{SegmentationGrowthDirection::All};
        bool matchedShortcut = true;
        const int key = event->key();
        if (key == vc3d::keybinds::keypress::GrowthLeft.key) {
            shortcutDirection = SegmentationGrowthDirection::Left;
        } else if (key == vc3d::keybinds::keypress::GrowthUp.key) {
            shortcutDirection = SegmentationGrowthDirection::Up;
        } else if (key == vc3d::keybinds::keypress::GrowthDown.key) {
            shortcutDirection = SegmentationGrowthDirection::Down;
        } else if (key == vc3d::keybinds::keypress::GrowthRight.key) {
            shortcutDirection = SegmentationGrowthDirection::Right;
        } else if (key == vc3d::keybinds::keypress::GrowthAll.key) {
            shortcutDirection = SegmentationGrowthDirection::All;
        } else {
            matchedShortcut = false;
        }

        if (matchedShortcut) {
            _pendingShortcutDirections = std::vector<SegmentationGrowthDirection>{shortcutDirection};
            const int steps = _widget ? std::max(1, _widget->growthSteps()) : std::max(1, _growthSteps);
            const SegmentationGrowthMethod method = _widget ? _widget->growthMethod() : _growthMethod;
            handleGrowSurfaceRequested(method, shortcutDirection, steps, false);
            event->accept();
            return true;
        }
    }

    return false;
}

bool SegmentationModule::handleKeyRelease(QKeyEvent* event)
{
    if (!event) {
        return false;
    }

    if (event->key() == vc3d::keybinds::keypress::LineDrawHold.key && !event->isAutoRepeat()) {
        _lineDrawKeyActive = false;
        if (_lineTool && _lineTool->strokeActive()) {
            _lineTool->finishStroke(_lineDrawKeyActive);
        }
        event->accept();
        return true;
    }

    if (event->key() == Qt::Key_Shift && !event->isAutoRepeat() && _shiftDrawMaskActive) {
        if (_surfaceMaskTool) {
            if (_surfaceMaskTool->strokeActive() || _surfaceMaskTool->hasPendingStroke()) {
                _surfaceMaskTool->finishStroke();
            }
            _surfaceMaskTool->setActive(_drawMaskEnabled);
        }
        _shiftDrawMaskActive = false;
        event->accept();
        return true;
    }

    const bool pushPullKey =
        (event->key() == vc3d::keybinds::keypress::PushPullIn.key ||
         event->key() == vc3d::keybinds::keypress::PushPullOut.key);
    const Qt::KeyboardModifiers pushPullMods = event->modifiers();
    const Qt::KeyboardModifiers disallowedMods = pushPullMods &
                                                 ~(Qt::ControlModifier | Qt::KeypadModifier);
    if (pushPullKey && disallowedMods == Qt::NoModifier) {
        const int direction = (event->key() == vc3d::keybinds::keypress::PushPullOut.key) ? 1 : -1;
        stopPushPull(direction);
        event->accept();
        return true;
    }

    return false;
}

void SegmentationModule::handleMousePress(CTiledVolumeViewer* viewer,
                                          const cv::Vec3f& worldPos,
                                          const cv::Vec3f& /*surfaceNormal*/,
                                          Qt::MouseButton button,
                                          Qt::KeyboardModifiers modifiers,
                                          const QPointF& scenePos)
{
    const bool isLeftButton = (button == Qt::LeftButton);
    const bool isRightButton = (button == Qt::RightButton);

    if (_manualAddMode && handleManualAddMousePress(viewer, worldPos, button, modifiers, scenePos)) {
        return;
    }

    const bool drawMaskRequested = _drawMaskEnabled ||
                                   modifiers.testFlag(Qt::ShiftModifier);
    if (drawMaskRequested && isRightButton && viewer && isSegmentationViewer(viewer) && _surfaceMaskTool) {
        auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
        if (!surface && _editManager && _editManager->hasSession()) {
            surface = _editManager->baseSurface().get();
        }
        if (surface) {
            _shiftDrawMaskActive = !_drawMaskEnabled && modifiers.testFlag(Qt::ShiftModifier);
            _surfaceMaskTool->setSurface(surface);
            _surfaceMaskTool->setActive(true);
            const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);
            _surfaceMaskTool->startStroke(QPointF(surfCoords[0], surfCoords[1]));
        }
        return;
    }

    // Handle approval mask editing mode - works independently of surface editing
    if (isEditingApprovalMask() && isLeftButton) {
        if (modifiers.testFlag(Qt::ControlModifier) || modifiers.testFlag(Qt::AltModifier)) {
            return;
        }
        if (_approvalTool) {
            // Check if this is a plane viewer (XY/XZ/YZ) or segmentation view
            Surface* viewerSurf = viewer->currentSurface();
            auto* planeSurf = dynamic_cast<PlaneSurface*>(viewerSurf);

            if (planeSurf) {
                // For plane viewers, use cylinder-based painting
                const float brushRadiusSteps = _approvalMaskBrushRadius;
                const float worldRadius = brushRadiusSteps * 1.0f;
                const cv::Vec3f planeNormal = planeSurf->normal({0, 0, 0});
                _approvalTool->startStrokeFromPlane(worldPos, planeNormal, worldRadius);
            } else {
                // Flattened view - convert scene coordinates to surface coordinates
                const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);
                _approvalTool->startStroke(worldPos, QPointF(surfCoords[0], surfCoords[1]));
            }
        }
        return;
    }

    // Surface editing requires _editingEnabled
    if (!_editingEnabled) {
        return;
    }

    if (isLeftButton && isNearRotationHandle(viewer, worldPos)) {
        return;
    }

    if (_corrections && _corrections->annotateMode()) {
        if (!isLeftButton) {
            return;
        }
        if (modifiers.testFlag(Qt::ControlModifier)) {
            handleCorrectionPointRemove(worldPos);
            updateCorrectionsWidget();
            return;
        }
        // Shift+click+drag: start correction drag with anchor2d
        // Click without Shift: add correction point directly (old behavior)
        if (modifiers.testFlag(Qt::ShiftModifier) && _editManager) {
            auto gridIndex = _editManager->worldToGridIndex(worldPos);
            if (gridIndex) {
                beginCorrectionDrag(gridIndex->first, gridIndex->second, viewer, worldPos);
                return;
            }
        }
        // Default: add correction point at clicked position
        handleCorrectionPointAdded(worldPos);
        updateCorrectionsWidget();
        return;
    }

    // Handle cell reoptimization mode
    if (_cellReoptMode && isLeftButton) {
        if (modifiers.testFlag(Qt::ControlModifier) || modifiers.testFlag(Qt::AltModifier)) {
            return;
        }
        if (_cellReoptTool && _editManager) {
            auto gridIndex = _editManager->worldToGridIndex(worldPos);
            if (gridIndex) {
                // Update the tool's surface reference and config
                if (_editManager->baseSurface()) {
                    _cellReoptTool->setSurface(_editManager->baseSurface().get());
                }
                if (_widget) {
                    CellReoptimizationTool::Config config;
                    config.maxFloodSteps = _widget->cellReoptMaxSteps();
                    config.maxCorrectionPoints = _widget->cellReoptMaxPoints();
                    config.minBoundarySpacing = _widget->cellReoptMinSpacing();
                    config.perimeterOffset = _widget->cellReoptPerimeterOffset();
                    _cellReoptTool->setConfig(config);
                }
                _cellReoptTool->executeAtGridPosition(gridIndex->first, gridIndex->second);
            }
        }
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

    if (_lineDrawKeyActive) {
        stopAllPushPull();
        if (_drag.active) {
            cancelDrag();
        }
        if (_lineTool) {
            _lineTool->startStroke(worldPos);
        }
        return;
    }

    stopAllPushPull();
    auto gridIndex = _editManager->worldToGridIndex(worldPos);
    if (!gridIndex) {
        return;
    }

    if (_activeFalloff != FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }

    if (!_editManager->beginActiveDrag(*gridIndex)) {
        return;
    }

    beginDrag(gridIndex->first, gridIndex->second, viewer, worldPos);
    refreshOverlay();
}

void SegmentationModule::handleMouseMove(CTiledVolumeViewer* viewer,
                                         const cv::Vec3f& worldPos,
                                         Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers modifiers,
                                         const QPointF& scenePos)
{
    if (_manualAddMode && handleManualAddMouseMove(viewer, buttons, scenePos)) {
        return;
    }

    const bool maskStrokeActive = _surfaceMaskTool && _surfaceMaskTool->strokeActive();
    if (maskStrokeActive) {
        if (buttons.testFlag(Qt::RightButton) &&
            (_drawMaskEnabled || modifiers.testFlag(Qt::ShiftModifier))) {
            const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);
            _surfaceMaskTool->extendStroke(QPointF(surfCoords[0], surfCoords[1]), false);
        } else {
            _surfaceMaskTool->finishStroke();
            if (_shiftDrawMaskActive) {
                _surfaceMaskTool->setActive(_drawMaskEnabled);
                _shiftDrawMaskActive = false;
            }
        }
        return;
    }

    // Handle approval mask mode
    const bool approvalStrokeActive = _approvalTool && _approvalTool->strokeActive();
    if (approvalStrokeActive) {
        if (buttons.testFlag(Qt::LeftButton)) {
            if (_approvalTool) {
                // Check if this is a plane viewer (XY/XZ/YZ) or segmentation view
                Surface* viewerSurf = viewer->currentSurface();
                auto* planeSurf = dynamic_cast<PlaneSurface*>(viewerSurf);

                if (planeSurf) {
                    // For plane viewers, use cylinder-based painting
                    const float brushRadiusSteps = _approvalMaskBrushRadius;
                    const float worldRadius = brushRadiusSteps * 1.0f;
                    const cv::Vec3f planeNormal = planeSurf->normal({0, 0, 0});
                    _approvalTool->extendStrokeFromPlane(worldPos, planeNormal, worldRadius, false);
                } else {
                    // Convert scene coordinates to surface coordinates for grid mapping
                    const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);
                    _approvalTool->extendStroke(worldPos, QPointF(surfCoords[0], surfCoords[1]), false);
                }
            }
        } else {
            if (_approvalTool) {
                // Check viewer type to call appropriate finish method
                Surface* viewerSurf = viewer->currentSurface();
                auto* planeSurf = dynamic_cast<PlaneSurface*>(viewerSurf);
                if (planeSurf) {
                    _approvalTool->finishStrokeFromPlane();
                } else {
                    _approvalTool->finishStroke();
                }
            }
        }
        return;
    }

    // Update hover position for approval brush circle when in edit approval mode but not stroking
    // Only update if position changed significantly to avoid expensive refreshOverlay on every mouse move
    if (isEditingApprovalMask() && _approvalTool && !buttons.testFlag(Qt::LeftButton)) {
        const auto lastHover = _approvalTool->hoverWorldPos();
        const float minMoveThreshold = 2.0f;  // Native voxels
        bool shouldUpdate = !lastHover.has_value();
        if (lastHover) {
            const cv::Vec3f delta = worldPos - *lastHover;
            shouldUpdate = delta.dot(delta) >= minMoveThreshold * minMoveThreshold;
        }
        if (shouldUpdate) {
            const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);

            // Get plane normal if this is a plane viewer (XY/XZ/YZ)
            std::optional<cv::Vec3f> planeNormal;
            Surface* viewerSurf = viewer->currentSurface();
            if (auto* planeSurf = dynamic_cast<PlaneSurface*>(viewerSurf)) {
                planeNormal = planeSurf->normal({0, 0, 0});
            }

            _approvalTool->setHoverWorldPos(worldPos, _approvalMaskBrushRadius, QPointF(surfCoords[0], surfCoords[1]), planeNormal);
            refreshOverlay();
        }
    }

    const bool lineStrokeActive = _lineTool && _lineTool->strokeActive();
    if (lineStrokeActive) {
        if (buttons.testFlag(Qt::LeftButton)) {
            if (_lineTool) {
                _lineTool->extendStroke(worldPos, false);
            }
        } else {
            if (_lineTool) {
                _lineTool->finishStroke(_lineDrawKeyActive);
            }
        }
        return;
    }

    if (_drag.active) {
        updateDrag(worldPos);
        return;
    }

    if (_correctionDrag.active) {
        updateCorrectionDrag(worldPos);
        return;
    }

    if (_corrections && _corrections->annotateMode()) {
        return;
    }

    if (!buttons.testFlag(Qt::LeftButton)) {
        recordPointerSample(viewer, worldPos);
        updateHover(viewer, worldPos, scenePos);
    }
}

void SegmentationModule::handleMouseRelease(CTiledVolumeViewer* viewer,
                                            const cv::Vec3f& worldPos,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers /*modifiers*/,
                                            const QPointF& scenePos)
{
    if (_manualAddMode) {
        Q_UNUSED(viewer);
        Q_UNUSED(worldPos);
        Q_UNUSED(button);
        Q_UNUSED(scenePos);
        return;
    }

    const bool maskStrokeActive = _surfaceMaskTool && _surfaceMaskTool->strokeActive();
    if (maskStrokeActive && button == Qt::RightButton) {
        if (_surfaceMaskTool && viewer) {
            const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);
            _surfaceMaskTool->extendStroke(QPointF(surfCoords[0], surfCoords[1]), true);
            if (_shiftDrawMaskActive) {
                _surfaceMaskTool->finishStroke();
                _surfaceMaskTool->setActive(_drawMaskEnabled);
                _shiftDrawMaskActive = false;
            } else {
                _surfaceMaskTool->finishStroke();
            }
        }
        return;
    }

    // Handle approval mask mode
    const bool approvalStrokeActive = _approvalTool && _approvalTool->strokeActive();
    if (approvalStrokeActive && button == Qt::LeftButton) {
        if (_approvalTool && viewer) {
            // Check if this is a plane viewer (XY/XZ/YZ) or segmentation view
            Surface* viewerSurf = viewer->currentSurface();
            auto* planeSurf = dynamic_cast<PlaneSurface*>(viewerSurf);

            if (planeSurf) {
                // For plane viewers, use cylinder-based painting
                const float brushRadiusSteps = _approvalMaskBrushRadius;
                const float worldRadius = brushRadiusSteps * 1.0f;
                const cv::Vec3f planeNormal = planeSurf->normal({0, 0, 0});
                _approvalTool->extendStrokeFromPlane(worldPos, planeNormal, worldRadius, true);
                _approvalTool->finishStrokeFromPlane();
            } else {
                // Convert scene coordinates to surface coordinates for grid mapping
                const cv::Vec2f surfCoords = viewer->sceneToSurfaceCoords(scenePos);
                _approvalTool->extendStroke(worldPos, QPointF(surfCoords[0], surfCoords[1]), true);
                _approvalTool->finishStroke();
            }
            // Don't apply immediately - wait for user to press Apply button
        }
        return;
    }

    const bool lineStrokeActive = _lineTool && _lineTool->strokeActive();
    if (lineStrokeActive && button == Qt::LeftButton) {
        if (_lineTool) {
            _lineTool->extendStroke(worldPos, true);
            _lineTool->finishStroke(_lineDrawKeyActive);
        }
        return;
    }

    if (_correctionDrag.active && button == Qt::LeftButton) {
        updateCorrectionDrag(worldPos);
        finishCorrectionDrag();
        return;
    }

    if (!_drag.active || button != Qt::LeftButton) {
        if (_corrections && _corrections->annotateMode() && button == Qt::LeftButton) {
            return;
        }
        return;
    }

    updateDrag(worldPos);
    finishDrag();
}

void SegmentationModule::handleWheel(CTiledVolumeViewer* viewer,
                                     int deltaSteps,
                                     const QPointF& scenePos,
                                     const cv::Vec3f& worldPos)
{
    if (!_editingEnabled) {
        return;
    }
    const float step = deltaSteps * 0.25f;
    FalloffTool targetTool = FalloffTool::Drag;
    const bool lineStrokeActive = _lineTool && _lineTool->strokeActive();
    if (_lineDrawKeyActive || lineStrokeActive) {
        targetTool = FalloffTool::Line;
    } else if (_pushPullTool && _pushPullTool->isActive()) {
        targetTool = FalloffTool::PushPull;
    }

    const float currentRadius = falloffRadius(targetTool);
    const float newRadius = currentRadius + step;

    switch (targetTool) {
    case FalloffTool::Drag:
        setDragRadius(newRadius);
        break;
    case FalloffTool::Line:
        setLineRadius(newRadius);
        break;
    case FalloffTool::PushPull:
        setPushPullRadius(newRadius);
        break;
    }

    recordPointerSample(viewer, worldPos);
    updateHover(viewer, worldPos, scenePos);
    const float updatedRadius = falloffRadius(targetTool);
    QString label;
    switch (targetTool) {
    case FalloffTool::Drag:
        label = tr("Drag brush radius");
        break;
    case FalloffTool::Line:
        label = tr("Line brush radius");
        break;
    case FalloffTool::PushPull:
        label = tr("Push/Pull radius");
        break;
    }
    emit statusMessageRequested(tr("%1: %2 steps").arg(label).arg(updatedRadius, 0, 'f', 2), kStatusShort);
}
