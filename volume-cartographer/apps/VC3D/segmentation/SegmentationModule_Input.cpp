#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "SegmentationBrushTool.hpp"
#include "SegmentationCorrections.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationLineTool.hpp"
#include "SegmentationPushPullTool.hpp"
#include "SegmentationWidget.hpp"

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

    if (event->key() == Qt::Key_S && !event->isAutoRepeat()) {
        if (_editingEnabled && !_growthInProgress && _editManager && _editManager->hasSession()) {
            _lineDrawKeyActive = true;
            stopAllPushPull();
            const bool brushActive = _brushTool && _brushTool->brushActive();
            if (brushActive) {
                deactivateInvalidationBrush();
            }
            clearLineDragStroke();
            cancelDrag();
            event->accept();
            return true;
        }
        _lineDrawKeyActive = false;
    }

    if (event->key() == Qt::Key_E && !event->isAutoRepeat()) {
        const Qt::KeyboardModifiers mods = event->modifiers();
        if (mods == Qt::NoModifier || mods == Qt::ShiftModifier) {
            if (_brushTool && _brushTool->applyPending(_dragRadiusSteps)) {
                event->accept();
                return true;
            }
        }
    }

    if (!event->isAutoRepeat() && event->key() == Qt::Key_G &&
        event->modifiers() == Qt::ControlModifier) {
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

        handleGrowSurfaceRequested(method, direction, steps);
        event->accept();
        return true;
    }

    if (event->key() == Qt::Key_Escape) {
        if (_drag.active) {
            cancelDrag();
            return true;
        }
    }

    const bool pushPullKey = (event->key() == Qt::Key_A || event->key() == Qt::Key_D);
    const Qt::KeyboardModifiers pushPullMods = event->modifiers();
    const bool controlActive = pushPullMods.testFlag(Qt::ControlModifier);
    const Qt::KeyboardModifiers disallowedMods = pushPullMods &
                                                 ~(Qt::ControlModifier | Qt::KeypadModifier);
    if (pushPullKey && disallowedMods == Qt::NoModifier) {
        if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
            return false;
        }

        const int direction = (event->key() == Qt::Key_D) ? 1 : -1;
        const std::optional<bool> alphaOverride = controlActive ? std::optional<bool>{true} : std::nullopt;
        if (startPushPull(direction, alphaOverride)) {
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
        if (event->key() == Qt::Key_6) {
            const SegmentationGrowthMethod method = _widget ? _widget->growthMethod() : _growthMethod;
            handleGrowSurfaceRequested(method, SegmentationGrowthDirection::All, 1);
            event->accept();
            return true;
        }

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

    if (event->key() == Qt::Key_S && !event->isAutoRepeat()) {
        _lineDrawKeyActive = false;
        if (_lineTool && _lineTool->strokeActive()) {
            _lineTool->finishStroke(_lineDrawKeyActive);
        }
        event->accept();
        return true;
    }

    const bool pushPullKey = (event->key() == Qt::Key_A || event->key() == Qt::Key_D);
    const Qt::KeyboardModifiers pushPullMods = event->modifiers();
    const Qt::KeyboardModifiers disallowedMods = pushPullMods &
                                                 ~(Qt::ControlModifier | Qt::KeypadModifier);
    if (pushPullKey && disallowedMods == Qt::NoModifier) {
        const int direction = (event->key() == Qt::Key_D) ? 1 : -1;
        stopPushPull(direction);
        event->accept();
        return true;
    }

    return false;
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

    if (_corrections && _corrections->annotateMode()) {
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

    const bool brushActive = _brushTool && _brushTool->brushActive();

    if (_lineDrawKeyActive) {
        stopAllPushPull();
        if (brushActive) {
            deactivateInvalidationBrush();
        }
        if (_drag.active) {
            cancelDrag();
        }
        if (_lineTool) {
            _lineTool->startStroke(worldPos);
        }
        return;
    }

    if (brushActive) {
        stopAllPushPull();
        if (_brushTool) {
            _brushTool->startStroke(worldPos);
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

void SegmentationModule::handleMouseMove(CVolumeViewer* viewer,
                                         const cv::Vec3f& worldPos,
                                         Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);

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

    const bool paintStrokeActive = _brushTool && _brushTool->strokeActive();
    if (paintStrokeActive) {
        if (buttons.testFlag(Qt::LeftButton)) {
            if (_brushTool) {
                _brushTool->extendStroke(worldPos, false);
            }
        } else {
            if (_brushTool) {
                _brushTool->finishStroke();
            }
        }
        return;
    }

    if (_drag.active) {
        updateDrag(worldPos);
        return;
    }

    if (_corrections && _corrections->annotateMode()) {
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
    const bool lineStrokeActive = _lineTool && _lineTool->strokeActive();
    if (lineStrokeActive && button == Qt::LeftButton) {
        if (_lineTool) {
            _lineTool->extendStroke(worldPos, true);
            _lineTool->finishStroke(_lineDrawKeyActive);
        }
        return;
    }

    const bool paintStrokeActive = _brushTool && _brushTool->strokeActive();
    if (paintStrokeActive && button == Qt::LeftButton) {
        if (_brushTool) {
            _brushTool->extendStroke(worldPos, true);
            _brushTool->finishStroke();
        }
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

void SegmentationModule::handleWheel(CVolumeViewer* viewer,
                                     int deltaSteps,
                                     const QPointF& /*scenePos*/,
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

    updateHover(viewer, worldPos);
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
