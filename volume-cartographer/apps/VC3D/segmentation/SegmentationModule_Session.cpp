#include "SegmentationModule.hpp"

#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "ApprovalMaskBrushTool.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include <QLoggingCategory>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

bool SegmentationModule::beginEditingSession(QuadSurface* surface)
{
    if (!_editManager || !surface) {
        return false;
    }

    stopAllPushPull();
    clearUndoStack();
    clearInvalidationBrush();
    setInvalidationBrushActive(false);
    resetHoverLookupDetail();
    _hoverPointer.valid = false;
    _hoverPointer.viewer = nullptr;
    if (!_editManager->beginSession(surface)) {
        qCWarning(lcSegModule) << "Failed to begin segmentation editing session";
        return false;
    }

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false, true);
    }

    if (_overlay) {
        _overlay->setEditingEnabled(_editingEnabled);
    }

    useFalloff(_activeFalloff);

    // Set surface on approval tool if edit approval mask mode is active
    if (isEditingApprovalMask() && _approvalTool) {
        _approvalTool->setSurface(_editManager->baseSurface());
    }

    if (_overlay) {
        refreshOverlay();
    }

    emitPendingChanges();
    _pendingAutosave = false;
    _autosaveNotifiedFailure = false;
    updateAutosaveState();
    return true;
}

void SegmentationModule::endEditingSession()
{
    stopAllPushPull();
    clearUndoStack();
    cancelDrag();
    clearInvalidationBrush();
    clearLineDragStroke();
    setInvalidationBrushActive(false);
    _lineDrawKeyActive = false;
    resetHoverLookupDetail();
    _hoverPointer.valid = false;
    _hoverPointer.viewer = nullptr;
    refreshOverlay();
    QuadSurface* baseSurface = _editManager ? _editManager->baseSurface() : nullptr;
    QuadSurface* previewSurface = _editManager ? _editManager->previewSurface() : nullptr;

    if (_surfaces && previewSurface) {
        Surface* currentSurface = _surfaces->surface("segmentation");
        if (currentSurface == previewSurface) {
            const bool previousGuard = _ignoreSegSurfaceChange;
            _ignoreSegSurfaceChange = true;
            _surfaces->setSurface("segmentation", baseSurface, false, false, true);
            _ignoreSegSurfaceChange = previousGuard;
        }
    }

    if (_pendingAutosave) {
        performAutosave();
    }

    if (_editManager) {
        _editManager->endSession();
    }

    updateAutosaveState();
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
    endEditingSession();
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

    return _undoHistory.capture(previewPoints);
}

void SegmentationModule::discardLastUndoSnapshot()
{
    _undoHistory.discardLast();
}

bool SegmentationModule::restoreUndoSnapshot()
{
    if (_suppressUndoCapture) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    auto state = _undoHistory.takeLast();
    if (!state) {
        return false;
    }

    cv::Mat_<cv::Vec3f> points = std::move(*state);
    if (points.empty()) {
        return false;
    }

    _suppressUndoCapture = true;
    std::optional<cv::Rect> undoBounds;
    bool applied = _editManager->setPreviewPoints(points, false, &undoBounds);
    if (applied) {
        _editManager->applyPreview();
        if (_surfaces) {
            auto* preview = _editManager->previewSurface();

            // Queue affected cells for incremental R-tree update
            if (preview && undoBounds && undoBounds->width > 0 && undoBounds->height > 0 && _viewerManager) {
                if (auto* index = _viewerManager->surfacePatchIndex()) {
                    index->queueCellRangeUpdate(preview,
                                              undoBounds->y,
                                              undoBounds->y + undoBounds->height,
                                              undoBounds->x,
                                              undoBounds->x + undoBounds->width);
                }
            }

            _surfaces->setSurface("segmentation", preview, false, false, true);
        }
        clearInvalidationBrush();
        refreshOverlay();
        emitPendingChanges();
        markAutosaveNeeded();
    } else {
        _undoHistory.pushBack(std::move(points));
    }
    _suppressUndoCapture = false;

    return applied;
}

void SegmentationModule::clearUndoStack()
{
    _undoHistory.clear();
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, false, true);
    }

    // Update approval tool surface if editing approval mask
    if (isEditingApprovalMask() && _approvalTool) {
        _approvalTool->setSurface(surface);
    }

    // Reload approval mask image if showing approval mask
    if (_showApprovalMask && _overlay) {
        _overlay->loadApprovalMaskImage(surface);
    }

    refreshOverlay();
    emitPendingChanges();
}

bool SegmentationModule::applySurfaceUpdateFromGrowth(const cv::Rect& vertexRect)
{
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }
    if (!_editManager->applyExternalSurfaceUpdate(vertexRect)) {
        return false;
    }

    // Update approval tool surface if editing approval mask
    if (isEditingApprovalMask() && _approvalTool) {
        _approvalTool->setSurface(_editManager->baseSurface());
    }

    // Reload approval mask image if showing approval mask
    if (_showApprovalMask && _overlay) {
        _overlay->loadApprovalMaskImage(_editManager->baseSurface());
    }

    refreshOverlay();
    emitPendingChanges();
    return true;
}

void SegmentationModule::requestAutosaveFromGrowth()
{
    markAutosaveNeeded();
}

void SegmentationModule::updateApprovalToolAfterGrowth(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    // Use base surface if there's an active editing session, otherwise use the provided surface
    QuadSurface* approvalSurface = surface;
    if (_editManager && _editManager->hasSession()) {
        approvalSurface = _editManager->baseSurface();
    }

    if (!approvalSurface) {
        return;
    }

    // Update approval tool surface if editing approval mask
    if (isEditingApprovalMask() && _approvalTool) {
        _approvalTool->setSurface(approvalSurface);
    }

    // Reload approval mask image if showing approval mask
    if (_showApprovalMask && _overlay) {
        _overlay->loadApprovalMaskImage(approvalSurface);
    }
}
