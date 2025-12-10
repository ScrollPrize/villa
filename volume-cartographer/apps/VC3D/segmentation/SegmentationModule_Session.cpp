#include "SegmentationModule.hpp"

#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "ApprovalMaskBrushTool.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include <QLoggingCategory>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

bool SegmentationModule::beginEditingSession(std::shared_ptr<QuadSurface> surface)
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }

    if (_overlay) {
        _overlay->setEditingEnabled(_editingEnabled);
    }

    useFalloff(_activeFalloff);

    // Set surface on approval tool if edit approval mask mode is active
    if (isEditingApprovalMask() && _approvalTool) {
        _approvalTool->setSurface(_editManager->baseSurface().get());
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
    auto baseSurface = _editManager ? _editManager->baseSurface() : nullptr;
    auto previewSurface = _editManager ? _editManager->previewSurface() : nullptr;

    if (_surfaces && previewSurface) {
        auto currentSurface = _surfaces->surface("segmentation");
        if (currentSurface.get() == previewSurface.get()) {
            const bool previousGuard = _ignoreSegSurfaceChange;
            _ignoreSegSurfaceChange = true;
            _surfaces->setSurface("segmentation", baseSurface, false, true);
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

void SegmentationModule::onSurfaceCollectionChanged(std::string name, std::shared_ptr<Surface> surface)
{
    if (name != "segmentation" || !_editingEnabled || _ignoreSegSurfaceChange) {
        return;
    }

    if (!_editManager) {
        setEditingEnabled(false);
        return;
    }

    auto previewSurface = _editManager->previewSurface();
    auto baseSurface = _editManager->baseSurface();

    if (surface.get() == previewSurface.get() || surface.get() == baseSurface.get()) {
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

bool SegmentationModule::captureUndoDelta()
{
    if (_suppressUndoCapture) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    const auto editedVerts = _editManager->editedVertices();
    if (editedVerts.empty()) {
        return false;
    }

    // Convert to delta format (storing original positions for undo)
    std::vector<segmentation::VertexDelta> deltas;
    deltas.reserve(editedVerts.size());
    for (const auto& edit : editedVerts) {
        deltas.push_back({edit.row, edit.col, edit.originalWorld});
    }

    return _undoHistory.captureDelta(deltas);
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

    if (_undoHistory.empty()) {
        return false;
    }

    _suppressUndoCapture = true;
    bool applied = false;
    std::optional<cv::Rect> undoBounds;

    // Check if this is a delta-based entry or full snapshot
    if (_undoHistory.lastIsDelta()) {
        auto deltas = _undoHistory.takeLastDelta();
        if (deltas && !deltas->empty()) {
            // Apply deltas to restore previous positions
            auto& previewPoints = _editManager->previewPointsMutable();
            int minRow = INT_MAX, maxRow = INT_MIN;
            int minCol = INT_MAX, maxCol = INT_MIN;

            for (const auto& delta : *deltas) {
                if (delta.row >= 0 && delta.row < previewPoints.rows &&
                    delta.col >= 0 && delta.col < previewPoints.cols) {
                    previewPoints(delta.row, delta.col) = delta.previousWorld;
                    minRow = std::min(minRow, delta.row);
                    maxRow = std::max(maxRow, delta.row);
                    minCol = std::min(minCol, delta.col);
                    maxCol = std::max(maxCol, delta.col);
                }
            }

            if (minRow <= maxRow && minCol <= maxCol) {
                undoBounds = cv::Rect(minCol, minRow, maxCol - minCol + 1, maxRow - minRow + 1);
            }

            _editManager->applyPreview();
            applied = true;
        }
    } else {
        // Legacy full snapshot restore
        auto state = _undoHistory.takeLast();
        if (state && !state->empty()) {
            applied = _editManager->setPreviewPoints(*state, false, &undoBounds);
            if (applied) {
                _editManager->applyPreview();
            } else {
                _undoHistory.pushBack(std::move(*state));
            }
        }
    }

    if (applied) {
        if (_surfaces) {
            auto preview = _editManager->previewSurface();

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

            _surfaces->setSurface("segmentation", preview, false, true);
        }

        // Also undo the corresponding auto-approval if approval mask is active
        if (_overlay && _overlay->hasApprovalMaskData() && _overlay->canUndoAutoApproval()) {
            _overlay->undoLastAutoApproval();
            // Schedule save to persist the undo
            if (_editManager && _editManager->baseSurface()) {
                _overlay->scheduleDebouncedSave(_editManager->baseSurface().get());
            }
        }

        clearInvalidationBrush();
        refreshOverlay();
        emitPendingChanges();
        markAutosaveNeeded();
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
    return _editManager ? _editManager->baseSurface().get() : nullptr;
}

std::shared_ptr<QuadSurface> SegmentationModule::activeBaseSurfaceShared() const
{
    return _editManager ? _editManager->baseSurface() : nullptr;
}

void SegmentationModule::refreshSessionFromSurface(QuadSurface* surface)
{
    if (!_editManager || !surface) {
        return;
    }
    if (_editManager->baseSurface().get() != surface) {
        return;
    }
    cancelDrag();
    _editManager->clearInvalidatedEdits();
    _editManager->refreshFromBaseSurface();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
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

    // Auto-approve the growth region if approval mask is active (growth = reviewed/corrected)
    // We paint into pending, then save immediately so the changes persist after reload
    auto* baseSurf = _editManager->baseSurface().get();
    if (_overlay && _overlay->hasApprovalMaskData() && vertexRect.area() > 0) {
        std::vector<std::pair<int, int>> gridPositions;
        gridPositions.reserve(static_cast<size_t>(vertexRect.area()));
        for (int row = vertexRect.y; row < vertexRect.y + vertexRect.height; ++row) {
            for (int col = vertexRect.x; col < vertexRect.x + vertexRect.width; ++col) {
                gridPositions.emplace_back(row, col);
            }
        }
        constexpr uint8_t kApproved = 255;
        constexpr float kRadius = 1.0f;
        constexpr bool kIsAutoApproval = true;
        const QColor brushColor = approvalBrushColor();
        _overlay->paintApprovalMaskDirect(gridPositions, kRadius, kApproved, brushColor, false, 0.0f, 0.0f, kIsAutoApproval);
        // Save immediately to persist through the upcoming reload
        _overlay->saveApprovalMaskToSurface(baseSurf);
        _overlay->clearApprovalMaskUndoHistory();
        qCInfo(lcSegModule) << "Auto-approved growth region:" << vertexRect.width << "x" << vertexRect.height;
    }

    // Update approval tool surface if editing approval mask
    if (isEditingApprovalMask() && _approvalTool) {
        _approvalTool->setSurface(baseSurf);
    }

    // Reload approval mask image if showing approval mask
    if (_showApprovalMask && _overlay) {
        _overlay->loadApprovalMaskImage(baseSurf);
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
        approvalSurface = _editManager->baseSurface().get();
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
