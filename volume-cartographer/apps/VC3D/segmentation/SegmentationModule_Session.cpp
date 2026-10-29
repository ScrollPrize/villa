#include "SegmentationModule.hpp"
#include "SurfaceCorrections.hpp"
#include "SurfaceStoreEdit.hpp"

#include "../CState.hpp"
#include "SegmentationWidget.hpp"
#include "tools/SegmentationEditManager.hpp"
#include "tools/SurfaceMaskBrushTool.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include <QLoggingCategory>
#include <QTimer>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

bool SegmentationModule::prepareEditingDestination(const std::shared_ptr<QuadSurface>& surface)
{
    if (!_editingDestinationResolver) return true;
    // Save workers capture their destination; drain before creating its copy.
    try {
        if (_saveFuture.isRunning()) _saveFuture.waitForFinished();
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("Finish the failed surface save before changing destinations: %1")
            .arg(QString::fromUtf8(error.what())), 15000);
        return false;
    }
    return _editingDestinationResolver(surface);
}

bool SegmentationModule::beginEditingSession(std::shared_ptr<QuadSurface> surface)
{
    if (!_editManager || !surface) {
        return false;
    }

    if (!prepareEditingDestination(surface)) return false;
    // Editing starts from the revision on screen and holds the object until it ends.
    const auto storeEdit = storeEditFor(surface.get());
    if (!storeEdit) {
        emit statusMessageRequested(tr("Only surfaces opened from the object store can be edited."), kStatusLong);
        return false;
    }
    try {
        if (!_state->objectStoreSession()->writable(storeEdit->pin().attachment))
            throw std::runtime_error("The object store is attached read-only");
        _editLease = _state->objectStoreSession()->holdForEditing(storeEdit->pin());
    } catch (const std::exception& error) {
        emit statusMessageRequested(tr("This surface cannot be edited now: %1").arg(QString::fromUtf8(error.what())),
                                    kStatusLong);
        return false;
    }
    stopAllPushPull();
    clearUndoStack();
    _hoverPointer.valid = false;
    _hoverPointer.viewer = nullptr;
    // The edit's draft is declared in the frame the surface was read through.
    SegmentationEditManager::DraftTarget draft{_state->objectStoreSession(), {}, storeEdit->pin().attachment};
    if (const auto& view = storeEdit->view()) {
        draft.coordinates = vc::objects::volumeCoordinates(view->volume, view->scale);
    } else {
        draft.session->read(draft.attachment, [&](const auto& db, const auto& txn, const auto&) {
            if (const auto record = db.objectByUuid(txn, storeEdit->uuid()))
                draft.coordinates = nlohmann::json::parse(record->meta).value("coordinates", nlohmann::json::object());
        });
    }
    if (!_editManager->beginSession(surface, std::move(draft))) {
        qCWarning(lcSegModule) << "Failed to begin segmentation editing session";
        return false;
    }
    _autosaveState.beginSession();

    if (_state) {
        _state->setActiveSurfaceEditing(true);
        // The preview is the base surface itself, unchanged. Re-announcing the
        // surface already on screen as an edit would drop every flattened-view
        // tile and flash the pane black until they refill.
        if (_state->surface("segmentation") != _editManager->previewSurface()) {
            _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
        }
    }

    if (_overlay) {
        _overlay->setEditingEnabled(_editingEnabled);
    }

    useFalloff(_activeFalloff);

    if (_surfaceMaskTool) {
        _surfaceMaskTool->setSurface(_editManager->baseSurface().get());
        _surfaceMaskTool->setActive(_editingEnabled && _drawMaskEnabled);
    }

    if (_overlay) {
        refreshOverlay();
    }

    // The surface's stored corrections, auto-saved while it stays active.
    loadCorrections(surface.get());

    emitPendingChanges();
    _pendingAutosaveVertexUpdates.clear();
    _pendingAutosaveVertexUpdateIndex.clear();
    _saveSnapshot.reset();
    updateAutosaveState();
    return true;
}

void SegmentationModule::endEditingSession()
{
    _manualAddContinuation.reset();
    stopAllPushPull();
    clearUndoStack();
    cancelDrag();
    if (_surfaceMaskTool) {
        _surfaceMaskTool->setActive(false);
    }
    _hoverPointer.valid = false;
    _hoverPointer.viewer = nullptr;
    refreshOverlay();
    // Edits land in the base surface in place (the preview is the base), so the
    // surface on screen stays as it is: nothing to restore or re-announce.

    // Save any pending corrections immediately and clear auto-save state
    if (_correctionsSaveTimer && _correctionsSaveTimer->isActive()) {
        _correctionsSaveTimer->stop();
    }
    saveCorrections();

    if (_autosaveState.pending()) {
        performAutosave();
    }

    if (_editManager) {
        _editManager->endSession();
    }
    if (_state) {
        _state->setActiveSurfaceEditing(false);
    }
    _autosaveState.endSession();
    _saveSnapshot.reset();
    _pendingAutosaveVertexUpdates.clear();
    _pendingAutosaveVertexUpdateIndex.clear();
    _editLease = {};
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
    std::vector<SegmentationEditManager::VertexEdit> autosaveEdits;
    const bool editingWasEnabled = _editingEnabled;

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
                    autosaveEdits.push_back({delta.row, delta.col, {}, delta.previousWorld, false});
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
            if (auto surface = _editManager->previewSurface()) {
                surface->invalidateCache();
            }
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
        auto restoredSurface = _editManager->previewSurface();
        if (restoredSurface) {
            restoredSurface->invalidateCache();
        }

        if (_editManager) {
            _editManager->refreshFromBaseSurface();
        }

        if (_surfaceMaskTool) {
            _surfaceMaskTool->setSurface(restoredSurface.get());
            _surfaceMaskTool->refreshFromSurface();
            _surfaceMaskTool->setActive(editingWasEnabled && _drawMaskEnabled && hasActiveSession());
        }

        if (_state) {
            auto preview = _editManager->previewSurface();


            _state->setSurface("segmentation", preview, false, true);
        }

        if (editingWasEnabled && !_editingEnabled) {
            setEditingEnabled(true);
            if (_widget) {
                _widget->setEditingEnabled(true);
            }
        }

        refreshOverlay();
        emitPendingChanges();
        queueAutosaveVertexUpdates(autosaveEdits);
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
    if (_state) {
        _state->setSurface("segmentation", _editManager->previewSurface(), false, true);
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

    refreshOverlay();
    emitPendingChanges();
    return true;
}

void SegmentationModule::requestAutosaveFromGrowth()
{
    markAutosaveNeeded();
}

void SegmentationModule::applyCorrectionAnchorOffset(float offsetX, float offsetY)
{
    if (_pointCollection) {
        _pointCollection->applyAnchorOffset(offsetX, offsetY);
    }
}

void SegmentationModule::loadCorrections(QuadSurface* surface)
{
    _correctionsSurfaceId.clear();
    if (!_pointCollection) {
        return;
    }
    std::map<uint64_t, utils::Json> stored;
    if (const auto storeEdit = storeEditFor(surface)) {
        try {
            stored = vc3d::corrections::read(*_state->objectStoreSession(), storeEdit->pin().attachment, storeEdit->uuid(),
                                             storeEdit->view());
            _correctionsSurfaceId = storeEdit->uuid();
        } catch (const std::exception& error) {
            emit statusMessageRequested(tr("Cannot read correction points: %1").arg(QString::fromUtf8(error.what())), kStatusLong);
        }
    }
    _pointCollection->replaceAnchoredCollections(stored);
}

void SegmentationModule::saveCorrections()
{
    if (_correctionsSaveTimer && _correctionsSaveTimer->isActive()) {
        _correctionsSaveTimer->stop();
    }
    if (!_pointCollection || _correctionsSurfaceId.empty() || !_storeEdit || _storeEdit->uuid() != _correctionsSurfaceId ||
        !_state || !_state->annotationVolume()) {
        return;
    }
    try {
        vc3d::corrections::write(*_state->objectStoreSession(), _storeEdit->pin().attachment, _correctionsSurfaceId,
                                 _pointCollection->anchoredCollections(), *_state->annotationVolume(), _storeEdit->view());
    } catch (const std::exception& error) {
        qCWarning(lcSegModule) << "Saving correction points failed:" << error.what();
        emit statusMessageRequested(tr("Failed to save correction points: %1").arg(QString::fromUtf8(error.what())), kStatusLong);
    }
}
