#include "SegmentationModule.hpp"

#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include <QLoggingCategory>
#include <QMessageBox>

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

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    if (_overlay) {
        _overlay->setEditingEnabled(_editingEnabled);
    }

    useFalloff(_activeFalloff);

    if (_overlay) {
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
    clearLineDragStroke();
    setInvalidationBrushActive(false);
    _lineDrawKeyActive = false;
    refreshOverlay();
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
    if (name != "segmentation" || _ignoreSegSurfaceChange) {
        return;
    }

    // If growth is running, prevent switching the active segmentation surface.
    if (_growthInProgress) {
        QuadSurface* previewSurface = _editManager ? _editManager->previewSurface() : nullptr;
        QuadSurface* baseSurface = _editManager ? _editManager->baseSurface() : nullptr;
        QuadSurface* target = previewSurface ? previewSurface : baseSurface;

        if (_surfaces && target && surface != target) {
            const bool previousGuard = _ignoreSegSurfaceChange;
            _ignoreSegSurfaceChange = true;
            _surfaces->setSurface("segmentation", target);
            _ignoreSegSurfaceChange = previousGuard;
            emit statusMessageRequested(tr("Cannot switch active surface while growth is running."),
                                        0);
        }
        return;
    }

    if (!_editingEnabled) {
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
    bool applied = _editManager->setPreviewPoints(points, false);
    if (applied) {
        _editManager->applyPreview();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        clearInvalidationBrush();
        refreshOverlay();
        emitPendingChanges();
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    refreshOverlay();
    emitPendingChanges();
}
