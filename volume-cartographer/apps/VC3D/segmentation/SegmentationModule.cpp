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

#include <QCursor>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLoggingCategory>
#include <QPointer>

#include <algorithm>
#include <cmath>
#include <optional>
#include <limits>
#include <utility>
#include <vector>


Q_LOGGING_CATEGORY(lcSegModule, "vc.segmentation.module")

namespace
{
constexpr int kStatusShort = 1500;
constexpr int kStatusMedium = 2000;
constexpr int kStatusLong = 5000;

bool nearlyEqual(float lhs, float rhs);

float averageScale(const cv::Vec2f& scale)
{
    const float sx = std::abs(scale[0]);
    const float sy = std::abs(scale[1]);
    const float avg = 0.5f * (sx + sy);
    return (avg > 1e-4f) ? avg : 1.0f;
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
    bool initialAlphaPushPullEnabled = false;
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
        initialAlphaPushPullEnabled = _widget->alphaPushPullEnabled();
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
    _pushPullTool->setAlphaEnabled(initialAlphaPushPullEnabled);
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
}

void SegmentationModule::setRotationHandleHitTester(std::function<bool(CVolumeViewer*, const cv::Vec3f&)> tester)
{
    _rotationHandleHitTester = std::move(tester);
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
    connect(_widget, &SegmentationWidget::alphaPushPullModeChanged,
            this, &SegmentationModule::setAlphaPushPullEnabled);
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
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
        clearLineDragStroke();
        _lineDrawKeyActive = false;
        clearUndoStack();
    }
    updateCorrectionsWidget();
    refreshOverlay();
    emit editingEnabledChanged(enabled);
}

namespace
{
bool nearlyEqual(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < 1e-4f;
}
}

void SegmentationModule::setDragRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (nearlyEqual(sanitized, _dragRadiusSteps)) {
        return;
    }
    _dragRadiusSteps = sanitized;
    if (_activeFalloff == FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }
    if (_widget) {
        _widget->setDragRadius(_dragRadiusSteps);
    }
}

void SegmentationModule::setDragSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (nearlyEqual(sanitized, _dragSigmaSteps)) {
        return;
    }
    _dragSigmaSteps = sanitized;
    if (_activeFalloff == FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }
    if (_widget) {
        _widget->setDragSigma(_dragSigmaSteps);
    }
}

void SegmentationModule::setLineRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (nearlyEqual(sanitized, _lineRadiusSteps)) {
        return;
    }
    _lineRadiusSteps = sanitized;
    if (_activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Line);
    }
    if (_widget) {
        _widget->setLineRadius(_lineRadiusSteps);
    }
}

void SegmentationModule::setLineSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (nearlyEqual(sanitized, _lineSigmaSteps)) {
        return;
    }
    _lineSigmaSteps = sanitized;
    if (_activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Line);
    }
    if (_widget) {
        _widget->setLineSigma(_lineSigmaSteps);
    }
}

void SegmentationModule::setPushPullRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (nearlyEqual(sanitized, _pushPullRadiusSteps)) {
        return;
    }
    _pushPullRadiusSteps = sanitized;
    if (_activeFalloff == FalloffTool::PushPull) {
        useFalloff(FalloffTool::PushPull);
    }
    if (_widget) {
        _widget->setPushPullRadius(_pushPullRadiusSteps);
    }
}

void SegmentationModule::setPushPullSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (nearlyEqual(sanitized, _pushPullSigmaSteps)) {
        return;
    }
    _pushPullSigmaSteps = sanitized;
    if (_activeFalloff == FalloffTool::PushPull) {
        useFalloff(FalloffTool::PushPull);
    }
    if (_widget) {
        _widget->setPushPullSigma(_pushPullSigmaSteps);
    }
}

float SegmentationModule::falloffRadius(FalloffTool tool) const
{
    switch (tool) {
    case FalloffTool::Drag:
        return _dragRadiusSteps;
    case FalloffTool::Line:
        return _lineRadiusSteps;
    case FalloffTool::PushPull:
        return _pushPullRadiusSteps;
    }
    return _dragRadiusSteps;
}

float SegmentationModule::falloffSigma(FalloffTool tool) const
{
    switch (tool) {
    case FalloffTool::Drag:
        return _dragSigmaSteps;
    case FalloffTool::Line:
        return _lineSigmaSteps;
    case FalloffTool::PushPull:
        return _pushPullSigmaSteps;
    }
    return _dragSigmaSteps;
}

void SegmentationModule::updateOverlayFalloff(FalloffTool)
{
    refreshOverlay();
}

void SegmentationModule::useFalloff(FalloffTool tool)
{
    _activeFalloff = tool;
    const float radius = falloffRadius(tool);
    const float sigma = falloffSigma(tool);
    if (_editManager) {
        _editManager->setRadius(radius);
        _editManager->setSigma(sigma);
    }
    updateOverlayFalloff(tool);
}

void SegmentationModule::setPushPullStepMultiplier(float multiplier)
{
    const float sanitized = std::clamp(multiplier, 0.05f, 10.0f);
    if (_pushPullTool && std::fabs(sanitized - _pushPullTool->stepMultiplier()) < 1e-4f) {
        if (_widget && std::fabs(_widget->pushPullStep() - sanitized) >= 1e-4f) {
            _widget->setPushPullStep(sanitized);
        }
        return;
    }
    if (_pushPullTool) {
        _pushPullTool->setStepMultiplier(sanitized);
    }
    if (_widget && std::fabs(_widget->pushPullStep() - sanitized) >= 1e-4f) {
        _widget->setPushPullStep(sanitized);
    }
}

void SegmentationModule::setSmoothingStrength(float strength)
{
    const float clamped = std::clamp(strength, 0.0f, 1.0f);
    if (std::fabs(clamped - _smoothStrength) < 1e-4f) {
        return;
    }
    _smoothStrength = clamped;
    if (_widget) {
        _widget->setSmoothingStrength(_smoothStrength);
    }
    if (_lineTool) {
        _lineTool->setSmoothing(_smoothStrength, _smoothIterations);
    }
}

void SegmentationModule::setSmoothingIterations(int iterations)
{
    const int clamped = std::clamp(iterations, 1, 25);
    if (_smoothIterations == clamped) {
        return;
    }
    _smoothIterations = clamped;
    if (_widget) {
        _widget->setSmoothingIterations(_smoothIterations);
    }
    if (_lineTool) {
        _lineTool->setSmoothing(_smoothStrength, _smoothIterations);
    }
}

void SegmentationModule::setAlphaPushPullEnabled(bool enabled)
{
    if (_pushPullTool && _pushPullTool->alphaEnabled() == enabled) {
        if (_widget && _widget->alphaPushPullEnabled() != enabled) {
            _widget->setAlphaPushPullEnabled(enabled);
        }
        return;
    }
    if (_pushPullTool) {
        _pushPullTool->setAlphaEnabled(enabled);
    }
    if (_widget && _widget->alphaPushPullEnabled() != enabled) {
        _widget->setAlphaPushPullEnabled(enabled);
    }
}

void SegmentationModule::setAlphaPushPullConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = SegmentationPushPullTool::sanitizeConfig(config);
    if (_pushPullTool && SegmentationPushPullTool::configsEqual(_pushPullTool->alphaConfig(), sanitized)) {
        if (_widget) {
            _widget->setAlphaPushPullConfig(sanitized);
        }
        return;
    }
    if (_pushPullTool) {
        _pushPullTool->setAlphaConfig(sanitized);
    }
    if (_widget) {
        _widget->setAlphaPushPullConfig(sanitized);
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
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
                setInvalidationBrushActive(false);
                clearInvalidationBrush();
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

    if (event->key() == Qt::Key_Escape) {
        if (_drag.active) {
            cancelDrag();
            return true;
        }
    }

    if ((event->key() == Qt::Key_F || event->key() == Qt::Key_G) && event->modifiers() == Qt::NoModifier) {
        if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
            return false;
        }

        const int direction = (event->key() == Qt::Key_G) ? 1 : -1;
        if (startPushPull(direction)) {
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
        // Directional growth shortcuts (1-5).
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

std::optional<std::vector<SegmentationGrowthDirection>> SegmentationModule::takeShortcutDirectionOverride()
{
    if (!_pendingShortcutDirections) {
        return std::nullopt;
    }
    auto result = std::move(*_pendingShortcutDirections);
    _pendingShortcutDirections.reset();
    return result;
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

    if ((event->key() == Qt::Key_F || event->key() == Qt::Key_G) && event->modifiers() == Qt::NoModifier) {
        const int direction = (event->key() == Qt::Key_G) ? 1 : -1;
        stopPushPull(direction);
        event->accept();
        return true;
    }

    return false;
}

void SegmentationModule::markNextEditsFromGrowth()
{
    if (_editManager) {
        _editManager->markNextEditsAsGrowth();
    }
}

void SegmentationModule::setGrowthInProgress(bool running)
{
    _growthInProgress = running;
    if (_widget) {
        _widget->setGrowthInProgress(running);
    }
    if (_corrections) {
        _corrections->setGrowthInProgress(running);
    }
    if (running) {
        setCorrectionsAnnotateMode(false, false);
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
        clearLineDragStroke();
        _lineDrawKeyActive = false;
    }
    updateCorrectionsWidget();
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
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
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
            setInvalidationBrushActive(false);
            clearInvalidationBrush();
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
                                                    int steps)
{
    qCInfo(lcSegModule) << "Grow request" << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(direction)
                        << "steps" << steps;

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

    _growthMethod = method;
    _growthSteps = std::max(1, steps);
    markNextEditsFromGrowth();
    emit growSurfaceRequested(method, direction, _growthSteps);
}

void SegmentationModule::setInvalidationBrushActive(bool active)
{
    if (!_brushTool) {
        return;
    }

    const bool shouldEnable = active && _editingEnabled && !_growthInProgress && !(_corrections && _corrections->annotateMode()) &&
                              _editManager && _editManager->hasSession();

    if (!shouldEnable) {
        if (_brushTool->brushActive()) {
            _brushTool->setActive(false);
        }
        _brushTool->clear();
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

void SegmentationModule::clearLineDragStroke()
{
    if (_lineTool) {
        _lineTool->clear();
    }
    if (!_lineDrawKeyActive && _activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Drag);
    }
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
            setInvalidationBrushActive(false);
            clearInvalidationBrush();
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
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
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
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
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

bool SegmentationModule::startPushPull(int direction)
{
    return _pushPullTool ? _pushPullTool->start(direction) : false;
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



