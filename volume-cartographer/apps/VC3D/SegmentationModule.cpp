#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "SegmentationWidget.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Slicing.hpp"

#include <QColor>
#include <QCursor>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLoggingCategory>
#include <QPointer>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <optional>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

Q_LOGGING_CATEGORY(lcSegModule, "vc.segmentation.module")

namespace
{
constexpr int kStatusShort = 1500;
constexpr int kStatusMedium = 2000;
constexpr int kStatusLong = 5000;
constexpr int kMaxUndoStates = 5;
constexpr float kBrushSampleSpacing = 2.0f;

constexpr float kAlphaMinStep = 0.05f;
constexpr float kAlphaMaxStep = 20.0f;
constexpr float kAlphaMinRange = 0.01f;
constexpr float kAlphaDefaultHighDelta = 0.05f;
constexpr float kAlphaBorderLimit = 20.0f;
constexpr int kAlphaBlurRadiusMax = 15;

bool nearlyEqual(float lhs, float rhs);

float averageScale(const cv::Vec2f& scale)
{
    const float sx = std::abs(scale[0]);
    const float sy = std::abs(scale[1]);
    const float avg = 0.5f * (sx + sy);
    return (avg > 1e-4f) ? avg : 1.0f;
}

AlphaPushPullConfig sanitizeAlphaConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = config;

    sanitized.start = std::clamp(sanitized.start, -128.0f, 128.0f);
    sanitized.stop = std::clamp(sanitized.stop, -128.0f, 128.0f);
    if (sanitized.start > sanitized.stop) {
        std::swap(sanitized.start, sanitized.stop);
    }

    const float magnitude = std::clamp(std::fabs(sanitized.step), kAlphaMinStep, kAlphaMaxStep);
    sanitized.step = (sanitized.step < 0.0f) ? -magnitude : magnitude;

    sanitized.low = std::clamp(sanitized.low, 0.0f, 1.0f);
    sanitized.high = std::clamp(sanitized.high, 0.0f, 1.0f);
    if (sanitized.high <= sanitized.low + kAlphaMinRange) {
        sanitized.high = std::min(1.0f, sanitized.low + kAlphaDefaultHighDelta);
    }

    sanitized.borderOffset = std::clamp(sanitized.borderOffset, -kAlphaBorderLimit, kAlphaBorderLimit);
    sanitized.blurRadius = std::clamp(sanitized.blurRadius, 0, kAlphaBlurRadiusMax);

    return sanitized;
}

bool alphaConfigsEqual(const AlphaPushPullConfig& lhs, const AlphaPushPullConfig& rhs)
{
    return nearlyEqual(lhs.start, rhs.start) &&
           nearlyEqual(lhs.stop, rhs.stop) &&
           nearlyEqual(lhs.step, rhs.step) &&
           nearlyEqual(lhs.low, rhs.low) &&
           nearlyEqual(lhs.high, rhs.high) &&
           nearlyEqual(lhs.borderOffset, rhs.borderOffset) &&
           lhs.blurRadius == rhs.blurRadius &&
           lhs.perVertex == rhs.perVertex;
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
    if (_widget) {
        _dragRadiusSteps = _widget->dragRadius();
        _dragSigmaSteps = _widget->dragSigma();
        _lineRadiusSteps = _widget->lineRadius();
        _lineSigmaSteps = _widget->lineSigma();
        _pushPullRadiusSteps = _widget->pushPullRadius();
        _pushPullSigmaSteps = _widget->pushPullSigma();
        _pushPullStepMultiplier = std::clamp(_widget->pushPullStep(), 0.05f, 10.0f);
        _smoothStrength = std::clamp(_widget->smoothingStrength(), 0.0f, 1.0f);
        _smoothIterations = std::clamp(_widget->smoothingIterations(), 1, 25);
        _alphaPushPullEnabled = _widget->alphaPushPullEnabled();
        _alphaPushPullConfig = sanitizeAlphaConfig(_widget->alphaPushPullConfig());
    }

    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
    }

    useFalloff(FalloffTool::Drag);

    _pushPullTimer = new QTimer(this);
    _pushPullTimer->setInterval(30);
    connect(_pushPullTimer, &QTimer::timeout,
            this, &SegmentationModule::onPushPullTick);

    bindWidgetSignals();

    if (_viewerManager) {
        _viewerManager->setSegmentationModule(this);
    }

    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationModule::onSurfaceCollectionChanged);
    }

    if (_pointCollection) {
        const auto& collections = _pointCollection->getAllCollections();
        for (const auto& entry : collections) {
            _pendingCorrectionIds.push_back(entry.first);
        }

        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t id) {
            const uint64_t sentinel = std::numeric_limits<uint64_t>::max();
            if (id == sentinel) {
                _pendingCorrectionIds.clear();
                _managedCorrectionIds.clear();
                _activeCorrectionId = 0;
                setCorrectionsAnnotateMode(false, false);
            } else {
                auto eraseIt = std::remove(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), id);
                if (eraseIt != _pendingCorrectionIds.end()) {
                    _pendingCorrectionIds.erase(eraseIt, _pendingCorrectionIds.end());
                    _managedCorrectionIds.erase(id);
                    if (_activeCorrectionId == id) {
                        _activeCorrectionId = 0;
                        setCorrectionsAnnotateMode(false, false);
                    }
                }
            }
            updateCorrectionsWidget();
        });

        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t id) {
            if (id == std::numeric_limits<uint64_t>::max()) {
                return;
            }
            if (std::find(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), id) != _pendingCorrectionIds.end()) {
                updateCorrectionsWidget();
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

void SegmentationModule::updateOverlayFalloff(FalloffTool tool)
{
    if (!_overlay) {
        return;
    }
    _overlay->setGaussianParameters(falloffRadius(tool), falloffSigma(tool), gridStepWorld());
    _overlay->refreshAll();
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
    if (std::fabs(sanitized - _pushPullStepMultiplier) < 1e-4f) {
        return;
    }
    _pushPullStepMultiplier = sanitized;
    if (_widget) {
        _widget->setPushPullStep(_pushPullStepMultiplier);
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
}

void SegmentationModule::setAlphaPushPullEnabled(bool enabled)
{
    if (_alphaPushPullEnabled == enabled) {
        return;
    }
    _alphaPushPullEnabled = enabled;
    if (_widget && _widget->alphaPushPullEnabled() != enabled) {
        _widget->setAlphaPushPullEnabled(enabled);
    }
}

void SegmentationModule::setAlphaPushPullConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = sanitizeAlphaConfig(config);
    if (alphaConfigsEqual(_alphaPushPullConfig, sanitized)) {
        return;
    }
    _alphaPushPullConfig = sanitized;
    if (_widget) {
        _widget->setAlphaPushPullConfig(_alphaPushPullConfig);
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
    if (_overlay) {
        _overlay->setActiveVertex(std::nullopt);
        _overlay->setTouchedVertices({});
        _overlay->refreshAll();
    }
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

    UndoState state;
    state.points = previewPoints.clone();
    if (state.points.empty()) {
        return false;
    }

    if (_undoStack.size() >= static_cast<std::size_t>(kMaxUndoStates)) {
        _undoStack.pop_front();
    }
    _undoStack.push_back(std::move(state));
    return true;
}

void SegmentationModule::discardLastUndoSnapshot()
{
    if (!_undoStack.empty()) {
        _undoStack.pop_back();
    }
}

bool SegmentationModule::restoreUndoSnapshot()
{
    if (_undoStack.empty()) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    UndoState state = std::move(_undoStack.back());
    _undoStack.pop_back();
    if (state.points.empty()) {
        return false;
    }

    _suppressUndoCapture = true;
    bool applied = _editManager->setPreviewPoints(state.points, false);
    if (applied) {
        _editManager->applyPreview();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        clearInvalidationBrush();
        refreshOverlay();
        emitPendingChanges();
        _pushPullUndoCaptured = false;
    } else {
        _undoStack.push_back(std::move(state));
    }
    _suppressUndoCapture = false;

    return applied;
}

void SegmentationModule::clearUndoStack()
{
    _undoStack.clear();
    _pushPullUndoCaptured = false;
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
            if (_invalidationBrushActive) {
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
            if (applyInvalidationBrush()) {
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
        if (_lineStrokeActive) {
            finishLineDragStroke();
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
    if (running) {
        setCorrectionsAnnotateMode(false, false);
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
        clearLineDragStroke();
        _lineDrawKeyActive = false;
    }
    updateCorrectionsWidget();
}

SegmentationCorrectionsPayload SegmentationModule::buildCorrectionsPayload() const
{
    SegmentationCorrectionsPayload payload;
    if (!_pointCollection) {
        return payload;
    }

    const auto& collections = _pointCollection->getAllCollections();
    for (uint64_t id : _pendingCorrectionIds) {
        auto it = collections.find(id);
        if (it == collections.end()) {
            continue;
        }

        SegmentationCorrectionsPayload::Collection entry;
        entry.id = it->second.id;
        entry.name = it->second.name;
        entry.metadata = it->second.metadata;
        entry.color = it->second.color;

        std::vector<ColPoint> points;
        points.reserve(it->second.points.size());
        for (const auto& pair : it->second.points) {
            points.push_back(pair.second);
        }
        std::sort(points.begin(), points.end(), [](const ColPoint& a, const ColPoint& b) {
            return a.id < b.id;
        });
        if (points.empty()) {
            continue;
        }
        entry.points = std::move(points);
        payload.collections.push_back(std::move(entry));
    }

    return payload;
}

void SegmentationModule::clearPendingCorrections()
{
    setCorrectionsAnnotateMode(false, false);

    if (_pointCollection) {
        for (uint64_t id : _pendingCorrectionIds) {
            if (_managedCorrectionIds.count(id) > 0) {
                _pointCollection->clearCollection(id);
            }
        }
    }

    _pendingCorrectionIds.clear();
    _managedCorrectionIds.clear();
    _activeCorrectionId = 0;
    updateCorrectionsWidget();
}

std::optional<std::pair<int, int>> SegmentationModule::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
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
    if (!_overlay || !_editManager) {
        return;
    }

    std::optional<SegmentationOverlayController::VertexMarker> activeMarker;
    if (_drag.active) {
        if (auto world = _editManager->vertexWorldPosition(_drag.row, _drag.col)) {
            activeMarker = SegmentationOverlayController::VertexMarker{
                .row = _drag.row,
                .col = _drag.col,
                .world = *world,
                .isActive = true,
                .isGrowth = false
            };
        }
    } else if (_hover.valid) {
        activeMarker = SegmentationOverlayController::VertexMarker{
            .row = _hover.row,
            .col = _hover.col,
            .world = _hover.world,
            .isActive = false,
            .isGrowth = false
        };
    }

    std::vector<SegmentationOverlayController::VertexMarker> neighbours;
    if (_editManager && _drag.active) {
        const auto touched = _editManager->recentTouched();
        neighbours.reserve(touched.size());
        for (const auto& key : touched) {
            if (key.row == _drag.row && key.col == _drag.col) {
                continue;
            }
            if (auto world = _editManager->vertexWorldPosition(key.row, key.col)) {
                neighbours.push_back({key.row, key.col, *world, false, false});
            }
        }
    }

    std::vector<cv::Vec3f> maskPoints;
    maskPoints.reserve(_paintOverlayPoints.size() + _currentPaintStroke.size() +
                       _lineStrokeOverlayPoints.size());
    maskPoints.insert(maskPoints.end(), _paintOverlayPoints.begin(), _paintOverlayPoints.end());
    maskPoints.insert(maskPoints.end(), _currentPaintStroke.begin(), _currentPaintStroke.end());
    maskPoints.insert(maskPoints.end(), _lineStrokeOverlayPoints.begin(), _lineStrokeOverlayPoints.end());

    const bool maskVisible = !maskPoints.empty();
    const bool hasLineStroke = !_lineStrokeOverlayPoints.empty();

    FalloffTool overlayTool = _activeFalloff;
    if (hasLineStroke) {
        overlayTool = FalloffTool::Line;
    } else if (!_paintOverlayPoints.empty() || !_currentPaintStroke.empty() ||
               _paintStrokeActive || _invalidationBrushActive) {
        overlayTool = FalloffTool::Drag;
    } else if (_pushPull.active) {
        overlayTool = FalloffTool::PushPull;
    }

    const float overlayRadiusSteps = falloffRadius(overlayTool);
    const float brushPixelRadius = std::clamp(overlayRadiusSteps * 1.5f, 3.0f, 18.0f);
    const bool drawingOverlay = _invalidationBrushActive || _paintStrokeActive || _lineStrokeActive ||
                                hasLineStroke;
    const float brushOpacity = drawingOverlay ? 0.6f : 0.45f;
    const auto renderMode = hasLineStroke ? ViewerOverlayControllerBase::PathRenderMode::LineStrip
                                          : ViewerOverlayControllerBase::PathRenderMode::Points;
    const float lineWidth = hasLineStroke ? 3.0f : std::max(brushPixelRadius * 0.5f, 2.0f);
    const float pointRadius = hasLineStroke ? std::max(brushPixelRadius * 0.35f, 2.0f) : brushPixelRadius;
    const QColor overlayColor = hasLineStroke ? QColor(80, 170, 255)
                                              : QColor(255, 140, 0);
    const float overlayOpacity = hasLineStroke ? 0.85f : brushOpacity;

    _overlay->setActiveVertex(activeMarker);
    _overlay->setTouchedVertices(neighbours);
    if (maskVisible) {
        _overlay->setMaskOverlay(maskPoints,
                                 true,
                                 pointRadius,
                                 overlayOpacity,
                                 renderMode,
                                 lineWidth,
                                 overlayColor);
    } else {
        _overlay->setMaskOverlay({},
                                 false,
                                 0.0f,
                                 0.0f,
                                 ViewerOverlayControllerBase::PathRenderMode::Points,
                                 0.0f,
                                 QColor());
    }
    _overlay->setGaussianParameters(falloffRadius(_activeFalloff), falloffSigma(_activeFalloff), gridStepWorld());
    _overlay->refreshAll();
}

void SegmentationModule::refreshMaskOverlay()
{
    if (_overlay) {
        _overlay->setMaskOverlay({},
                                 false,
                                 0.0f,
                                 0.0f,
                                 ViewerOverlayControllerBase::PathRenderMode::Points,
                                 0.0f,
                                 QColor());
    }
}

void SegmentationModule::updateCorrectionsWidget()
{
    if (!_widget) {
        return;
    }

    pruneMissingCorrections();

    const bool correctionsAvailable = (_pointCollection != nullptr) && !_growthInProgress;
    QVector<QPair<uint64_t, QString>> entries;
    if (_pointCollection) {
        const auto& collections = _pointCollection->getAllCollections();
        entries.reserve(static_cast<int>(_pendingCorrectionIds.size()));
        for (uint64_t id : _pendingCorrectionIds) {
            auto it = collections.find(id);
            if (it != collections.end()) {
                entries.append({id, QString::fromStdString(it->second.name)});
            }
        }
    }

    std::optional<uint64_t> active;
    if (_activeCorrectionId != 0) {
        active = _activeCorrectionId;
    }

    _widget->setCorrectionCollections(entries, active);
    _widget->setCorrectionsEnabled(correctionsAvailable);
    _widget->setCorrectionsAnnotateChecked(_correctionsAnnotateMode && correctionsAvailable);
}

void SegmentationModule::setCorrectionsAnnotateMode(bool enabled, bool userInitiated)
{
    if (!_pointCollection || _growthInProgress || !_editingEnabled) {
        enabled = false;
    }

    if (enabled && _activeCorrectionId == 0) {
        if (!createCorrectionCollection(false)) {
            enabled = false;
        }
    }

    if (_correctionsAnnotateMode == enabled) {
        updateCorrectionsWidget();
        return;
    }

    _correctionsAnnotateMode = enabled;
    if (_widget) {
        _widget->setCorrectionsAnnotateChecked(enabled);
    }

    if (enabled) {
        setInvalidationBrushActive(false);
        clearInvalidationBrush();
    }

    if (userInitiated) {
        const QString message = enabled ? tr("Correction annotation enabled")
                                        : tr("Correction annotation disabled");
        emit statusMessageRequested(message, kStatusShort);
    }

    updateCorrectionsWidget();
}

void SegmentationModule::setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated)
{
    if (!_pointCollection) {
        return;
    }

    if (collectionId == 0) {
        _activeCorrectionId = 0;
        setCorrectionsAnnotateMode(false, false);
        updateCorrectionsWidget();
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    if (collections.find(collectionId) == collections.end()) {
        pruneMissingCorrections();
        emit statusMessageRequested(tr("Selected correction set no longer exists."), kStatusShort);
        updateCorrectionsWidget();
        return;
    }

    if (std::find(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), collectionId) == _pendingCorrectionIds.end()) {
        _pendingCorrectionIds.push_back(collectionId);
    }

    _activeCorrectionId = collectionId;

    if (userInitiated) {
        emit statusMessageRequested(tr("Active correction set changed."), kStatusShort);
    }

    updateCorrectionsWidget();
}

uint64_t SegmentationModule::createCorrectionCollection(bool announce)
{
    if (!_pointCollection) {
        return 0;
    }

    const std::string newName = _pointCollection->generateNewCollectionName("correction");
    const uint64_t newId = _pointCollection->addCollection(newName);
    if (newId == 0) {
        return 0;
    }

    if (std::find(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), newId) == _pendingCorrectionIds.end()) {
        _pendingCorrectionIds.push_back(newId);
    }
    _managedCorrectionIds.insert(newId);
    _activeCorrectionId = newId;

    if (announce) {
        emit statusMessageRequested(tr("Created correction set '%1'.").arg(QString::fromStdString(newName)), kStatusShort);
    }

    updateCorrectionsWidget();
    return newId;
}

void SegmentationModule::handleCorrectionPointAdded(const cv::Vec3f& worldPos)
{
    if (!_pointCollection || _activeCorrectionId == 0) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    auto it = collections.find(_activeCorrectionId);
    if (it == collections.end()) {
        pruneMissingCorrections();
        updateCorrectionsWidget();
        return;
    }

    _pointCollection->addPoint(it->second.name, worldPos);
}

void SegmentationModule::handleCorrectionPointRemove(const cv::Vec3f& worldPos)
{
    if (!_pointCollection || _activeCorrectionId == 0) {
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    auto it = collections.find(_activeCorrectionId);
    if (it == collections.end()) {
        pruneMissingCorrections();
        updateCorrectionsWidget();
        return;
    }

    const auto& points = it->second.points;
    if (points.empty()) {
        return;
    }

    uint64_t closestId = 0;
    float closestDistance = std::numeric_limits<float>::max();
    for (const auto& entry : points) {
        const float dist = cv::norm(entry.second.p - worldPos);
        if (dist < closestDistance) {
            closestDistance = dist;
            closestId = entry.second.id;
        }
    }

    if (closestId != 0) {
        _pointCollection->removePoint(closestId);
    }
}

void SegmentationModule::pruneMissingCorrections()
{
    if (!_pointCollection) {
        _pendingCorrectionIds.clear();
        _managedCorrectionIds.clear();
        _activeCorrectionId = 0;
        _correctionsAnnotateMode = false;
        return;
    }

    const auto& collections = _pointCollection->getAllCollections();
    auto endIt = std::remove_if(_pendingCorrectionIds.begin(), _pendingCorrectionIds.end(), [&](uint64_t id) {
        const bool missing = collections.find(id) == collections.end();
        if (missing) {
            _managedCorrectionIds.erase(id);
            if (_activeCorrectionId == id) {
                _activeCorrectionId = 0;
                _correctionsAnnotateMode = false;
            }
        }
        return missing;
    });
    _pendingCorrectionIds.erase(endIt, _pendingCorrectionIds.end());

    if (_activeCorrectionId != 0 && collections.find(_activeCorrectionId) == collections.end()) {
        _activeCorrectionId = 0;
        _correctionsAnnotateMode = false;
    }
}

void SegmentationModule::onCorrectionsCreateRequested()
{
    if (createCorrectionCollection(true) != 0) {
        setCorrectionsAnnotateMode(true, false);
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
    _correctionsZRangeEnabled = enabled;
    if (zMin > zMax) {
        std::swap(zMin, zMax);
    }
    _correctionsZMin = zMin;
    _correctionsZMax = zMax;
    if (enabled) {
        _correctionsRange = std::make_pair(_correctionsZMin, _correctionsZMax);
    } else {
        _correctionsRange.reset();
    }
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
    applyInvalidationBrush();

    _growthMethod = method;
    _growthSteps = std::max(1, steps);
    markNextEditsFromGrowth();
    emit growSurfaceRequested(method, direction, _growthSteps);
}

void SegmentationModule::setInvalidationBrushActive(bool active)
{
    const bool shouldEnable = active && _editingEnabled && !_growthInProgress && !_correctionsAnnotateMode &&
                              _editManager && _editManager->hasSession();
    if (_invalidationBrushActive == shouldEnable) {
        if (_widget) {
            _widget->setEraseBrushActive(shouldEnable);
        }
        return;
    }

    _invalidationBrushActive = shouldEnable;
    if (!_invalidationBrushActive) {
        _hasLastPaintSample = false;
        if (_activeFalloff == FalloffTool::Drag) {
            // Keep drag falloff active by default.
        }
    }
    if (_invalidationBrushActive && _activeFalloff != FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }

    if (_widget) {
        _widget->setEraseBrushActive(_invalidationBrushActive);
    }

    refreshOverlay();
}

void SegmentationModule::clearInvalidationBrush()
{
    _paintStrokeActive = false;
    _currentPaintStroke.clear();
    _pendingPaintStrokes.clear();
    _paintOverlayPoints.clear();
    _hasLastPaintSample = false;

    if (!_invalidationBrushActive && _widget) {
        _widget->setEraseBrushActive(false);
    }

    refreshOverlay();
}

void SegmentationModule::startPaintStroke(const cv::Vec3f& worldPos)
{
    if (_activeFalloff != FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }
    _paintStrokeActive = true;
    _currentPaintStroke.clear();
    _currentPaintStroke.push_back(worldPos);
    _paintOverlayPoints.push_back(worldPos);
    _lastPaintSample = worldPos;
    _hasLastPaintSample = true;
    refreshOverlay();
}

void SegmentationModule::extendPaintStroke(const cv::Vec3f& worldPos, bool forceSample)
{
    if (!_paintStrokeActive) {
        return;
    }

    const float spacing = kBrushSampleSpacing;
    const float spacingSq = spacing * spacing;

    if (_hasLastPaintSample) {
        const cv::Vec3f delta = worldPos - _lastPaintSample;
        const float distanceSq = delta.dot(delta);
        if (!forceSample && distanceSq < spacingSq) {
            return;
        }

        const float distance = std::sqrt(distanceSq);
        if (distance > spacing) {
            const cv::Vec3f direction = delta / distance;
            float travelled = spacing;
            while (travelled < distance) {
                const cv::Vec3f intermediate = _lastPaintSample + direction * travelled;
                _currentPaintStroke.push_back(intermediate);
                _paintOverlayPoints.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _currentPaintStroke.push_back(worldPos);
    _paintOverlayPoints.push_back(worldPos);
    _lastPaintSample = worldPos;
    _hasLastPaintSample = true;
    refreshOverlay();
}

void SegmentationModule::finishPaintStroke()
{
    if (!_paintStrokeActive) {
        return;
    }

    _paintStrokeActive = false;
    if (!_currentPaintStroke.empty()) {
        _pendingPaintStrokes.push_back(_currentPaintStroke);
    }
    _currentPaintStroke.clear();
    _hasLastPaintSample = false;
    refreshOverlay();
}

void SegmentationModule::startLineDragStroke(const cv::Vec3f& worldPos)
{
    useFalloff(FalloffTool::Line);
    _lineStrokeActive = true;
    _lineStrokePoints.clear();
    _lineStrokeOverlayPoints.clear();
    _lineStrokePoints.push_back(worldPos);
    _lineStrokeOverlayPoints.push_back(worldPos);
    _lastLineSample = worldPos;
    _hasLastLineSample = true;
    refreshOverlay();
}

void SegmentationModule::extendLineDragStroke(const cv::Vec3f& worldPos, bool forceSample)
{
    if (!_lineStrokeActive) {
        return;
    }

    const float spacing = kBrushSampleSpacing;
    const float spacingSq = spacing * spacing;

    if (_hasLastLineSample) {
        const cv::Vec3f delta = worldPos - _lastLineSample;
        const float distanceSq = delta.dot(delta);
        if (!forceSample && distanceSq < spacingSq) {
            return;
        }

        const float distance = std::sqrt(distanceSq);
        if (distance > spacing) {
            const cv::Vec3f direction = delta / distance;
            float travelled = spacing;
            while (travelled < distance) {
                const cv::Vec3f intermediate = _lastLineSample + direction * travelled;
                _lineStrokePoints.push_back(intermediate);
                _lineStrokeOverlayPoints.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _lineStrokePoints.push_back(worldPos);
    _lineStrokeOverlayPoints.push_back(worldPos);
    _lastLineSample = worldPos;
    _hasLastLineSample = true;
    refreshOverlay();
}

void SegmentationModule::finishLineDragStroke()
{
    if (!_lineStrokeActive) {
        return;
    }

    _lineStrokeActive = false;
    const std::vector<cv::Vec3f> strokeCopy = _lineStrokePoints;
    applyLineDragStroke(strokeCopy);
    clearLineDragStroke();
    if (!_lineDrawKeyActive) {
        useFalloff(FalloffTool::Drag);
    }
}

bool SegmentationModule::applyLineDragStroke(const std::vector<cv::Vec3f>& stroke)
{
    useFalloff(FalloffTool::Line);
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }
    if (stroke.size() < 2) {
        return false;
    }

    using GridKey = SegmentationEditManager::GridKey;
    using GridKeyHash = SegmentationEditManager::GridKeyHash;

    std::unordered_set<GridKey, GridKeyHash> visited;
    visited.reserve(stroke.size());

    bool snapshotCaptured = false;
    bool anyMoved = false;

    for (const auto& world : stroke) {
        auto gridIndex = _editManager->worldToGridIndex(world);
        if (!gridIndex) {
            continue;
        }

        GridKey key{gridIndex->first, gridIndex->second};
        if (!visited.insert(key).second) {
            continue;
        }

        if (!_editManager->beginActiveDrag(*gridIndex)) {
            continue;
        }

        bool capturedThisSample = false;
        if (!snapshotCaptured) {
            snapshotCaptured = captureUndoSnapshot();
            capturedThisSample = snapshotCaptured;
        }

        if (!_editManager->updateActiveDrag(world)) {
            _editManager->cancelActiveDrag();
            if (capturedThisSample) {
                discardLastUndoSnapshot();
                snapshotCaptured = false;
            }
            continue;
        }

        if (_smoothStrength > 0.0f && _smoothIterations > 0) {
            _editManager->smoothRecentTouched(_smoothStrength, _smoothIterations);
        }

        _editManager->commitActiveDrag();
        anyMoved = true;
    }

    if (!anyMoved) {
        if (snapshotCaptured) {
            discardLastUndoSnapshot();
        }
        return false;
    }

    _editManager->applyPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    refreshOverlay();
    emitPendingChanges();
    emit statusMessageRequested(tr("Applied segmentation drag along path."), kStatusShort);
    return true;
}

void SegmentationModule::clearLineDragStroke()
{
    _lineStrokeActive = false;
    _lineStrokePoints.clear();
    _lineStrokeOverlayPoints.clear();
    _hasLastLineSample = false;
    refreshOverlay();
    if (!_lineDrawKeyActive && _activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Drag);
    }
}

bool SegmentationModule::applyInvalidationBrush()
{
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (_paintStrokeActive) {
        finishPaintStroke();
    }

    if (_pendingPaintStrokes.empty()) {
        return false;
    }

    using GridKey = SegmentationEditManager::GridKey;
    using GridKeyHash = SegmentationEditManager::GridKeyHash;

    std::unordered_set<GridKey, GridKeyHash> targets;
    std::size_t estimate = 0;
    for (const auto& stroke : _pendingPaintStrokes) {
        estimate += stroke.size();
    }
    targets.reserve(estimate);

    const float stepWorld = gridStepWorld();
    const float dragRadius = std::max(_dragRadiusSteps, 0.5f);
    const float maxDistance = stepWorld * std::max(dragRadius * 6.0f, 15.0f);

    for (const auto& stroke : _pendingPaintStrokes) {
        for (const auto& world : stroke) {
            float gridDistance = 0.0f;
            auto grid = _editManager->worldToGridIndex(world, &gridDistance);
            if (!grid) {
                continue;
            }
            if (maxDistance > 0.0f && gridDistance > maxDistance) {
                continue;
            }
            targets.insert(GridKey{grid->first, grid->second});
        }
    }

    if (targets.empty()) {
        clearInvalidationBrush();
        return false;
    }

    bool snapshotCaptured = captureUndoSnapshot();
    const float brushRadiusSteps = dragRadius;
    bool anyChanged = false;
    for (const auto& key : targets) {
        if (_editManager->markInvalidRegion(key.row, key.col, brushRadiusSteps)) {
            anyChanged = true;
        }
    }

    clearInvalidationBrush();

    if (!anyChanged) {
        if (snapshotCaptured) {
            discardLastUndoSnapshot();
        }
        return false;
    }

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    emitPendingChanges();
    emit statusMessageRequested(tr("Invalidated %1 brush target(s).").arg(static_cast<int>(targets.size())),
                                kStatusMedium);
    return true;
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

    if (_correctionsAnnotateMode) {
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

    if (_lineDrawKeyActive) {
        stopAllPushPull();
        if (_invalidationBrushActive) {
            setInvalidationBrushActive(false);
            clearInvalidationBrush();
        }
        if (_drag.active) {
            cancelDrag();
        }
        startLineDragStroke(worldPos);
        return;
    }

    if (_invalidationBrushActive) {
        stopAllPushPull();
        startPaintStroke(worldPos);
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

    if (_lineStrokeActive) {
        if (buttons.testFlag(Qt::LeftButton)) {
            extendLineDragStroke(worldPos);
        } else {
            finishLineDragStroke();
        }
        return;
    }

    if (_paintStrokeActive) {
        if (buttons.testFlag(Qt::LeftButton)) {
            extendPaintStroke(worldPos);
        } else {
            finishPaintStroke();
        }
        return;
    }

    if (_drag.active) {
        updateDrag(worldPos);
        return;
    }

    if (_correctionsAnnotateMode) {
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
    if (_lineStrokeActive && button == Qt::LeftButton) {
        extendLineDragStroke(worldPos, true);
        finishLineDragStroke();
        return;
    }

    if (_paintStrokeActive && button == Qt::LeftButton) {
        extendPaintStroke(worldPos, true);
        finishPaintStroke();
        return;
    }

    if (!_drag.active || button != Qt::LeftButton) {
        if (_correctionsAnnotateMode && button == Qt::LeftButton) {
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
    if (_lineDrawKeyActive || _lineStrokeActive) {
        targetTool = FalloffTool::Line;
    } else if (_pushPull.active) {
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
    if (!_editManager || !_editManager->hasSession()) {
        _hover.clear();
        if (_overlay) {
            _overlay->setActiveVertex(std::nullopt);
            _overlay->refreshViewer(viewer);
        }
        return;
    }

    auto gridIndex = _editManager->worldToGridIndex(worldPos);
    if (!gridIndex) {
        _hover.clear();
        if (_overlay) {
            _overlay->setActiveVertex(std::nullopt);
            _overlay->refreshViewer(viewer);
        }
        return;
    }

    if (auto world = _editManager->vertexWorldPosition(gridIndex->first, gridIndex->second)) {
        _hover.set(gridIndex->first, gridIndex->second, *world, viewer);
        if (_overlay) {
            _overlay->setActiveVertex(SegmentationOverlayController::VertexMarker{
                .row = _hover.row,
                .col = _hover.col,
                .world = *world,
                .isActive = false,
                .isGrowth = false
            });
            _overlay->refreshViewer(viewer);
        }
    }
}

bool SegmentationModule::startPushPull(int direction)
{
    if (direction == 0) {
        return false;
    }

    if (_pushPull.active && _pushPull.direction == direction) {
        if (_pushPullTimer && !_pushPullTimer->isActive()) {
            _pushPullTimer->start();
        }
        return true;
    }

    if (!_hover.valid || !_hover.viewer || !isSegmentationViewer(_hover.viewer)) {
        return false;
    }
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    _pushPull.active = true;
    _pushPull.direction = direction;
    _pushPullUndoCaptured = false;
    useFalloff(FalloffTool::PushPull);

    if (_pushPullTimer && !_pushPullTimer->isActive()) {
        _pushPullTimer->start();
    }

    if (!applyPushPullStep()) {
        stopAllPushPull();
        return false;
    }

    return true;
}

void SegmentationModule::stopPushPull(int direction)
{
    if (!_pushPull.active) {
        return;
    }
    if (direction != 0 && direction != _pushPull.direction) {
        return;
    }
    stopAllPushPull();
}

void SegmentationModule::stopAllPushPull()
{
    _pushPull.active = false;
    _pushPull.direction = 0;
    if (_pushPullTimer && _pushPullTimer->isActive()) {
        _pushPullTimer->stop();
    }
    _pushPullUndoCaptured = false;
    if (_activeFalloff == FalloffTool::PushPull) {
        useFalloff(FalloffTool::Drag);
    }
}

bool SegmentationModule::applyPushPullStep()
{
    if (!_pushPull.active || !_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (!_hover.valid || !_hover.viewer || !isSegmentationViewer(_hover.viewer)) {
        return false;
    }

    const int row = _hover.row;
    const int col = _hover.col;

    bool snapshotCapturedThisStep = false;
    if (!_pushPullUndoCaptured) {
        snapshotCapturedThisStep = captureUndoSnapshot();
        if (snapshotCapturedThisStep) {
            _pushPullUndoCaptured = true;
        }
    }

    if (!_editManager->beginActiveDrag({row, col})) {
        if (snapshotCapturedThisStep) {
            discardLastUndoSnapshot();
            _pushPullUndoCaptured = false;
        }
        return false;
    }

    auto centerWorldOpt = _editManager->vertexWorldPosition(row, col);
    if (!centerWorldOpt) {
        _editManager->cancelActiveDrag();
        if (snapshotCapturedThisStep) {
            discardLastUndoSnapshot();
            _pushPullUndoCaptured = false;
        }
        return false;
    }
    const cv::Vec3f centerWorld = *centerWorldOpt;

    QuadSurface* baseSurface = _editManager->baseSurface();
    if (!baseSurface) {
        _editManager->cancelActiveDrag();
        return false;
    }

    cv::Vec3f ptr = baseSurface->pointer();
    baseSurface->pointTo(ptr, centerWorld, std::numeric_limits<float>::max(), 400);
    cv::Vec3f normal = baseSurface->normal(ptr);
    if (std::isnan(normal[0]) || std::isnan(normal[1]) || std::isnan(normal[2])) {
        _editManager->cancelActiveDrag();
        return false;
    }

    const float norm = cv::norm(normal);
    if (norm <= 1e-4f) {
        _editManager->cancelActiveDrag();
        return false;
    }
    normal /= norm;

    cv::Vec3f targetWorld = centerWorld;
    bool usedAlphaPushPull = false;
    bool alphaUnavailable = false;

    if (_alphaPushPullEnabled) {
        if (_alphaPushPullConfig.perVertex) {
            const auto& drag = _editManager->activeDrag();
            const auto& samples = drag.samples;
            if (samples.empty()) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    discardLastUndoSnapshot();
                    _pushPullUndoCaptured = false;
                }
                return false;
            }

            std::vector<cv::Vec3f> perVertexTargets;
            perVertexTargets.reserve(samples.size());
            bool anyMovement = false;

            for (const auto& sample : samples) {
                cv::Vec3f sampleNormal = normal;
                cv::Vec3f samplePtr = baseSurface->pointer();
                baseSurface->pointTo(samplePtr, sample.baseWorld, std::numeric_limits<float>::max(), 400);
                cv::Vec3f candidateNormal = baseSurface->normal(samplePtr);
                if (std::isfinite(candidateNormal[0]) &&
                    std::isfinite(candidateNormal[1]) &&
                    std::isfinite(candidateNormal[2])) {
                    const float candidateNorm = cv::norm(candidateNormal);
                    if (candidateNorm > 1e-4f) {
                        sampleNormal = candidateNormal / candidateNorm;
                    }
                }

                bool sampleUnavailable = false;
                auto sampleTarget = computeAlphaPushPullTarget(sample.baseWorld,
                                                               sampleNormal,
                                                               _pushPull.direction,
                                                               baseSurface,
                                                               _hover.viewer,
                                                               &sampleUnavailable);
                if (sampleUnavailable) {
                    alphaUnavailable = true;
                    break;
                }

                cv::Vec3f newWorld = sample.baseWorld;
                if (sampleTarget) {
                    newWorld = *sampleTarget;
                    if (cv::norm(newWorld - sample.baseWorld) >= 1e-4f) {
                        anyMovement = true;
                    }
                }

                perVertexTargets.push_back(newWorld);
            }

            if (alphaUnavailable) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    discardLastUndoSnapshot();
                    _pushPullUndoCaptured = false;
                }
                return false;
            }

            if (!anyMovement) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    discardLastUndoSnapshot();
                    _pushPullUndoCaptured = false;
                }
                return false;
            }

            if (!_editManager->updateActiveDragTargets(perVertexTargets)) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    discardLastUndoSnapshot();
                    _pushPullUndoCaptured = false;
                }
                return false;
            }

            usedAlphaPushPull = true;
        } else {
            auto alphaTarget = computeAlphaPushPullTarget(centerWorld,
                                                         normal,
                                                         _pushPull.direction,
                                                         baseSurface,
                                                         _hover.viewer,
                                                         &alphaUnavailable);
            if (alphaTarget) {
                targetWorld = *alphaTarget;
                usedAlphaPushPull = true;
            } else if (!alphaUnavailable) {
                _editManager->cancelActiveDrag();
                if (snapshotCapturedThisStep) {
                    discardLastUndoSnapshot();
                    _pushPullUndoCaptured = false;
                }
                return false;
            }
        }
    }

    if (!usedAlphaPushPull) {
        const float stepWorld = gridStepWorld() * _pushPullStepMultiplier;
        if (stepWorld <= 0.0f) {
            _editManager->cancelActiveDrag();
            return false;
        }
        targetWorld = centerWorld + normal * (static_cast<float>(_pushPull.direction) * stepWorld);
    }

    if (!usedAlphaPushPull && !_editManager->updateActiveDrag(targetWorld)) {
        _editManager->cancelActiveDrag();
        if (snapshotCapturedThisStep) {
            discardLastUndoSnapshot();
            _pushPullUndoCaptured = false;
        }
        return false;
    }

    _editManager->commitActiveDrag();
    _editManager->applyPreview();

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    refreshOverlay();
    emitPendingChanges();
    return true;
}

std::optional<cv::Vec3f> SegmentationModule::computeAlphaPushPullTarget(const cv::Vec3f& centerWorld,
                                                                        const cv::Vec3f& normal,
                                                                        int direction,
                                                                        QuadSurface* surface,
                                                                        CVolumeViewer* viewer,
                                                                        bool* outUnavailable) const
{
    if (outUnavailable) {
        *outUnavailable = false;
    }

    if (!_alphaPushPullEnabled || !viewer || !surface) {
        return std::nullopt;
    }

    std::shared_ptr<Volume> volume = viewer->currentVolume();
    if (!volume) {
        if (outUnavailable) {
            *outUnavailable = true;
        }
        return std::nullopt;
    }

    const size_t scaleCount = volume->numScales();
    int datasetIndex = viewer->datasetScaleIndex();
    if (scaleCount == 0) {
        datasetIndex = 0;
    } else {
        datasetIndex = std::clamp(datasetIndex, 0, static_cast<int>(scaleCount) - 1);
    }

    z5::Dataset* dataset = volume->zarrDataset(datasetIndex);
    if (!dataset) {
        dataset = volume->zarrDataset(0);
    }
    if (!dataset) {
        if (outUnavailable) {
            *outUnavailable = true;
        }
        return std::nullopt;
    }

    float scale = viewer->datasetScaleFactor();
    if (!std::isfinite(scale) || scale <= 0.0f) {
        scale = 1.0f;
    }

    ChunkCache* cache = viewer->chunkCachePtr();

    AlphaPushPullConfig cfg = sanitizeAlphaConfig(_alphaPushPullConfig);

    cv::Vec3f orientedNormal = normal * static_cast<float>(direction);
    const float norm = cv::norm(orientedNormal);
    if (norm <= 1e-4f) {
        return std::nullopt;
    }
    orientedNormal /= norm;

    const int radius = std::max(cfg.blurRadius, 0);
    const int kernel = radius * 2 + 1;
    const cv::Size patchSize(kernel, kernel);

    PlaneSurface plane(centerWorld, orientedNormal);
    cv::Mat_<cv::Vec3f> coords;
    plane.gen(&coords, nullptr, patchSize, cv::Vec3f(0, 0, 0), scale, cv::Vec3f(0, 0, 0));
    coords *= scale;

    const cv::Point2i centerIndex(radius, radius);
    const float range = std::max(cfg.high - cfg.low, kAlphaMinRange);

    float transparent = 1.0f;
    float integ = 0.0f;

    const float start = cfg.start;
    const float stop = cfg.stop;
    const float step = std::fabs(cfg.step);

    for (float offset = start; offset <= stop + 1e-4f; offset += step) {
        cv::Mat_<uint8_t> slice;
        cv::Mat_<cv::Vec3f> offsetMat(patchSize, orientedNormal * (offset * scale));
        readInterpolated3D(slice, dataset, coords + offsetMat, cache);
        if (slice.empty()) {
            continue;
        }

        cv::Mat sliceFloat;
        slice.convertTo(sliceFloat, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(sliceFloat, sliceFloat, cv::Size(kernel, kernel), 0);

        cv::Mat_<float> opaq = sliceFloat;
        opaq = (opaq - cfg.low) / range;
        cv::min(opaq, 1.0f, opaq);
        cv::max(opaq, 0.0f, opaq);

        const float centerOpacity = opaq(centerIndex);
        const float joint = transparent * centerOpacity;
        integ += joint * offset;
        transparent -= joint;

        if (transparent <= 1e-3f) {
            break;
        }
    }

    if (transparent >= 1.0f) {
        return std::nullopt;
    }

    const float denom = 1.0f - transparent;
    if (denom < 1e-5f) {
        return std::nullopt;
    }

    const float expected = integ / denom;
    if (!std::isfinite(expected)) {
        return std::nullopt;
    }

    const float totalOffset = expected + cfg.borderOffset;
    if (!std::isfinite(totalOffset) || totalOffset <= 0.0f) {
        return std::nullopt;
    }

    const cv::Vec3f targetWorld = centerWorld + orientedNormal * totalOffset;
    if (!std::isfinite(targetWorld[0]) || !std::isfinite(targetWorld[1]) || !std::isfinite(targetWorld[2])) {
        return std::nullopt;
    }

    const cv::Vec3f delta = targetWorld - centerWorld;
    if (cv::norm(delta) < 1e-4f) {
        return std::nullopt;
    }

    return targetWorld;
}

void SegmentationModule::onPushPullTick()
{
    if (!applyPushPullStep()) {
        stopAllPushPull();
    }
}
