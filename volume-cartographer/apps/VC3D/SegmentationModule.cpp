#include "SegmentationModule.hpp"

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "SegmentationWidget.hpp"
#include "ViewerManager.hpp"

#include "vc/ui/VCCollection.hpp"

#include "vc/core/util/Surface.hpp"

#include <QApplication>
#include <QGraphicsSimpleTextItem>
#include <QKeyEvent>
#include <QColor>
#include <QFont>
#include <QLoggingCategory>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPointer>
#include <QStringList>
#include <QTimer>
#include <QVariant>
#include <QVector>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

Q_LOGGING_CATEGORY(lcSegEdit, "vc.segmentation.edit");

namespace
{
constexpr float kMinRadius = 1.0f;
constexpr int kMaxRadiusSteps = 32;
constexpr int kStatusShort = 1500;
constexpr int kStatusMedium = 2000;
constexpr int kStatusLong = 5000;

QString vecToString(const cv::Vec3f& v)
{
    return QStringLiteral("(%1,%2,%3)")
        .arg(static_cast<double>(v[0]), 0, 'f', 3)
        .arg(static_cast<double>(v[1]), 0, 'f', 3)
        .arg(static_cast<double>(v[2]), 0, 'f', 3);
}

QString modifiersToString(Qt::KeyboardModifiers mods)
{
    QStringList parts;
    if (mods.testFlag(Qt::ShiftModifier)) {
        parts.append(QStringLiteral("Shift"));
    }
    if (mods.testFlag(Qt::ControlModifier)) {
        parts.append(QStringLiteral("Control"));
    }
    if (mods.testFlag(Qt::AltModifier)) {
        parts.append(QStringLiteral("Alt"));
    }
    if (mods.testFlag(Qt::MetaModifier)) {
        parts.append(QStringLiteral("Meta"));
    }
    if (parts.isEmpty()) {
        return QStringLiteral("None");
    }
    return parts.join(QChar('+'));
}

QString axisToString(std::optional<SegmentationRowColAxis> axis)
{
    if (!axis.has_value()) {
        return QStringLiteral("auto");
    }
    switch (*axis) {
    case SegmentationRowColAxis::Row:
        return QStringLiteral("row");
    case SegmentationRowColAxis::Column:
        return QStringLiteral("column");
    case SegmentationRowColAxis::Both:
        return QStringLiteral("both");
    }
    return QStringLiteral("auto");
}

struct ClickLog
{
    explicit ClickLog(QString header)
        : _header(std::move(header))
    {
    }

    ~ClickLog()
    {
        if (_header.isEmpty()) {
            return;
        }
        if (_details.isEmpty()) {
            qCInfo(lcSegEdit).noquote() << _header;
        } else {
            qCInfo(lcSegEdit).noquote() << _header + QStringLiteral(" | ") + _details.join(QStringLiteral(" | "));
        }
    }

    void add(const QString& message)
    {
        _details.append(message);
    }

private:
    QString _header;
    QStringList _details;
};
}

SegmentationModule::SegmentationModule(SegmentationWidget* widget,
                                       SegmentationEditManager* editManager,
                                       SegmentationOverlayController* overlay,
                                       ViewerManager* viewerManager,
                                       CSurfaceCollection* surfaces,
                                       VCCollection* pointCollection,
                                       bool editingEnabled,
                                       int downsample,
                                       float radius,
                                       float sigma,
                                       QObject* parent)
    : QObject(parent)
    , _widget(widget)
    , _editManager(editManager)
    , _overlay(overlay)
    , _viewerManager(viewerManager)
    , _surfaces(surfaces)
    , _pointCollection(pointCollection)
    , _editingEnabled(editingEnabled)
    , _downsample(downsample)
    , _radius(static_cast<float>(std::clamp(static_cast<int>(std::lround(radius)), 1, kMaxRadiusSteps)))
    , _sigma(std::clamp(sigma, 0.10f, 2.0f))
{
    if (_overlay && _editManager) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
        _overlay->setDownsample(_downsample);
        _overlay->setRadius(_radius);
    }

    bindWidgetSignals();

    if (_widget) {
        _holeSearchRadius = _widget->holeSearchRadius();
        _holeSmoothIterations = _widget->holeSmoothIterations();
        _showHandlesAlways = _widget->handlesAlwaysVisible();
        _handleDisplayDistance = _widget->handleDisplayDistance();
        _influenceMode = _widget->influenceMode();
        _rowColMode = _widget->rowColMode();
        _sliceFadeDistance = _widget->sliceFadeDistance();
        _sliceDisplayMode = _widget->sliceDisplayMode();
        _highlightDistance = _widget->highlightDistance();
        _fillInvalidRegions = _widget->fillInvalidRegions();
        _handlesLocked = _widget->handlesLocked();
        _usingCorrectionsGrowth = (_widget->growthMethod() == SegmentationGrowthMethod::Corrections);
    }
    if (_editManager) {
        _editManager->setHoleSearchRadius(_holeSearchRadius);
        _editManager->setHoleSmoothIterations(_holeSmoothIterations);
        _editManager->setInfluenceMode(_influenceMode);
        _editManager->setRowColMode(_rowColMode);
        _editManager->setFillInvalidCells(_fillInvalidRegions);
    }
    if (_overlay) {
        _overlay->setHandleVisibility(_showHandlesAlways, _handleDisplayDistance);
        _overlay->setSliceFadeDistance(_sliceFadeDistance);
        _overlay->setSliceDisplayMode(_sliceDisplayMode);
        _overlay->setCursorWorld(cv::Vec3f(0, 0, 0), false);
    }

    updateHandleVisibility();

    if (_viewerManager) {
        _viewerManager->setSegmentationOverlay(_overlay);
        _viewerManager->setSegmentationModule(this);
    }

    if (_pointCollection) {
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

    setHandlesLocked(_handlesLocked, false);
    updateCorrectionsWidget();
}

void SegmentationModule::bindWidgetSignals()
{
    if (!_widget) {
        return;
    }

    connect(_widget, &SegmentationWidget::editingModeChanged,
            this, &SegmentationModule::setEditingEnabled,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::downsampleChanged,
            this, &SegmentationModule::setDownsample,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::radiusChanged,
            this, &SegmentationModule::setRadius,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::sigmaChanged,
            this, &SegmentationModule::setSigma,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::influenceModeChanged,
            this, &SegmentationModule::setInfluenceMode,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::rowColModeChanged,
            this, &SegmentationModule::setRowColMode,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::highlightDistanceChanged,
            this, &SegmentationModule::setHighlightDistance,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::sliceFadeDistanceChanged,
            this, &SegmentationModule::setSliceFadeDistance,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::sliceDisplayModeChanged,
            this, &SegmentationModule::setSliceDisplayMode,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::holeSearchRadiusChanged,
            this, &SegmentationModule::setHoleSearchRadius,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::holeSmoothIterationsChanged,
            this, &SegmentationModule::setHoleSmoothIterations,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::handlesAlwaysVisibleChanged,
            this, &SegmentationModule::setHandlesAlwaysVisible,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::handleDisplayDistanceChanged,
            this, &SegmentationModule::setHandleDisplayDistance,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::fillInvalidRegionsChanged,
            this, &SegmentationModule::setFillInvalidRegions,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::applyRequested,
            this, &SegmentationModule::applyEdits,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::resetRequested,
            this, &SegmentationModule::resetEdits,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::stopToolsRequested,
            this, &SegmentationModule::stopTools,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::growSurfaceRequested,
            this, &SegmentationModule::handleGrowSurfaceRequested,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::growthMethodChanged,
            this, &SegmentationModule::onGrowthMethodChanged,
            Qt::UniqueConnection);

    connect(_widget, &SegmentationWidget::correctionsCreateRequested,
            this, &SegmentationModule::onCorrectionsCreateRequested,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::correctionsCollectionSelected,
            this, &SegmentationModule::onCorrectionsCollectionSelected,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::correctionsAnnotateToggled,
            this, &SegmentationModule::onCorrectionsAnnotateToggled,
            Qt::UniqueConnection);
    connect(_widget, &SegmentationWidget::correctionsZRangeChanged,
            this, &SegmentationModule::onCorrectionsZRangeChanged,
            Qt::UniqueConnection);

    if (auto zRange = _widget->correctionsZRange()) {
        onCorrectionsZRangeChanged(true, zRange->first, zRange->second);
    } else {
        onCorrectionsZRangeChanged(false, 0, 0);
    }
}

void SegmentationModule::onGrowthMethodChanged(SegmentationGrowthMethod method)
{
    const bool usingCorrections = (method == SegmentationGrowthMethod::Corrections);
    if (_usingCorrectionsGrowth == usingCorrections) {
        return;
    }
    _usingCorrectionsGrowth = usingCorrections;
    updateHandleVisibility();
}

void SegmentationModule::onCorrectionsCreateRequested()
{
    const uint64_t id = createCorrectionCollection(true);
    if (id == 0) {
        emit statusMessageRequested(tr("Unable to create a correction set. Make sure point collections are available."), kStatusShort);
        return;
    }
    setCorrectionsAnnotateMode(true, false);
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
    if (!enabled) {
        _correctionsZRangeEnabled = false;
        return;
    }

    if (zMin > zMax) {
        std::swap(zMin, zMax);
    }

    _correctionsZRangeEnabled = true;
    _correctionsZMin = std::max(0, zMin);
    _correctionsZMax = std::max(_correctionsZMin, zMax);
}

std::optional<std::pair<int, int>> SegmentationModule::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

void SegmentationModule::updateHandleVisibility()
{
    if (!_overlay) {
        return;
    }
    const bool showHandles = _editingEnabled && !_correctionsAnnotateMode && !_usingCorrectionsGrowth;
    _overlay->setHandlesVisible(showHandles);
}

void SegmentationModule::handleGrowSurfaceRequested(SegmentationGrowthMethod method,
                                                    SegmentationGrowthDirection direction,
                                                    int steps)
{
    qCInfo(lcSegEdit) << "Grow request received" << segmentationGrowthMethodToString(method)
                      << segmentationGrowthDirectionToString(direction)
                      << "steps" << steps;
    if (_growthInProgress) {
        qCInfo(lcSegEdit) << "Ignoring grow request because growth is already running";
        emit statusMessageRequested(tr("Surface growth already in progress"), kStatusMedium);
        return;
    }
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        qCInfo(lcSegEdit) << "Rejecting grow request because editing session is unavailable"
                          << "editingEnabled" << _editingEnabled
                          << "hasEditManager" << (_editManager != nullptr)
                          << "hasSession" << (_editManager && _editManager->hasSession());
        emit statusMessageRequested(tr("Enable segmentation editing before growing surfaces"), kStatusMedium);
        return;
    }
    qCInfo(lcSegEdit) << "Forwarding grow request to window";
    emit growSurfaceRequested(method, direction, steps);
}

void SegmentationModule::bindViewerSignals(CVolumeViewer* viewer)
{
    if (!viewer || viewer->property("vc_segmentation_bound").toBool()) {
        return;
    }

    connect(viewer, &CVolumeViewer::sendMousePressVolume,
            this, [this, viewer](cv::Vec3f worldPos, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                handleMousePress(viewer, worldPos, normal, button, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
            this, [this, viewer](cv::Vec3f worldPos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                handleMouseMove(viewer, worldPos, buttons, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
            this, [this, viewer](cv::Vec3f worldPos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                handleMouseRelease(viewer, worldPos, button, modifiers);
            });
    connect(viewer, &CVolumeViewer::sendSegmentationRadiusWheel,
            this, [this, viewer](int steps, QPointF scenePoint, cv::Vec3f worldPos) {
                handleRadiusWheel(viewer, steps, scenePoint, worldPos);
            });

    viewer->setProperty("vc_segmentation_bound", true);
    viewer->setSegmentationEditActive(_editingEnabled);
}

void SegmentationModule::attachViewer(CVolumeViewer* viewer)
{
    bindViewerSignals(viewer);
    updateViewerCursors();
}

void SegmentationModule::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;

    if (!enabled) {
        setCorrectionsAnnotateMode(false, false);
        setHandlesLocked(false, false);
    }
    updateCorrectionsWidget();

    if (_overlay) {
        _overlay->setEditingEnabled(enabled);
        _overlay->setDownsample(_downsample);
        _overlay->setRadius(_radius);
        _overlay->refreshAll();
    }

    updateHandleVisibility();

    if (!enabled) {
        resetInteractionState();
    } else {
        if (_viewerManager) {
            _viewerManager->forEachViewer([](CVolumeViewer* v) {
                if (v) {
                    v->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }
        updateViewerCursors();
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationEditActive(enabled);
    }

    emit editingEnabledChanged(enabled);
}

void SegmentationModule::setDownsample(int value)
{
    if (value == _downsample) {
        return;
    }
    _downsample = value;
    if (_widget && _widget->downsample() != value) {
        _widget->setDownsample(value);
    }
    if (_editManager && _editManager->hasSession()) {
        _editManager->setDownsample(value);
        emitPendingChanges();
    }
    if (_overlay) {
        _overlay->setDownsample(value);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setRadius(float radius)
{
    const int snapped = std::clamp(static_cast<int>(std::lround(radius)), 1, kMaxRadiusSteps);
    const float snappedRadius = static_cast<float>(snapped);
    if (std::fabs(snappedRadius - _radius) < 1e-4f) {
        return;
    }
    _radius = snappedRadius;

    if (_widget && std::fabs(_widget->radius() - _radius) > 1e-4f) {
        _widget->setRadius(_radius);
    }

    if (_editManager && _editManager->hasSession()) {
        _editManager->setRadius(_radius);
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
    }

    if (_overlay) {
        _overlay->setRadius(_radius);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setSigma(float sigma)
{
    const float clamped = std::clamp(sigma, 0.10f, 2.0f);
    if (std::fabs(clamped - _sigma) < 1e-4f) {
        return;
    }
    _sigma = clamped;

    if (_widget && std::fabs(_widget->sigma() - _sigma) > 1e-4f) {
        _widget->setSigma(_sigma);
    }

    if (_editManager && _editManager->hasSession()) {
        _editManager->setSigma(_sigma);
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
    }
}

void SegmentationModule::setInfluenceMode(SegmentationInfluenceMode mode)
{
    if (_influenceMode == mode) {
        return;
    }
    _influenceMode = mode;
    if (_editManager) {
        _editManager->setInfluenceMode(mode);
        if (_editManager->hasSession()) {
            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            emitPendingChanges();
            refreshOverlay();
        }
    }
}

void SegmentationModule::setSliceFadeDistance(float distance)
{
    const float clamped = std::clamp(distance, 0.1f, 500.0f);
    if (std::fabs(clamped - _sliceFadeDistance) < 1e-4f) {
        return;
    }
    _sliceFadeDistance = clamped;
    if (_widget && std::fabs(_widget->sliceFadeDistance() - clamped) > 1e-4f) {
        _widget->setSliceFadeDistance(clamped);
    }
    if (_overlay) {
        _overlay->setSliceFadeDistance(_sliceFadeDistance);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setSliceDisplayMode(SegmentationSliceDisplayMode mode)
{
    if (_sliceDisplayMode == mode) {
        return;
    }
    _sliceDisplayMode = mode;
    if (_widget && _widget->sliceDisplayMode() != mode) {
        _widget->setSliceDisplayMode(mode);
    }
    if (_overlay) {
        _overlay->setSliceDisplayMode(_sliceDisplayMode);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setRowColMode(SegmentationRowColMode mode)
{
    if (_rowColMode == mode) {
        return;
    }
    _rowColMode = mode;
    if (_widget && _widget->rowColMode() != mode) {
        _widget->setRowColMode(mode);
    }
    if (_editManager) {
        _editManager->setRowColMode(mode);
        if (_editManager->hasSession()) {
            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            emitPendingChanges();
            refreshOverlay();
        }
    }
}

void SegmentationModule::setHoleSearchRadius(int radius)
{
    const int clamped = std::clamp(radius, 1, 64);
    if (clamped == _holeSearchRadius) {
        return;
    }
    _holeSearchRadius = clamped;
    if (_widget && _widget->holeSearchRadius() != clamped) {
        _widget->setHoleSearchRadius(clamped);
    }
    if (_editManager) {
        _editManager->setHoleSearchRadius(_holeSearchRadius);
    }
}

void SegmentationModule::setHoleSmoothIterations(int iterations)
{
    const int clamped = std::clamp(iterations, 1, 200);
    if (clamped == _holeSmoothIterations) {
        return;
    }
    _holeSmoothIterations = clamped;
    if (_widget && _widget->holeSmoothIterations() != clamped) {
        _widget->setHoleSmoothIterations(clamped);
    }
    if (_editManager) {
        _editManager->setHoleSmoothIterations(_holeSmoothIterations);
    }
}

void SegmentationModule::setHandlesAlwaysVisible(bool value)
{
    if (_showHandlesAlways == value) {
        return;
    }
    _showHandlesAlways = value;
    if (_widget && _widget->handlesAlwaysVisible() != value) {
        _widget->setHandlesAlwaysVisible(value);
    }
    if (_overlay) {
        _overlay->setHandleVisibility(_showHandlesAlways, _handleDisplayDistance);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setHandleDisplayDistance(float distance)
{
    const float clamped = std::max(1.0f, distance);
    if (std::fabs(clamped - _handleDisplayDistance) < 1e-4f) {
        return;
    }
    _handleDisplayDistance = clamped;
    if (_widget && std::fabs(_widget->handleDisplayDistance() - clamped) > 1e-4f) {
        _widget->setHandleDisplayDistance(clamped);
    }
    if (_overlay) {
        _overlay->setHandleVisibility(_showHandlesAlways, _handleDisplayDistance);
        _overlay->refreshAll();
    }
}

void SegmentationModule::setHighlightDistance(float distance)
{
    const float clamped = std::clamp(distance, 0.5f, 500.0f);
    if (std::fabs(clamped - _highlightDistance) < 1e-4f) {
        return;
    }
    _highlightDistance = clamped;
    if (_widget && std::fabs(_widget->highlightDistance() - clamped) > 1e-4f) {
        _widget->setHighlightDistance(clamped);
    }
    if (_hover.valid) {
        _hover.clear();
        if (_overlay) {
            _overlay->setHoverHandle(std::nullopt);
            _overlay->refreshAll();
        }
    }
}

void SegmentationModule::setFillInvalidRegions(bool enabled)
{
    if (_fillInvalidRegions == enabled) {
        return;
    }
    _fillInvalidRegions = enabled;

    if (_widget && _widget->fillInvalidRegions() != enabled) {
        _widget->setFillInvalidRegions(enabled);
    }

    if (_editManager) {
        _editManager->setFillInvalidCells(enabled);
    }
}

void SegmentationModule::setGrowthInProgress(bool running)
{
    _growthInProgress = running;
    if (running) {
        setCorrectionsAnnotateMode(false, false);
    }
    updateCorrectionsWidget();
}

void SegmentationModule::setHandlesLocked(bool locked, bool userInitiated)
{
    if (_handlesLocked == locked) {
        if (_widget && _widget->handlesLocked() != locked) {
            _widget->setHandlesLocked(locked);
        }
        return;
    }

    _handlesLocked = locked;

    if (_widget && _widget->handlesLocked() != locked) {
        _widget->setHandlesLocked(locked);
    }

    if (locked) {
        resetInteractionState();
        setPointAddMode(false, true);
    }

    if (userInitiated) {
        emit statusMessageRequested(locked ? tr("Handle edits locked") : tr("Handle edits unlocked"), kStatusShort);
    }

    updateViewerCursors();
}

void SegmentationModule::applyEdits()
{
    emit statusMessageRequested(tr("Applying segmentation edits"), kStatusMedium);
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    _editManager->applyPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();

    if (auto* base = _editManager->baseSurface()) {
        try {
            base->saveOverwrite();
        } catch (const std::exception& e) {
            emit statusMessageRequested(tr("Failed to save segmentation: ") + e.what(), kStatusLong);
        }
    }

    resetInteractionState();
}

void SegmentationModule::resetEdits()
{
    emit statusMessageRequested(tr("Segmentation edits reset"), kStatusMedium);
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    _editManager->resetPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();

    resetInteractionState();
}

void SegmentationModule::stopTools()
{
    emit stopToolsRequested();
}

void SegmentationModule::updateViewerCursors()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachViewer([this](CVolumeViewer* viewer) {
        if (!viewer || !viewer->fGraphicsView) {
            return;
        }
        if (!_editingEnabled) {
            viewer->fGraphicsView->unsetCursor();
            return;
        }
        if (_drag.active && viewer == _drag.viewer) {
            return;
        }
        if (_pointAddMode) {
            viewer->fGraphicsView->setCursor(addCursor());
        } else {
            viewer->fGraphicsView->setCursor(Qt::ArrowCursor);
        }
    });
}

void SegmentationModule::setPointAddMode(bool enabled, bool silent)
{
    if (enabled && (_handlesLocked || _correctionsAnnotateMode)) {
        if (!silent) {
            const QString message = _correctionsAnnotateMode
                                        ? tr("Point add mode is unavailable while corrections are active.")
                                        : tr("Unlock handles to add manual points.");
            emit statusMessageRequested(message, kStatusShort);
        }
        return;
    }
    if (_pointAddMode == enabled) {
        return;
    }
    _pointAddMode = enabled;
    updateViewerCursors();
    if (!silent) {
        const auto message = enabled ? tr("Segmentation point-add mode enabled")
                                     : tr("Segmentation point-add mode disabled");
        emit statusMessageRequested(message, kStatusShort);
    }
}

void SegmentationModule::togglePointAddMode()
{
    setPointAddMode(!_pointAddMode);
}

bool SegmentationModule::handleKeyPress(QKeyEvent* event)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        return false;
    }

    if (_correctionsAnnotateMode) {
        return false;
    }

    if (_handlesLocked && event->key() != Qt::Key_T) {
        return false;
    }

    bool handled = false;
    if (event->key() == Qt::Key_Shift && event->modifiers() == Qt::ShiftModifier && !event->isAutoRepeat()) {
        togglePointAddMode();
        handled = true;
    } else if (event->key() == Qt::Key_R && event->modifiers() == Qt::NoModifier) {
        std::optional<std::pair<int, int>> target;
        if (_drag.active) {
            target = std::make_pair(_drag.row, _drag.col);
        } else if (_hover.valid) {
            target = std::make_pair(_hover.row, _hover.col);
        } else if (_cursorValid) {
            const SegmentationEditManager::Handle* nearest = nullptr;
            if (_cursorViewer) {
                nearest = screenClosestHandle(_cursorViewer, _cursorWorld, _highlightDistance);
            }
            if (!nearest) {
                nearest = _editManager->findNearestHandle(_cursorWorld, radiusWorldExtent(_radius));
            }
            if (nearest) {
                target = std::make_pair(nearest->row, nearest->col);
            }
        }

        if (target) {
            if (auto pos = _editManager->handleWorldPosition(target->first, target->second)) {
                if (_surfaces) {
                    POI* poi = _surfaces->poi("focus");
                    if (!poi) {
                        poi = new POI;
                    }
                    poi->p = *pos;
                    poi->src = _surfaces->surface("segmentation");
                    _surfaces->setPOI("focus", poi);
                }

                if (!_hover.valid || _hover.row != target->first || _hover.col != target->second) {
                    _hover.set(target->first, target->second, *pos);
                }
                if (_overlay) {
                    _overlay->setKeyboardHandle(target);
                    _overlay->setHoverHandle(target);
                }
                emit focusPoiRequested(*pos, _editManager->baseSurface());
                handled = true;
            }
        }
    } else if ((event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) && event->modifiers() == Qt::NoModifier) {
        std::optional<std::pair<int, int>> target;
        if (_drag.active) {
            target = std::make_pair(_drag.row, _drag.col);
        } else if (_hover.valid) {
            target = std::make_pair(_hover.row, _hover.col);
        }
        if (target) {
            if (_editManager->removeHandle(target->first, target->second)) {
                _hover.clear();
                _drag.reset();
                if (_surfaces) {
                    _surfaces->setSurface("segmentation", _editManager->previewSurface());
                }
                emitPendingChanges();
                if (_overlay) {
                    _overlay->setActiveHandle(std::nullopt, false);
                    _overlay->setHoverHandle(std::nullopt, false);
                    _overlay->setKeyboardHandle(std::nullopt, false);
                    _overlay->refreshAll();
                }
                handled = true;
            }
        }
    } else if (event->key() == Qt::Key_T && event->modifiers() == Qt::NoModifier) {
        if (createCorrectionCollection(true)) {
            setCorrectionsAnnotateMode(true, false);
            handled = true;
        }
    }

    if (handled) {
        event->accept();
    }
    return handled;
}

bool SegmentationModule::beginEditingSession(QuadSurface* activeSurface)
{
    if (!_editingEnabled || !_editManager || !activeSurface) {
        return false;
    }

    if (!_editManager->beginSession(activeSurface, _downsample)) {
        return false;
    }

    _editManager->setRadius(_radius);
    _editManager->setSigma(_sigma);
    _editManager->setDownsample(_downsample);
    _editManager->setHoleSearchRadius(_holeSearchRadius);
    _editManager->setHoleSmoothIterations(_holeSmoothIterations);
    _editManager->setRowColMode(_rowColMode);
    _editManager->setInfluenceMode(_influenceMode);

    if (_surfaces) {
        if (auto* preview = _editManager->previewSurface()) {
            _surfaces->setSurface("segmentation", preview);
        }
    }

    emitPendingChanges();
    refreshOverlay();
    updateViewerCursors();

    return true;
}

void SegmentationModule::endEditingSession()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    QuadSurface* base = _editManager->baseSurface();
    _editManager->endSession();
    if (_surfaces && base) {
        _surfaces->setSurface("segmentation", base);
    }

    emitPendingChanges();
    resetInteractionState();
    updateViewerCursors();
}

bool SegmentationModule::hasActiveSession() const
{
    return _editManager && _editManager->hasSession();
}

QuadSurface* SegmentationModule::activeBaseSurface() const
{
    if (!_editManager || !_editManager->hasSession()) {
        return nullptr;
    }
    return _editManager->baseSurface();
}

void SegmentationModule::markNextHandlesFromGrowth()
{
    if (_editManager && _editManager->hasSession()) {
        _editManager->markNextRefreshHandlesAsGrowth();
    }
}

void SegmentationModule::refreshSessionFromSurface(QuadSurface* surface)
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    Q_UNUSED(surface);

    _editManager->refreshFromBaseSurface();
    if (_surfaces) {
        if (auto* preview = _editManager->previewSurface()) {
            _surfaces->setSurface("segmentation", preview);
        }
    }
    emitPendingChanges();
    refreshOverlay();
}

void SegmentationModule::refreshOverlay()
{
    if (_overlay) {
        _overlay->refreshAll();
    }
}

void SegmentationModule::emitPendingChanges()
{
    if (_widget && _editManager) {
        const bool pending = _editManager->hasPendingChanges();
        _widget->setPendingChanges(pending);
        emit pendingChangesChanged(pending);
    }
}

void SegmentationModule::resetOverlayHandles()
{
    _hover.clear();

    if (_overlay) {
        _overlay->setActiveHandle(std::nullopt, false);
        _overlay->setHoverHandle(std::nullopt, false);
        _overlay->setKeyboardHandle(std::nullopt, false);
        _overlay->refreshAll();
    }
}

void SegmentationModule::resetInteractionState()
{
    _drag.reset();
    _hover.clear();
    _cursorValid = false;
    _cursorViewer = nullptr;

    if (_overlay) {
        _overlay->setActiveHandle(std::nullopt, false);
        _overlay->setHoverHandle(std::nullopt, false);
        _overlay->setKeyboardHandle(std::nullopt, false);
        _overlay->setCursorWorld(cv::Vec3f(0, 0, 0), false);
        _overlay->refreshAll();
    }

    if (_pointAddMode) {
        setPointAddMode(false, true);
    }

    if (_viewerManager) {
        _viewerManager->forEachViewer([](CVolumeViewer* v) {
            if (!v) {
                return;
            }
            v->clearOverlayGroup("segmentation_radius_indicator");
            if (v->fGraphicsView) {
                v->fGraphicsView->unsetCursor();
            }
        });
    }
}

float SegmentationModule::gridStepWorld() const
{
    if (_editManager && _editManager->hasSession()) {
        if (auto* surface = _editManager->baseSurface()) {
            const cv::Vec2f scale = surface->scale();
            const float sx = std::fabs(scale[0]);
            const float sy = std::fabs(scale[1]);
            const float step = std::max(sx, sy);
            if (std::isfinite(step) && step > 1e-4f) {
                return step;
            }
        }
    }
    return 1.0f;
}

float SegmentationModule::radiusWorldExtent(float gridRadius) const
{
    const float step = gridStepWorld();
    const float cells = std::max(gridRadius, 1.0f);
    const float baseExtent = (cells + 0.5f) * step;
    const float minExtent = std::max(step, 3.0f);
    return std::max(baseExtent, minExtent);
}

void SegmentationModule::handleMousePress(CVolumeViewer* viewer,
                                          const cv::Vec3f& worldPos,
                                          const cv::Vec3f& normal,
                                          Qt::MouseButton button,
                                          Qt::KeyboardModifiers modifiers)
{
    const QString viewerName = viewer ? QString::fromStdString(viewer->surfName()) : QStringLiteral("<null>");
    const bool canEdit = _editingEnabled && _editManager && _editManager->hasSession();
    const QString header = QStringLiteral("[SegEdit] click viewer=%1 button=%2 pointAdd=%3 pos=%4 mods=%5")
                               .arg(viewerName,
                                    button == Qt::LeftButton ? QStringLiteral("Left") : QStringLiteral("Other"),
                                    _pointAddMode ? QStringLiteral("true") : QStringLiteral("false"),
                                    vecToString(worldPos),
                                    modifiersToString(modifiers));

    if (!canEdit) {
        qCInfo(lcSegEdit).noquote() << header + QStringLiteral(" | ignored=no-active-session");
        return;
    }

    ClickLog logger(header);

    Q_UNUSED(normal);

    if (button != Qt::LeftButton) {
        logger.add(QStringLiteral("ignored: button not left"));
        return;
    }

    if (_correctionsAnnotateMode) {
        if (!_pointCollection || _activeCorrectionId == 0) {
            logger.add(QStringLiteral("correction annotate aborted: no active collection"));
            setCorrectionsAnnotateMode(false, false);
            emit statusMessageRequested(tr("Create or select a correction set before annotating."), kStatusShort);
            return;
        }

        if (!modifiers.testFlag(Qt::ControlModifier) && viewer) {
            const uint64_t highlightedId = viewer->highlightedPointId();
            if (highlightedId != 0 && _pointCollection) {
                if (auto pointOpt = _pointCollection->getPoint(highlightedId)) {
                    if (pointOpt->collectionId == _activeCorrectionId) {
                        logger.add(QStringLiteral("correction drag: existing point highlighted"));
                        return;
                    }
                }
            }
        }

        if (modifiers.testFlag(Qt::ControlModifier)) {
            handleCorrectionPointRemove(worldPos);
            logger.add(QStringLiteral("correction point removed"));
        } else {
            handleCorrectionPointAdded(worldPos);
            logger.add(QStringLiteral("correction point added"));
        }

        updateCorrectionsWidget();
        return;
    }

    if (_handlesLocked) {
        logger.add(QStringLiteral("ignored: handles locked"));
        return;
    }

    if (viewer) {
        _cursorViewer = viewer;
    }

    PlaneSurface* planeSurface = viewer ? dynamic_cast<PlaneSurface*>(viewer->currentSurface()) : nullptr;
    Qt::KeyboardModifiers effectiveModifiers = modifiers & ~Qt::ShiftModifier;

    if (effectiveModifiers.testFlag(Qt::ControlModifier)) {
        const float tolerance = radiusWorldExtent(_radius);
        if (auto* handle = _editManager->findNearestHandle(worldPos, tolerance)) {
            if (_editManager->removeHandle(handle->row, handle->col)) {
                logger.add(QStringLiteral("removed handle row=%1 col=%2").arg(handle->row).arg(handle->col));
                _hover.clear();
                emitPendingChanges();
                if (_surfaces) {
                    _surfaces->setSurface("segmentation", _editManager->previewSurface());
                }
                if (_overlay) {
                    _overlay->setActiveHandle(std::nullopt, false);
                    _overlay->setHoverHandle(std::nullopt, false);
                    _overlay->setKeyboardHandle(std::nullopt, false);
                    _overlay->refreshAll();
                }
            } else {
                logger.add(QStringLiteral("remove handle failed row=%1 col=%2").arg(handle->row).arg(handle->col));
            }
        } else {
            logger.add(QStringLiteral("remove skipped: no handle within %1mm").arg(tolerance, 0, 'f', 2));
        }
        return;
    }
    if (!viewer) {
        logger.add(QStringLiteral("ignored: viewer null"));
        return;
    }

    logger.add(QStringLiteral("viewer-surface=%1").arg(viewerName));

    std::optional<SegmentationRowColAxis> axis;
    if (_influenceMode == SegmentationInfluenceMode::RowColumn) {
        axis = rowColAxisForViewer(viewer);
    }
    logger.add(QStringLiteral("axis-mode=%1").arg(axisToString(axis)));

    if (_pointAddMode) {
        const float addTolerance = -1.0f;
        const float addPlaneTolerance = planeSurface ? -1.0f : 0.0f;
        if (auto added = _editManager->addHandleAtWorld(worldPos,
                                                        addTolerance,
                                                        planeSurface,
                                                        addPlaneTolerance,
                                                        true,
                                                        false,
                                                        axis)) {
            cv::Vec3f handleWorld = worldPos;
            if (auto world = _editManager->handleWorldPosition(added->first, added->second)) {
                handleWorld = *world;
            }
            _hover.set(added->first, added->second, handleWorld);
            emitPendingChanges();
            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            if (_overlay) {
                _overlay->setActiveHandle(*added);
                _overlay->setHoverHandle(*added);
                _overlay->setKeyboardHandle(*added, false);
                _overlay->refreshAll();
            }
            logger.add(QStringLiteral("added handle row=%1 col=%2").arg(added->first).arg(added->second));

            _editManager->bakePreviewToOriginal();

            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            resetOverlayHandles();
        } else {
            emit statusMessageRequested(tr("Failed to add handle; see terminal for details"), kStatusMedium);
            logger.add(QStringLiteral("add handle failed"));
        }
        return;
    }

    const float screenRadius = std::max(_highlightDistance, 1.0f);
    const float extendedRadius = screenRadius * 1.75f;
    const bool segmentationView = isSegmentationViewer(viewer);
    logger.add(QStringLiteral("segmentation-view=%1").arg(segmentationView ? QStringLiteral("true") : QStringLiteral("false")));
    if (segmentationView) {
        logger.add(QStringLiteral("mass-pull disabled on segmentation surface"));
    }

    int handleRow = -1;
    int handleCol = -1;
    cv::Vec3f initialWorld{0.0f, 0.0f, 0.0f};
    bool haveHandle = false;
    bool repositionImmediately = false;

    if (const auto* handle = screenClosestHandle(viewer, worldPos, screenRadius)) {
        handleRow = handle->row;
        handleCol = handle->col;
        initialWorld = handle->currentWorld;
        haveHandle = true;
        logger.add(QStringLiteral("picked handle row=%1 col=%2 (screen radius)")
                       .arg(handleRow)
                       .arg(handleCol));
    }

    if (!haveHandle) {
        if (const auto* handle = screenClosestHandle(viewer, worldPos, extendedRadius)) {
            handleRow = handle->row;
            handleCol = handle->col;
            initialWorld = handle->currentWorld;
            haveHandle = true;
            repositionImmediately = true;
            logger.add(QStringLiteral("picked handle row=%1 col=%2 (extended radius)")
                           .arg(handleRow)
                           .arg(handleCol));
        }
    }

    if (!haveHandle && !segmentationView) {
        const auto pullResult = pullHandlesTowards(worldPos, axis);
        if (pullResult.moved) {
            logger.add(QStringLiteral("pulled handles candidates=%1 applied=%2 avgWeight=%3")
                           .arg(pullResult.candidateCount)
                           .arg(pullResult.appliedCount)
                           .arg(pullResult.averageWeight, 0, 'f', 3));
            if (_surfaces) {
                _surfaces->setSurface("segmentation", _editManager->previewSurface());
            }
            emitPendingChanges();
            if (_overlay) {
                const auto* hoverHandle = screenClosestHandle(viewer, worldPos, screenRadius);
                if (hoverHandle) {
                    if (auto world = _editManager->handleWorldPosition(hoverHandle->row, hoverHandle->col)) {
                        _hover.set(hoverHandle->row, hoverHandle->col, *world);
                    } else {
                        _hover.set(hoverHandle->row, hoverHandle->col, hoverHandle->currentWorld);
                    }
                    _overlay->setHoverHandle(std::make_pair(hoverHandle->row, hoverHandle->col));
                } else {
                    _hover.clear();
                    _overlay->setHoverHandle(std::nullopt);
                }
                _overlay->setActiveHandle(std::nullopt, false);
                _overlay->refreshViewer(viewer);
            }
            return;
        } else {
            logger.add(QStringLiteral("pull skipped candidates=%1 applied=%2")
                           .arg(pullResult.candidateCount)
                           .arg(pullResult.appliedCount));
        }
    }

    if (!haveHandle) {
        const float ensureTolerance = -1.0f;
        const float ensurePlaneTolerance = planeSurface ? -1.0f : 0.0f;
        if (auto ensured = _editManager->addHandleAtWorld(worldPos,
                                                          ensureTolerance,
                                                          planeSurface,
                                                          ensurePlaneTolerance,
                                                          true,
                                                          true,
                                                          axis)) {
            handleRow = ensured->first;
            handleCol = ensured->second;
            if (auto world = _editManager->handleWorldPosition(handleRow, handleCol)) {
                initialWorld = *world;
            } else {
                initialWorld = worldPos;
            }
            haveHandle = true;
            repositionImmediately = true;
            logger.add(QStringLiteral("ensured handle row=%1 col=%2")
                           .arg(handleRow)
                           .arg(handleCol));
        } else if (auto* nearest = _editManager->findNearestHandle(worldPos, -1.0f)) {
            handleRow = nearest->row;
            handleCol = nearest->col;
            initialWorld = nearest->currentWorld;
            haveHandle = true;
            repositionImmediately = true;
            logger.add(QStringLiteral("fallback to nearest handle row=%1 col=%2")
                           .arg(handleRow)
                           .arg(handleCol));
        }
    }

    if (!haveHandle) {
        logger.add(QStringLiteral("no handle found; click ignored"));
        return;
    }

    logger.add(QStringLiteral("begin drag row=%1 col=%2 immediate=%3")
                   .arg(handleRow)
                   .arg(handleCol)
                   .arg(repositionImmediately));
    _drag.active = true;
    _drag.row = handleRow;
    _drag.col = handleCol;
    _drag.viewer = viewer;
    _drag.startWorld = initialWorld;
    _drag.moved = false;

    _hover.clear();
    if (_overlay) {
        _overlay->setHoverHandle(std::nullopt);
        _overlay->setActiveHandle(std::make_pair(handleRow, handleCol));
        _overlay->refreshViewer(viewer);
    }

    if (viewer->fGraphicsView) {
        viewer->fGraphicsView->setCursor(Qt::ClosedHandCursor);
    }

    if (repositionImmediately) {
        const bool moved = _editManager->updateHandleWorldPosition(handleRow, handleCol, worldPos, axis);
        if (!moved) {
            emit statusMessageRequested(tr("Handle move failed; see terminal for details"), kStatusMedium);
            logger.add(QStringLiteral("immediate reposition failed"));
            _drag.reset();
            if (_overlay) {
                _overlay->setActiveHandle(std::nullopt, false);
                _overlay->refreshViewer(viewer);
            }
            if (viewer->fGraphicsView) {
                viewer->fGraphicsView->setCursor(Qt::ArrowCursor);
            }
            return;
        }
        _drag.moved = true;
        logger.add(QStringLiteral("immediate reposition applied"));
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
        if (auto world = _editManager->handleWorldPosition(handleRow, handleCol)) {
            _hover.set(handleRow, handleCol, *world);
        }
        if (_overlay) {
            _overlay->setActiveHandle(std::make_pair(handleRow, handleCol), false);
            _overlay->setHoverHandle(std::nullopt, false);
            _overlay->refreshAll();
        }

        _editManager->bakePreviewToOriginal();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        resetOverlayHandles();
    }
}

void SegmentationModule::handleMouseMove(CVolumeViewer* viewer,
                                         const cv::Vec3f& worldPos,
                                         Qt::MouseButtons buttons,
                                         Qt::KeyboardModifiers /*modifiers*/)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession()) {
        return;
    }

    _cursorViewer = viewer;
    _cursorWorld = worldPos;
    _cursorValid = true;
    if (_overlay) {
        _overlay->setCursorWorld(worldPos, true);
        if (!_showHandlesAlways) {
            _overlay->refreshViewer(viewer);
        }
    }

    if (_correctionsAnnotateMode) {
        return;
    }

    if (_handlesLocked) {
        return;
    }

    if (_drag.active && viewer == _drag.viewer) {
        if (!(buttons & Qt::LeftButton)) {
            return;
        }

        std::optional<SegmentationRowColAxis> axis;
        if (_influenceMode == SegmentationInfluenceMode::RowColumn) {
            axis = rowColAxisForViewer(viewer);
        }
        const bool moved = _editManager->updateHandleWorldPosition(_drag.row, _drag.col, worldPos, axis);
        if (!moved) {
            emit statusMessageRequested(tr("Handle move failed; see terminal for details"), kStatusMedium);
            return;
        }
        _drag.moved = true;
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        emitPendingChanges();
        if (auto world = _editManager->handleWorldPosition(_drag.row, _drag.col)) {
            _hover.set(_drag.row, _drag.col, *world);
        }
        if (_overlay) {
            _overlay->setActiveHandle(std::make_pair(_drag.row, _drag.col), false);
            _overlay->setHoverHandle(std::nullopt, false);
            _overlay->refreshAll();
        }

        _editManager->bakePreviewToOriginal();
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
        resetOverlayHandles();
        return;
    }

    const float screenRadius = std::max(_highlightDistance, 1.0f);
    const SegmentationEditManager::Handle* handle = screenClosestHandle(viewer, worldPos, screenRadius);
    if (handle) {
        const bool changed = !_hover.valid || _hover.row != handle->row || _hover.col != handle->col;
        if (auto world = _editManager->handleWorldPosition(handle->row, handle->col)) {
            _hover.set(handle->row, handle->col, *world);
        } else {
            _hover.set(handle->row, handle->col, handle->currentWorld);
        }
        if (_overlay) {
            _overlay->setHoverHandle(std::make_pair(handle->row, handle->col));
            if (changed) {
                _overlay->refreshViewer(viewer);
            }
        }
    } else if (_hover.valid) {
        _hover.clear();
        if (_overlay) {
            _overlay->setHoverHandle(std::nullopt);
            _overlay->refreshViewer(viewer);
        }
    }
}

void SegmentationModule::handleMouseRelease(CVolumeViewer* viewer,
                                            const cv::Vec3f& worldPos,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers /*modifiers*/)
{
    if (_correctionsAnnotateMode) {
        return;
    }

    if (_handlesLocked) {
        return;
    }
    if (viewer) {
        _cursorViewer = viewer;
    }
    if (!_drag.active || viewer != _drag.viewer) {
        return;
    }
    if (button != Qt::LeftButton) {
        return;
    }

    if (viewer && viewer->fGraphicsView) {
        viewer->fGraphicsView->setCursor(Qt::ArrowCursor);
    }

    if (_pointAddMode && _editManager && _editManager->hasSession() && !_drag.moved) {
        std::optional<SegmentationRowColAxis> axis;
        if (_influenceMode == SegmentationInfluenceMode::RowColumn) {
            axis = rowColAxisForViewer(viewer);
        }
        const bool moved = _editManager->updateHandleWorldPosition(_drag.row, _drag.col, worldPos, axis);
        if (!moved) {
            emit statusMessageRequested(tr("Handle move failed; see terminal for details"), kStatusMedium);
        }
        if (auto world = _editManager->handleWorldPosition(_drag.row, _drag.col)) {
            _hover.set(_drag.row, _drag.col, *world);
        }
    }

    if (_editManager && _editManager->hasSession()) {
        if (_surfaces) {
            _surfaces->setSurface("segmentation", _editManager->previewSurface());
        }
    }

    emitPendingChanges();

    if (_editManager) {
        _editManager->bakePreviewToOriginal();
    }

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    resetOverlayHandles();

    if (_overlay) {
        _overlay->setHoverHandle(std::nullopt, false);
    }

    _drag.reset();
    updateViewerCursors();
}

void SegmentationModule::handleRadiusWheel(CVolumeViewer* viewer,
                                           int steps,
                                           const QPointF& scenePoint,
                                           const cv::Vec3f& /*worldPos*/)
{
    if (!_editingEnabled || !_editManager || !_editManager->hasSession() || _handlesLocked || _correctionsAnnotateMode) {
        return;
    }
    if (steps == 0) {
        return;
    }

    int deltaSteps = steps > 0 ? 1 : -1;
    if (QApplication::keyboardModifiers().testFlag(Qt::ControlModifier)) {
        deltaSteps *= 2;
    }

    const int currentSteps = std::clamp(static_cast<int>(std::lround(_radius)), 1, kMaxRadiusSteps);
    const int newSteps = std::clamp(currentSteps + deltaSteps, 1, kMaxRadiusSteps);
    if (newSteps == currentSteps) {
        return;
    }

    _radius = static_cast<float>(newSteps);
    if (_widget && std::fabs(_widget->radius() - _radius) > 1e-4f) {
        _widget->setRadius(_radius);
    }
    _editManager->setRadius(_radius);
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();

    if (_overlay) {
        _overlay->setRadius(_radius);
        _overlay->refreshAll();
    }

    if (viewer) {
        showRadiusIndicator(viewer, scenePoint, _radius);
    }
}

void SegmentationModule::showRadiusIndicator(CVolumeViewer* viewer,
                                             const QPointF& scenePoint,
                                             float radius)
{
    if (!viewer) {
        return;
    }

    viewer->clearOverlayGroup("segmentation_radius_indicator");

    const int steps = std::clamp(static_cast<int>(std::lround(radius)), 1, kMaxRadiusSteps);
    const QString label = steps == 1 ? tr("1 step") : tr("%1 steps").arg(steps);
    auto* textItem = new QGraphicsSimpleTextItem(label);
    QFont font = textItem->font();
    font.setPointSizeF(11.0);
    textItem->setFont(font);
    textItem->setBrush(QColor(255, 255, 255));
    textItem->setPen(QPen(Qt::black, 0.8));
    textItem->setZValue(150.0);

    const QPointF offset(12.0, -12.0);
    textItem->setPos(scenePoint + offset);

    viewer->setOverlayGroup("segmentation_radius_indicator", {textItem});

    QPointer<CVolumeViewer> guard(viewer);
    QTimer::singleShot(800, this, [guard]() {
        if (guard) {
            guard->clearOverlayGroup("segmentation_radius_indicator");
        }
    });
}

const QCursor& SegmentationModule::addCursor()
{
    static bool initialized = false;
    static QCursor cursor;
    if (!initialized) {
        QPixmap pixmap(32, 32);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing);
        QPen pen(Qt::white, 2);
        painter.setPen(pen);
        painter.drawEllipse(QPointF(16, 16), 12, 12);
        painter.drawLine(QPointF(16, 6), QPointF(16, 26));
        painter.drawLine(QPointF(6, 16), QPointF(26, 16));
        painter.end();
        cursor = QCursor(pixmap, 16, 16);
        initialized = true;
    }
    return cursor;
}

bool SegmentationModule::isSegmentationViewer(const CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    std::string name = viewer->surfName();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return name == "segmentation";
}

const SegmentationEditManager::Handle* SegmentationModule::screenClosestHandle(CVolumeViewer* viewer,
                                                                               const cv::Vec3f& referenceWorld,
                                                                               float maxScreenDistance) const
{
    if (!viewer || !_editManager || !_editManager->hasSession()) {
        return nullptr;
    }

    const auto& handles = _editManager->handles();
    if (handles.empty()) {
        return nullptr;
    }

    const float radius = std::max(maxScreenDistance, 1.0f);
    float bestDistSq = radius * radius;
    const QPointF cursorScene = viewer->volumePointToScene(referenceWorld);
    const SegmentationEditManager::Handle* bestHandle = nullptr;

    for (const auto& handle : handles) {
        const cv::Vec3f& world = handle.currentWorld;
        if (!std::isfinite(world[0]) || !std::isfinite(world[1]) || !std::isfinite(world[2])) {
            continue;
        }
        const QPointF scenePos = viewer->volumePointToScene(world);
        const QPointF diff = cursorScene - scenePos;
        const float distSq = static_cast<float>(QPointF::dotProduct(diff, diff));
        if (distSq <= bestDistSq) {
            bestDistSq = distSq;
            bestHandle = &handle;
        }
    }

    return bestHandle;
}

SegmentationModule::PullResult SegmentationModule::pullHandlesTowards(const cv::Vec3f& worldPos,
                                                                      std::optional<SegmentationRowColAxis> axis)
{
    PullResult result;
    if (!_editManager || !_editManager->hasSession()) {
        return result;
    }

    const auto& handles = _editManager->handles();
    if (handles.empty()) {
        return result;
    }

    const float worldRadius = radiusWorldExtent(_radius);
    if (worldRadius <= 0.0f || !std::isfinite(worldRadius)) {
        return result;
    }

    const float baseStrength = std::clamp(_sigma, 0.10f, 2.0f);
    struct HandleUpdate {
        int row;
        int col;
        cv::Vec3f target;
        float weight;
    };
    std::vector<HandleUpdate> updates;
    updates.reserve(handles.size());

    for (const auto& handle : handles) {
        const cv::Vec3f& current = handle.currentWorld;
        if (!std::isfinite(current[0]) || !std::isfinite(current[1]) || !std::isfinite(current[2])) {
            continue;
        }
        const float dist = static_cast<float>(cv::norm(current - worldPos));
        if (!std::isfinite(dist) || dist > worldRadius) {
            continue;
        }
        const float falloff = std::max(0.0f, 1.0f - (dist / worldRadius));
        if (falloff <= 1e-3f) {
            continue;
        }
        float weight = std::clamp(baseStrength * falloff, 0.0f, 0.95f);
        if (weight <= 1e-4f) {
            continue;
        }
        cv::Vec3f target = current + (worldPos - current) * weight;
        updates.push_back({handle.row, handle.col, target, weight});
    }

    result.candidateCount = static_cast<int>(updates.size());
    if (updates.empty()) {
        return result;
    }

    float weightSum = 0.0f;
    for (const auto& upd : updates) {
        if (_editManager->updateHandleWorldPosition(upd.row, upd.col, upd.target, axis)) {
            result.appliedCount += 1;
            weightSum += upd.weight;
        }
    }

    if (result.appliedCount > 0) {
        result.moved = true;
        result.averageWeight = weightSum / static_cast<float>(result.appliedCount);
    }

    return result;
}

SegmentationRowColAxis SegmentationModule::rowColAxisForViewer(const CVolumeViewer* viewer) const
{
    switch (_rowColMode) {
    case SegmentationRowColMode::RowOnly:
        return SegmentationRowColAxis::Row;
    case SegmentationRowColMode::ColumnOnly:
        return SegmentationRowColAxis::Column;
    case SegmentationRowColMode::Dynamic:
    default:
        break;
    }

    if (!viewer) {
        return SegmentationRowColAxis::Row;
    }

    std::string name = viewer->surfName();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (name == "seg yz" || name == "seg xz") {
        return SegmentationRowColAxis::Column;
    }
    if (name == "xy plane" || name == "xy / slices") {
        return SegmentationRowColAxis::Row;
    }
    return SegmentationRowColAxis::Row;
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

    if (_correctionsAnnotateMode) {
        resetInteractionState();
    }

    updateHandleVisibility();

    if (!enabled) {
        _widget->setCorrectionsAnnotateChecked(false);
    } else {
        _widget->setCorrectionsAnnotateChecked(true);
        if (_pointAddMode) {
            setPointAddMode(false, true);
        }
    }

    if (userInitiated) {
        const QString message = enabled ? tr("Correction annotation enabled")
                                        : tr("Correction annotation disabled");
        emit statusMessageRequested(message, kStatusShort);
    }

    updateViewerCursors();
    updateCorrectionsWidget();
}

void SegmentationModule::setActiveCorrectionCollection(uint64_t collectionId, bool userInitiated)
{
    if (collectionId == 0 || !_pointCollection) {
        if (_activeCorrectionId != 0) {
            _activeCorrectionId = 0;
            setCorrectionsAnnotateMode(false, false);
            updateCorrectionsWidget();
        }
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
        entry.points.reserve(it->second.points.size());

        std::vector<ColPoint> sortedPoints;
        sortedPoints.reserve(it->second.points.size());
        for (const auto& pair : it->second.points) {
            sortedPoints.push_back(pair.second);
        }
        std::sort(sortedPoints.begin(), sortedPoints.end(), [](const ColPoint& a, const ColPoint& b) {
            return a.id < b.id;
        });
        if (sortedPoints.empty()) {
            continue;
        }
        entry.points = std::move(sortedPoints);

        payload.collections.push_back(std::move(entry));
    }

    return payload;
}

void SegmentationModule::clearPendingCorrections()
{
    if (!_pendingCorrectionIds.empty()) {
        setCorrectionsAnnotateMode(false, false);
    }

    if (_pointCollection) {
        std::vector<uint64_t> toClear = _pendingCorrectionIds;
        for (uint64_t id : toClear) {
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
