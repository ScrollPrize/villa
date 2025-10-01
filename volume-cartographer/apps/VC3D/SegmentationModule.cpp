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

#include <QCursor>
#include <QKeyEvent>
#include <QLoggingCategory>
#include <QPointer>

#include <algorithm>
#include <cmath>
#include <optional>
#include <limits>

Q_LOGGING_CATEGORY(lcSegModule, "vc.segmentation.module")

namespace
{
constexpr int kStatusShort = 1500;
constexpr int kStatusMedium = 2000;
constexpr int kStatusLong = 5000;

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

void SegmentationModule::HoverState::set(int r, int c, const cv::Vec3f& w)
{
    valid = true;
    row = r;
    col = c;
    world = w;
}

void SegmentationModule::HoverState::clear()
{
    valid = false;
}

SegmentationModule::SegmentationModule(SegmentationWidget* widget,
                                       SegmentationEditManager* editManager,
                                       SegmentationOverlayController* overlay,
                                       ViewerManager* viewerManager,
                                       CSurfaceCollection* surfaces,
                                       VCCollection* pointCollection,
                                       bool editingEnabled,
                                       float radiusSteps,
                                       float sigmaSteps,
                                       QObject* parent)
    : QObject(parent)
    , _widget(widget)
    , _editManager(editManager)
    , _overlay(overlay)
    , _viewerManager(viewerManager)
    , _surfaces(surfaces)
    , _pointCollection(pointCollection)
    , _editingEnabled(editingEnabled)
    , _radiusSteps(radiusSteps)
    , _sigmaSteps(sigmaSteps)
    , _growthMethod(_widget ? _widget->growthMethod() : SegmentationGrowthMethod::Tracer)
    , _growthSteps(_widget ? _widget->growthSteps() : 5)
{
    if (_overlay) {
        _overlay->setEditManager(_editManager);
        _overlay->setEditingEnabled(_editingEnabled);
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
    }

    if (_editManager) {
        _editManager->setRadius(_radiusSteps);
        _editManager->setSigma(_sigmaSteps);
    }

    bindWidgetSignals();

    if (_viewerManager) {
        _viewerManager->setSegmentationModule(this);
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

void SegmentationModule::bindWidgetSignals()
{
    if (!_widget) {
        return;
    }

    connect(_widget, &SegmentationWidget::editingModeChanged,
            this, &SegmentationModule::setEditingEnabled);
    connect(_widget, &SegmentationWidget::radiusChanged,
            this, &SegmentationModule::setRadius);
    connect(_widget, &SegmentationWidget::sigmaChanged,
            this, &SegmentationModule::setSigma);
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
    connect(_widget, &SegmentationWidget::correctionsCreateRequested,
            this, &SegmentationModule::onCorrectionsCreateRequested);
    connect(_widget, &SegmentationWidget::correctionsCollectionSelected,
            this, &SegmentationModule::onCorrectionsCollectionSelected);
    connect(_widget, &SegmentationWidget::correctionsAnnotateToggled,
            this, &SegmentationModule::onCorrectionsAnnotateToggled);
    connect(_widget, &SegmentationWidget::correctionsZRangeChanged,
            this, &SegmentationModule::onCorrectionsZRangeChanged);
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
        setCorrectionsAnnotateMode(false, false);
    }
    updateCorrectionsWidget();
    emit editingEnabledChanged(enabled);
}

void SegmentationModule::setRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (std::fabs(sanitized - _radiusSteps) < 1e-4f) {
        return;
    }
    _radiusSteps = sanitized;
    if (_editManager) {
        _editManager->setRadius(_radiusSteps);
    }
    if (_overlay) {
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
        _overlay->refreshAll();
    }
    if (_widget) {
        _widget->setRadius(_radiusSteps);
    }
}

void SegmentationModule::setSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (std::fabs(sanitized - _sigmaSteps) < 1e-4f) {
        return;
    }
    _sigmaSteps = sanitized;
    if (_editManager) {
        _editManager->setSigma(_sigmaSteps);
    }
    if (_overlay) {
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
        _overlay->refreshAll();
    }
    if (_widget) {
        _widget->setSigma(_sigmaSteps);
    }
}

void SegmentationModule::applyEdits()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    _editManager->applyPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    emitPendingChanges();
}

void SegmentationModule::resetEdits()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }
    cancelDrag();
    _editManager->resetPreview();
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }
    refreshOverlay();
    emitPendingChanges();
}

void SegmentationModule::stopTools()
{
    cancelDrag();
    emit stopToolsRequested();
}

bool SegmentationModule::beginEditingSession(QuadSurface* surface)
{
    if (!_editManager || !surface) {
        return false;
    }

    if (!_editManager->beginSession(surface)) {
        qCWarning(lcSegModule) << "Failed to begin segmentation editing session";
        return false;
    }

    _editManager->setRadius(_radiusSteps);
    _editManager->setSigma(_sigmaSteps);

    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface());
    }

    if (_overlay) {
        _overlay->setEditingEnabled(_editingEnabled);
        _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
        refreshOverlay();
    }

    emitPendingChanges();
    return true;
}

void SegmentationModule::endEditingSession()
{
    cancelDrag();
    if (_overlay) {
        _overlay->setActiveVertex(std::nullopt);
        _overlay->setTouchedVertices({});
        _overlay->refreshAll();
    }
    if (_editManager) {
        _editManager->endSession();
    }
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

    if (event->key() == Qt::Key_Escape) {
        if (_drag.active) {
            cancelDrag();
            return true;
        }
    }

    if (event->key() == Qt::Key_T && event->modifiers() == Qt::NoModifier) {
        onCorrectionsCreateRequested();
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

    _overlay->setActiveVertex(activeMarker);
    _overlay->setTouchedVertices(neighbours);
    _overlay->setGaussianParameters(_radiusSteps, _sigmaSteps, gridStepWorld());
    _overlay->refreshAll();
}

void SegmentationModule::refreshMaskOverlay()
{
    if (_overlay) {
        _overlay->setMaskOverlay({}, false, 0.0f, 0.0f);
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

    _growthMethod = method;
    _growthSteps = std::max(1, steps);
    markNextEditsFromGrowth();
    emit growSurfaceRequested(method, direction, _growthSteps);
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

    if (_correctionsAnnotateMode) {
        if (button != Qt::LeftButton) {
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

    if (button != Qt::LeftButton) {
        return;
    }

    if (modifiers.testFlag(Qt::ControlModifier) || modifiers.testFlag(Qt::AltModifier)) {
        return;
    }

    auto gridIndex = _editManager->worldToGridIndex(worldPos);
    if (!gridIndex) {
        return;
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
    setRadius(_radiusSteps + step);
    updateHover(viewer, worldPos);
    emit statusMessageRequested(tr("Gaussian radius: %1 steps").arg(_radiusSteps, 0, 'f', 2), kStatusShort);
}

bool SegmentationModule::isSegmentationViewer(const CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    return viewer->surfName() == "segmentation";
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

    if (!_editManager->updateActiveDrag(worldPos)) {
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

    _editManager->commitActiveDrag();
    _drag.reset();
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
        _hover.set(gridIndex->first, gridIndex->second, *world);
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
