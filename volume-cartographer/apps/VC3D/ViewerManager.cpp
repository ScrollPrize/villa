#include "ViewerManager.hpp"

#include "VCSettings.hpp"
#include "VolumeViewerBase.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"
#include "streaming/CChunkedVolumeViewer.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/RawPointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "CState.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Logging.hpp"

#include <QMdiArea>
#include <QCoreApplication>
#include <QThread>
#include <QMdiSubWindow>
#include <QSettings>
#include <QtConcurrent/QtConcurrent>
#include <QLoggingCategory>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <unordered_set>
#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"

Q_LOGGING_CATEGORY(lcViewerManager, "vc.viewer.manager")

#define VC3D_DEBUG_QCINFO(category) if (!DebugLoggingEnabled()) {} else qCInfo(category)


ViewerManager::ViewerManager(CState* state,
                             VCCollection* points,
                             QObject* parent)
    : QObject(parent)
    , _state(state)
    , _points(points)
{
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const int savedOpacityPercent = settings.value(viewer::INTERSECTION_OPACITY, viewer::INTERSECTION_OPACITY_DEFAULT).toInt();
    const float normalized = static_cast<float>(savedOpacityPercent) / 100.0f;
    _intersectionOpacity = std::clamp(normalized, 0.0f, 1.0f);

    const float storedBaseLow = settings.value(viewer::BASE_WINDOW_LOW, viewer::BASE_WINDOW_LOW_DEFAULT).toFloat();
    const float storedBaseHigh = settings.value(viewer::BASE_WINDOW_HIGH, viewer::BASE_WINDOW_HIGH_DEFAULT).toFloat();
    _volumeWindowLow = std::clamp(storedBaseLow, 0.0f, 255.0f);
    const float minHigh = std::min(_volumeWindowLow + 1.0f, 255.0f);
    _volumeWindowHigh = std::clamp(storedBaseHigh, minHigh, 255.0f);

    const bool strideUserSet = settings.value(viewer::INTERSECTION_SAMPLING_STRIDE_USER_SET, false).toBool();
    const int storedSampling = settings.value(viewer::INTERSECTION_SAMPLING_STRIDE,
                                              viewer::INTERSECTION_SAMPLING_STRIDE_DEFAULT).toInt();
    _surfacePatchSamplingStride = std::max(1, strideUserSet
                                                  ? storedSampling
                                                  : viewer::INTERSECTION_SAMPLING_STRIDE_DEFAULT);
    const float storedThickness = settings.value(viewer::INTERSECTION_THICKNESS, viewer::INTERSECTION_THICKNESS_DEFAULT).toFloat();
    _intersectionThickness = std::max(0.0f, storedThickness);
    _intersectionMaxSurfaces = std::max(0, settings.value(viewer::INTERSECTION_MAX_SURFACES, viewer::INTERSECTION_MAX_SURFACES_DEFAULT).toInt());

    _surfacePatchIndexWatcher =
        new QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>(this);
    connect(_surfacePatchIndexWatcher,
            &QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>::finished,
            this,
            &ViewerManager::handleSurfacePatchIndexPrimeFinished);

    _surfacePatchIndexTaskWatcher =
        new QFutureWatcher<SurfacePatchIndexTaskResult>(this);
    connect(_surfacePatchIndexTaskWatcher,
            &QFutureWatcher<SurfacePatchIndexTaskResult>::finished,
            this,
            &ViewerManager::handleSurfacePatchIndexTaskFinished);

    if (_state) {
        connect(_state,
                &CState::surfaceChanged,
                this,
                &ViewerManager::handleSurfaceChanged);
        connect(_state,
                &CState::surfaceWillBeDeleted,
                this,
                &ViewerManager::handleSurfaceWillBeDeleted);
    }
}

bool ViewerManager::useChunkedViewer() const
{
    const auto* app = QCoreApplication::instance();
    return app && app->property("vc3d/useChunkedViewer").toBool();
}

VolumeViewerBase* ViewerManager::createViewer(const std::string& surfaceName,
                                              const QString& title,
                                              QMdiArea* mdiArea)
{
    if (!mdiArea || !_state) {
        return nullptr;
    }

    QWidget* widget = nullptr;
    VolumeViewerBase* baseViewer = nullptr;
    CTiledVolumeViewer* adaptiveViewer = nullptr;
    CChunkedVolumeViewer* chunkedViewer = nullptr;

    if (useChunkedViewer()) {
        chunkedViewer = new CChunkedVolumeViewer(_state, this, mdiArea);
        widget = chunkedViewer;
        baseViewer = chunkedViewer;
    } else {
        adaptiveViewer = new CTiledVolumeViewer(_state, this, mdiArea);
        widget = adaptiveViewer;
        baseViewer = adaptiveViewer;
    }

    auto* win = mdiArea->addSubWindow(widget);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::WindowTitleHint | Qt::WindowMinMaxButtonsHint);
    win->installEventFilter(widget);

    if (adaptiveViewer) {
        adaptiveViewer->setPointCollection(_points);
    }
    if (chunkedViewer) {
        chunkedViewer->setPointCollection(_points);
    }

    if (_state) {
        if (adaptiveViewer) {
            connect(_state, &CState::surfaceChanged, adaptiveViewer, &CTiledVolumeViewer::onSurfaceChanged);
            connect(_state, &CState::surfaceWillBeDeleted, adaptiveViewer, &CTiledVolumeViewer::onSurfaceWillBeDeleted);
            connect(_state, &CState::poiChanged, adaptiveViewer, &CTiledVolumeViewer::onPOIChanged);
        }
        if (chunkedViewer) {
            connect(_state, &CState::surfaceChanged, chunkedViewer, &CChunkedVolumeViewer::onSurfaceChanged);
            connect(_state, &CState::surfaceWillBeDeleted, chunkedViewer, &CChunkedVolumeViewer::onSurfaceWillBeDeleted);
            connect(_state, &CState::poiChanged, chunkedViewer, &CChunkedVolumeViewer::onPOIChanged);
            connect(_state, &CState::volumeChanged, chunkedViewer, &CChunkedVolumeViewer::OnVolumeChanged);
            connect(_state, &CState::volumeClosing, chunkedViewer, &CChunkedVolumeViewer::onVolumeClosing);
        }
    }

    // Restore persisted viewer preferences
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showHints = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        baseViewer->setShowDirectionHints(showHints);
        bool showNormals = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        baseViewer->setShowSurfaceNormals(showNormals);
    }

    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool resetView = settings.value(viewer::RESET_VIEW_ON_SURFACE_CHANGE, viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
        baseViewer->setResetViewOnSurfaceChange(resetView);
        _resetDefaults[baseViewer] = resetView;
    }

    baseViewer->setSurface(surfaceName);
    baseViewer->setSegmentationEditActive(_segmentationEditActive);
    baseViewer->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);

    if (adaptiveViewer) {
        _viewers.push_back(adaptiveViewer);
    }
    _baseViewers.push_back(baseViewer);

    // Clean up when viewer is destroyed (e.g. MDI sub-window closed)
    connect(widget, &QObject::destroyed, this, [this, baseViewer, adaptiveViewer]() {
        if (adaptiveViewer) {
            _viewers.erase(std::remove(_viewers.begin(), _viewers.end(), adaptiveViewer), _viewers.end());
        }
        _resetDefaults.erase(baseViewer);
        _baseViewers.erase(std::remove(_baseViewers.begin(), _baseViewers.end(), baseViewer), _baseViewers.end());
    });

    for (auto* overlay : _allOverlays) {
        overlay->attachViewer(baseViewer);
    }

    baseViewer->setIntersectionOpacity(_intersectionOpacity);
    baseViewer->setIntersectionThickness(_intersectionThickness);
    baseViewer->setSurfacePatchSamplingStride(_surfacePatchSamplingStride);
    baseViewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
    baseViewer->setOverlayVolume(_overlayVolume);
    baseViewer->setOverlayOpacity(_overlayOpacity);
    baseViewer->setOverlayColormap(_overlayColormapId);
    baseViewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);

    if (_segmentationModule && adaptiveViewer) {
        _segmentationModule->attachViewer(adaptiveViewer);
    }
    if (adaptiveViewer) {
        emit viewerCreated(adaptiveViewer);
    }
    emit baseViewerCreated(baseViewer);
    return baseViewer;
}

void ViewerManager::registerOverlay(ViewerOverlayControllerBase* overlay)
{
    if (!overlay) {
        return;
    }
    if (std::find(_allOverlays.begin(), _allOverlays.end(), overlay) != _allOverlays.end()) {
        return;
    }
    _allOverlays.push_back(overlay);
    overlay->bindToViewerManager(this);
}

void ViewerManager::setSegmentationOverlay(SegmentationOverlayController* overlay)
{
    _segmentationOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setSegmentationEditActive(bool active)
{
    _segmentationEditActive = active;
    forEachBaseViewer([active](VolumeViewerBase* v) { v->setSegmentationEditActive(active); });
}

void ViewerManager::setSegmentationModule(SegmentationModule* module)
{
    _segmentationModule = module;
    if (!_segmentationModule) {
        return;
    }

    forEachViewer([this](CTiledVolumeViewer* v) { _segmentationModule->attachViewer(v); });
}

void ViewerManager::setPointsOverlay(PointsOverlayController* overlay)
{
    _pointsOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setRawPointsOverlay(RawPointsOverlayController* overlay)
{
    _rawPointsOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setPathsOverlay(PathsOverlayController* overlay)
{
    _pathsOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setBBoxOverlay(BBoxOverlayController* overlay)
{
    _bboxOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setVectorOverlay(VectorOverlayController* overlay)
{
    _vectorOverlay = overlay;
    registerOverlay(overlay);
}

void ViewerManager::setVolumeOverlay(VolumeOverlayController* overlay)
{
    _volumeOverlay = overlay;
    if (_volumeOverlay) {
        _volumeOverlay->syncWindowFromManager(_overlayWindowLow, _overlayWindowHigh);
    }
}

void ViewerManager::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_OPACITY,
                      static_cast<int>(std::lround(_intersectionOpacity * 100.0f)));

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setIntersectionOpacity(_intersectionOpacity); });
}

void ViewerManager::setIntersectionThickness(float thickness)
{
    const float clamped = std::clamp(thickness, 0.0f, 100.0f);
    if (std::abs(clamped - _intersectionThickness) < 1e-6f) {
        return;
    }
    _intersectionThickness = clamped;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_THICKNESS, _intersectionThickness);

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setIntersectionThickness(_intersectionThickness); });
}

void ViewerManager::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    forEachBaseViewer([&ids](VolumeViewerBase* v) { v->setHighlightedSurfaceIds(ids); });
}

void ViewerManager::setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId)
{
    _overlayVolume = std::move(volume);
    _overlayVolumeId = volumeId;
    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayVolume(_overlayVolume); });

    emit overlayVolumeAvailabilityChanged(static_cast<bool>(_overlayVolume));
}

void ViewerManager::setOverlayOpacity(float opacity)
{
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayOpacity(_overlayOpacity); });
}

void ViewerManager::setOverlayColormap(const std::string& colormapId)
{
    _overlayColormapId = colormapId;
    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayColormap(_overlayColormapId); });
}

void ViewerManager::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void ViewerManager::setOverlayWindow(float low, float high)
{
    constexpr float kMaxOverlayValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxOverlayValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxOverlayValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxOverlayValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;

    if (_volumeOverlay) {
        _volumeOverlay->syncWindowFromManager(_overlayWindowLow, _overlayWindowHigh);
    }

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh); });

    emit overlayWindowChanged(_overlayWindowLow, _overlayWindowHigh);
}

void ViewerManager::setVolumeWindow(float low, float high)
{
    constexpr float kMaxValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _volumeWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _volumeWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _volumeWindowLow = clampedLow;
    _volumeWindowHigh = clampedHigh;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::BASE_WINDOW_LOW, _volumeWindowLow);
    settings.setValue(vc3d::settings::viewer::BASE_WINDOW_HIGH, _volumeWindowHigh);

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh); });

    emit volumeWindowChanged(_volumeWindowLow, _volumeWindowHigh);
}

void ViewerManager::setSurfacePatchSamplingStride(int stride, bool userInitiated)
{
    stride = std::max(1, stride);
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (userInitiated) {
        settings.setValue(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE, _surfacePatchSamplingStride);
        settings.setValue(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE_USER_SET, true);
    }

    if (_surfacePatchIndex.setSamplingStride(_surfacePatchSamplingStride)) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.clear();
        // Index was cleared — remove stale intersection lines immediately.
        // New lines will appear once the async rebuild completes.
        forEachBaseViewer([](VolumeViewerBase* v) { v->invalidateIntersect(); });
    }

    forEachBaseViewer([this](VolumeViewerBase* v) { v->setSurfacePatchSamplingStride(_surfacePatchSamplingStride); });

    emit samplingStrideChanged(_surfacePatchSamplingStride);
}

void ViewerManager::setIntersectionMaxSurfaces(int limit)
{
    limit = std::max(0, limit);
    if (_intersectionMaxSurfaces == limit) {
        return;
    }
    _intersectionMaxSurfaces = limit;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_MAX_SURFACES, limit);
}

SurfacePatchIndex* ViewerManager::surfacePatchIndex()
{
    rebuildSurfacePatchIndexIfNeeded();
    if (_surfacePatchIndex.empty()) {
        return nullptr;
    }
    return &_surfacePatchIndex;
}

SurfacePatchIndex* ViewerManager::surfacePatchIndexIfReady()
{
    if (_surfacePatchIndex.empty()) {
        return nullptr;
    }
    return &_surfacePatchIndex;
}

void ViewerManager::refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface)
{
    if (!surface) {
        return;
    }
    const std::string surfId = surface->id;
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.erase(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Deferred surface index refresh for" << surfId.c_str()
                                << "(global rebuild pending)";
        return;
    }

    if (_surfacePatchIndex.updateSurface(surface)) {
        _indexedSurfaceIds.insert(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Rebuilt SurfacePatchIndex entries for surface" << surfId.c_str();
        return;
    }

    _surfacePatchIndexNeedsRebuild = true;
    _indexedSurfaceIds.erase(surfId);
    VC3D_DEBUG_QCINFO(lcViewerManager) << "Failed to rebuild SurfacePatchIndex for surface" << surfId.c_str()
                            << "- marking index for rebuild";
}

void ViewerManager::refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface, const cv::Rect& changedRegion)
{
    if (!surface) {
        return;
    }

    // Empty rect means no changes
    if (changedRegion.empty()) {
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Skipped SurfacePatchIndex update (no changes)";
        return;
    }

    const std::string surfId = surface->id;
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.erase(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Deferred surface index refresh for" << surfId.c_str()
                                << "(global rebuild pending)";
        return;
    }

    // Use region-based update
    const int rowStart = changedRegion.y;
    const int rowEnd = changedRegion.y + changedRegion.height;
    const int colStart = changedRegion.x;
    const int colEnd = changedRegion.x + changedRegion.width;

    if (_surfacePatchIndex.updateSurfaceRegion(surface, rowStart, rowEnd, colStart, colEnd)) {
        _indexedSurfaceIds.insert(surfId);
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Updated SurfacePatchIndex region for" << surfId.c_str()
                                << "rows" << rowStart << "-" << rowEnd
                                << "cols" << colStart << "-" << colEnd;
        return;
    }

    // Region update failed, fall back to full surface update
    refreshSurfacePatchIndex(surface);
}

void ViewerManager::primeSurfacePatchIndicesAsync()
{
    if (!_surfacePatchIndexWatcher) {
        return;
    }
    if (_surfacePatchIndexWatcher->isRunning()) {
        _surfacePatchIndexWatcher->cancel();
    }
    if (!_state) {
        return;
    }
    auto allSurfaces = _state->surfaces();
    std::vector<SurfacePatchIndex::SurfacePtr> quadSurfaces;
    std::vector<std::string> surfaceIds;
    quadSurfaces.reserve(allSurfaces.size());
    surfaceIds.reserve(allSurfaces.size());
    // Track seen surfaces to avoid duplicates (e.g., "segmentation" alias)
    std::unordered_set<SurfacePatchIndex::SurfacePtr> seenSurfaces;
    for (const auto& surface : allSurfaces) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surface)) {
            // Skip if we've already seen this surface (shared_ptr hash uses underlying pointer)
            if (seenSurfaces.insert(quad).second) {
                quadSurfaces.push_back(quad);
                surfaceIds.push_back(surface->id);
            }
        }
    }
    // Apply max surfaces limit
    if (_intersectionMaxSurfaces > 0 && quadSurfaces.size() > static_cast<size_t>(_intersectionMaxSurfaces)) {
        quadSurfaces.resize(_intersectionMaxSurfaces);
        surfaceIds.resize(_intersectionMaxSurfaces);
    }
    _pendingSurfacePatchIndexSurfaceIds = surfaceIds;
    if (quadSurfaces.empty()) {
        _surfacePatchIndex.clear();
        _indexedSurfaceIds.clear();
        _surfacePatchIndexNeedsRebuild = false;
        return;
    }

    // Clear rebuild flag since we're about to do an async build
    // (prevents rebuildSurfacePatchIndexIfNeeded from triggering a synchronous build)
    _surfacePatchIndexNeedsRebuild = false;

    // Clear any surfaces queued from a previous rebuild cycle
    _surfacesQueuedDuringRebuild.clear();

    // Build task captures shared_ptrs - surfaces stay alive throughout async operation
    const int stride = _surfacePatchSamplingStride;
    auto future = QtConcurrent::run([quadSurfaces, stride]() -> std::shared_ptr<SurfacePatchIndex> {
        try {
            auto index = std::make_shared<SurfacePatchIndex>();
            index->setSamplingStride(stride);
            index->rebuild(quadSurfaces);
            return index;
        } catch (const std::exception& e) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex async rebuild failed:" << e.what();
        } catch (...) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex async rebuild failed with an unknown exception";
        }
        return nullptr;
    });
    _surfacePatchIndexWatcher->setFuture(future);
}

void ViewerManager::rebuildSurfacePatchIndexIfNeeded()
{
    if (!_surfacePatchIndexNeedsRebuild) {
        return;
    }
    // Called from the render hot path via surfacePatchIndex(). Do not do
    // a synchronous rtree rebuild here — a 2K² surface takes seconds and
    // would freeze the GUI. primeSurfacePatchIndicesAsync() clears the
    // flag inside itself; readers will see the stale-but-valid index
    // until the worker swap completes.
    primeSurfacePatchIndicesAsync();
}

void ViewerManager::handleSurfacePatchIndexPrimeFinished()
{
    if (!_surfacePatchIndexWatcher) {
        return;
    }
    auto result = _surfacePatchIndexWatcher->future().result();
    if (!result) {
        _pendingSurfacePatchIndexSurfaceIds.clear();
        return;
    }
    _surfacePatchIndex = std::move(*result);
    _surfacePatchIndexNeedsRebuild = false;
    _indexedSurfaceIds.clear();
    _indexedSurfaceIds.insert(_pendingSurfacePatchIndexSurfaceIds.begin(),
                              _pendingSurfacePatchIndexSurfaceIds.end());

    auto queuedDuringRebuild = std::move(_surfacesQueuedDuringRebuild);
    _surfacesQueuedDuringRebuild.clear();

    VC3D_DEBUG_QCINFO(lcViewerManager) << "Asynchronously rebuilt SurfacePatchIndex for"
                            << _indexedSurfaceIds.size() << "surfaces"
                            << "at stride" << _surfacePatchSamplingStride;

    _pendingSurfacePatchIndexSurfaceIds.clear();

    // Surfaces added/removed while the full async rebuild was running do not
    // require another full rebuild. Apply just those deltas on the worker.
    if (queuedDuringRebuild.empty()) {
        forEachBaseViewer([](VolumeViewerBase* v) { v->renderIntersections(); });
    } else {
        forEachBaseViewer([](VolumeViewerBase* v) { v->invalidateIntersect(); });
        for (auto& task : queuedDuringRebuild) {
            queueSurfacePatchIndexTask(std::move(task));
        }
    }
}

void ViewerManager::queueSurfacePatchIndexTask(SurfacePatchIndexTask task)
{
    if (!task.surface) {
        return;
    }

    const QuadSurface* raw = task.surface.get();
    if (task.type == SurfacePatchIndexTaskType::Remove) {
        _indexedSurfaceIds.erase(task.id);
        _pendingSurfacePatchIndexTasks.erase(
            std::remove_if(_pendingSurfacePatchIndexTasks.begin(),
                           _pendingSurfacePatchIndexTasks.end(),
                           [raw](const SurfacePatchIndexTask& pending) {
                               return pending.surface.get() == raw;
                           }),
            _pendingSurfacePatchIndexTasks.end());
    } else {
        for (auto& pending : _pendingSurfacePatchIndexTasks) {
            if (pending.type == SurfacePatchIndexTaskType::Update &&
                pending.surface.get() == raw) {
                pending = std::move(task);
                startNextSurfacePatchIndexTask();
                return;
            }
        }
    }

    _pendingSurfacePatchIndexTasks.push_back(std::move(task));
    startNextSurfacePatchIndexTask();
}

void ViewerManager::startNextSurfacePatchIndexTask()
{
    if (!_surfacePatchIndexTaskWatcher ||
        _surfacePatchIndexTaskWatcher->isRunning() ||
        _pendingSurfacePatchIndexTasks.empty()) {
        return;
    }

    SurfacePatchIndexTask task = std::move(_pendingSurfacePatchIndexTasks.front());
    _pendingSurfacePatchIndexTasks.erase(_pendingSurfacePatchIndexTasks.begin());

    auto* index = &_surfacePatchIndex;
    auto future = QtConcurrent::run([index, task = std::move(task)]() mutable -> SurfacePatchIndexTaskResult {
        SurfacePatchIndexTaskResult result;
        result.type = task.type;
        result.id = std::move(task.id);
        result.surface = std::move(task.surface);

        try {
            if (result.type == SurfacePatchIndexTaskType::Update) {
                result.success = index->updateSurface(result.surface);
            } else {
                result.success = index->removeSurface(result.surface);
            }
        } catch (const std::exception& e) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex single-surface task failed:" << e.what();
        } catch (...) {
            qCWarning(lcViewerManager) << "SurfacePatchIndex single-surface task failed with an unknown exception";
        }

        return result;
    });
    _surfacePatchIndexTaskWatcher->setFuture(future);
}

void ViewerManager::handleSurfacePatchIndexTaskFinished()
{
    if (!_surfacePatchIndexTaskWatcher) {
        return;
    }

    const auto result = _surfacePatchIndexTaskWatcher->future().result();
    if (result.success) {
        if (result.type == SurfacePatchIndexTaskType::Update) {
            _indexedSurfaceIds.insert(result.id);
            VC3D_DEBUG_QCINFO(lcViewerManager) << "Updated SurfacePatchIndex for surface"
                                               << result.id.c_str();
            forEachBaseViewer([](VolumeViewerBase* v) { v->renderIntersections(); });
        } else {
            _indexedSurfaceIds.erase(result.id);
            VC3D_DEBUG_QCINFO(lcViewerManager) << "Removed surface from SurfacePatchIndex"
                                               << result.id.c_str();
            forEachBaseViewer([](VolumeViewerBase* v) {
                v->invalidateIntersect();
                v->renderIntersections();
            });
        }
    } else if (result.type == SurfacePatchIndexTaskType::Update) {
        _indexedSurfaceIds.erase(result.id);
        _surfacePatchIndexNeedsRebuild = true;
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Failed to update SurfacePatchIndex for surface"
                                           << result.id.c_str()
                                           << "- marking index for rebuild";
    }

    startNextSurfacePatchIndexTask();
}

bool ViewerManager::updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool isEditUpdate)
{
    if (!quad) {
        return false;
    }

    const std::string surfId = quad->id;
    const bool alreadyIndexed = _surfacePatchIndex.containsSurface(quad);

    // Check if async rebuild is in progress
    const bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                        _surfacePatchIndexWatcher->isRunning();

    // Editing tools queue the exact touched cells as vertices move. Flush those
    // cells into the current index immediately so plane intersections update
    // without turning every brush/push-pull tick into a global async rebuild.
    if (_surfacePatchIndex.hasPendingUpdates(quad)) {
        const bool flushed = _surfacePatchIndex.flushPendingUpdates(quad);
        if (flushed) {
            _indexedSurfaceIds.insert(surfId);
        }
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild && !flushed;
        return flushed || isEditUpdate;
    }

    // Non-edit surfaceChanged signals are also used for UI alias/selection
    // updates, especially the "segmentation" alias. If the surface is already
    // present in the index and no cell updates are pending, there is no rtree
    // work to do.
    if (!isEditUpdate && alreadyIndexed) {
        _indexedSurfaceIds.insert(surfId);
        return true;
    }

    if (isEditUpdate && alreadyIndexed) {
        return true;
    }

    if (isEditUpdate && !_surfacePatchIndex.empty()) {
        if (_surfacePatchIndex.updateSurface(quad)) {
            _indexedSurfaceIds.insert(surfId);
            if (asyncRebuildInProgress) {
                _surfacesQueuedDuringRebuild.push_back(
                    {SurfacePatchIndexTaskType::Update, surfId, quad});
            }
            VC3D_DEBUG_QCINFO(lcViewerManager) << "Inserted active edit surface into SurfacePatchIndex"
                                    << surfId.c_str();
            return true;
        }
        _indexedSurfaceIds.erase(surfId);
        _surfacePatchIndexNeedsRebuild = true;
        VC3D_DEBUG_QCINFO(lcViewerManager) << "Failed to insert active edit surface into SurfacePatchIndex"
                                << surfId.c_str() << "- marking index for rebuild";
        return false;
    }

    if (asyncRebuildInProgress) {
        // An async rebuild is already running. Apply this surface as a
        // single-surface delta after the rebuilt index is swapped in.
        _surfacesQueuedDuringRebuild.push_back(
            {SurfacePatchIndexTaskType::Update, surfId, quad});
        return true;
    }

    if (!_surfacePatchIndex.empty()) {
        queueSurfacePatchIndexTask(
            {SurfacePatchIndexTaskType::Update, surfId, quad});
        return true;
    }

    _surfacePatchIndexNeedsRebuild = true;
    return true;
}

void ViewerManager::handleSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate)
{
    bool affectsSurfaceIndex = false;
    bool regionUpdated = false;

    if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
        affectsSurfaceIndex = true;
        if (updateSurfacePatchIndexForSurface(quad, isEditUpdate)) {
            regionUpdated = true;  // Signal that work was done (prevents marking index for rebuild)
        }
    } else if (!surf) {
        // Surface was removed - the handleSurfaceWillBeDeleted already cleaned up the index
        affectsSurfaceIndex = true;
        regionUpdated = true;  // Incremental removal already done - don't trigger full rebuild
        _indexedSurfaceIds.erase(name);
    }

    if (affectsSurfaceIndex) {
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild || !regionUpdated;
    }
}

void ViewerManager::handleSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf)
{
    // Fast path on app shutdown: don't bother maintaining the rtree when
    // everything is about to be freed anyway. A single large surface would
    // otherwise cost ~11s of per-cell tree->remove() while the user stares
    // at a frozen window.
    if (_shuttingDown.load(std::memory_order_relaxed)) {
        return;
    }

    // Called BEFORE surface deletion - remove from R-tree index
    auto quad = std::dynamic_pointer_cast<QuadSurface>(surf);

    // Only process cleanup if we're deleting under the surface's actual ID.
    // Aliases like "segmentation" just point to surfaces that exist under their
    // own IDs - we don't want to remove from the index when an alias changes.
    const bool isDeletingByActualId = quad && (name == quad->id);

    if (isDeletingByActualId) {
        // Track whether this surface was ever actually indexed. If not,
        // the R-tree has nothing to remove — skip the whole mask walk.
        const bool wasIndexed = (_indexedSurfaceIds.find(name)
                                 != _indexedSurfaceIds.end());

        // Remove from indexed surface IDs
        _indexedSurfaceIds.erase(name);

        // Remove from queued-for-add IDs
        auto removeFromVector = [&name](std::vector<std::string>& vec) {
            vec.erase(std::remove(vec.begin(), vec.end(), name), vec.end());
        };
        removeFromVector(_pendingSurfacePatchIndexSurfaceIds);
        _surfacesQueuedDuringRebuild.erase(
            std::remove_if(_surfacesQueuedDuringRebuild.begin(),
                           _surfacesQueuedDuringRebuild.end(),
                           [&name](const SurfacePatchIndexTask& task) {
                               return task.id == name;
                           }),
            _surfacesQueuedDuringRebuild.end());

        // If an async rebuild is in progress, queue for removal from the new
        // index when it completes. Store the shared_ptr so the surface stays
        // alive for the R-tree removal even after CState drops it.
        bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                       _surfacePatchIndexWatcher->isRunning();
        if (asyncRebuildInProgress) {
            _surfacesQueuedDuringRebuild.push_back(
                {SurfacePatchIndexTaskType::Remove, name, quad});
        } else if (wasIndexed) {
            queueSurfacePatchIndexTask(
                {SurfacePatchIndexTaskType::Remove, name, quad});
        } else {
            std::fprintf(stderr,
                "[ViewerManager::handleSurfaceWillBeDeleted] name=%s skipping "
                "removeSurface (never indexed)\n", name.c_str());
        }

        if (asyncRebuildInProgress || wasIndexed) {
            // Hide stale lines immediately; the async removal will update the
            // R-tree before intersections are rendered again.
            forEachBaseViewer([](VolumeViewerBase* v) {
                v->invalidateIntersect();
            });
        }
    }
}

bool ViewerManager::resetDefaultFor(VolumeViewerBase* viewer) const
{
    auto it = _resetDefaults.find(viewer);
    return it != _resetDefaults.end() ? it->second : true;
}

void ViewerManager::setResetDefaultFor(VolumeViewerBase* viewer, bool value)
{
    if (!viewer) {
        return;
    }
    _resetDefaults[viewer] = value;
}

void ViewerManager::setSegmentationCursorMirroring(bool enabled)
{
    _mirrorCursorToSegmentation = enabled;
    forEachBaseViewer([enabled](VolumeViewerBase* v) { v->setSegmentationCursorMirroring(enabled); });
}

void ViewerManager::broadcastLinkedCursor(VolumeViewerBase* source,
                                          const std::optional<cv::Vec3f>& point)
{
    forEachBaseViewer([source, &point](VolumeViewerBase* viewer) {
        if (viewer != source) {
            viewer->setLinkedCursorVolumePoint(point);
        }
    });
}

void ViewerManager::setSliceStepSize(int size)
{
    _sliceStepSize = std::max(1, size);
}

void ViewerManager::forEachViewer(const std::function<void(CTiledVolumeViewer*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _viewers) {
        if (viewer) {
            fn(viewer);
        }
    }
}

void ViewerManager::forEachBaseViewer(const std::function<void(VolumeViewerBase*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _baseViewers) {
        if (viewer) {
            fn(viewer);
        }
    }
}
