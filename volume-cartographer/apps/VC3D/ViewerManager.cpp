#include "ViewerManager.hpp"

#include "VCSettings.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"
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

#include <QMdiArea>
#include <QThread>
#include <QMdiSubWindow>
#include <QSettings>
#include <QtConcurrent/QtConcurrent>
#include <QLoggingCategory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <unordered_set>
#include "utils/Json.hpp"
#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"

Q_LOGGING_CATEGORY(lcViewerManager, "vc.viewer.manager")


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

    const int storedSampling = settings.value(viewer::INTERSECTION_SAMPLING_STRIDE, viewer::INTERSECTION_SAMPLING_STRIDE_DEFAULT).toInt();
    _surfacePatchSamplingStride = std::max(1, storedSampling);
    const float storedThickness = settings.value(viewer::INTERSECTION_THICKNESS, viewer::INTERSECTION_THICKNESS_DEFAULT).toFloat();
    _intersectionThickness = std::max(0.0f, storedThickness);
    _intersectionMaxSurfaces = std::max(0, settings.value(viewer::INTERSECTION_MAX_SURFACES, viewer::INTERSECTION_MAX_SURFACES_DEFAULT).toInt());

    _surfacePatchIndexWatcher =
        new QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>(this);
    connect(_surfacePatchIndexWatcher,
            &QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>::finished,
            this,
            &ViewerManager::handleSurfacePatchIndexPrimeFinished);

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

CTiledVolumeViewer* ViewerManager::createViewer(const std::string& surfaceName,
                                                const QString& title,
                                                QMdiArea* mdiArea)
{
    if (!mdiArea || !_state) {
        return nullptr;
    }

    auto* viewer = new CTiledVolumeViewer(_state, this, mdiArea);
    auto* win = mdiArea->addSubWindow(viewer);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::WindowTitleHint | Qt::WindowMinMaxButtonsHint);
    win->installEventFilter(viewer);

    viewer->setPointCollection(_points);

    if (_state) {
        connect(_state, &CState::surfaceChanged, viewer, &CTiledVolumeViewer::onSurfaceChanged);
        connect(_state, &CState::surfaceWillBeDeleted, viewer, &CTiledVolumeViewer::onSurfaceWillBeDeleted);
        connect(_state, &CState::poiChanged, viewer, &CTiledVolumeViewer::onPOIChanged);
    }

    // Restore persisted viewer preferences
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showHints = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        viewer->setShowDirectionHints(showHints);
        bool showNormals = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        viewer->setShowSurfaceNormals(showNormals);
    }

    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool resetView = settings.value(viewer::RESET_VIEW_ON_SURFACE_CHANGE, viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
        viewer->setResetViewOnSurfaceChange(resetView);
        _resetDefaults[viewer] = resetView;
    }

    viewer->setSurface(surfaceName);
    viewer->setSegmentationEditActive(_segmentationEditActive);
    viewer->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);

    _viewers.push_back(viewer);

    // Clean up when viewer is destroyed (e.g. MDI sub-window closed)
    connect(viewer, &QObject::destroyed, this, [this, viewer]() {
        _viewers.erase(std::remove(_viewers.begin(), _viewers.end(), viewer), _viewers.end());
        _resetDefaults.erase(viewer);
    });

    for (auto* overlay : _allOverlays) {
        overlay->attachViewer(viewer);
    }

    viewer->setIntersectionOpacity(_intersectionOpacity);
    viewer->setIntersectionThickness(_intersectionThickness);
    viewer->setSurfacePatchSamplingStride(_surfacePatchSamplingStride);
    viewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
    viewer->setOverlayVolume(_overlayVolume);
    viewer->setOverlayOpacity(_overlayOpacity);
    viewer->setOverlayColormap(_overlayColormapId);
    viewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);

    if (_segmentationModule) {
        _segmentationModule->attachViewer(viewer);
    }
    emit viewerCreated(viewer);
    return viewer;
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
    forEachViewer([active](CTiledVolumeViewer* v) { v->setSegmentationEditActive(active); });
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

    forEachViewer([this](CTiledVolumeViewer* v) { v->setIntersectionOpacity(_intersectionOpacity); });
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

    forEachViewer([this](CTiledVolumeViewer* v) { v->setIntersectionThickness(_intersectionThickness); });
}

void ViewerManager::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    forEachViewer([&ids](CTiledVolumeViewer* v) { v->setHighlightedSurfaceIds(ids); });
}

void ViewerManager::setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId)
{
    _overlayVolume = std::move(volume);
    _overlayVolumeId = volumeId;
    forEachViewer([this](CTiledVolumeViewer* v) { v->setOverlayVolume(_overlayVolume); });

    emit overlayVolumeAvailabilityChanged(static_cast<bool>(_overlayVolume));
}

void ViewerManager::setOverlayOpacity(float opacity)
{
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    forEachViewer([this](CTiledVolumeViewer* v) { v->setOverlayOpacity(_overlayOpacity); });
}

void ViewerManager::setOverlayColormap(const std::string& colormapId)
{
    _overlayColormapId = colormapId;
    forEachViewer([this](CTiledVolumeViewer* v) { v->setOverlayColormap(_overlayColormapId); });
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

    forEachViewer([this](CTiledVolumeViewer* v) { v->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh); });

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

    forEachViewer([this](CTiledVolumeViewer* v) { v->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh); });

    emit volumeWindowChanged(_volumeWindowLow, _volumeWindowHigh);
}

void ViewerManager::setSurfacePatchSamplingStride(int stride, bool userInitiated)
{
    stride = std::max(1, stride);
    if (userInitiated) {
        _surfacePatchStrideUserSet = true;
        _targetRefinedStride = 0;
    }
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE, _surfacePatchSamplingStride);

    if (_surfacePatchIndex.setSamplingStride(_surfacePatchSamplingStride)) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.clear();
        // Index was cleared — remove stale intersection lines immediately.
        // New lines will appear once the async rebuild completes.
        forEachViewer([](CTiledVolumeViewer* v) { v->invalidateIntersect(); });
    }

    forEachViewer([this](CTiledVolumeViewer* v) { v->setSurfacePatchSamplingStride(_surfacePatchSamplingStride); });

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

void ViewerManager::refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface)
{
    if (!surface) {
        return;
    }
    const std::string surfId = surface->id;
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.erase(surfId);
        qCInfo(lcViewerManager) << "Deferred surface index refresh for" << surfId.c_str()
                                << "(global rebuild pending)";
        return;
    }

    if (_surfacePatchIndex.updateSurface(surface)) {
        _indexedSurfaceIds.insert(surfId);
        qCInfo(lcViewerManager) << "Rebuilt SurfacePatchIndex entries for surface" << surfId.c_str();
        return;
    }

    _surfacePatchIndexNeedsRebuild = true;
    _indexedSurfaceIds.erase(surfId);
    qCInfo(lcViewerManager) << "Failed to rebuild SurfacePatchIndex for surface" << surfId.c_str()
                            << "- marking index for rebuild";
}

void ViewerManager::refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface, const cv::Rect& changedRegion)
{
    if (!surface) {
        return;
    }

    // Empty rect means no changes
    if (changedRegion.empty()) {
        qCInfo(lcViewerManager) << "Skipped SurfacePatchIndex update (no changes)";
        return;
    }

    const std::string surfId = surface->id;
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaceIds.erase(surfId);
        qCInfo(lcViewerManager) << "Deferred surface index refresh for" << surfId.c_str()
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
        qCInfo(lcViewerManager) << "Updated SurfacePatchIndex region for" << surfId.c_str()
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

    // Apply tiered default stride based on surface count (if not user-set)
    const size_t surfaceCount = quadSurfaces.size();
    _targetRefinedStride = 0;  // Reset refinement target

    if (!_surfacePatchStrideUserSet) {
        // TODO: re-enable finer strides (2, and eventually 1) once
        // SurfacePatchIndex rebuild + rtree mutation are cheap enough that
        // the higher entry count doesn't stall the GUI. Right now 4 is the
        // floor across every tier — lower strides produce millions of
        // entries on 2K² surfaces for no visible win in intersection
        // drawing.
        int defaultStride;
        int refinedStride;
        if (surfaceCount > 2500) {
            defaultStride = 32;
            refinedStride = 16;
        } else if (surfaceCount >= 500) {
            defaultStride = 16;
            refinedStride = 8;
        } else if (surfaceCount >= 100) {
            defaultStride = 8;
            refinedStride = 4;
        } else if (surfaceCount >= 30) {
            defaultStride = 4;
            refinedStride = 4;
        } else {
            defaultStride = 4;
            refinedStride = 4;
        }

        // Choose the coarse auto stride once per surface-count tier, then
        // allow the completed async build to refine to refinedStride. Without
        // this guard, every later prime resets the stride to defaultStride and
        // handleSurfacePatchIndexPrimeFinished() immediately refines again,
        // causing an endless rebuild loop.
        const bool autoTierChanged =
            !_surfacePatchAutoStrideInitialized ||
            _surfacePatchAutoDefaultStride != defaultStride;
        if (autoTierChanged) {
            _surfacePatchAutoStrideInitialized = true;
            _surfacePatchAutoDefaultStride = defaultStride;
            setSurfacePatchSamplingStride(defaultStride, false);
            if (defaultStride > refinedStride) {
                _targetRefinedStride = refinedStride;
            }
        } else if (_surfacePatchSamplingStride == defaultStride &&
                   defaultStride > refinedStride) {
            _targetRefinedStride = refinedStride;
        }
    }

    // Clear rebuild flag since we're about to do an async build
    // (prevents rebuildSurfacePatchIndexIfNeeded from triggering a synchronous build)
    _surfacePatchIndexNeedsRebuild = false;

    // Clear any surfaces queued from a previous rebuild cycle
    _surfacesQueuedDuringRebuildIds.clear();
    _surfacesQueuedForRemovalDuringRebuild.clear();

    // Build task captures shared_ptrs - surfaces stay alive throughout async operation
    const int stride = _surfacePatchSamplingStride;
    auto future = QtConcurrent::run([quadSurfaces, stride]() -> std::shared_ptr<SurfacePatchIndex> {
        auto index = std::make_shared<SurfacePatchIndex>();
        index->setSamplingStride(stride);
        index->rebuild(quadSurfaces);
        return index;
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

    // If surfaces were added or removed during the async rebuild, re-prime
    // instead of doing incremental remove/update on the main thread. The
    // rebuild runs on a worker thread; per-surface rtree mutations here
    // were a measurable main-thread stall (~14% of profile) on segments
    // with large coords grids.
    const bool queuesDirty = !_surfacesQueuedForRemovalDuringRebuild.empty()
                          || !_surfacesQueuedDuringRebuildIds.empty();
    _surfacesQueuedForRemovalDuringRebuild.clear();
    _surfacesQueuedDuringRebuildIds.clear();

    qCInfo(lcViewerManager) << "Asynchronously rebuilt SurfacePatchIndex for"
                            << _indexedSurfaceIds.size() << "surfaces"
                            << "at stride" << _surfacePatchSamplingStride;
    forEachViewer([](CTiledVolumeViewer* v) { v->renderIntersections(); });

    // Check if progressive refinement is needed. Queued changes also fall
    // through this path by re-priming at the (possibly same) current stride.
    const bool refineRequested =
        _targetRefinedStride > 0 && _surfacePatchSamplingStride > _targetRefinedStride;
    if (refineRequested || queuesDirty) {
        if (refineRequested) {
            qCInfo(lcViewerManager) << "Starting progressive refinement from stride"
                                    << _surfacePatchSamplingStride << "to" << _targetRefinedStride;
            const int targetStride = _targetRefinedStride;
            _targetRefinedStride = 0;  // Clear target to prevent infinite loop
            setSurfacePatchSamplingStride(targetStride, false);
        }
        _pendingSurfacePatchIndexSurfaceIds.clear();
        primeSurfacePatchIndicesAsync();
    } else {
        _pendingSurfacePatchIndexSurfaceIds.clear();
    }
}

bool ViewerManager::updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool isEditUpdate)
{
    if (!quad) {
        return false;
    }

    const std::string surfId = quad->id;
    const bool alreadyIndexed = _indexedSurfaceIds.count(surfId) != 0;

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

    if (isEditUpdate && alreadyIndexed) {
        return true;
    }

    if (asyncRebuildInProgress) {
        // An async rebuild is already running; tell it to re-prime when it
        // finishes so this surface's changes land without blocking main.
        _surfacesQueuedDuringRebuildIds.push_back(surfId);
        return true;
    }

    // No async work in flight. Previously we ran updateSurface /
    // flushPendingUpdates / insertCells synchronously on the main thread —
    // a full rtree insert pass for a large segment costs ~3% of a frame
    // and shows up as a visible GUI stall when many surfaces load in
    // quick succession. Defer to the async primer instead: mark dirty and
    // kick off a worker-thread rebuild. Main stays responsive; the new
    // surface's entries become visible one rebuild cycle later.
    _indexedSurfaceIds.erase(surfId);  // will be re-populated by async primer
    _surfacePatchIndexNeedsRebuild = true;
    primeSurfacePatchIndicesAsync();
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
        removeFromVector(_surfacesQueuedDuringRebuildIds);

        // If an async rebuild is in progress, queue for removal from the new
        // index when it completes. Store the shared_ptr so the surface stays
        // alive for the R-tree removal even after CState drops it.
        bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                       _surfacePatchIndexWatcher->isRunning();
        if (asyncRebuildInProgress) {
            _surfacesQueuedForRemovalDuringRebuild.emplace_back(name, surf);
        }

        if (wasIndexed) {
            // Remove from the current R-tree index
            _surfacePatchIndex.removeSurface(quad);
        } else {
            std::fprintf(stderr,
                "[ViewerManager::handleSurfaceWillBeDeleted] name=%s skipping "
                "removeSurface (never indexed)\n", name.c_str());
        }

        // Invalidate intersection lines on all viewers so stale lines from the
        // deleted surface don't persist on screen.
        forEachViewer([](CTiledVolumeViewer* v) {
            v->invalidateIntersect();
            v->renderIntersections();
        });
    }
}

bool ViewerManager::resetDefaultFor(CTiledVolumeViewer* viewer) const
{
    auto it = _resetDefaults.find(viewer);
    return it != _resetDefaults.end() ? it->second : true;
}

void ViewerManager::setResetDefaultFor(CTiledVolumeViewer* viewer, bool value)
{
    if (!viewer) {
        return;
    }
    _resetDefaults[viewer] = value;
}

void ViewerManager::setSegmentationCursorMirroring(bool enabled)
{
    _mirrorCursorToSegmentation = enabled;
    forEachViewer([enabled](CTiledVolumeViewer* v) { v->setSegmentationCursorMirroring(enabled); });
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
