#include "ViewerManager.hpp"

#include "VCSettings.hpp"
#include "CVolumeViewer.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/Volume.hpp"

#include <QMdiArea>
#include <QMdiSubWindow>
#include <QSettings>
#include <QtConcurrent/QtConcurrent>
#include <QLoggingCategory>
#include <algorithm>
#include <cmath>
#include <optional>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"

Q_LOGGING_CATEGORY(lcViewerManager, "vc.viewer.manager")

namespace {
struct CellRegion {
    int rowStart = 0;
    int rowEnd = 0;
    int colStart = 0;
    int colEnd = 0;
};

} // namespace

ViewerManager::ViewerManager(CSurfaceCollection* surfaces,
                             VCCollection* points,
                             ChunkCache<uint8_t>* cache,
                             QObject* parent)
    : QObject(parent)
    , _surfaces(surfaces)
    , _points(points)
    , _chunkCache(cache)
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

    _surfacePatchIndexWatcher =
        new QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>(this);
    connect(_surfacePatchIndexWatcher,
            &QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>::finished,
            this,
            &ViewerManager::handleSurfacePatchIndexPrimeFinished);

    if (_surfaces) {
        connect(_surfaces,
                &CSurfaceCollection::sendSurfaceChanged,
                this,
                &ViewerManager::handleSurfaceChanged);
        connect(_surfaces,
                &CSurfaceCollection::sendSurfaceWillBeDeleted,
                this,
                &ViewerManager::handleSurfaceWillBeDeleted);
    }
}

CVolumeViewer* ViewerManager::createViewer(const std::string& surfaceName,
                                           const QString& title,
                                           QMdiArea* mdiArea)
{
    if (!mdiArea || !_surfaces) {
        return nullptr;
    }

    auto* viewer = new CVolumeViewer(_surfaces, this, mdiArea);
    auto* win = mdiArea->addSubWindow(viewer);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::WindowTitleHint | Qt::WindowMinMaxButtonsHint);

    viewer->setCache(_chunkCache);
    viewer->setPointCollection(_points);

    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged, viewer, &CVolumeViewer::onSurfaceChanged);
        connect(_surfaces, &CSurfaceCollection::sendSurfaceWillBeDeleted, viewer, &CVolumeViewer::onSurfaceWillBeDeleted);
        connect(_surfaces, &CSurfaceCollection::sendPOIChanged, viewer, &CVolumeViewer::onPOIChanged);
    }

    // Restore persisted viewer preferences
    {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showHints = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        viewer->setShowDirectionHints(showHints);
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

    if (_segmentationOverlay) {
        _segmentationOverlay->attachViewer(viewer);
    }

    if (_pointsOverlay) {
        _pointsOverlay->attachViewer(viewer);
    }

    if (_pathsOverlay) {
        _pathsOverlay->attachViewer(viewer);
    }

    if (_bboxOverlay) {
        _bboxOverlay->attachViewer(viewer);
    }

    if (_vectorOverlay) {
        _vectorOverlay->attachViewer(viewer);
    }

    viewer->setIntersectionOpacity(_intersectionOpacity);
    viewer->setIntersectionThickness(_intersectionThickness);
    viewer->setSurfacePatchSamplingStride(_surfacePatchSamplingStride);
    viewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
    viewer->setOverlayVolume(_overlayVolume);
    viewer->setOverlayOpacity(_overlayOpacity);
    viewer->setOverlayColormap(_overlayColormapId);
    viewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);

    _viewers.push_back(viewer);
    if (_segmentationModule) {
        _segmentationModule->attachViewer(viewer);
    }
    emit viewerCreated(viewer);
    return viewer;
}

void ViewerManager::setSegmentationOverlay(SegmentationOverlayController* overlay)
{
    _segmentationOverlay = overlay;
    if (!_segmentationOverlay) {
        return;
    }
    _segmentationOverlay->bindToViewerManager(this);
}

void ViewerManager::setSegmentationEditActive(bool active)
{
    _segmentationEditActive = active;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setSegmentationEditActive(active);
        }
    }
}

void ViewerManager::setSegmentationModule(SegmentationModule* module)
{
    _segmentationModule = module;
    if (!_segmentationModule) {
        return;
    }

    for (auto* viewer : _viewers) {
        _segmentationModule->attachViewer(viewer);
    }
}

void ViewerManager::setPointsOverlay(PointsOverlayController* overlay)
{
    _pointsOverlay = overlay;
    if (!_pointsOverlay) {
        return;
    }
    _pointsOverlay->bindToViewerManager(this);
}

void ViewerManager::setPathsOverlay(PathsOverlayController* overlay)
{
    _pathsOverlay = overlay;
    if (!_pathsOverlay) {
        return;
    }
    _pathsOverlay->bindToViewerManager(this);
}

void ViewerManager::setBBoxOverlay(BBoxOverlayController* overlay)
{
    _bboxOverlay = overlay;
    if (!_bboxOverlay) {
        return;
    }
    _bboxOverlay->bindToViewerManager(this);
}

void ViewerManager::setVectorOverlay(VectorOverlayController* overlay)
{
    _vectorOverlay = overlay;
    if (!_vectorOverlay) {
        return;
    }
    _vectorOverlay->bindToViewerManager(this);
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

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setIntersectionOpacity(_intersectionOpacity);
        }
    }
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

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setIntersectionThickness(_intersectionThickness);
        }
    }
}

void ViewerManager::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setHighlightedSurfaceIds(ids);
        }
    }
}

void ViewerManager::setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId)
{
    _overlayVolume = std::move(volume);
    _overlayVolumeId = volumeId;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayVolume(_overlayVolume);
        }
    }

    emit overlayVolumeAvailabilityChanged(static_cast<bool>(_overlayVolume));
}

void ViewerManager::setOverlayOpacity(float opacity)
{
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayOpacity(_overlayOpacity);
        }
    }
}

void ViewerManager::setOverlayColormap(const std::string& colormapId)
{
    _overlayColormapId = colormapId;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayColormap(_overlayColormapId);
        }
    }
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

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);
        }
    }

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

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
        }
    }

    emit volumeWindowChanged(_volumeWindowLow, _volumeWindowHigh);
}

void ViewerManager::setSurfacePatchSamplingStride(int stride, bool userInitiated)
{
    stride = std::max(1, stride);
    if (userInitiated) {
        _surfacePatchStrideUserSet = true;
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
    }

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setSurfacePatchSamplingStride(_surfacePatchSamplingStride);
        }
    }

    emit samplingStrideChanged(_surfacePatchSamplingStride);
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

void ViewerManager::waitForPendingIndexRebuild()
{
    if (_surfacePatchIndexWatcher && _surfacePatchIndexWatcher->isRunning()) {
        _surfacePatchIndexWatcher->waitForFinished();
    }
}

void ViewerManager::primeSurfacePatchIndicesAsync()
{
    if (!_surfacePatchIndexWatcher) {
        return;
    }
    if (_surfacePatchIndexWatcher->isRunning()) {
        _surfacePatchIndexWatcher->waitForFinished();
    }
    if (!_surfaces) {
        return;
    }
    auto allSurfaces = _surfaces->surfaces();
    std::vector<SurfacePatchIndex::SurfacePtr> quadSurfaces;
    std::vector<std::string> surfaceIds;
    quadSurfaces.reserve(allSurfaces.size());
    surfaceIds.reserve(allSurfaces.size());
    for (const auto& surface : allSurfaces) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surface)) {
            quadSurfaces.push_back(quad);
            surfaceIds.push_back(surface->id);
        }
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
        int defaultStride;
        if (surfaceCount > 2500) {
            // > 2500: build at 8x initially, then refine to 4x
            defaultStride = 8;
            _targetRefinedStride = 4;
        } else if (surfaceCount >= 500) {
            // 500-2500: build at 4x initially, then refine to 2x
            defaultStride = 4;
            _targetRefinedStride = 2;
        } else {
            // < 500: build at 1x (full resolution), no progressive loading
            defaultStride = 1;
        }
        setSurfacePatchSamplingStride(defaultStride, false);
    }

    // Clear rebuild flag since we're about to do an async build
    // (prevents rebuildSurfacePatchIndexIfNeeded from triggering a synchronous build)
    _surfacePatchIndexNeedsRebuild = false;

    // Clear any surfaces queued from a previous rebuild cycle
    _surfacesQueuedDuringRebuildIds.clear();
    _surfacesQueuedForRemovalDuringRebuildIds.clear();

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
    _surfacePatchIndexNeedsRebuild = false;

    if (!_surfaces) {
        _surfacePatchIndex.clear();
        _indexedSurfaceIds.clear();
        qCInfo(lcViewerManager) << "SurfacePatchIndex cleared (no surface collection)";
        return;
    }

    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    std::vector<std::string> surfaceIds;
    for (const auto& surf : _surfaces->surfaces()) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            surfaces.push_back(quad);
            surfaceIds.push_back(surf->id);
        }
    }

    if (surfaces.empty()) {
        _surfacePatchIndex.clear();
        _indexedSurfaceIds.clear();
        qCInfo(lcViewerManager) << "SurfacePatchIndex cleared (no QuadSurfaces to index)";
        return;
    }

    qCInfo(lcViewerManager) << "Rebuilding SurfacePatchIndex for" << surfaces.size() << "surfaces";
    _surfacePatchIndex.rebuild(surfaces);
    _indexedSurfaceIds.clear();
    _indexedSurfaceIds.insert(surfaceIds.begin(), surfaceIds.end());
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

    // Process any surfaces that were removed during the async rebuild
    for (const std::string& idToRemove : _surfacesQueuedForRemovalDuringRebuildIds) {
        // Look up the surface by ID to remove from index
        auto surf = _surfaces ? _surfaces->surface(idToRemove) : nullptr;
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            _surfacePatchIndex.removeSurface(quad);
        }
        _indexedSurfaceIds.erase(idToRemove);
    }
    _surfacesQueuedForRemovalDuringRebuildIds.clear();

    // Merge any surfaces that were added during the async rebuild
    for (const std::string& queuedId : _surfacesQueuedDuringRebuildIds) {
        auto surf = _surfaces ? _surfaces->surface(queuedId) : nullptr;
        if (auto queued = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            if (_surfacePatchIndex.updateSurface(queued)) {
                _indexedSurfaceIds.insert(queuedId);
                qCInfo(lcViewerManager) << "Indexed queued surface" << queuedId.c_str()
                                        << "after async rebuild";
            }
        }
    }
    _surfacesQueuedDuringRebuildIds.clear();

    qCInfo(lcViewerManager) << "Asynchronously rebuilt SurfacePatchIndex for"
                            << _indexedSurfaceIds.size() << "surfaces"
                            << "at stride" << _surfacePatchSamplingStride;
    forEachViewer([](CVolumeViewer* v) { v->renderIntersections(); });

    // Check if progressive refinement is needed
    if (_targetRefinedStride > 0 && _surfacePatchSamplingStride > _targetRefinedStride) {
        qCInfo(lcViewerManager) << "Starting progressive refinement from stride"
                                << _surfacePatchSamplingStride << "to" << _targetRefinedStride;
        const int targetStride = _targetRefinedStride;
        _targetRefinedStride = 0;  // Clear target to prevent infinite loop
        setSurfacePatchSamplingStride(targetStride, false);

        // Trigger another async rebuild at the refined stride
        // Collect current surfaces - shared_ptrs keep surfaces alive
        std::vector<SurfacePatchIndex::SurfacePtr> surfacesForTask;
        std::vector<std::string> surfaceIdsForTask;
        if (_surfaces) {
            for (const auto& surf : _surfaces->surfaces()) {
                if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
                    surfacesForTask.push_back(quad);
                    surfaceIdsForTask.push_back(surf->id);
                }
            }
        }
        _pendingSurfacePatchIndexSurfaceIds = surfaceIdsForTask;

        auto future = QtConcurrent::run([surfacesForTask, targetStride]() -> std::shared_ptr<SurfacePatchIndex> {
            auto index = std::make_shared<SurfacePatchIndex>();
            index->setSamplingStride(targetStride);
            index->rebuild(surfacesForTask);
            return index;
        });
        _surfacePatchIndexWatcher->setFuture(future);
    } else {
        _pendingSurfacePatchIndexSurfaceIds.clear();
    }
}

bool ViewerManager::updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool /*isEditUpdate*/)
{
    if (!quad) {
        return false;
    }

    const std::string surfId = quad->id;
    const bool alreadyIndexed = _indexedSurfaceIds.count(surfId) != 0;

    // Check if async rebuild is in progress
    const bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                        _surfacePatchIndexWatcher->isRunning();

    // Flush any pending cell updates
    if (_surfacePatchIndex.hasPendingUpdates(quad)) {
        bool flushed = _surfacePatchIndex.flushPendingUpdates(quad);
        if (flushed) {
            _indexedSurfaceIds.insert(surfId);
        }
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild && !flushed;
        return flushed;
    }

    // First-time indexing
    if (!alreadyIndexed) {
        // If async rebuild is in progress, queue this surface for later
        // Don't add to current tree - it will be replaced when rebuild finishes
        if (asyncRebuildInProgress) {
            _surfacesQueuedDuringRebuildIds.push_back(surfId);
            qCInfo(lcViewerManager)
                << "Queued surface" << surfId.c_str()
                << "for indexing after async rebuild completes";
            return true;
        }

        bool updated = _surfacePatchIndex.updateSurface(quad);
        if (updated) {
            _indexedSurfaceIds.insert(surfId);
            qCInfo(lcViewerManager)
                << "Indexed surface" << surfId.c_str()
                << "into SurfacePatchIndex (first time)";
        }
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild && !updated;
        return updated;
    }

    // Already indexed and no pending updates - nothing to do
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
    // Called BEFORE surface deletion - remove from R-tree index
    auto quad = std::dynamic_pointer_cast<QuadSurface>(surf);

    // Only process cleanup if we're deleting under the surface's actual ID.
    // Aliases like "segmentation" just point to surfaces that exist under their
    // own IDs - we don't want to remove from the index when an alias changes.
    const bool isDeletingByActualId = quad && (name == quad->id);

    if (isDeletingByActualId) {
        // Remove from indexed surface IDs
        _indexedSurfaceIds.erase(name);

        // Remove from queued IDs
        auto removeFromVector = [&name](std::vector<std::string>& vec) {
            vec.erase(std::remove(vec.begin(), vec.end(), name), vec.end());
        };
        removeFromVector(_pendingSurfacePatchIndexSurfaceIds);
        removeFromVector(_surfacesQueuedDuringRebuildIds);
        removeFromVector(_surfacesQueuedForRemovalDuringRebuildIds);

        // Remove from the R-tree index
        _surfacePatchIndex.removeSurface(quad);
    }
}

bool ViewerManager::resetDefaultFor(CVolumeViewer* viewer) const
{
    auto it = _resetDefaults.find(viewer);
    return it != _resetDefaults.end() ? it->second : true;
}

void ViewerManager::setResetDefaultFor(CVolumeViewer* viewer, bool value)
{
    if (!viewer) {
        return;
    }
    _resetDefaults[viewer] = value;
}

void ViewerManager::setSegmentationCursorMirroring(bool enabled)
{
    _mirrorCursorToSegmentation = enabled;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setSegmentationCursorMirroring(enabled);
        }
    }
}

void ViewerManager::setSliceStepSize(int size)
{
    _sliceStepSize = std::max(1, size);
}

void ViewerManager::forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _viewers) {
        fn(viewer);
    }
}
