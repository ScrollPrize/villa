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
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const int savedOpacityPercent = settings.value("viewer/intersection_opacity", 100).toInt();
    const float normalized = static_cast<float>(savedOpacityPercent) / 100.0f;
    _intersectionOpacity = std::clamp(normalized, 0.0f, 1.0f);

    const float storedBaseLow = settings.value("viewer/base_window_low", 0.0f).toFloat();
    const float storedBaseHigh = settings.value("viewer/base_window_high", 255.0f).toFloat();
    _volumeWindowLow = std::clamp(storedBaseLow, 0.0f, 255.0f);
    const float minHigh = std::min(_volumeWindowLow + 1.0f, 255.0f);
    _volumeWindowHigh = std::clamp(storedBaseHigh, minHigh, 255.0f);

    const int storedSampling = settings.value("viewer/intersection_sampling_stride", 1).toInt();
    _surfacePatchSamplingStride = std::max(1, storedSampling);
    const float storedThickness = settings.value("viewer/intersection_thickness", 0.0f).toFloat();
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
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showHints = settings.value("viewer/show_direction_hints", true).toBool();
        viewer->setShowDirectionHints(showHints);
    }

    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool resetView = settings.value("viewer/reset_view_on_surface_change", true).toBool();
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
    settings.setValue("viewer/intersection_opacity",
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
    settings.setValue("viewer/intersection_thickness", _intersectionThickness);

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
    settings.setValue("viewer/base_window_low", _volumeWindowLow);
    settings.setValue("viewer/base_window_high", _volumeWindowHigh);

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
    settings.setValue("viewer/intersection_sampling_stride", _surfacePatchSamplingStride);

    if (_surfacePatchIndex.setSamplingStride(_surfacePatchSamplingStride)) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaces.clear();
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

void ViewerManager::refreshSurfacePatchIndex(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    if (_surfacePatchIndexNeedsRebuild || _surfacePatchIndex.empty()) {
        _surfacePatchIndexNeedsRebuild = true;
        _indexedSurfaces.erase(surface);
        qCInfo(lcViewerManager) << "Deferred surface index refresh for" << surface->id.c_str()
                                << "(global rebuild pending)";
        return;
    }

    if (_surfacePatchIndex.updateSurface(surface)) {
        _indexedSurfaces.insert(surface);
        qCInfo(lcViewerManager) << "Rebuilt SurfacePatchIndex entries for surface" << surface->id.c_str();
        return;
    }

    _surfacePatchIndexNeedsRebuild = true;
    _indexedSurfaces.erase(surface);
    qCInfo(lcViewerManager) << "Failed to rebuild SurfacePatchIndex for surface" << surface->id.c_str()
                            << "- marking index for rebuild";
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
    std::vector<QuadSurface*> quadSurfaces;
    quadSurfaces.reserve(allSurfaces.size());
    for (Surface* surface : allSurfaces) {
        if (auto* quad = dynamic_cast<QuadSurface*>(surface)) {
            quadSurfaces.push_back(quad);
        }
    }
    _pendingSurfacePatchIndexSurfaces = quadSurfaces;
    if (_pendingSurfacePatchIndexSurfaces.empty()) {
        _surfacePatchIndex.clear();
        _indexedSurfaces.clear();
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
    _surfacesQueuedDuringRebuild.clear();
    _surfacesQueuedForRemovalDuringRebuild.clear();

    auto surfacesForTask = _pendingSurfacePatchIndexSurfaces;
    const int stride = _surfacePatchSamplingStride;
    auto future = QtConcurrent::run([surfacesForTask, stride]() mutable -> std::shared_ptr<SurfacePatchIndex> {
        auto index = std::make_shared<SurfacePatchIndex>();
        index->setSamplingStride(stride);
        index->rebuild(surfacesForTask);
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
        _indexedSurfaces.clear();
        qCInfo(lcViewerManager) << "SurfacePatchIndex cleared (no surface collection)";
        return;
    }

    std::vector<QuadSurface*> surfaces;
    for (Surface* surf : _surfaces->surfaces()) {
        if (auto* quad = dynamic_cast<QuadSurface*>(surf)) {
            surfaces.push_back(quad);
        }
    }

    if (surfaces.empty()) {
        _surfacePatchIndex.clear();
        _indexedSurfaces.clear();
        qCInfo(lcViewerManager) << "SurfacePatchIndex cleared (no QuadSurfaces to index)";
        return;
    }

    qCInfo(lcViewerManager) << "Rebuilding SurfacePatchIndex for" << surfaces.size() << "surfaces";
    _surfacePatchIndex.rebuild(surfaces);
    _indexedSurfaces.clear();
    _indexedSurfaces.insert(surfaces.begin(), surfaces.end());
}

void ViewerManager::handleSurfacePatchIndexPrimeFinished()
{
    if (!_surfacePatchIndexWatcher) {
        return;
    }
    auto result = _surfacePatchIndexWatcher->future().result();
    if (!result) {
        _pendingSurfacePatchIndexSurfaces.clear();
        return;
    }
    _surfacePatchIndex = std::move(*result);
    _surfacePatchIndexNeedsRebuild = false;
    _indexedSurfaces.clear();
    _indexedSurfaces.insert(_pendingSurfacePatchIndexSurfaces.begin(),
                            _pendingSurfacePatchIndexSurfaces.end());

    // Process any surfaces that were removed during the async rebuild
    for (QuadSurface* toRemove : _surfacesQueuedForRemovalDuringRebuild) {
        _surfacePatchIndex.removeSurface(toRemove);
        _indexedSurfaces.erase(toRemove);
    }
    _surfacesQueuedForRemovalDuringRebuild.clear();

    // Merge any surfaces that were added during the async rebuild
    for (QuadSurface* queued : _surfacesQueuedDuringRebuild) {
        if (_surfacePatchIndex.updateSurface(queued)) {
            _indexedSurfaces.insert(queued);
            qCInfo(lcViewerManager) << "Indexed queued surface" << queued->id.c_str()
                                    << "after async rebuild";
        }
    }
    _surfacesQueuedDuringRebuild.clear();

    qCInfo(lcViewerManager) << "Asynchronously rebuilt SurfacePatchIndex for"
                            << _indexedSurfaces.size() << "surfaces"
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
        auto surfacesForTask = _pendingSurfacePatchIndexSurfaces;
        if (surfacesForTask.empty()) {
            // Re-collect surfaces if needed
            if (_surfaces) {
                for (Surface* surf : _surfaces->surfaces()) {
                    if (auto* quad = dynamic_cast<QuadSurface*>(surf)) {
                        surfacesForTask.push_back(quad);
                    }
                }
            }
        }
        _pendingSurfacePatchIndexSurfaces = surfacesForTask;

        auto future = QtConcurrent::run([surfacesForTask, targetStride]() mutable -> std::shared_ptr<SurfacePatchIndex> {
            auto index = std::make_shared<SurfacePatchIndex>();
            index->setSamplingStride(targetStride);
            index->rebuild(surfacesForTask);
            return index;
        });
        _surfacePatchIndexWatcher->setFuture(future);
    } else {
        _pendingSurfacePatchIndexSurfaces.clear();
    }
}

bool ViewerManager::updateSurfacePatchIndexForSurface(QuadSurface* quad, bool /*isEditUpdate*/)
{
    if (!quad) {
        return false;
    }

    const bool alreadyIndexed = _indexedSurfaces.count(quad) != 0;

    // Check if async rebuild is in progress
    const bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                        _surfacePatchIndexWatcher->isRunning();

    // Flush any pending cell updates
    if (_surfacePatchIndex.hasPendingUpdates(quad)) {
        bool flushed = _surfacePatchIndex.flushPendingUpdates(quad);
        if (flushed) {
            _indexedSurfaces.insert(quad);
        }
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild && !flushed;
        return flushed;
    }

    // First-time indexing
    if (!alreadyIndexed) {
        // If async rebuild is in progress, queue this surface for later
        // Don't add to current tree - it will be replaced when rebuild finishes
        if (asyncRebuildInProgress) {
            _surfacesQueuedDuringRebuild.push_back(quad);
            qCInfo(lcViewerManager)
                << "Queued surface" << quad->id.c_str()
                << "for indexing after async rebuild completes";
            return true;
        }

        bool updated = _surfacePatchIndex.updateSurface(quad);
        if (updated) {
            _indexedSurfaces.insert(quad);
            qCInfo(lcViewerManager)
                << "Indexed surface" << quad->id.c_str()
                << "into SurfacePatchIndex (first time)";
        }
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild && !updated;
        return updated;
    }

    // Already indexed and no pending updates - nothing to do
    return true;
}

void ViewerManager::handleSurfaceChanged(std::string /*name*/, Surface* surf, bool isEditUpdate)
{
    bool affectsSurfaceIndex = false;
    bool regionUpdated = false;
    bool indexUpdated = false;

    if (auto* quad = dynamic_cast<QuadSurface*>(surf)) {
        affectsSurfaceIndex = true;
        if (updateSurfacePatchIndexForSurface(quad, isEditUpdate)) {
            regionUpdated = true;  // Signal that work was done (prevents marking index for rebuild)
        }
    } else if (!surf) {
        affectsSurfaceIndex = true;
        const bool asyncRebuildInProgress = _surfacePatchIndexWatcher &&
                                            _surfacePatchIndexWatcher->isRunning();
        if (_surfaces) {
            std::unordered_set<const QuadSurface*> liveSurfaces;
            auto surfaces = _surfaces->surfaces();
            liveSurfaces.reserve(surfaces.size());
            for (Surface* candidate : surfaces) {
                if (auto* quadSurface = dynamic_cast<QuadSurface*>(candidate)) {
                    liveSurfaces.insert(quadSurface);
                }
            }
            for (auto it = _indexedSurfaces.begin(); it != _indexedSurfaces.end();) {
                if (!liveSurfaces.count(*it)) {
                    QuadSurface* toRemove = *it;
                    it = _indexedSurfaces.erase(it);
                    // Remove from the R-tree (queue if async rebuild in progress)
                    if (asyncRebuildInProgress) {
                        _surfacesQueuedForRemovalDuringRebuild.push_back(toRemove);
                    } else {
                        _surfacePatchIndex.removeSurface(toRemove);
                    }
                } else {
                    ++it;
                }
            }
        } else {
            _indexedSurfaces.clear();
        }
    }

    if (affectsSurfaceIndex) {
        _surfacePatchIndexNeedsRebuild = _surfacePatchIndexNeedsRebuild || !(regionUpdated || indexUpdated);
    }
}

void ViewerManager::handleSurfaceWillBeDeleted(std::string /*name*/, Surface* surf)
{
    // Called BEFORE surface deletion - clear all references to prevent use-after-free
    auto* quad = dynamic_cast<QuadSurface*>(surf);
    if (!quad) {
        return;
    }

    // Remove from indexed surfaces set
    _indexedSurfaces.erase(quad);

    // Remove from pending surfaces vector
    auto removeFromVector = [quad](std::vector<QuadSurface*>& vec) {
        vec.erase(std::remove(vec.begin(), vec.end(), quad), vec.end());
    };
    removeFromVector(_pendingSurfacePatchIndexSurfaces);
    removeFromVector(_surfacesQueuedDuringRebuild);
    removeFromVector(_surfacesQueuedForRemovalDuringRebuild);

    // Remove from the R-tree index
    _surfacePatchIndex.removeSurface(quad);
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
