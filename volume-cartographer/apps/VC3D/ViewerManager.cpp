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

struct DirtyBoundsInfo {
    cv::Rect rect;
    int version = 0;
};

std::optional<DirtyBoundsInfo> readDirtyBounds(QuadSurface* surface)
{
    if (!surface || !surface->meta || !surface->meta->is_object()) {
        return std::nullopt;
    }

    nlohmann::json& meta = *surface->meta;
    auto it = meta.find("dirty_bounds");
    if (it == meta.end() || !it->is_object()) {
        return std::nullopt;
    }

    const int rowStart = it->value("row_start", -1);
    const int rowEnd = it->value("row_end", -1);
    const int colStart = it->value("col_start", -1);
    const int colEnd = it->value("col_end", -1);

    if (rowStart < 0 || colStart < 0 || rowEnd <= rowStart || colEnd <= colStart) {
        return std::nullopt;
    }

    int version = it->value("version", 0);
    if (version <= 0) {
        auto versionIt = meta.find("dirty_bounds_version");
        if (versionIt != meta.end() && versionIt->is_number_integer()) {
            version = versionIt->get<int>();
        }
    }

    DirtyBoundsInfo info;
    info.rect = cv::Rect(colStart, rowStart, colEnd - colStart, rowEnd - rowStart);
    info.version = std::max(version, 1);
    return info;
}

std::optional<CellRegion> vertexRectToCellRegion(const cv::Rect& vertexRect,
                                                 QuadSurface* surface)
{
    if (!surface) {
        return std::nullopt;
    }
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->rows < 2 || points->cols < 2) {
        return std::nullopt;
    }

    const int cellRowCount = points->rows - 1;
    const int cellColCount = points->cols - 1;

    CellRegion region;
    region.rowStart = std::max(0, vertexRect.y - 1);
    region.rowEnd = std::min(cellRowCount, vertexRect.y + vertexRect.height);
    region.colStart = std::max(0, vertexRect.x - 1);
    region.colEnd = std::min(cellColCount, vertexRect.x + vertexRect.width);

    if (region.rowStart >= region.rowEnd || region.colStart >= region.colEnd) {
        return std::nullopt;
    }

    return region;
}
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

void ViewerManager::setSurfacePatchSamplingStride(int stride)
{
    stride = std::max(1, stride);
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("viewer/intersection_sampling_stride", _surfacePatchSamplingStride);

    if (_surfacePatchIndex.setSamplingStride(_surfacePatchSamplingStride)) {
        _surfacePatchIndexDirty = true;
        _indexedSurfaces.clear();
    }

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setSurfacePatchSamplingStride(_surfacePatchSamplingStride);
        }
    }
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
    if (_surfacePatchIndexDirty || _surfacePatchIndex.empty()) {
        _surfacePatchIndexDirty = true;
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

    _surfacePatchIndexDirty = true;
    _indexedSurfaces.erase(surface);
    qCInfo(lcViewerManager) << "Failed to rebuild SurfacePatchIndex for surface" << surface->id.c_str()
                            << "- marking index dirty";
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
        _surfacePatchIndexDirty = false;
        return;
    }

    auto surfacesForTask = _pendingSurfacePatchIndexSurfaces;
    auto future = QtConcurrent::run([surfacesForTask]() mutable -> std::shared_ptr<SurfacePatchIndex> {
        auto index = std::make_shared<SurfacePatchIndex>();
        index->rebuild(surfacesForTask);
        return index;
    });
    _surfacePatchIndexWatcher->setFuture(future);
}

void ViewerManager::rebuildSurfacePatchIndexIfNeeded()
{
    if (!_surfacePatchIndexDirty) {
        return;
    }
    _surfacePatchIndexDirty = false;

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
    _surfacePatchIndexDirty = false;
    _indexedSurfaces.clear();
    _indexedSurfaces.insert(_pendingSurfacePatchIndexSurfaces.begin(),
                            _pendingSurfacePatchIndexSurfaces.end());
    _pendingSurfacePatchIndexSurfaces.clear();
    qCInfo(lcViewerManager) << "Asynchronously rebuilt SurfacePatchIndex for"
                            << _indexedSurfaces.size() << "surfaces";
}

void ViewerManager::handleSurfaceChanged(std::string /*name*/, Surface* surf)
{
    bool affectsSurfaceIndex = false;
    bool regionUpdated = false;
    bool indexUpdated = false;

    if (auto* quad = dynamic_cast<QuadSurface*>(surf)) {
        affectsSurfaceIndex = true;
        std::optional<cv::Rect> dirtyVertices;
        bool dirtyBoundsRegressed = false;
        const bool alreadyIndexed = _indexedSurfaces.count(quad) != 0;
        if (auto dirtyInfo = readDirtyBounds(quad)) {
            int lastVersion = 0;
            auto it = _surfaceDirtyBoundsVersions.find(quad);
            if (it != _surfaceDirtyBoundsVersions.end()) {
                lastVersion = it->second;
            }
            if (dirtyInfo->version > lastVersion) {
                dirtyVertices = dirtyInfo->rect;
            } else if (dirtyInfo->version < lastVersion) {
                dirtyBoundsRegressed = true;
            }
            _surfaceDirtyBoundsVersions[quad] = dirtyInfo->version;
        } else {
            _surfaceDirtyBoundsVersions.erase(quad);
        }

        bool skippedDueToExistingIndex = false;
        if (!_surfacePatchIndexDirty) {
            if (dirtyVertices) {
                if (auto cellRegion = vertexRectToCellRegion(*dirtyVertices, quad)) {
                    regionUpdated = _surfacePatchIndex.updateSurfaceRegion(
                        quad,
                        cellRegion->rowStart,
                        cellRegion->rowEnd,
                        cellRegion->colStart,
                        cellRegion->colEnd);
                    if (regionUpdated) {
                        qCInfo(lcViewerManager)
                            << "Updated SurfacePatchIndex region for surface" << quad->id.c_str()
                            << "rows" << cellRegion->rowStart << "to" << cellRegion->rowEnd
                            << "cols" << cellRegion->colStart << "to" << cellRegion->colEnd;
                    }
                }
            }
            if (!regionUpdated && (!alreadyIndexed || dirtyBoundsRegressed)) {
                indexUpdated = _surfacePatchIndex.updateSurface(quad);
                if (indexUpdated) {
                    qCInfo(lcViewerManager)
                        << "Rebuilt SurfacePatchIndex entries for surface" << quad->id.c_str()
                        << "due to missing or regressed dirty bounds";
                }
            } else if (!regionUpdated && alreadyIndexed && !dirtyBoundsRegressed) {
                skippedDueToExistingIndex = true;
            }
        }
        if (dirtyBoundsRegressed) {
            _surfacePatchIndexDirty = true;
            _indexedSurfaces.erase(quad);
            qCInfo(lcViewerManager)
                << "Dirty bounds regressed for surface" << quad->id.c_str()
                << "- scheduling global SurfacePatchIndex rebuild";
        }
        if (skippedDueToExistingIndex) {
            regionUpdated = true;
        }
        if (regionUpdated || indexUpdated) {
            _indexedSurfaces.insert(quad);
            qCInfo(lcViewerManager) << "SurfacePatchIndex updated for surface" << quad->id.c_str();
        }
    } else if (!surf) {
        affectsSurfaceIndex = true;
        if (_surfaces) {
            std::unordered_set<const QuadSurface*> liveSurfaces;
            auto surfaces = _surfaces->surfaces();
            liveSurfaces.reserve(surfaces.size());
            for (Surface* candidate : surfaces) {
                if (auto* quadSurface = dynamic_cast<QuadSurface*>(candidate)) {
                    liveSurfaces.insert(quadSurface);
                }
            }
            for (auto it = _surfaceDirtyBoundsVersions.begin();
                 it != _surfaceDirtyBoundsVersions.end();) {
                if (!liveSurfaces.count(it->first)) {
                    it = _surfaceDirtyBoundsVersions.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = _indexedSurfaces.begin(); it != _indexedSurfaces.end();) {
                if (!liveSurfaces.count(*it)) {
                    it = _indexedSurfaces.erase(it);
                } else {
                    ++it;
                }
            }
        } else {
            _surfaceDirtyBoundsVersions.clear();
            _indexedSurfaces.clear();
        }
    }

    if (affectsSurfaceIndex) {
        _surfacePatchIndexDirty = _surfacePatchIndexDirty || !(regionUpdated || indexUpdated);
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

void ViewerManager::forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _viewers) {
        fn(viewer);
    }
}
