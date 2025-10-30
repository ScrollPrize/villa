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
#include <algorithm>
#include <cmath>

ViewerManager::ViewerManager(CSurfaceCollection* surfaces,
                             VCCollection* points,
                             ChunkCache* cache,
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
}

CVolumeViewer* ViewerManager::createViewer(const std::string& surfaceName,
                                           const QString& title,
                                           QMdiArea* mdiArea)
{
    if (!mdiArea || !_surfaces) {
        return nullptr;
    }

    auto* viewer = new CVolumeViewer(_surfaces, mdiArea);
    auto* win = mdiArea->addSubWindow(viewer);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::WindowTitleHint | Qt::WindowMinMaxButtonsHint);

    viewer->setCache(_chunkCache);
    viewer->setPointCollection(_points);

    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged, viewer, &CVolumeViewer::onSurfaceChanged);
        connect(_surfaces, &CSurfaceCollection::sendPOIChanged, viewer, &CVolumeViewer::onPOIChanged);
        connect(_surfaces, &CSurfaceCollection::sendIntersectionChanged, viewer, &CVolumeViewer::onIntersectionChanged);
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

void ViewerManager::forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _viewers) {
        fn(viewer);
    }
}
