#include "ViewerManager.hpp"

#include "CVolumeViewer.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "SegmentationModule.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QMdiArea>
#include <QMdiSubWindow>
#include <QSettings>

ViewerManager::ViewerManager(CSurfaceCollection* surfaces,
                             VCCollection* points,
                             ChunkCache* cache,
                             QObject* parent)
    : QObject(parent)
    , _surfaces(surfaces)
    , _points(points)
    , _chunkCache(cache)
{
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
        QSettings settings("VC.ini", QSettings::IniFormat);
        bool showHints = settings.value("viewer/show_direction_hints", true).toBool();
        viewer->setShowDirectionHints(showHints);
    }

    {
        QSettings settings("VC.ini", QSettings::IniFormat);
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
    for (auto* viewer : _viewers) {
        if (viewer) {
            _segmentationOverlay->attachViewer(viewer);
        }
    }
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
