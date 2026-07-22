#include "ViewerManager.hpp"

ViewerManager::ViewerManager(CState* state, VCCollection* points, QObject* parent)
    : QObject(parent)
    , _state(state)
    , _points(points)
{
}

ViewerManager::~ViewerManager() = default;

void ViewerManager::forEachBaseViewer(const std::function<void(VolumeViewerBase*)>&) const {}

SurfacePatchIndex* ViewerManager::surfacePatchIndex()
{
    return &_surfacePatchIndex;
}

void ViewerManager::handleSurfacePatchIndexPrimeFinished() {}

void ViewerManager::handleSurfacePatchIndexTaskFinished() {}

void ViewerManager::handleFocusPoiChanged(std::string, POI*) {}

void ViewerManager::handleVolumeClicked(cv::Vec3f, cv::Vec3f, Surface*,
                                        Qt::MouseButton, Qt::KeyboardModifiers) {}

void ViewerManager::handleSurfaceChanged(std::string, std::shared_ptr<Surface>, bool) {}

void ViewerManager::handleSurfaceWillBeDeleted(std::string, std::shared_ptr<Surface>) {}

void ViewerManager::onGlobalTick() {}
