#include "ViewerManager.hpp"

void ViewerManager::forEachBaseViewer(const std::function<void(VolumeViewerBase*)>&) const {}

SurfacePatchIndex* ViewerManager::surfacePatchIndex()
{
    return nullptr;
}

void ViewerManager::handleSurfacePatchIndexPrimeFinished() {}

void ViewerManager::handleSurfacePatchIndexTaskFinished() {}

void ViewerManager::handleSurfaceChanged(std::string, std::shared_ptr<Surface>, bool) {}

void ViewerManager::handleSurfaceWillBeDeleted(std::string, std::shared_ptr<Surface>) {}

void ViewerManager::onGlobalTick() {}
