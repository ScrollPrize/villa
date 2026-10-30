#include "ViewerManager.hpp"
#include "test_viewer_manager_stub.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>

namespace {
std::shared_ptr<vc::objects::ObjectStoreSession> locateSession;
std::string locateVolume;
}

void vc3d_test::setLocateSession(std::shared_ptr<vc::objects::ObjectStoreSession> session, std::string volume)
{
    locateSession = std::move(session);
    locateVolume = std::move(volume);
}

ViewerManager::ViewerManager(CState* state, VCCollection* points, QObject* parent)
    : QObject(parent)
    , _state(state)
    , _points(points)
{
}

ViewerManager::~ViewerManager() = default;

void ViewerManager::forEachBaseViewer(const std::function<void(VolumeViewerBase*)>&) const {}

std::optional<ViewerManager::SurfaceLocation> ViewerManager::locate(const QuadSurface& surface, const cv::Vec3f& point,
                                                                    float tolerance) const
{
    return locate(surface, std::vector<cv::Vec3f>{point}, tolerance).front();
}

std::vector<std::optional<ViewerManager::SurfaceLocation>> ViewerManager::locate(const QuadSurface& surface,
                                                                                 const std::vector<cv::Vec3f>& points,
                                                                                 float tolerance) const
{
    std::vector<std::optional<SurfaceLocation>> result(points.size());
    if (!locateSession || !vc::objects::util::isUuid(surface.id)) return result;
    vc::objects::SurfaceProximityQuery query;
    query.volume = locateVolume;
    query.uuids = std::vector{surface.id};
    query.tolerance = tolerance;
    std::vector<std::array<double, 3>> queried;
    for (const auto& p : points) queried.push_back({p[0], p[1], p[2]});
    const auto hits = locateSession->nearestSurfaces({}, query, queried);
    for (size_t i = 0; i < points.size(); ++i)
        if (!hits[i].empty()) {
            const auto& hit = hits[i].front().hit;
            result[i] = SurfaceLocation{surface.gridToPtr({float(hit.grid[0]), float(hit.grid[1])}), float(hit.distance)};
        }
    return result;
}

void ViewerManager::handleFocusPoiChanged(std::string, POI*) {}

void ViewerManager::handleVolumeClicked(cv::Vec3f, cv::Vec3f, Surface*,
                                        Qt::MouseButton, Qt::KeyboardModifiers) {}

void ViewerManager::onGlobalTick() {}
