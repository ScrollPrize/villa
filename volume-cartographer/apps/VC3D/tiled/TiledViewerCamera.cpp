#include "TiledViewerCamera.hpp"
#include "TileScene.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

// ---------------------------------------------------------------------------
// Coordinate-transform free functions (Qt bridge)
// ---------------------------------------------------------------------------

QPointF tiledVolumeToScene(Surface* surf, TileScene* tileScene,
                           SurfacePatchIndex* patchIndex,
                           const cv::Vec3f& volPoint)
{
    if (!surf || !tileScene) return QPointF();

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf)) {
        cv::Vec3f surfPos = plane->project(volPoint, 1.0, 1.0);
        return tileScene->surfaceToScene(surfPos[0], surfPos[1]);
    }

    if (auto* quad = dynamic_cast<QuadSurface*>(surf)) {
        cv::Vec3f ptr(0, 0, 0);
        surf->pointTo(ptr, volPoint, 4.0, 100, patchIndex);
        cv::Vec3f loc = surf->loc(ptr);
        return tileScene->surfaceToScene(loc[0], loc[1]);
    }

    return QPointF();
}

bool tiledSceneToVolume(Surface* surf, TileScene* tileScene,
                        const QPointF& scenePos,
                        cv::Vec3f& outPos, cv::Vec3f& outNormal)
{
    if (!surf || !tileScene) {
        outPos = cv::Vec3f(0, 0, 0);
        outNormal = cv::Vec3f(0, 0, 1);
        return false;
    }

    try {
        cv::Vec2f surfParam = tileScene->sceneToSurface(scenePos);
        cv::Vec3f surfLoc = {surfParam[0], surfParam[1], 0};
        cv::Vec3f ptr(0, 0, 0);
        outNormal = surf->normal(ptr, surfLoc);
        outPos = surf->coord(ptr, surfLoc);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}
