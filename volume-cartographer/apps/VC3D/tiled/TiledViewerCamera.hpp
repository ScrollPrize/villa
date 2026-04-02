#pragma once

#include "vc/core/render/TiledViewerCamera.hpp"  // Camera struct lives here now

#include <QPointF>
#include <opencv2/core.hpp>

class Surface;
class PlaneSurface;
class QuadSurface;
class TileScene;
class SurfacePatchIndex;

// ---------------------------------------------------------------------------
// Coordinate-transform helpers (Qt bridge functions)
// ---------------------------------------------------------------------------
// These use QPointF and depend on TileScene (Qt), so they stay in the app layer.

// Map a volume (world) coordinate to scene pixel coordinates.
// Returns a null QPointF if the surface is nullptr.
QPointF tiledVolumeToScene(Surface* surf, TileScene* tileScene,
                           SurfacePatchIndex* patchIndex,
                           const cv::Vec3f& volPoint);

// Map scene pixel coordinates to a volume (world) position + surface normal.
// Returns false if the conversion fails (null surface, out-of-range, etc.).
bool tiledSceneToVolume(Surface* surf, TileScene* tileScene,
                        const QPointF& scenePos,
                        cv::Vec3f& outPos, cv::Vec3f& outNormal);
