#include "TileRenderer.hpp"
#include "PostProcess.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"

#include <algorithm>
#include <cmath>

TileRenderResult TileRenderer::renderTile(
    const TileRenderParams& params,
    const std::shared_ptr<Surface>& surface,
    Volume* volume)
{
    TileRenderResult result;
    result.worldKey = params.worldKey;
    result.epoch = params.epoch;
    result.scale = params.scale;
    result.zOff = params.zOff;
    result.dsScaleIdx = params.dsScaleIdx;

    if (!surface || !volume) {
        return result;
    }

    if (!volume->zarrDataset(params.dsScaleIdx)) {
        return result;
    }

    // Check for composite rendering (needed before generateTileCoords to
    // avoid calling gen() twice for the QuadSurface composite path)
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surface.get());
    const bool useComposite = (params.compositeSettings.enabled &&
                               (params.compositeSettings.layersFront > 0 ||
                                params.compositeSettings.layersBehind > 0));
    const bool usePlaneComposite = (plane != nullptr &&
                                    params.compositeSettings.planeEnabled &&
                                    (params.compositeSettings.planeLayersFront > 0 ||
                                     params.compositeSettings.planeLayersBehind > 0));

    // Generate coordinates for this tile.
    // For the QuadSurface composite path we also need normals, so request both
    // in a single gen() call instead of calling gen() twice.
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    if (useComposite && !plane) {
        surface->gen(&coords, &normals, cv::Size(params.tileW, params.tileH),
                     cv::Vec3f(0, 0, 0), params.scale,
                     {params.surfaceROI.x * params.scale,
                      params.surfaceROI.y * params.scale,
                      params.zOff});
    } else {
        generateTileCoords(coords, params, surface);
    }

    if (coords.empty()) {
        return result;
    }

    // Quick reject: if the tile's world AABB doesn't intersect data bounds,
    // skip the entire sampling pass (avoids cache lookups and prefetch requests).
    // Sample corners, edge midpoints, and center to handle curved surfaces
    // where interior points can extend beyond the corner-only AABB.
    const auto& db = volume->dataBounds();
    if (db.valid) {
        int midR = coords.rows / 2;
        int midC = coords.cols / 2;
        int lastR = coords.rows - 1;
        int lastC = coords.cols - 1;

        // 4 corners + 4 edge midpoints + center = 9 sample points
        const cv::Vec3f* samples[] = {
            &coords(0, 0), &coords(0, lastC),
            &coords(lastR, 0), &coords(lastR, lastC),
            &coords(0, midC), &coords(lastR, midC),
            &coords(midR, 0), &coords(midR, lastC),
            &coords(midR, midC)
        };

        float tMinX = samples[0]->val[0], tMaxX = tMinX;
        float tMinY = samples[0]->val[1], tMaxY = tMinY;
        float tMinZ = samples[0]->val[2], tMaxZ = tMinZ;
        for (int i = 1; i < 9; i++) {
            const auto& s = *samples[i];
            tMinX = std::min(tMinX, s[0]); tMaxX = std::max(tMaxX, s[0]);
            tMinY = std::min(tMinY, s[1]); tMaxY = std::max(tMaxY, s[1]);
            tMinZ = std::min(tMinZ, s[2]); tMaxZ = std::max(tMaxZ, s[2]);
        }

        // Conservative margin for interpolation + composite layers
        float margin = 2.0f;
        if (params.compositeSettings.enabled || params.compositeSettings.planeEnabled) {
            margin += static_cast<float>(std::max({
                params.compositeSettings.layersFront,
                params.compositeSettings.layersBehind,
                params.compositeSettings.planeLayersFront,
                params.compositeSettings.planeLayersBehind}));
        }

        if (tMaxX < db.minX - margin || tMinX > db.maxX + margin ||
            tMaxY < db.minY - margin || tMinY > db.maxY + margin ||
            tMaxZ < db.minZ - margin || tMinZ > db.maxZ + margin) {
            return result;
        }
    }

    // Sample volume data
    cv::Mat_<uint8_t> gray;

    if (useComposite && !plane) {
        // QuadSurface composite: coords and normals already generated above

        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.composite = params.compositeSettings.params;
        sp.zStart = params.compositeSettings.reverseDirection
                        ? -params.compositeSettings.layersBehind
                        : -params.compositeSettings.layersFront;
        sp.zEnd = params.compositeSettings.reverseDirection
                      ? params.compositeSettings.layersFront
                      : params.compositeSettings.layersBehind;

        result.actualLevel = volume->sampleCompositeBestEffort(
            gray, coords, normals, sp);
    } else if (usePlaneComposite) {
        cv::Vec3f planeNormal = plane->normal(cv::Vec3f(0, 0, 0));
        cv::Mat_<cv::Vec3f> normals(coords.size(), planeNormal);

        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.composite = params.compositeSettings.params;
        sp.zStart = params.compositeSettings.reverseDirection
                        ? -params.compositeSettings.planeLayersBehind
                        : -params.compositeSettings.planeLayersFront;
        sp.zEnd = params.compositeSettings.reverseDirection
                      ? params.compositeSettings.planeLayersFront
                      : params.compositeSettings.planeLayersBehind;

        result.actualLevel = volume->sampleCompositeBestEffort(
            gray, coords, normals, sp);
    } else {
        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.method = (params.useFastInterpolation || params.dsScaleIdx >= 3)
                        ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

        result.actualLevel = volume->sampleBestEffort(gray, coords, sp);
    }

    // Post-process

    if (gray.empty()) {
        return result;
    }

    // Unified post-processing: produces QImage::Format_RGB32 directly,
    // bypassing all cvtColor conversions and RGB888→RGB32 expansion.
    PostProcessParams pp;
    pp.isoCutoff = params.compositeSettings.params.isoCutoff;
    pp.windowLow = params.windowLow;
    pp.windowHigh = params.windowHigh;
    pp.stretchValues = params.stretchValues;
    pp.colormapId = params.colormapId;
    pp.postStretchValues = params.compositeSettings.postStretchValues;
    pp.removeSmallComponents = params.compositeSettings.postRemoveSmallComponents;
    pp.minComponentSize = params.compositeSettings.postMinComponentSize;
    result.image = applyPostProcess(gray, pp);
    return result;
}

void TileRenderer::generateTileCoords(
    cv::Mat_<cv::Vec3f>& coords,
    const TileRenderParams& params,
    const std::shared_ptr<Surface>& surface)
{
    const cv::Size tileSize(params.tileW, params.tileH);

    // Both PlaneSurface and QuadSurface use the same gen() call:
    // ptr = (0,0,0), offset = (surfaceROI * scale, zOff)
    // For QuadSurface, this is equivalent to using surfacePtr with canvas-relative offsets
    // because surfacePtr cancels out: ptr + (ptr*scale + dx + px) / scale = ptr + ptr + (dx+px)/scale
    // Using ptr=0: (surfROI*scale + px) / scale = surfROI + px/scale (same surface param)
    surface->gen(&coords, nullptr, tileSize, cv::Vec3f(0, 0, 0), params.scale,
                 {params.surfaceROI.x * params.scale,
                  params.surfaceROI.y * params.scale,
                  params.zOff});
}

