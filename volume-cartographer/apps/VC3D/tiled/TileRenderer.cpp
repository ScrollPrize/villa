#include "TileRenderer.hpp"
#include "PostProcess.hpp"
#include "VolumeViewerCmaps.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"

#include <algorithm>
#include <cmath>

namespace {

void blendOverlayImage(QImage* base,
                       const QImage& overlay,
                       const cv::Mat_<uint8_t>& overlayMask,
                       float opacity)
{
    if (!base || base->isNull() || overlay.isNull() || overlayMask.empty()) {
        return;
    }

    const float alpha = std::clamp(opacity, 0.0f, 1.0f);
    if (alpha <= 0.0f) {
        return;
    }

    const float ia = 1.0f - alpha;
    const int rows = std::min(base->height(), overlay.height());
    const int cols = std::min(base->width(), overlay.width());
    for (int y = 0; y < rows; ++y) {
        auto* dst = reinterpret_cast<uint32_t*>(base->scanLine(y));
        const auto* src = reinterpret_cast<const uint32_t*>(overlay.constScanLine(y));
        const auto* mask = overlayMask.ptr<uint8_t>(y);
        for (int x = 0; x < cols; ++x) {
            if (mask[x] == 0) {
                continue;
            }

            const uint32_t d = dst[x];
            const uint32_t s = src[x];

            const uint32_t dr = (d >> 16) & 0xFFu;
            const uint32_t dg = (d >> 8) & 0xFFu;
            const uint32_t db = d & 0xFFu;

            const uint32_t sr = (s >> 16) & 0xFFu;
            const uint32_t sg = (s >> 8) & 0xFFu;
            const uint32_t sb = s & 0xFFu;

            const uint32_t r = static_cast<uint32_t>(std::clamp(sr * alpha + dr * ia, 0.0f, 255.0f));
            const uint32_t g = static_cast<uint32_t>(std::clamp(sg * alpha + dg * ia, 0.0f, 255.0f));
            const uint32_t b = static_cast<uint32_t>(std::clamp(sb * alpha + db * ia, 0.0f, 255.0f));
            dst[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
    }
}

} // namespace

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
    PlaneSurface* plane = params.isPlaneSurface ? static_cast<PlaneSurface*>(surface.get()) : nullptr;
    const bool useComposite = (params.compositeSettings.enabled &&
                               (params.compositeSettings.layersFront > 0 ||
                                params.compositeSettings.layersBehind > 0));
    const bool usePlaneComposite = (plane != nullptr &&
                                    params.compositeSettings.planeEnabled &&
                                    (params.compositeSettings.planeLayersFront > 0 ||
                                     params.compositeSettings.planeLayersBehind > 0));

    // Determine whether we need surface normals.  Normals are required for
    // composite sampling (layers along normal) and for directional lighting
    // even when composite mode is off.
    const bool needNormals = (useComposite && !plane) ||
                             params.compositeSettings.params.lightingEnabled;

    // Fused plane path: for PlaneSurface without composite, we can skip
    // generating the intermediate coords Mat entirely by computing coordinates
    // inline during sampling. This eliminates 786KB of allocation + two
    // full passes over transient data, cutting cache pressure significantly.
    // When overlays are active we need coords for overlay sampling, so disable
    // the fused path to avoid generating coords, releasing them, then
    // regenerating them (which defeats the memory savings).
    const bool hasOverlay = params.overlayVolume && params.overlayOpacity > 0.0f;
    const bool useFusedPlane = (plane != nullptr && !useComposite && !usePlaneComposite && !hasOverlay);

    // Compute plane affine parameters (world-space) for PlaneSurface paths.
    // These are used by both the fused path and the AABB quick-reject.
    cv::Vec3f planeOrigin, planeVxStep, planeVyStep, planeNormal;
    if (plane) {
        float m = 1.0f / params.scale;
        cv::Vec3f totalOffset(params.surfaceROI.x, params.surfaceROI.y, params.zOff);
        cv::Vec3f vx = plane->basisX();
        cv::Vec3f vy = plane->basisY();
        planeNormal = plane->normal(cv::Vec3f(0, 0, 0));
        cv::Vec3f useOrigin = plane->origin() + planeNormal * totalOffset[2];
        planeVxStep = vx * m;
        planeVyStep = vy * m;
        planeOrigin = vx * totalOffset[0] + vy * totalOffset[1] + useOrigin;
    }

    // Generate coordinates for non-fused paths.
    // Thread-local to reuse buffers across tiles (gen() uses create() internally).
    thread_local cv::Mat_<cv::Vec3f> coords;
    thread_local cv::Mat_<cv::Vec3f> normals;
    if (!useFusedPlane) {
        if (needNormals && !plane) {
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
    } else {
        // Fused path skips coord generation; mark empty so overlay
        // logic knows to generate coords lazily if needed.
        coords.release();
    }

    // Quick reject: if the tile's world AABB doesn't intersect data bounds,
    // skip the entire sampling pass (avoids cache lookups and prefetch requests).
    const auto& db = volume->dataBounds();
    if (db.valid) {
        float tMinX, tMaxX, tMinY, tMaxY, tMinZ, tMaxZ;

        if (plane) {
            // Plane: compute AABB analytically from 4 corners (exact for affine)
            cv::Vec3f c0 = planeOrigin;
            cv::Vec3f c1 = planeOrigin + planeVxStep * static_cast<float>(params.tileW - 1);
            cv::Vec3f c2 = planeOrigin + planeVyStep * static_cast<float>(params.tileH - 1);
            cv::Vec3f c3 = c1 + planeVyStep * static_cast<float>(params.tileH - 1);
            tMinX = std::min({c0[0], c1[0], c2[0], c3[0]});
            tMaxX = std::max({c0[0], c1[0], c2[0], c3[0]});
            tMinY = std::min({c0[1], c1[1], c2[1], c3[1]});
            tMaxY = std::max({c0[1], c1[1], c2[1], c3[1]});
            tMinZ = std::min({c0[2], c1[2], c2[2], c3[2]});
            tMaxZ = std::max({c0[2], c1[2], c2[2], c3[2]});
        } else {
            // QuadSurface: sample corners, edge midpoints, and center
            int midR = coords.rows / 2;
            int midC = coords.cols / 2;
            int lastR = coords.rows - 1;
            int lastC = coords.cols - 1;

            const cv::Vec3f* samples[] = {
                &coords(0, 0), &coords(0, lastC),
                &coords(lastR, 0), &coords(lastR, lastC),
                &coords(0, midC), &coords(lastR, midC),
                &coords(midR, 0), &coords(midR, lastC),
                &coords(midR, midC)
            };

            tMinX = samples[0]->val[0]; tMaxX = tMinX;
            tMinY = samples[0]->val[1]; tMaxY = tMinY;
            tMinZ = samples[0]->val[2]; tMaxZ = tMinZ;
            for (int i = 1; i < 9; i++) {
                const auto& s = *samples[i];
                tMinX = std::min(tMinX, s[0]); tMaxX = std::max(tMaxX, s[0]);
                tMinY = std::min(tMinY, s[1]); tMaxY = std::max(tMaxY, s[1]);
                tMinZ = std::min(tMinZ, s[2]); tMaxZ = std::max(tMaxZ, s[2]);
            }
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

    // Sample volume data — thread-local buffers avoid per-tile malloc/free.
    // The sampling functions reuse the buffer when size matches.
    thread_local cv::Mat_<uint8_t> gray;
    thread_local cv::Mat_<uint8_t> overlayGray;

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
        if (normals.size() != coords.size()) {
            normals.create(coords.size());
        }
        normals.setTo(planeNormal);

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
    } else if (useFusedPlane) {
        // Fused plane path: no intermediate coords Mat
        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.method = (params.useFastInterpolation || params.dsScaleIdx >= 3)
                        ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

        // Fully fused ARGB32 path: sample voxels and apply window/level LUT
        // in a single pass, writing directly to QImage memory. Eliminates
        // the intermediate cv::Mat and the applyPostProcess second pass.
        // Valid when no colormap, no stretch, core preprocessing is a no-op.
        // Lighting is fused into the LUT (plane has constant normal).
        const bool canFuseARGB32 =
            params.colormapId.empty() && !params.stretchValues &&
            params.compositeSettings.params.isoCutoff == 0 &&
            !params.compositeSettings.postStretchValues &&
            !params.compositeSettings.postRemoveSmallComponents;

        if (canFuseARGB32) {
            // Compute lighting factor for constant plane normal (fused into LUT)
            float lightFactor = 1.0f;
            if (params.compositeSettings.params.lightingEnabled) {
                lightFactor = computeLightingFactor(planeNormal,
                    params.compositeSettings.params);
            }

            std::array<uint32_t, 256> lut;
            buildWindowLevelLut(lut, params.windowLow, params.windowHigh,
                                lightFactor);

            // Write directly into QImage scanline buffer -- one pass, zero copies.
            result.image = allocTileImage(params.tileW, params.tileH);
            auto* bits = reinterpret_cast<uint32_t*>(result.image.bits());
            const int stride = result.image.bytesPerLine() / 4;

            result.actualLevel = volume->samplePlaneBestEffortARGB32(
                bits, stride, planeOrigin, planeVxStep, planeVyStep,
                params.tileW, params.tileH, sp, lut.data());
            // Skip the gray post-process path; jump directly to overlay
            goto overlay;
        }

        result.actualLevel = volume->samplePlaneBestEffort(
            gray, planeOrigin, planeVxStep, planeVyStep,
            params.tileW, params.tileH, sp);

        // Apply directional lighting when enabled outside composite mode.
        if (params.compositeSettings.params.lightingEnabled && !gray.empty()) {
            const auto& cp = params.compositeSettings.params;
            float factor = computeLightingFactor(planeNormal, cp);
            if (factor < 1.0f) {
                for (int y = 0; y < gray.rows; ++y) {
                    auto* row = gray.ptr<uint8_t>(y);
                    for (int x = 0; x < gray.cols; ++x) {
                        row[x] = static_cast<uint8_t>(
                            std::clamp(row[x] * factor, 0.0f, 255.0f));
                    }
                }
            }
        }
    } else {
        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.method = (params.useFastInterpolation || params.dsScaleIdx >= 3)
                        ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

        result.actualLevel = volume->sampleBestEffort(gray, coords, sp);

        // Apply directional lighting when enabled outside composite mode.
        // The composite path handles this internally; here we do it as a
        // post-step using the surface normals generated above.
        if (params.compositeSettings.params.lightingEnabled && !gray.empty()) {
            const auto& cp = params.compositeSettings.params;
            if (plane) {
                // PlaneSurface: constant normal, compute factor once
                float factor = computeLightingFactor(planeNormal, cp);
                if (factor < 1.0f) {
                    for (int y = 0; y < gray.rows; ++y) {
                        auto* row = gray.ptr<uint8_t>(y);
                        for (int x = 0; x < gray.cols; ++x) {
                            row[x] = static_cast<uint8_t>(
                                std::clamp(row[x] * factor, 0.0f, 255.0f));
                        }
                    }
                }
            } else if (!normals.empty()) {
                // QuadSurface: per-pixel normal
                for (int y = 0; y < gray.rows; ++y) {
                    auto* row = gray.ptr<uint8_t>(y);
                    for (int x = 0; x < gray.cols; ++x) {
                        const cv::Vec3f& n = normals(y, x);
                        if (!std::isfinite(n[0])) continue;
                        float factor = computeLightingFactor(n, cp);
                        row[x] = static_cast<uint8_t>(
                            std::clamp(row[x] * factor, 0.0f, 255.0f));
                    }
                }
            }
        }
    }

    // Post-process

    if (gray.empty()) {
        return result;
    }

    {
        // Unified post-processing: produces QImage::Format_RGB32 directly,
        // bypassing all cvtColor conversions and RGB888->RGB32 expansion.
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
    }

overlay:
    if (params.overlayVolume && params.overlayOpacity > 0.0f) {
        // Overlay needs coords -- generate them lazily if the fused path skipped it
        if (coords.empty() && plane) {
            generateTileCoords(coords, params, surface);
        }

        vc::SampleParams overlaySp;
        overlaySp.level = params.dsScaleIdx;
        const bool categoricalOverlay =
            !params.overlayColormapId.empty() &&
            volume_viewer_cmaps::resolve(params.overlayColormapId).kind ==
                volume_viewer_cmaps::OverlayColormapKind::DiscreteLut;
        overlaySp.method = (categoricalOverlay || params.useFastInterpolation || params.dsScaleIdx >= 3)
                               ? vc::Sampling::Nearest
                               : vc::Sampling::Trilinear;

        // Avoid double I/O when overlay is the same volume with matching
        // sample params -- just copy the already-sampled base data.
        vc::SampleParams baseSp;
        baseSp.level = params.dsScaleIdx;
        baseSp.method = (params.useFastInterpolation || params.dsScaleIdx >= 3)
                            ? vc::Sampling::Nearest : vc::Sampling::Trilinear;
        if (params.overlayVolume.get() == volume &&
            overlaySp.level == baseSp.level &&
            overlaySp.method == baseSp.method &&
            !gray.empty()) {
            overlayGray = gray;
        } else {
            params.overlayVolume->sampleBestEffort(overlayGray, coords, overlaySp);
        }
        if (!overlayGray.empty()) {
            PostProcessParams overlayParams;
            overlayParams.windowLow = params.overlayWindowLow;
            overlayParams.windowHigh = params.overlayWindowHigh;
            overlayParams.colormapId = params.overlayColormapId;
            overlayParams.stretchValues = false;

            QImage overlayImage = applyPostProcess(overlayGray, overlayParams);
            blendOverlayImage(&result.image, overlayImage, overlayGray, params.overlayOpacity);
        }
    }

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
