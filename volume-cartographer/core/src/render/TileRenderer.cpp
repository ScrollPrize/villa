#include "vc/core/render/TileRenderer.hpp"
#include "vc/core/render/PostProcess.hpp"
#include "vc/core/render/Colormaps.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"

#include <algorithm>
#include <cmath>

namespace vc {

namespace {

void blendOverlayImage(uint32_t* base, int baseStride,
                       const uint32_t* overlay, int overlayStride,
                       const cv::Mat_<uint8_t>& overlayMask,
                       float opacity, int rows, int cols)
{
    if (!base || !overlay || overlayMask.empty()) {
        return;
    }

    const float alpha = std::clamp(opacity, 0.0f, 1.0f);
    if (alpha <= 0.0f) {
        return;
    }

    const float ia = 1.0f - alpha;
    for (int y = 0; y < rows; ++y) {
        auto* dst = base + y * baseStride;
        const auto* src = overlay + y * overlayStride;
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
    result.width = params.tileW;
    result.height = params.tileH;

    if (!surface || !volume) {
        fprintf(stderr, "[render] null surface or volume\n");
        return result;
    }

    if (!volume->zarrDataset(params.dsScaleIdx)) {
        fprintf(stderr, "[render] no zarrDataset at level %d\n", params.dsScaleIdx);
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
        cv::Vec3f totalOffset(params.surfaceROI.x, params.surfaceROI.y, params.zOff);
        cv::Vec3f vx = plane->basisX();
        cv::Vec3f vy = plane->basisY();
        planeNormal = plane->normal(cv::Vec3f(0, 0, 0));
        cv::Vec3f useOrigin = plane->origin() + planeNormal * totalOffset[2];
        // Division is more numerically stable than multiply-by-reciprocal
        planeVxStep = vx / params.scale;
        planeVyStep = vy / params.scale;
        planeOrigin = vx * totalOffset[0] + vy * totalOffset[1] + useOrigin;

        static int coordDbg = 0;
        if (coordDbg++ < 5)
            fprintf(stderr, "[render] tile(%d,%d) scale=%.3f dsIdx=%d origin=[%.1f,%.1f,%.1f] vxStep=[%.4f,%.4f,%.4f] vyStep=[%.4f,%.4f,%.4f] surfROI=[%.1f,%.1f] planeOrig=[%.1f,%.1f,%.1f]\n",
                params.worldKey.worldCol, params.worldKey.worldRow,
                params.scale, params.dsScaleIdx,
                planeOrigin[0], planeOrigin[1], planeOrigin[2],
                planeVxStep[0], planeVxStep[1], planeVxStep[2],
                planeVyStep[0], planeVyStep[1], planeVyStep[2],
                params.surfaceROI.x, params.surfaceROI.y,
                plane->origin()[0], plane->origin()[1], plane->origin()[2]);
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
            static int rejectCount = 0;
            if (rejectCount++ < 20)
                fprintf(stderr, "[render] AABB reject: tile(%d,%d) AABB[%.0f..%.0f, %.0f..%.0f, %.0f..%.0f] db[%.0f..%.0f, %.0f..%.0f, %.0f..%.0f] scale=%.3f dsIdx=%d\n",
                    params.worldKey.worldCol, params.worldKey.worldRow,
                    tMinX, tMaxX, tMinY, tMaxY, tMinZ, tMaxZ,
                    db.minX, db.maxX, db.minY, db.maxY, db.minZ, db.maxZ,
                    params.scale, params.dsScaleIdx);
            return result;
        }
    }

    // Sample volume data -- thread-local buffers avoid per-tile malloc/free.
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
        // in a single pass, writing directly to the pixel buffer. Eliminates
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
            vc::buildWindowLevelLut(lut, params.windowLow, params.windowHigh,
                                lightFactor);

            // Write directly into pixel buffer -- one pass, zero copies.
            result.pixels.resize(params.tileW * params.tileH);
            auto* bits = result.pixels.data();
            const int stride = params.tileW;

            result.actualLevel = volume->samplePlaneBestEffortARGB32(
                bits, stride, planeOrigin, planeVxStep, planeVyStep,
                params.tileW, params.tileH, sp, lut.data());

            static int pixDbg = 0;
            if (pixDbg++ < 5) {
                int nonBlack = 0;
                uint32_t bg = lut[0];
                for (int i = 0; i < params.tileW * params.tileH; i++)
                    if (bits[i] != bg) nonBlack++;
                fprintf(stderr, "[render] tile(%d,%d) actualLevel=%d nonBlack=%d/%d lut0=0x%08x wLo=%.1f wHi=%.1f\n",
                    params.worldKey.worldCol, params.worldKey.worldRow,
                    result.actualLevel, nonBlack, params.tileW * params.tileH,
                    bg, params.windowLow, params.windowHigh);
            }

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
        // Unified post-processing: produces ARGB32 directly,
        // bypassing all cvtColor conversions and RGB888->RGB32 expansion.
        RenderPostProcessParams pp;
        pp.isoCutoff = params.compositeSettings.params.isoCutoff;
        pp.windowLow = params.windowLow;
        pp.windowHigh = params.windowHigh;
        pp.stretchValues = params.stretchValues;
        pp.colormapId = params.colormapId;
        pp.postStretchValues = params.compositeSettings.postStretchValues;
        pp.removeSmallComponents = params.compositeSettings.postRemoveSmallComponents;
        pp.minComponentSize = params.compositeSettings.postMinComponentSize;
        result.pixels.resize(params.tileW * params.tileH);
        vc::applyRenderPostProcess(gray, pp, result.pixels.data(), params.tileW);
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
            vc::resolve(params.overlayColormapId).kind ==
                vc::OverlayColormapKind::DiscreteLut;
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
            RenderPostProcessParams overlayParams;
            overlayParams.windowLow = params.overlayWindowLow;
            overlayParams.windowHigh = params.overlayWindowHigh;
            overlayParams.colormapId = params.overlayColormapId;
            overlayParams.stretchValues = false;

            // Render overlay into a temporary buffer, then blend into result.pixels
            const int ow = overlayGray.cols;
            const int oh = overlayGray.rows;
            thread_local std::vector<uint32_t> overlayBuf;
            overlayBuf.resize(ow * oh);
            vc::applyRenderPostProcess(overlayGray, overlayParams, overlayBuf.data(), ow);
            blendOverlayImage(result.pixels.data(), params.tileW,
                              overlayBuf.data(), ow,
                              overlayGray, params.overlayOpacity,
                              std::min(result.height, oh),
                              std::min(result.width, ow));
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

}  // namespace vc
