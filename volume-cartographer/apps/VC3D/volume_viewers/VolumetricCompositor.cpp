#include "VolumetricCompositor.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>

namespace vc3d::volumetric {

namespace {

constexpr float kMaxTiltDeg = 85.0f;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

// Orthonormal screen basis for the turntable camera: the patch is spun by
// azimuth about the normal, then the view tips by tilt about the
// screen-horizontal axis. e1 = screen x, e2 = screen y (down), both in slab
// coordinates; reduces to (1,0,0)/(0,1,0) at zero azimuth and tilt.
struct ScreenBasis {
    std::array<float, 3> e1;
    std::array<float, 3> e2;
};

ScreenBasis screenBasis(const CameraParams& cam)
{
    const float tilt = std::clamp(cam.tiltDeg, 0.0f, kMaxTiltDeg) * kDegToRad;
    const float az = cam.azimuthDeg * kDegToRad;
    const float ct = std::cos(tilt), st = std::sin(tilt);
    const float ca = std::cos(az), sa = std::sin(az);
    return {{ca, -sa, 0.0f},
            {sa * ct, ca * ct, -st}};
}

} // namespace

std::array<float, 3> viewDirection(const CameraParams& cam)
{
    // d = e1 x e2 x ... = the view axis of the turntable camera (toward the
    // camera, d_w = cos(tilt) > 0). In slab coordinates the tilt leans toward
    // the (sin(az), cos(az)) in-plane direction.
    const float tilt = std::clamp(cam.tiltDeg, 0.0f, kMaxTiltDeg) * kDegToRad;
    const float az = cam.azimuthDeg * kDegToRad;
    return {std::sin(tilt) * std::sin(az),
            std::sin(tilt) * std::cos(az),
            std::cos(tilt)};
}

SlabProjection slabProjection(const CameraParams& cam,
                              int numLayers,
                              int zStart,
                              float outputScale,
                              float centerU,
                              float centerV)
{
    const auto d = viewDirection(cam);
    const float slopeU = d[0] / d[2];
    const float slopeV = d[1] / d[2];
    const auto [e1, e2] = screenBasis(cam);

    // A screen point s (relative to the view center) maps onto the layer
    // plane at height w by following the parallel ray s1*e1 + s2*e2 + t*d
    // to w: q_uv = M*s + w*(d_uv/d_w), with
    // M = [e_j(uv) - e_j(w) * d_uv/d_w].
    SlabProjection proj;
    proj.m00 = e1[0] - e1[2] * slopeU;
    proj.m01 = e2[0] - e2[2] * slopeU;
    proj.m10 = e1[1] - e1[2] * slopeV;
    proj.m11 = e2[1] - e2[2] * slopeV;

    // Fold the rotation-center terms into the per-layer offset so the
    // compositor can apply q = M*p + off directly on raw pixel coords.
    const float centerOffU = centerU - (proj.m00 * centerU + proj.m01 * centerV);
    const float centerOffV = centerV - (proj.m10 * centerU + proj.m11 * centerV);

    proj.layerOffsets.resize(std::max(numLayers, 0));
    for (int i = 0; i < numLayers; ++i) {
        const float wPx = float(zStart + i) * outputScale;
        proj.layerOffsets[i] = {centerOffU + wPx * slopeU,
                                centerOffV + wPx * slopeV};
    }
    return proj;
}

PerspectiveCamera perspectiveCamera(const CameraParams& cam,
                                    int numLayers,
                                    int zStart,
                                    float outputScale,
                                    float centerU,
                                    float centerV,
                                    float screenW,
                                    float screenH)
{
    const auto d = viewDirection(cam);
    const auto [e1, e2] = screenBasis(cam);

    // perspective maps to a half-FOV of perspective * 45 deg at the screen
    // edge; the camera distance D follows from the view size. Focal length
    // f = D makes magnification exactly 1 on the plane through the view
    // center perpendicular to the view axis, so the coverage there (and the
    // central ray) match the orthographic render.
    const float p = std::clamp(cam.perspective, 0.01f, 1.0f);
    const float halfSpan = 0.5f * std::max(std::max(screenW, screenH), 1.0f);
    const float dist = halfSpan / std::tan(p * 45.0f * kDegToRad);
    const float focal = dist;

    PerspectiveCamera pc;
    pc.pos = {centerU + dist * d[0], centerV + dist * d[1], dist * d[2]};
    pc.rayBase = {-d[0], -d[1], -d[2]};
    pc.e1OverF = {e1[0] / focal, e1[1] / focal, e1[2] / focal};
    pc.e2OverF = {e2[0] / focal, e2[1] / focal, e2[2] / focal};
    pc.centerU = centerU;
    pc.centerV = centerV;
    pc.layerNum.resize(std::max(numLayers, 0));
    for (int i = 0; i < numLayers; ++i) {
        pc.layerNum[i] = float(zStart + i) * outputScale - pc.pos[2];
    }
    return pc;
}

namespace {

// Shared coefficients for the per-point w-plane mappings. Work in the
// azimuth-rotated frame, where the tilt is purely about screen-x and the
// per-w shift purely vertical: for a screen point s and plane height wPx,
//   denom = ct + k*s_y
//   u = s_x * (ct - wPx*invD) / denom
//   v = (s_y * (1 - wPx*ct*invD) + wPx*st) / denom
// with ct/st = cos/sin(tilt), invD = 1/camera distance (0 = orthographic),
// k = st*invD. At wPx = 0 this is the render's view-center homography; the
// ortho limit reproduces slabProjection's affine exactly.
struct PointMapCoeffs {
    float ca, sa, ct, st, invD, k;
};

PointMapCoeffs pointMapCoeffs(const CameraParams& cam, float halfSpan)
{
    const float tilt = std::clamp(cam.tiltDeg, 0.0f, kMaxTiltDeg) * kDegToRad;
    const float az = cam.azimuthDeg * kDegToRad;
    PointMapCoeffs c{std::cos(az), std::sin(az), std::cos(tilt), std::sin(tilt), 0.0f, 0.0f};
    if (cam.perspective > 0.0f) {
        const float p = std::clamp(cam.perspective, 0.01f, 1.0f);
        c.invD = std::tan(p * 45.0f * kDegToRad) / std::max(halfSpan, 1.0f);
        c.k = c.st * c.invD;
    }
    return c;
}

float clampedDenom(float d)
{
    return std::max(d, 1e-4f);
}

} // namespace

std::array<float, 2> slabPointToScreen(const CameraParams& cam,
                                       float halfSpan,
                                       const std::array<float, 2>& slabUV,
                                       float wPx)
{
    const auto c = pointMapCoeffs(cam, halfSpan);
    const float u = c.ca * slabUV[0] - c.sa * slabUV[1];
    const float v = c.sa * slabUV[0] + c.ca * slabUV[1];
    const float sy = (v * c.ct - wPx * c.st) /
                     clampedDenom(1.0f - wPx * c.ct * c.invD - v * c.k);
    const float sx = u * (c.ct + c.k * sy) / clampedDenom(c.ct - wPx * c.invD);
    return {sx, sy};
}

std::array<float, 2> screenToSlabPoint(const CameraParams& cam,
                                       float halfSpan,
                                       const std::array<float, 2>& screenUV,
                                       float wPx)
{
    const auto c = pointMapCoeffs(cam, halfSpan);
    const float denom = clampedDenom(c.ct + c.k * screenUV[1]);
    const float u = screenUV[0] * (c.ct - wPx * c.invD) / denom;
    const float v = (screenUV[1] * (1.0f - wPx * c.ct * c.invD) + wPx * c.st) / denom;
    return {c.ca * u + c.sa * v, -c.sa * u + c.ca * v};
}

SlabMargins computeSlabMargins(const CameraParams& cam,
                               int numLayers,
                               int zStart,
                               float outputScale,
                               int outW,
                               int outH)
{
    SlabMargins m;
    if (numLayers <= 0 || outW <= 0 || outH <= 0)
        return m;

    const float cU = float(outW) * 0.5f;
    const float cV = float(outH) * 0.5f;
    const bool perspective = cam.perspective > 0.0f;
    const auto proj = slabProjection(cam, numLayers, zStart, outputScale, cU, cV);
    const auto pcam = perspective
        ? perspectiveCamera(cam, numLayers, zStart, outputScale, cU, cV,
                            float(outW), float(outH))
        : PerspectiveCamera{};

    // Each side is at most half a screen span: enough for any practical
    // tilt x slab-thickness product, and bounds the sampling cost (at most
    // ~4x the screen area) when the settings go extreme.
    const int maxMargin = std::max(outW, outH) / 2;

    float minU = 0.0f, maxU = float(outW);
    float minV = 0.0f, maxV = float(outH);
    const float xs[2] = {0.0f, float(outW)};
    const float ys[2] = {0.0f, float(outH)};
    const int extremeLayers[2] = {0, numLayers - 1};
    for (const int i : extremeLayers) {
        for (const float x : xs) {
            for (const float y : ys) {
                float qu, qv;
                if (perspective) {
                    const float s1 = x - pcam.centerU;
                    const float s2 = y - pcam.centerV;
                    const float rayU = pcam.rayBase[0] + s1 * pcam.e1OverF[0] +
                                       s2 * pcam.e2OverF[0];
                    const float rayV = pcam.rayBase[1] + s1 * pcam.e1OverF[1] +
                                       s2 * pcam.e2OverF[1];
                    const float rayW = pcam.rayBase[2] + s1 * pcam.e1OverF[2] +
                                       s2 * pcam.e2OverF[2];
                    if (rayW >= -1e-4f) {
                        // Past the horizon: the compositor skips these rays,
                        // but rays just inside can reach far — take the clamp.
                        minU = -float(maxMargin);
                        maxU = float(outW + maxMargin);
                        minV = -float(maxMargin);
                        maxV = float(outH + maxMargin);
                        continue;
                    }
                    const float t = pcam.layerNum[std::size_t(i)] / rayW;
                    qu = pcam.pos[0] + t * rayU;
                    qv = pcam.pos[1] + t * rayV;
                } else {
                    qu = proj.m00 * x + proj.m01 * y +
                         proj.layerOffsets[std::size_t(i)][0];
                    qv = proj.m10 * x + proj.m11 * y +
                         proj.layerOffsets[std::size_t(i)][1];
                }
                minU = std::min(minU, qu);
                maxU = std::max(maxU, qu);
                minV = std::min(minV, qv);
                maxV = std::max(maxV, qv);
            }
        }
    }

    // +1 for the bilinear tap's x0+1/y0+1 neighbour.
    auto side = [maxMargin](float overshoot) {
        if (overshoot <= 0.0f)
            return 0;
        return std::min(int(std::ceil(overshoot)) + 1, maxMargin);
    };
    m.left = side(-minU);
    m.top = side(-minV);
    m.right = side(maxU - float(outW));
    m.bottom = side(maxV - float(outH));
    return m;
}

std::array<float, 256> buildOpacityLut(float alphaMin,
                                       float alphaMax,
                                       float opacity,
                                       float gamma,
                                       uint8_t isoCutoff,
                                       float segmentLength)
{
    std::array<float, 256> lut{};
    const float lo = std::clamp(alphaMin, 0.0f, 1.0f) * 255.0f;
    const float hi = std::clamp(alphaMax, 0.0f, 1.0f) * 255.0f;
    const float range = std::max(hi - lo, 1e-3f);
    const float g = std::max(gamma, 1e-3f);
    const float scale = std::max(opacity, 0.0f) * std::max(segmentLength, 1.0f);
    for (int v = 0; v < 256; ++v) {
        if (v < int(isoCutoff)) {
            lut[v] = 0.0f;
            continue;
        }
        const float rho = std::clamp((float(v) - lo) / range, 0.0f, 1.0f);
        lut[v] = std::min(scale * std::pow(rho, g), 1.0f);
    }
    return lut;
}

void compositeVolumetric(const std::vector<cv::Mat_<uint8_t>>& layerValues,
                         const std::vector<cv::Mat_<uint8_t>>& layerCoverage,
                         const CameraParams& cam,
                         int zStart,
                         float outputScale,
                         const std::array<uint32_t, 256>& colorLut,
                         const std::array<float, 256>& opacityLut,
                         cv::Mat_<cv::Vec3b>& colorOut,
                         cv::Mat_<uint8_t>& coverageOut,
                         const SlabMargins& margins,
                         float lightingStrength)
{
    const int numLayers = int(layerValues.size());
    if (numLayers == 0 || layerCoverage.size() != layerValues.size() ||
        layerValues[0].empty()) {
        return;
    }
    // Layer buffers carry the sampling margins; the output raster is the
    // inner screen window at offset (left, top).
    const int rows = layerValues[0].rows;
    const int cols = layerValues[0].cols;
    const int outRows = rows - margins.top - margins.bottom;
    const int outCols = cols - margins.left - margins.right;
    if (outRows <= 0 || outCols <= 0)
        return;
    colorOut.create(outRows, outCols);
    colorOut.setTo(cv::Vec3b(0, 0, 0));
    coverageOut.create(outRows, outCols);
    coverageOut.setTo(uint8_t(0));

    // View center and camera anchor to the screen window (in layer-buffer
    // coordinates), so margins never change the on-screen framing.
    const float centerU = float(margins.left) + float(outCols) * 0.5f;
    const float centerV = float(margins.top) + float(outRows) * 0.5f;
    const bool perspective = cam.perspective > 0.0f;
    const auto proj = slabProjection(cam, numLayers, zStart, outputScale,
                                     centerU, centerV);
    const auto pcam = perspective
        ? perspectiveCamera(cam, numLayers, zStart, outputScale,
                            centerU, centerV,
                            float(outCols), float(outRows))
        : PerspectiveCamera{};

    // Per-value emission premultiplied by alpha, so the inner loop is one
    // multiply-add per channel.
    std::array<std::array<float, 3>, 256> premul;
    for (int v = 0; v < 256; ++v) {
        const uint32_t c = colorLut[v];
        const float a = opacityLut[v];
        premul[v] = {a * float((c >> 16) & 0xFFu),
                     a * float((c >> 8) & 0xFFu),
                     a * float(c & 0xFFu)};
    }

    constexpr float kEarlyOutT = 0.004f;

    // Rows are independent; each writes only its own output row.
    cv::parallel_for_(cv::Range(0, outRows), [&](const cv::Range& range) {
    for (int y = range.start; y < range.end; ++y) {
        auto* outRow = colorOut.ptr<cv::Vec3b>(y);
        auto* covRow = coverageOut.ptr<uint8_t>(y);
        const float py = float(y + margins.top);
        for (int x = 0; x < outCols; ++x) {
            const float px = float(x + margins.left);
            float r = 0.0f, g = 0.0f, b = 0.0f;
            float T = 1.0f;
            bool anyValid = false;
            // Orthographic: the linear part is shared by all layers; only the
            // offset varies. Perspective: the pixel's ray direction and the
            // 1/r_w division are per-pixel; per layer it's one multiply-add.
            float baseU = 0.0f, baseV = 0.0f;
            float rayU = 0.0f, rayV = 0.0f, invRayW = 0.0f;
            if (perspective) {
                const float s1 = px - pcam.centerU;
                const float s2 = py - pcam.centerV;
                rayU = pcam.rayBase[0] + s1 * pcam.e1OverF[0] + s2 * pcam.e2OverF[0];
                rayV = pcam.rayBase[1] + s1 * pcam.e1OverF[1] + s2 * pcam.e2OverF[1];
                const float rayW =
                    pcam.rayBase[2] + s1 * pcam.e1OverF[2] + s2 * pcam.e2OverF[2];
                if (rayW >= -1e-4f)
                    continue;  // ray points away from the slab
                invRayW = 1.0f / rayW;
            } else {
                baseU = proj.m00 * px + proj.m01 * py;
                baseV = proj.m10 * px + proj.m11 * py;
            }
            // Near-to-far: the camera sits on the +w side, so highest w first.
            for (int i = numLayers - 1; i >= 0; --i) {
                float qu, qv;
                if (perspective) {
                    const float t = pcam.layerNum[i] * invRayW;
                    qu = pcam.pos[0] + t * rayU;
                    qv = pcam.pos[1] + t * rayV;
                } else {
                    qu = baseU + proj.layerOffsets[i][0];
                    qv = baseV + proj.layerOffsets[i][1];
                }
                const int x0 = int(std::floor(qu));
                const int y0 = int(std::floor(qv));
                if (x0 < 0 || y0 < 0 || x0 + 1 >= cols || y0 + 1 >= rows)
                    continue;
                const float fx = qu - float(x0);
                const float fy = qv - float(y0);
                const auto& vals = layerValues[i];
                const auto& cov = layerCoverage[i];
                const uint8_t* v0 = vals.ptr<uint8_t>(y0) + x0;
                const uint8_t* v1 = vals.ptr<uint8_t>(y0 + 1) + x0;
                const uint8_t* c0 = cov.ptr<uint8_t>(y0) + x0;
                const uint8_t* c1 = cov.ptr<uint8_t>(y0 + 1) + x0;
                // Coverage-weighted bilinear: uncovered texels are fully
                // transparent, not value 0 (avoids dark halos at patch
                // borders under tilt).
                const float w00 = c0[0] ? (1.0f - fx) * (1.0f - fy) : 0.0f;
                const float w10 = c0[1] ? fx * (1.0f - fy) : 0.0f;
                const float w01 = c1[0] ? (1.0f - fx) * fy : 0.0f;
                const float w11 = c1[1] ? fx * fy : 0.0f;
                const float wsum = w00 + w10 + w01 + w11;
                if (wsum <= 1e-6f)
                    continue;
                const float value = (w00 * float(v0[0]) + w10 * float(v0[1]) +
                                     w01 * float(v1[0]) + w11 * float(v1[1])) / wsum;
                anyValid = true;
                const int idx = std::clamp(int(value + 0.5f), 0, 255);
                const float alpha = opacityLut[idx];
                if (alpha <= 0.0f)
                    continue;
                const auto& e = premul[idx];
                float shade = 1.0f;
                const float lightAmount = std::clamp(lightingStrength, 0.0f, 1.0f);
                if (lightAmount > 0.0f) {
                    // Central differences in the already-extracted voxel
                    // stack. Missing neighbours fall back to the center, so
                    // coverage boundaries do not create artificial normals.
                    auto sample = [&](int layer, int sx, int sy) {
                        if (layer < 0 || layer >= numLayers)
                            return value;
                        const int xx = std::clamp(x0 + sx, 0, cols - 1);
                        const int yy = std::clamp(y0 + sy, 0, rows - 1);
                        if (!layerCoverage[layer](yy, xx))
                            return value;
                        return float(layerValues[layer](yy, xx));
                    };
                    const float gx = sample(i, 1, 0) - sample(i, -1, 0);
                    const float gy = sample(i, 0, 1) - sample(i, 0, -1);
                    const float gz = sample(i + 1, 0, 0) - sample(i - 1, 0, 0);
                    const float len2 = gx * gx + gy * gy + gz * gz;
                    if (len2 > 1e-6f) {
                        const float invLen = 1.0f / std::sqrt(len2);
                        // Fixed upper-left raking light, 35 degrees above the
                        // slab. At maximum strength, 10% ambient preserves
                        // some detail on back-facing sides.
                        constexpr float lx = -0.579228f;
                        constexpr float ly = -0.579228f;
                        constexpr float lz =  0.573576f;
                        const float diffuse = std::max(
                            0.0f, (gx * lx + gy * ly + gz * lz) * invLen);
                        const float fullyLit = 0.10f + 0.90f * diffuse;
                        shade = 1.0f + lightAmount * (fullyLit - 1.0f);
                    }
                }
                r += T * e[0] * shade;
                g += T * e[1] * shade;
                b += T * e[2] * shade;
                T *= 1.0f - alpha;
                if (T < kEarlyOutT)
                    break;
            }
            if (anyValid) {
                outRow[x] = cv::Vec3b(uint8_t(std::min(r, 255.0f)),
                                      uint8_t(std::min(g, 255.0f)),
                                      uint8_t(std::min(b, 255.0f)));
                covRow[x] = 1;
            }
        }
    }
    });
}

} // namespace vc3d::volumetric
