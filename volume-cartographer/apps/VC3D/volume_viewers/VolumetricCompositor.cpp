#include <vc_reader/slab.hpp>
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

    std::vector<vcr::display::SlabLayer> layers;
    for (int i=0;i<numLayers;++i) {
        if (layerValues[i].size()!=layerValues[0].size() || layerCoverage[i].size()!=layerValues[0].size())
            throw std::invalid_argument("slab layer sizes differ");
        layers.push_back({layerValues[i].ptr<uint8_t>(),layerCoverage[i].ptr<uint8_t>(),
            layerValues[i].step1(),layerCoverage[i].step1()});
    }
    vcr::display::Slab slab;
    slab.layers=layers.data();slab.count=numLayers;slab.rows=rows;slab.cols=cols;
    slab.left=margins.left;slab.top=margins.top;slab.width=outCols;slab.height=outRows;
    slab.colors=colorLut.data();slab.opacity=opacityLut.data();slab.lighting=lightingStrength;
    auto& projection=slab.projection;
    projection.perspective=perspective;
    projection.m00=proj.m00;projection.m01=proj.m01;projection.m10=proj.m10;projection.m11=proj.m11;
    projection.layerOffsets=reinterpret_cast<const float*>(proj.layerOffsets.data());
    std::copy_n(pcam.pos.data(),3,projection.pos);
    std::copy_n(pcam.rayBase.data(),3,projection.rayBase);
    std::copy_n(pcam.e1OverF.data(),3,projection.e1OverF);
    std::copy_n(pcam.e2OverF.data(),3,projection.e2OverF);
    projection.centerU=pcam.centerU;projection.centerV=pcam.centerV;projection.layerNum=pcam.layerNum.data();
    vcr::display::compositeSlab(slab,colorOut.ptr<uint8_t>(),coverageOut.ptr<uint8_t>());
}

} // namespace vc3d::volumetric
