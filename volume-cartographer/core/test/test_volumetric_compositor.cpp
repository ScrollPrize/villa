#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "volume_viewers/VolumetricCompositor.hpp"

#include <algorithm>
#include <cmath>

using namespace vc3d::volumetric;

namespace
{

// Grayscale identity color LUT (the blit convention: 0xFFRRGGBB).
std::array<uint32_t, 256> identityColorLut()
{
    std::array<uint32_t, 256> lut{};
    for (int v = 0; v < 256; ++v) {
        lut[v] = 0xFF000000u | (uint32_t(v) << 16) | (uint32_t(v) << 8) | uint32_t(v);
    }
    return lut;
}

std::array<float, 256> constantOpacityLut(float alpha)
{
    std::array<float, 256> lut{};
    lut.fill(alpha);
    return lut;
}

std::vector<cv::Mat_<uint8_t>> uniformLayers(int numLayers, int rows, int cols, uint8_t value)
{
    std::vector<cv::Mat_<uint8_t>> layers;
    for (int i = 0; i < numLayers; ++i) {
        layers.emplace_back(rows, cols, value);
    }
    return layers;
}

} // namespace

TEST_CASE("viewDirection straight down and 45 degree tilt")
{
    CameraParams cam;
    auto d = viewDirection(cam);
    CHECK(d[0] == doctest::Approx(0.0f));
    CHECK(d[1] == doctest::Approx(0.0f));
    CHECK(d[2] == doctest::Approx(1.0f));

    // Turntable: at azimuth 0 the tilt leans toward +V (the screen-vertical
    // squash direction with no patch spin).
    cam.tiltDeg = 45.0f;
    d = viewDirection(cam);
    CHECK(d[1] / d[2] == doctest::Approx(1.0f));
    CHECK(d[0] == doctest::Approx(0.0f));
    CHECK(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] == doctest::Approx(1.0f));

    // Spinning the patch by 90 degrees moves the lean to +U.
    cam.azimuthDeg = 90.0f;
    d = viewDirection(cam);
    CHECK(d[1] == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(d[0] / d[2] == doctest::Approx(1.0f));

    // Tilt clamped to 85 degrees.
    cam.azimuthDeg = 0.0f;
    cam.tiltDeg = 90.0f;
    d = viewDirection(cam);
    CHECK(d[1] / d[2] == doctest::Approx(std::tan(85.0 * M_PI / 180.0)).epsilon(1e-3));
}

TEST_CASE("slabProjection is the identity at zero tilt")
{
    CameraParams cam;
    const auto proj = slabProjection(cam, 5, -2, 2.0f, 33.0f, 44.0f);
    CHECK(proj.m00 == doctest::Approx(1.0f));
    CHECK(proj.m01 == doctest::Approx(0.0f));
    CHECK(proj.m10 == doctest::Approx(0.0f));
    CHECK(proj.m11 == doctest::Approx(1.0f));
    REQUIRE(proj.layerOffsets.size() == size_t(5));
    for (const auto& off : proj.layerOffsets) {
        CHECK(off[0] == doctest::Approx(0.0f));
        CHECK(off[1] == doctest::Approx(0.0f));
    }
}

TEST_CASE("slabProjection tilt at zero azimuth: vertical cos foreshortening plus w*tan shift")
{
    CameraParams cam;
    cam.azimuthDeg = 0.0f;
    cam.tiltDeg = 45.0f;

    const float scale = 2.0f;
    const float centerU = 10.0f, centerV = 20.0f;
    const auto proj = slabProjection(cam, 3, -1, scale, centerU, centerV);

    // One screen pixel spans 1/cos(45) slab units along v — the slab appears
    // vertically compressed by cos(tilt) — and u is untouched.
    const float sqrt2 = std::sqrt(2.0f);
    CHECK(proj.m00 == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(proj.m01 == doctest::Approx(0.0f));
    CHECK(proj.m10 == doctest::Approx(0.0f));
    CHECK(proj.m11 == doctest::Approx(sqrt2).epsilon(1e-4));

    // The view center is the rotation center: it maps to itself plus the
    // per-layer w * tan(tilt) shift (in +V at zero azimuth).
    REQUIRE(proj.layerOffsets.size() == size_t(3));
    for (int i = 0; i < 3; ++i) {
        const float wPx = float(-1 + i) * scale;
        const float qu = proj.m00 * centerU + proj.m01 * centerV + proj.layerOffsets[i][0];
        const float qv = proj.m10 * centerU + proj.m11 * centerV + proj.layerOffsets[i][1];
        CHECK(qu == doctest::Approx(centerU).epsilon(1e-4));
        CHECK(qv == doctest::Approx(centerV + wPx).epsilon(1e-4));  // tan(45) = 1
    }
}

TEST_CASE("slabProjection azimuth at zero tilt is a pure in-plane rotation")
{
    CameraParams cam;
    cam.azimuthDeg = 90.0f;
    cam.tiltDeg = 0.0f;

    const auto proj = slabProjection(cam, 3, -1, 1.0f, 0.0f, 0.0f);
    // M = Rz(-90): screen x fetches slab -v, screen y fetches slab +u.
    CHECK(proj.m00 == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(proj.m01 == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(proj.m10 == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(proj.m11 == doctest::Approx(0.0f).epsilon(1e-5));
    // No tilt: no per-layer shift, whatever the spin.
    for (const auto& off : proj.layerOffsets) {
        CHECK(off[0] == doctest::Approx(0.0f));
        CHECK(off[1] == doctest::Approx(0.0f));
    }
}

TEST_CASE("slabProjection tilt after 90deg spin composes rotation and squash")
{
    CameraParams cam;
    cam.azimuthDeg = 90.0f;
    cam.tiltDeg = 30.0f;

    const auto proj = slabProjection(cam, 1, 0, 1.0f, 0.0f, 0.0f);
    const float rad = 30.0f * float(M_PI) / 180.0f;
    // M = Rz(-90) * diag(1, 1/cos(tilt)).
    CHECK(proj.m00 == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(proj.m01 == doctest::Approx(1.0f / std::cos(rad)).epsilon(1e-4));
    CHECK(proj.m10 == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(proj.m11 == doctest::Approx(0.0f).epsilon(1e-5));
}

namespace
{

// Map a raw output pixel through the perspective camera onto layer i.
std::array<float, 2> mapPerspective(const PerspectiveCamera& pc, int layer,
                                    float x, float y)
{
    const float s1 = x - pc.centerU;
    const float s2 = y - pc.centerV;
    const float ru = pc.rayBase[0] + s1 * pc.e1OverF[0] + s2 * pc.e2OverF[0];
    const float rv = pc.rayBase[1] + s1 * pc.e1OverF[1] + s2 * pc.e2OverF[1];
    const float rw = pc.rayBase[2] + s1 * pc.e1OverF[2] + s2 * pc.e2OverF[2];
    const float t = pc.layerNum[layer] / rw;
    return {pc.pos[0] + t * ru, pc.pos[1] + t * rv};
}

} // namespace

TEST_CASE("perspectiveCamera: identity at the anchor plane, magnification off it")
{
    CameraParams cam;
    cam.perspective = 0.5f;

    // Layers at w = -2, 0, +2 px (zStart -1, outputScale 2).
    const float cu = 50.0f, cv = 40.0f;
    const auto pc = perspectiveCamera(cam, 3, -1, 2.0f, cu, cv, 100.0f, 80.0f);

    // Zero tilt: the w = 0 layer maps every pixel to itself (focal = distance).
    for (const auto& px : {std::array<float, 2>{10.0f, 70.0f},
                           std::array<float, 2>{90.0f, 5.0f}}) {
        const auto q = mapPerspective(pc, 1, px[0], px[1]);
        CHECK(q[0] == doctest::Approx(px[0]).epsilon(1e-4));
        CHECK(q[1] == doctest::Approx(px[1]).epsilon(1e-4));
    }

    // Layers nearer the camera (w > 0) are magnified: the fetch position
    // moves toward the center. Farther layers shrink: it moves away.
    const auto qNear = mapPerspective(pc, 2, 90.0f, 40.0f);
    const auto qFar = mapPerspective(pc, 0, 90.0f, 40.0f);
    CHECK(qNear[0] > cu);
    CHECK(qNear[0] < 90.0f);
    CHECK(qFar[0] > 90.0f);
}

TEST_CASE("perspectiveCamera matches orthographic on the central ray and at small strength")
{
    CameraParams cam;
    cam.azimuthDeg = 155.0f;
    cam.tiltDeg = 27.0f;

    const float cu = 64.0f, cv = 64.0f;
    const auto proj = slabProjection(cam, 5, -2, 1.5f, cu, cv);

    // Central ray: identical to orthographic for any perspective strength.
    cam.perspective = 1.0f;
    const auto pcStrong = perspectiveCamera(cam, 5, -2, 1.5f, cu, cv, 128.0f, 128.0f);
    for (int i = 0; i < 5; ++i) {
        const auto q = mapPerspective(pcStrong, i, cu, cv);
        const float ou = proj.m00 * cu + proj.m01 * cv + proj.layerOffsets[i][0];
        const float ov = proj.m10 * cu + proj.m11 * cv + proj.layerOffsets[i][1];
        CHECK(q[0] == doctest::Approx(ou).epsilon(1e-3));
        CHECK(q[1] == doctest::Approx(ov).epsilon(1e-3));
    }

    // perspective -> 0: converges to the orthographic mapping everywhere
    // (within half a pixel at strength 0.01 — the residual is O(1/distance)).
    cam.perspective = 0.01f;
    const auto pcWeak = perspectiveCamera(cam, 5, -2, 1.5f, cu, cv, 128.0f, 128.0f);
    const auto q = mapPerspective(pcWeak, 4, 100.0f, 30.0f);
    const float ou = proj.m00 * 100.0f + proj.m01 * 30.0f + proj.layerOffsets[4][0];
    const float ov = proj.m10 * 100.0f + proj.m11 * 30.0f + proj.layerOffsets[4][1];
    CHECK(std::abs(q[0] - ou) < 0.5f);
    CHECK(std::abs(q[1] - ov) < 0.5f);
}

TEST_CASE("compositeVolumetric perspective at zero tilt keeps the w=0 layer pixel-aligned")
{
    const int rows = 8, cols = 8;
    std::vector<cv::Mat_<uint8_t>> values{cv::Mat_<uint8_t>(rows, cols, uint8_t(0))};
    std::vector<cv::Mat_<uint8_t>> coverage{cv::Mat_<uint8_t>(rows, cols, uint8_t(1))};
    values[0](2, 5) = 220;

    CameraParams cam;
    cam.perspective = 1.0f;
    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, 0, 1.0f,
                        identityColorLut(), constantOpacityLut(1.0f), color, cov);

    CHECK(int(color(2, 5)[0]) == 220);
    CHECK(int(color(2, 4)[0]) == 0);
    CHECK(cov(4, 4) == 1);
}

TEST_CASE("buildOpacityLut windows, gamma and cutoff")
{
    const auto lut = buildOpacityLut(0.25f, 0.75f, 1.0f, 1.0f, 32, 1.0f);
    // Below iso cutoff: fully transparent.
    CHECK(lut[0] == 0.0f);
    CHECK(lut[31] == 0.0f);
    // Below the alpha window: transparent.
    CHECK(lut[40] == doctest::Approx(0.0f));
    // Above the window: saturated at the opacity scale.
    CHECK(lut[250] == doctest::Approx(1.0f));
    // Midpoint of the window: ~0.5.
    CHECK(lut[127] == doctest::Approx(0.5f).epsilon(0.05));
    // Monotonic in between.
    for (int v = 33; v < 256; ++v) {
        CHECK(lut[v] >= lut[v - 1]);
    }

    // Gamma bends the ramp down; segment length scales alpha up (clamped).
    const auto gammaLut = buildOpacityLut(0.0f, 1.0f, 1.0f, 2.0f, 0, 1.0f);
    CHECK(gammaLut[128] == doctest::Approx(0.25f).epsilon(0.05));
    const auto segLut = buildOpacityLut(0.0f, 1.0f, 0.5f, 1.0f, 0, 2.0f);
    CHECK(segLut[255] == doctest::Approx(1.0f));
    CHECK(segLut[128] == doctest::Approx(0.5f).epsilon(0.05));
}

TEST_CASE("compositeVolumetric identity camera with opaque TF picks the near layer")
{
    const int rows = 8, cols = 8;
    auto values = uniformLayers(3, rows, cols, 0);
    auto coverage = uniformLayers(3, rows, cols, 1);
    values[2].setTo(uint8_t(200));  // nearest to the camera (highest w)
    values[1].setTo(uint8_t(100));
    values[0].setTo(uint8_t(50));

    CameraParams cam;  // yaw = pitch = 0, perspective = 0
    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, -1, 1.0f,
                        identityColorLut(), constantOpacityLut(1.0f), color, cov);

    REQUIRE(!color.empty());
    // Interior pixel (bilinear needs x0+1 < cols).
    CHECK(cov(4, 4) == 1);
    CHECK(int(color(4, 4)[0]) == 200);
    CHECK(int(color(4, 4)[1]) == 200);
    CHECK(int(color(4, 4)[2]) == 200);
}

TEST_CASE("compositeVolumetric zero opacity keeps coverage but composites black")
{
    const int rows = 6, cols = 6;
    auto values = uniformLayers(2, rows, cols, 180);
    auto coverage = uniformLayers(2, rows, cols, 1);

    CameraParams cam;
    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, 0, 1.0f,
                        identityColorLut(), constantOpacityLut(0.0f), color, cov);

    CHECK(cov(3, 3) == 1);
    CHECK(int(color(3, 3)[0]) == 0);
}

TEST_CASE("compositeVolumetric semi-transparent layers blend front to back")
{
    const int rows = 6, cols = 6;
    auto values = uniformLayers(2, rows, cols, 0);
    auto coverage = uniformLayers(2, rows, cols, 1);
    values[1].setTo(uint8_t(100));  // near layer
    values[0].setTo(uint8_t(200));  // far layer

    CameraParams cam;
    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, 0, 1.0f,
                        identityColorLut(), constantOpacityLut(0.5f), color, cov);

    // 0.5*100 + 0.5*0.5*200 = 100
    CHECK(int(color(3, 3)[0]) == doctest::Approx(100).epsilon(0.03));
}

TEST_CASE("compositeVolumetric oblique tilt shifts higher layers")
{
    const int rows = 8, cols = 8;
    auto values = uniformLayers(2, rows, cols, 0);
    auto coverage = uniformLayers(2, rows, cols, 1);
    // Bright texel in the w=1 layer, one pixel to the +v side of the center.
    values[1](5, 4) = 250;

    CameraParams cam;
    cam.tiltDeg = 45.0f;  // azimuth 0: layer at w=1 fetched at p + 1 in v
    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, 0, 1.0f,
                        identityColorLut(), constantOpacityLut(1.0f), color, cov);

    CHECK(int(color(4, 4)[0]) == 250);  // shifted into view at (4,4)
    CHECK(int(color(5, 4)[0]) == 0);    // and no longer at its own position
}

TEST_CASE("compositeVolumetric consumes expanded cache-style layer buffers")
{
    constexpr int outRows = 10;
    constexpr int outCols = 12;
    constexpr int numLayers = 3;
    constexpr int zStart = -1;
    constexpr float outputScale = 1.0f;

    CameraParams cam;
    cam.tiltDeg = 45.0f;
    cam.azimuthDeg = 20.0f;
    const SlabMargins margins = computeSlabMargins(
        cam, numLayers, zStart, outputScale, outCols, outRows);
    REQUIRE(margins.any());

    const int rows = outRows + margins.top + margins.bottom;
    const int cols = outCols + margins.left + margins.right;
    auto values = uniformLayers(numLayers, rows, cols, 0);
    auto coverage = uniformLayers(numLayers, rows, cols, 1);

    // Cache-backed layers are sampled over an expanded (u,v) window whose
    // on-screen origin is (left, top). Put a bright sample where the center
    // screen ray intersects the nearest layer.
    const float centerU = float(margins.left) + float(outCols) * 0.5f;
    const float centerV = float(margins.top) + float(outRows) * 0.5f;
    const auto projection = slabProjection(
        cam, numLayers, zStart, outputScale, centerU, centerV);
    const int nearLayer = numLayers - 1;
    const float qU = projection.m00 * centerU + projection.m01 * centerV +
                     projection.layerOffsets[nearLayer][0];
    const float qV = projection.m10 * centerU + projection.m11 * centerV +
                     projection.layerOffsets[nearLayer][1];
    const int qX = int(std::floor(qU));
    const int qY = int(std::floor(qV));
    REQUIRE(qX >= 0);
    REQUIRE(qY >= 0);
    REQUIRE(qX + 1 < cols);
    REQUIRE(qY + 1 < rows);
    for (int y = qY; y <= qY + 1; ++y)
        for (int x = qX; x <= qX + 1; ++x)
            values[nearLayer](y, x) = 250;

    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, zStart, outputScale,
                        identityColorLut(), constantOpacityLut(1.0f),
                        color, cov, margins);

    CHECK(color.rows == outRows);
    CHECK(color.cols == outCols);
    CHECK(cov.rows == outRows);
    CHECK(cov.cols == outCols);
    CHECK(cov(outRows / 2, outCols / 2) == 1);
    CHECK(int(color(outRows / 2, outCols / 2)[0]) == 250);
}

TEST_CASE("compositeVolumetric treats uncovered texels as transparent")
{
    const int rows = 6, cols = 6;
    auto values = uniformLayers(2, rows, cols, 0);
    auto coverage = uniformLayers(2, rows, cols, 1);
    values[1].setTo(uint8_t(0));
    coverage[1].setTo(uint8_t(0));  // near layer entirely uncovered
    values[0].setTo(uint8_t(150));

    CameraParams cam;
    cv::Mat_<cv::Vec3b> color;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, 0, 1.0f,
                        identityColorLut(), constantOpacityLut(1.0f), color, cov);

    // The far layer shows through instead of a black halo.
    CHECK(cov(3, 3) == 1);
    CHECK(int(color(3, 3)[0]) == 150);
}

TEST_CASE("compositeVolumetric optionally applies gradient Lambertian lighting")
{
    const int rows = 8, cols = 8;
    auto values = uniformLayers(3, rows, cols, 0);
    auto coverage = uniformLayers(3, rows, cols, 1);
    for (auto& layer : values) {
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x)
                layer(y, x) = uint8_t(200 - 10 * x);
    }

    CameraParams cam;
    cv::Mat_<cv::Vec3b> unlit;
    cv::Mat_<cv::Vec3b> lit;
    cv::Mat_<uint8_t> cov;
    compositeVolumetric(values, coverage, cam, -1, 1.0f,
                        identityColorLut(), constantOpacityLut(1.0f), unlit, cov);
    compositeVolumetric(values, coverage, cam, -1, 1.0f,
                        identityColorLut(), constantOpacityLut(1.0f), lit, cov,
                        {}, 1.0f);

    // The -U-facing ramp receives the upper-left raking light. Lighting is
    // intentionally non-destructive to opacity/coverage and modulates only
    // the emitted RGB.
    CHECK(cov(4, 3) == 1);
    CHECK(int(unlit(4, 3)[0]) == 170);
    CHECK(int(lit(4, 3)[0]) == doctest::Approx(105).epsilon(0.03));
    CHECK(int(lit(4, 3)[1]) == int(lit(4, 3)[0]));
    CHECK(int(lit(4, 3)[2]) == int(lit(4, 3)[0]));
}


TEST_CASE("slab point mapping matches the orthographic render and round-trips")
{
    CameraParams cam;
    cam.tiltDeg = 35.0f;
    cam.azimuthDeg = 25.0f;

    const int numLayers = 9;
    const int zStart = -4;
    const float outputScale = 2.0f;
    const float halfSpan = 128.0f;  // unused at perspective == 0

    // View center at the origin so raw coords are center-relative.
    const auto proj = slabProjection(cam, numLayers, zStart, outputScale, 0.0f, 0.0f);
    const float screen[2] = {37.0f, -12.0f};
    for (int i = 0; i < numLayers; i += 4) {
        const float wPx = float(zStart + i) * outputScale;
        const float qu = proj.m00 * screen[0] + proj.m01 * screen[1] +
                         proj.layerOffsets[std::size_t(i)][0];
        const float qv = proj.m10 * screen[0] + proj.m11 * screen[1] +
                         proj.layerOffsets[std::size_t(i)][1];

        const auto s = slabPointToScreen(cam, halfSpan, {qu, qv}, wPx);
        CHECK(s[0] == doctest::Approx(screen[0]).epsilon(1e-3));
        CHECK(s[1] == doctest::Approx(screen[1]).epsilon(1e-3));

        const auto q = screenToSlabPoint(cam, halfSpan, {screen[0], screen[1]}, wPx);
        CHECK(q[0] == doctest::Approx(qu).epsilon(1e-3));
        CHECK(q[1] == doctest::Approx(qv).epsilon(1e-3));
    }
}

TEST_CASE("slab point mapping matches the pinhole render and round-trips")
{
    CameraParams cam;
    cam.tiltDeg = 40.0f;
    cam.azimuthDeg = -60.0f;
    cam.perspective = 0.7f;

    const int numLayers = 11;
    const int zStart = -5;
    const float outputScale = 1.5f;
    const float screenW = 300.0f, screenH = 200.0f;
    const float halfSpan = 0.5f * std::max(screenW, screenH);

    const auto pc = perspectiveCamera(cam, numLayers, zStart, outputScale,
                                      0.0f, 0.0f, screenW, screenH);
    const float screen[2] = {-42.0f, 31.0f};
    const float rayU = pc.rayBase[0] + screen[0] * pc.e1OverF[0] + screen[1] * pc.e2OverF[0];
    const float rayV = pc.rayBase[1] + screen[0] * pc.e1OverF[1] + screen[1] * pc.e2OverF[1];
    const float rayW = pc.rayBase[2] + screen[0] * pc.e1OverF[2] + screen[1] * pc.e2OverF[2];
    REQUIRE(rayW < 0.0f);
    for (int i = 0; i < numLayers; i += 5) {
        const float wPx = float(zStart + i) * outputScale;
        const float t = pc.layerNum[std::size_t(i)] / rayW;
        const float qu = pc.pos[0] + t * rayU;
        const float qv = pc.pos[1] + t * rayV;

        const auto s = slabPointToScreen(cam, halfSpan, {qu, qv}, wPx);
        CHECK(s[0] == doctest::Approx(screen[0]).epsilon(1e-3));
        CHECK(s[1] == doctest::Approx(screen[1]).epsilon(1e-3));

        const auto q = screenToSlabPoint(cam, halfSpan, {screen[0], screen[1]}, wPx);
        CHECK(q[0] == doctest::Approx(qu).epsilon(1e-3));
        CHECK(q[1] == doctest::Approx(qv).epsilon(1e-3));
    }
}
