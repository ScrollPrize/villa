// Coverage for GenInterpolation::Smooth — the bicubic (Catmull-Rom) resampling
// mode of QuadSurface::gen(), where the normal is differentiated analytically
// from the same basis as the positions instead of being resampled
// nearest-neighbour from the per-vertex normal cache.
//
// The motivating defect: the source grid is far coarser than the render, so in
// Linear mode the normal is piecewise-constant over each source cell and any
// offset along it (the slice stack in vc_render_tifxyz) steps at cell edges.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/CubicInterpolation.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

// A gently wavy, everywhere-curved grid: bilinear and bicubic differ on it, and
// the true normal varies continuously.
cv::Mat_<cv::Vec3f> makeWavyGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const float u = float(c), v = float(r);
            m(r, c) = cv::Vec3f(u, v, 3.0f * std::sin(0.35f * u) * std::cos(0.27f * v));
        }
    return m;
}

cv::Mat_<cv::Vec3f> makeTiltedPlaneGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const float u = float(c), v = float(r);
            // An affine map of (u, v): Catmull-Rom reproduces it exactly.
            m(r, c) = cv::Vec3f(u + 0.25f * v, 0.5f * u + v, 2.0f + 0.3f * u - 0.4f * v);
        }
    return m;
}

// P(u,v) = (R cos(u*du), R sin(u*du), v). The geometric normal dP/du x dP/dv is
// the *outward* radial direction, which pins the sign as well as the axis.
cv::Mat_<cv::Vec3f> makeCylinderGrid(int rows, int cols, float R, float du)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const float a = float(c) * du;
            m(r, c) = cv::Vec3f(R * std::cos(a), R * std::sin(a), float(r));
        }
    return m;
}

// A patch of a sphere of radius R centred at the origin. Unlike the cylinder
// both principal curvatures are non-zero, so a swapped cross-product axis shows.
cv::Mat_<cv::Vec3f> makeSphereGrid(int rows, int cols, float R, float dth)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const float phi = float(c) * dth;                 // azimuth
            const float th  = 0.35f * kPi + float(r) * dth;   // polar, away from the poles
            m(r, c) = cv::Vec3f(R * std::sin(th) * std::cos(phi),
                                R * std::sin(th) * std::sin(phi),
                                R * std::cos(th));
        }
    return m;
}

bool finiteVec(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

// Bit-identical, treating NaN as equal to NaN (gen()'s invalid marker).
bool sameOrBothNaN(float a, float b)
{
    return a == b || (std::isnan(a) && std::isnan(b));
}

void checkMatsIdentical(const cv::Mat_<cv::Vec3f>& a, const cv::Mat_<cv::Vec3f>& b)
{
    REQUIRE(a.size() == b.size());
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < a.cols; ++c)
            for (int k = 0; k < 3; ++k)
                CHECK(sameOrBothNaN(a(r, c)[k], b(r, c)[k]));
}

struct GenResult {
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
};

// Canvas origin, as vc_render_tifxyz computes it (computeCanvasOrigin): centres
// the output on the surface instead of starting at its middle.
cv::Vec3f canvasOrigin(cv::Size size)
{
    return cv::Vec3f(-0.5f * (size.width - 1.0f), -0.5f * (size.height - 1.0f), 0.f);
}

GenResult genBoth(QuadSurface& s, cv::Size size, float scale)
{
    GenResult g;
    s.gen(&g.coords, &g.normals, size, cv::Vec3f(0, 0, 0), scale, canvasOrigin(size));
    // gen() returns views into thread-local scratch that the next call reuses.
    g.coords = g.coords.clone();
    g.normals = g.normals.clone();
    return g;
}

// The source-grid coordinate an output pixel samples, reproducing gen()'s
// mapping from its public inputs: ul = (offset/scale + center) * gridScale, and
// the 4px halo is cropped away, so pixel dx lands on ul + dx*(gridScale/scale).
// Used to check which pixels are backed by a complete 4x4 stencil.
cv::Vec2d sourceCoordOf(const QuadSurface& s, cv::Size size, float scale, int dx, int dy)
{
    const cv::Vec2f gs = s.scale();
    const cv::Vec3f ctr = s.center();
    const cv::Vec3f off = canvasOrigin(size);
    const double ulx = (double(off[0]) / double(scale) + double(ctr[0])) * double(gs[0]);
    const double uly = (double(off[1]) / double(scale) + double(ctr[1])) * double(gs[1]);
    return { ulx + double(dx) * (double(gs[0]) / double(scale)),
             uly + double(dy) * (double(gs[1]) / double(scale)) };
}

// Whether this output pixel takes the cubic path rather than the bilinear
// fallback. Quality claims only apply where the cubic stencil actually runs:
// on the fallback ring smooth output is *deliberately* identical to linear.
bool cubicSupported(const QuadSurface& s, const cv::Mat_<cv::Vec3f>& pts,
                   cv::Size size, float scale, int dx, int dy)
{
    const cv::Vec2d src = sourceCoordOf(s, size, scale, dx, dy);
    return loc_valid_cubic(pts, cv::Vec2d(src[1], src[0]));   // [y, x]
}

float angleBetween(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const float d = std::clamp(a.dot(b), -1.0f, 1.0f);
    return std::acos(d);
}

} // namespace

// ---------------------------------------------------------------------------
// Basis
// ---------------------------------------------------------------------------

TEST_CASE("Catmull-Rom weights satisfy the identities the smooth mode relies on")
{
    for (float f = 0.0f; f < 1.0f; f += 0.05f) {
        float w[4], dw[4];
        vc::interp::catmullRomWeights4WithDerivative(f, w, dw);

        // Partition of unity: the interpolant reproduces constants.
        CHECK(w[0] + w[1] + w[2] + w[3] == doctest::Approx(1.0f).epsilon(1e-5));
        // Derivative of a constant is zero.
        CHECK(dw[0] + dw[1] + dw[2] + dw[3] == doctest::Approx(0.0f).epsilon(1e-5));
        // Reproduces linears: sum of w[i]*tap_position == the sample position.
        const float pos = w[0] * -1.f + w[1] * 0.f + w[2] * 1.f + w[3] * 2.f;
        CHECK(pos == doctest::Approx(f).epsilon(1e-5));
        // ...and its derivative is 1.
        const float dpos = dw[0] * -1.f + dw[1] * 0.f + dw[2] * 1.f + dw[3] * 2.f;
        CHECK(dpos == doctest::Approx(1.0f).epsilon(1e-5));

        // Consistency with the plain weight accessor.
        float w2[4];
        vc::interp::catmullRomWeights4(f, w2);
        for (int i = 0; i < 4; ++i) CHECK(w[i] == w2[i]);
    }
}

TEST_CASE("At a grid node the cubic derivative reduces to the central difference")
{
    // This is what makes the smooth normal agree with grid_normal_int exactly
    // where the two must agree, so --flip-normals and slice ordering are safe.
    float w[4], dw[4];
    vc::interp::catmullRomWeights4WithDerivative(0.0f, w, dw);
    CHECK(w[0] == doctest::Approx(0.0f));
    CHECK(w[1] == doctest::Approx(1.0f));
    CHECK(w[2] == doctest::Approx(0.0f));
    CHECK(w[3] == doctest::Approx(0.0f));
    CHECK(dw[0] == doctest::Approx(-0.5f));
    CHECK(dw[1] == doctest::Approx(0.0f));
    CHECK(dw[2] == doctest::Approx(0.5f));
    CHECK(dw[3] == doctest::Approx(0.0f));
}

TEST_CASE("catmullRomWeight matches the tabulated Catmull-Rom basis")
{
    // Guards the extraction of this kernel out of Slicing.cpp's tricubic
    // volume sampler, which must stay bit-identical.
    struct { float t, w; } table[] = {
        {  0.00f,  1.0f     }, {  0.50f,  0.5625f }, {  1.00f,  0.0f   },
        {  1.50f, -0.0625f  }, {  2.00f,  0.0f    }, {  2.50f,  0.0f   },
        { -0.50f,  0.5625f  }, { -1.50f, -0.0625f }, { -1.00f,  0.0f   },
    };
    for (auto& e : table)
        CHECK(vc::interp::catmullRomWeight(e.t) == doctest::Approx(e.w).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// Default behaviour must not move
// ---------------------------------------------------------------------------

TEST_CASE("default interpolation is Linear and is bit-identical to today")
{
    auto pts = makeWavyGrid(24, 24);
    pts(7, 9)   = cv::Vec3f(-1.f, -1.f, -1.f);
    pts(13, 16) = cv::Vec3f(-1.f, -1.f, -1.f);

    QuadSurface implicitDefault(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface explicitLinear(pts, cv::Vec2f(1.f, 1.f));
    explicitLinear.setGenInterpolation(GenInterpolation::Linear);

    CHECK(implicitDefault.genInterpolation() == GenInterpolation::Linear);

    auto a = genBoth(implicitDefault, cv::Size(64, 64), 4.0f);
    auto b = genBoth(explicitLinear, cv::Size(64, 64), 4.0f);
    checkMatsIdentical(a.coords, b.coords);
    checkMatsIdentical(a.normals, b.normals);
}

TEST_CASE("Linear mode still produces piecewise-constant normals")
{
    // The defect the smooth mode exists to fix. If this ever stops holding,
    // the discrimination in the continuity test below is no longer meaningful.
    auto pts = makeWavyGrid(16, 16);
    QuadSurface s(pts, cv::Vec2f(1.f, 1.f));
    auto g = genBoth(s, cv::Size(120, 120), 10.0f);  // 10 output px per source cell

    int identicalNeighbours = 0, total = 0;
    const int row = g.normals.rows / 2;
    for (int c = 1; c < g.normals.cols; ++c) {
        if (!finiteVec(g.normals(row, c)) || !finiteVec(g.normals(row, c - 1))) continue;
        ++total;
        if (g.normals(row, c) == g.normals(row, c - 1)) ++identicalNeighbours;
    }
    REQUIRE(total > 20);
    // Nearest-neighbour resampling => most neighbours share a normal exactly.
    CHECK(identicalNeighbours * 2 > total);
}

// ---------------------------------------------------------------------------
// Smooth mode: geometry
// ---------------------------------------------------------------------------

TEST_CASE("smooth mode reproduces a plane exactly, with the exact constant normal")
{
    auto pts = makeTiltedPlaneGrid(20, 20);
    QuadSurface s(pts, cv::Vec2f(1.f, 1.f));
    s.setGenInterpolation(GenInterpolation::Smooth);
    auto g = genBoth(s, cv::Size(96, 96), 6.0f);

    // dP/du = (1, 0.5, 0.3), dP/dv = (0.25, 1, -0.4) from makeTiltedPlaneGrid.
    const cv::Vec3f du(1.0f, 0.5f, 0.3f), dv(0.25f, 1.0f, -0.4f);
    cv::Vec3f expect = du.cross(dv);
    expect /= float(cv::norm(expect));

    int checked = 0;
    for (int r = 0; r < g.coords.rows; ++r)
        for (int c = 0; c < g.coords.cols; ++c) {
            const cv::Vec3f& p = g.coords(r, c);
            if (!finiteVec(p)) continue;
            // Catmull-Rom reproduces affine maps exactly, so the rendered
            // point must satisfy the plane equation through p0 = (0, 0, 2)
            // (the grid value at u = v = 0) with the analytic normal.
            const cv::Vec3f d = p - cv::Vec3f(0.f, 0.f, 2.f);
            CHECK(std::abs(d.dot(expect)) < 1e-3f);
            if (finiteVec(g.normals(r, c))) {
                CHECK(angleBetween(g.normals(r, c), expect) < 1e-3f);
                ++checked;
            }
        }
    REQUIRE(checked > 1000);
}

TEST_CASE("smooth normals track a cylinder's analytic normal far better than linear")
{
    const float R = 40.0f, du = 0.05f;
    auto pts = makeCylinderGrid(20, 40, R, du);

    QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
    smo.setGenInterpolation(GenInterpolation::Smooth);

    auto gl = genBoth(lin, cv::Size(200, 100), 5.0f);
    auto gs = genBoth(smo, cv::Size(200, 100), 5.0f);

    double sumLin = 0.0, sumSmo = 0.0, maxSmo = 0.0;
    int n = 0;
    for (int r = 0; r < gs.coords.rows; ++r)
        for (int c = 0; c < gs.coords.cols; ++c) {
            const cv::Vec3f& p = gs.coords(r, c);
            if (!finiteVec(p) || !finiteVec(gs.normals(r, c))
                || !finiteVec(gl.normals(r, c))) continue;
            if (!cubicSupported(smo, pts, cv::Size(200, 100), 5.0f, c, r)) continue;
            // dP/du x dP/dv points outward, so the expected normal is radial
            // and outward — this pins the sign, not just the axis.
            cv::Vec3f radial(p[0], p[1], 0.f);
            const float len = float(cv::norm(radial));
            if (len < 1e-3f) continue;
            radial /= len;
            const double aSmo = angleBetween(gs.normals(r, c), radial);
            const double aLin = angleBetween(gl.normals(r, c), radial);
            sumSmo += aSmo; sumLin += aLin; maxSmo = std::max(maxSmo, aSmo);
            ++n;
        }
    REQUIRE(n > 1000);
    const double meanSmo = sumSmo / n, meanLin = sumLin / n;
    INFO("mean angular error: smooth=" << meanSmo << " linear=" << meanLin);
    CHECK(meanSmo < meanLin / 10.0);
    CHECK(maxSmo < 0.02);   // ~1 degree worst case
}

TEST_CASE("smooth normals track a sphere's analytic normal, with the right sign")
{
    const float R = 60.0f, dth = 0.04f;
    auto pts = makeSphereGrid(24, 24, R, dth);

    QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
    smo.setGenInterpolation(GenInterpolation::Smooth);

    auto gl = genBoth(lin, cv::Size(160, 160), 8.0f);
    auto gs = genBoth(smo, cv::Size(160, 160), 8.0f);

    double sumLin = 0.0, sumSmo = 0.0;
    int n = 0, signAgree = 0;
    for (int r = 0; r < gs.coords.rows; ++r)
        for (int c = 0; c < gs.coords.cols; ++c) {
            const cv::Vec3f& p = gs.coords(r, c);
            if (!finiteVec(p) || !finiteVec(gs.normals(r, c))
                || !finiteVec(gl.normals(r, c))) continue;
            if (!cubicSupported(smo, pts, cv::Size(160, 160), 8.0f, c, r)) continue;
            cv::Vec3f radial = p / float(cv::norm(p));
            // Both modes must agree on orientation; compare each against the
            // radial axis, allowing a consistent global sign.
            const cv::Vec3f& ns = gs.normals(r, c);
            const double aSmo = std::min(angleBetween(ns, radial),
                                         angleBetween(ns, -radial));
            const double aLin = std::min(angleBetween(gl.normals(r, c), radial),
                                         angleBetween(gl.normals(r, c), -radial));
            if (ns.dot(gl.normals(r, c)) > 0) ++signAgree;
            sumSmo += aSmo; sumLin += aLin;
            ++n;
        }
    REQUIRE(n > 1000);
    INFO("mean angular error: smooth=" << sumSmo / n << " linear=" << sumLin / n);
    CHECK(sumSmo / n < sumLin / n / 10.0);
    // Smooth must not flip the normal relative to Linear anywhere.
    CHECK(signAgree == n);
}

TEST_CASE("smooth normals are continuous across source-cell boundaries")
{
    auto pts = makeWavyGrid(16, 16);
    QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
    smo.setGenInterpolation(GenInterpolation::Smooth);

    const float scale = 10.0f;  // 10 output pixels per source cell
    auto gl = genBoth(lin, cv::Size(120, 120), scale);
    auto gs = genBoth(smo, cv::Size(120, 120), scale);

    // Only the cubic interior: on the fallback ring smooth deliberately
    // reproduces linear, including its step.
    auto maxJumpAlongRow = [&](const cv::Mat_<cv::Vec3f>& nrm, int row, int* n) {
        double worst = 0.0;
        for (int c = 1; c < nrm.cols; ++c) {
            const cv::Vec3f& a = nrm(row, c - 1);
            const cv::Vec3f& b = nrm(row, c);
            if (!finiteVec(a) || !finiteVec(b)) continue;
            if (!cubicSupported(smo, pts, cv::Size(120, 120), scale, c, row)
                || !cubicSupported(smo, pts, cv::Size(120, 120), scale, c - 1, row)) continue;
            ++*n;
            worst = std::max(worst, double(cv::norm(b - a)));
        }
        return worst;
    };

    const int row = 60;
    int nLin = 0, nSmo = 0;
    const double jumpLin = maxJumpAlongRow(gl.normals, row, &nLin);
    const double jumpSmo = maxJumpAlongRow(gs.normals, row, &nSmo);
    REQUIRE(nSmo > 40);
    INFO("max consecutive normal jump: smooth=" << jumpSmo << " linear=" << jumpLin);
    // Linear steps by a whole inter-vertex normal difference at each cell edge;
    // smooth changes by roughly 1/scale of that per pixel.
    CHECK(jumpSmo < jumpLin / 4.0);
    CHECK(jumpSmo < 0.05);
}

TEST_CASE("offset layers are continuous in smooth mode")
{
    // The defect this mode exists to fix, measured directly: vc_render_tifxyz
    // samples the volume at base + normal*offset for each slice, so a normal
    // that steps at cell edges tears every non-zero layer apart. Here we build
    // that offset surface ourselves and measure its worst pixel-to-pixel jump.
    auto pts = makeWavyGrid(16, 16);
    QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
    smo.setGenInterpolation(GenInterpolation::Smooth);

    const float scale = 10.0f;               // 10 output px per source cell
    const cv::Size size(120, 120);
    const float offset = 8.0f;               // a deep layer, like --num-slices 17

    auto gl = genBoth(lin, size, scale);
    auto gs = genBoth(smo, size, scale);

    auto worstJump = [&](const GenResult& g) {
        double worst = 0.0;
        const int row = size.height / 2;
        for (int c = 1; c < g.coords.cols; ++c) {
            auto layer = [&](int x) { return g.coords(row, x) + g.normals(row, x) * offset; };
            if (!finiteVec(g.coords(row, c)) || !finiteVec(g.normals(row, c))
                || !finiteVec(g.coords(row, c - 1)) || !finiteVec(g.normals(row, c - 1)))
                return worst;  // keep the scan contiguous
            if (!cubicSupported(smo, pts, size, scale, c, row)
                || !cubicSupported(smo, pts, size, scale, c - 1, row)) continue;
            worst = std::max(worst, double(cv::norm(layer(c) - layer(c - 1))));
        }
        return worst;
    };

    const double jumpLin = worstJump(gl);
    const double jumpSmo = worstJump(gs);
    INFO("worst offset-layer jump: smooth=" << jumpSmo << " linear=" << jumpLin);
    REQUIRE(jumpLin > 0.0);
    // Linear tears by a sizeable fraction of the offset distance; smooth should
    // move by roughly one pixel's worth of surface, i.e. ~1/scale.
    CHECK(jumpSmo < jumpLin / 4.0);
    CHECK(jumpSmo < 0.5);
}

// ---------------------------------------------------------------------------
// Smooth mode: validity, holes and footprint
// ---------------------------------------------------------------------------

TEST_CASE("smooth mode leaves the valid footprint unchanged")
{
    auto pts = makeWavyGrid(20, 20);
    pts(6, 6) = cv::Vec3f(-1.f, -1.f, -1.f);
    pts(6, 7) = cv::Vec3f(-1.f, -1.f, -1.f);
    pts(14, 3) = cv::Vec3f(-1.f, -1.f, -1.f);

    auto countFinite = [](const cv::Mat_<cv::Vec3f>& m) {
        int n = 0;
        for (int r = 0; r < m.rows; ++r)
            for (int c = 0; c < m.cols; ++c) n += finiteVec(m(r, c));
        return n;
    };

    for (bool strict : {false, true}) {
        QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
        QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
        lin.setStrictQuadRenderValidity(strict);
        smo.setStrictQuadRenderValidity(strict);
        smo.setGenInterpolation(GenInterpolation::Smooth);

        auto gl = genBoth(lin, cv::Size(100, 100), 5.0f);
        auto gs = genBoth(smo, cv::Size(100, 100), 5.0f);
        INFO("strict=" << strict);
        CHECK(countFinite(gl.coords) == countFinite(gs.coords));
    }
}

TEST_CASE("pixels without complete 4x4 support fall back to the exact Linear result")
{
    auto pts = makeWavyGrid(20, 20);
    pts(9, 9) = cv::Vec3f(-1.f, -1.f, -1.f);

    QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
    smo.setGenInterpolation(GenInterpolation::Smooth);

    const float scale = 5.0f;
    auto gl = genBoth(lin, cv::Size(100, 100), scale);
    auto gs = genBoth(smo, cv::Size(100, 100), scale);

    int fallbackPixels = 0, cubicPixels = 0;
    for (int r = 0; r < gs.coords.rows; ++r)
        for (int c = 0; c < gs.coords.cols; ++c) {
            const cv::Vec2d src = sourceCoordOf(smo, cv::Size(100, 100), scale, c, r);
            // loc_valid_cubic takes [y, x].
            const bool supported = loc_valid_cubic(pts, cv::Vec2d(src[1], src[0]));
            if (supported) { ++cubicPixels; continue; }
            ++fallbackPixels;
            for (int k = 0; k < 3; ++k) {
                CHECK(sameOrBothNaN(gs.coords(r, c)[k], gl.coords(r, c)[k]));
                CHECK(sameOrBothNaN(gs.normals(r, c)[k], gl.normals(r, c)[k]));
            }
        }
    // The hole plus the grid border must produce a meaningful fallback region,
    // and the interior must actually take the cubic path.
    REQUIRE(fallbackPixels > 100);
    REQUIRE(cubicPixels > 1000);
}

TEST_CASE("loc_valid_cubic rejects incomplete support, NaN and the -1 sentinel")
{
    auto pts = makeWavyGrid(10, 10);
    // Interior with full support.
    CHECK(loc_valid_cubic(pts, cv::Vec2d(4.5, 4.5)));
    // The stencil reaches [i-1, i+2], so the outer ring has no support.
    CHECK_FALSE(loc_valid_cubic(pts, cv::Vec2d(0.5, 4.5)));
    CHECK_FALSE(loc_valid_cubic(pts, cv::Vec2d(4.5, 0.5)));
    CHECK_FALSE(loc_valid_cubic(pts, cv::Vec2d(8.5, 4.5)));
    CHECK_FALSE(loc_valid_cubic(pts, cv::Vec2d(4.5, 8.5)));
    // Too small a grid entirely.
    CHECK_FALSE(loc_valid_cubic(makeWavyGrid(3, 3), cv::Vec2d(1.5, 1.5)));
    // Non-finite input.
    CHECK_FALSE(loc_valid_cubic(pts, cv::Vec2d(std::nan(""), 4.5)));

    // Both spellings of "missing" must be rejected: vc_render_tifxyz rewrites
    // the -1 sentinel to NaN before rendering, so both occur in practice.
    auto sentinel = makeWavyGrid(10, 10);
    sentinel(6, 6) = cv::Vec3f(-1.f, -1.f, -1.f);
    CHECK_FALSE(loc_valid_cubic(sentinel, cv::Vec2d(4.5, 4.5)));   // 6 is in [3,6]
    auto nanned = makeWavyGrid(10, 10);
    nanned(6, 6) = cv::Vec3f(NAN, NAN, NAN);
    CHECK_FALSE(loc_valid_cubic(nanned, cv::Vec2d(4.5, 4.5)));

    // The xy-swapped form must agree with the coordinates transposed.
    auto asym = makeWavyGrid(12, 12);
    asym(3, 7) = cv::Vec3f(-1.f, -1.f, -1.f);
    CHECK(loc_valid_cubic(asym, cv::Vec2d(4.5, 6.5))
          == loc_valid_cubic_xy(asym, cv::Vec2d(6.5, 4.5)));
}

TEST_CASE("smooth mode does not blow up near holes")
{
    // Catmull-Rom overshoots on high-curvature data; the fallback rule is what
    // keeps the riskiest neighbourhoods (hole edges) on the bilinear path.
    auto pts = makeWavyGrid(24, 24);
    for (int r = 8; r < 12; ++r)
        for (int c = 8; c < 12; ++c) pts(r, c) = cv::Vec3f(-1.f, -1.f, -1.f);

    QuadSurface lin(pts, cv::Vec2f(1.f, 1.f));
    QuadSurface smo(pts, cv::Vec2f(1.f, 1.f));
    smo.setGenInterpolation(GenInterpolation::Smooth);
    auto gl = genBoth(lin, cv::Size(120, 120), 5.0f);
    auto gs = genBoth(smo, cv::Size(120, 120), 5.0f);

    double worst = 0.0;
    for (int r = 0; r < gs.coords.rows; ++r)
        for (int c = 0; c < gs.coords.cols; ++c) {
            if (!finiteVec(gs.coords(r, c)) || !finiteVec(gl.coords(r, c))) continue;
            worst = std::max(worst, double(cv::norm(gs.coords(r, c) - gl.coords(r, c))));
        }
    INFO("max |P_smooth - P_linear| = " << worst);
    // One source cell is 1 unit across in these fixtures; the cubic correction
    // must stay a fraction of that, not a wild excursion.
    CHECK(worst < 0.5);
    CHECK(worst > 0.0);   // ...but it must actually differ, or nothing happened
}

TEST_CASE("smooth mode keeps component seams intact")
{
    const int rows = 20, cols = 24;
    auto pts = makeWavyGrid(rows, cols);
    std::vector<std::pair<int, int>> comps = {{0, 12}, {12, 24}};

    auto render = [&](const cv::Mat_<cv::Vec3f>& p) {
        auto s = std::make_unique<QuadSurface>(p, cv::Vec2f(1.f, 1.f));
        s->setComponents(comps);
        s->setGenInterpolation(GenInterpolation::Smooth);
        return genBoth(*s, cv::Size(120, 100), 5.0f);
    };

    auto base = render(pts);
    // Perturb the far side of the seam only. Nothing on the near side may move.
    auto perturbed = pts.clone();
    for (int r = 0; r < rows; ++r)
        for (int c = 12; c < cols; ++c) perturbed(r, c)[2] += 5.0f;
    auto after = render(perturbed);

    QuadSurface mapping(pts, cv::Vec2f(1.f, 1.f));
    mapping.setComponents(comps);
    int comparedLeft = 0;
    for (int r = 0; r < base.coords.rows; ++r)
        for (int c = 0; c < base.coords.cols; ++c) {
            const cv::Vec2d src = sourceCoordOf(mapping, cv::Size(120, 100), 5.0f, c, r);
            if (src[0] >= 11.0) continue;   // only the left component
            if (!finiteVec(base.coords(r, c))) continue;
            ++comparedLeft;
            for (int k = 0; k < 3; ++k)
                CHECK(sameOrBothNaN(base.coords(r, c)[k], after.coords(r, c)[k]));
        }
    INFO("left-component pixels compared: " << comparedLeft);
    REQUIRE(comparedLeft > 500);
}

TEST_CASE("smooth mode is safe under concurrent gen() calls")
{
    // The renderer calls gen() per tile from an OMP parallel loop, so the new
    // support-mask cache is built under contention exactly like _normalCache.
    auto pts = makeWavyGrid(40, 40);
    pts(10, 10) = cv::Vec3f(-1.f, -1.f, -1.f);
    QuadSurface s(pts, cv::Vec2f(1.f, 1.f));
    s.setGenInterpolation(GenInterpolation::Smooth);

    const cv::Size size(96, 96);
    const float scale = 4.0f;

    // Reference, computed single-threaded on a warm cache.
    auto ref = genBoth(s, size, scale);

    std::vector<GenResult> out(8);
    std::vector<std::thread> threads;
    QuadSurface fresh(pts, cv::Vec2f(1.f, 1.f));   // cold: all 8 race the build
    fresh.setGenInterpolation(GenInterpolation::Smooth);
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([&, i] { out[i] = genBoth(fresh, size, scale); });
    for (auto& t : threads) t.join();

    for (int i = 0; i < 8; ++i) {
        INFO("thread " << i);
        checkMatsIdentical(ref.coords, out[i].coords);
        checkMatsIdentical(ref.normals, out[i].normals);
    }
}
