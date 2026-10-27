#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/SheetStitch.hpp"
#include "vc/core/types/SheetCollection.hpp"

#include <cmath>
#include <vector>

using vc::sheets::Mesh;
using vc::sheets::Point;
using vc::sheets::stitchSheets;
using vc::sheets::StitchResult;

namespace
{
// A full (no holes) nu*nv planar patch: vertex (i, j) sits at
//   base + i*step*a + j*step*b + height(i, j)*nrm
// with (a, b, nrm) an orthonormal frame. Vertices are in row-major scan order,
// matching the .asv decode convention the stitcher assumes.
template <typename HeightFn>
Mesh planarPatch(int64_t nu, int64_t nv, Point base, Point a, Point b, Point nrm,
                 double step, HeightFn height)
{
    Mesh m;
    m.nu = nu;
    m.nv = nv;
    m.foot.assign(size_t(nu * nv), 1);
    for (int64_t i = 0; i < nu; ++i)
        for (int64_t j = 0; j < nv; ++j) {
            const double h = height(i, j);
            m.verts.push_back({base[0] + i * step * a[0] + j * step * b[0] + h * nrm[0],
                               base[1] + i * step * a[1] + j * step * b[1] + h * nrm[1],
                               base[2] + i * step * a[2] + j * step * b[2] + h * nrm[2]});
        }
    return m;
}

// Distance from a world point to the plane z = z0 (the test patches live in z).
double distToZ(const Point& p, double z0) { return std::abs(p[2] - z0); }
}  // namespace

TEST_CASE("single flat sheet reproduces its plane")
{
    const Point X{1, 0, 0}, Y{0, 1, 0}, Z{0, 0, 1};
    Mesh m = planarPatch(6, 7, {100, 200, 50}, X, Y, Z, 10.0, [](int64_t, int64_t) { return 0.0; });

    StitchResult r = stitchSheets({m});
    REQUIRE(r.ok());
    CHECK(r.cell == doctest::Approx(10.0));
    CHECK(r.foldCells == 0);
    // Every filled node lies on z = 50.
    double worst = 0.0;
    for (int rr = 0; rr < r.rows; ++rr)
        for (int cc = 0; cc < r.cols; ++cc)
            if (r.validAt(rr, cc))
                worst = std::max(worst, distToZ(r.at(rr, cc), 50.0));
    CHECK(worst < 1e-3);
    // The footprint (6x7 at step 10, plus a margin) is substantially covered.
    CHECK(r.filledCells >= 6 * 7);
}

TEST_CASE("two overlapping coplanar sheets blend seamlessly")
{
    const Point X{1, 0, 0}, Y{0, 1, 0}, Z{0, 0, 1};
    // Both patches lie on z = 100; patch B starts 30 voxels along +x so they overlap.
    Mesh a = planarPatch(8, 8, {0, 0, 100}, X, Y, Z, 10.0, [](int64_t, int64_t) { return 0.0; });
    Mesh b = planarPatch(8, 8, {30, 0, 100}, X, Y, Z, 10.0, [](int64_t, int64_t) { return 0.0; });

    StitchResult r = stitchSheets({a, b});
    REQUIRE(r.ok());
    CHECK(r.foldCells == 0);        // coplanar -> no multivalued cells
    CHECK(r.maxSpreadVox < 1.0);
    // Union spans x in [0, 100] (8 cols each, offset 30) -> wider than one patch.
    CHECK(r.cols > 8);
    // Seamless: every blended node is on the shared plane, including the overlap band.
    double worst = 0.0;
    for (int rr = 0; rr < r.rows; ++rr)
        for (int cc = 0; cc < r.cols; ++cc)
            if (r.validAt(rr, cc))
                worst = std::max(worst, distToZ(r.at(rr, cc), 100.0));
    CHECK(worst < 1e-2);
}

TEST_CASE("two sheets stacked along the normal are flagged as a fold")
{
    const Point X{1, 0, 0}, Y{0, 1, 0}, Z{0, 0, 1};
    // Wide in x/y so PCA picks Z as the normal; two parallel planes 40 apart over the
    // same footprint are multivalued over the plane -> must be reported.
    Mesh lo = planarPatch(20, 20, {0, 0, 100}, X, Y, Z, 10.0, [](int64_t, int64_t) { return 0.0; });
    Mesh hi = planarPatch(20, 20, {0, 0, 140}, X, Y, Z, 10.0, [](int64_t, int64_t) { return 0.0; });

    StitchResult r = stitchSheets({lo, hi});
    REQUIRE(r.ok());
    CHECK(r.foldCells > 0);
    CHECK(r.maxSpreadVox > 20.0);
}

TEST_CASE("empty input is handled")
{
    CHECK_FALSE(stitchSheets({}).ok());
    Mesh empty;
    empty.nu = empty.nv = 0;
    CHECK_FALSE(stitchSheets({empty}).ok());
}
