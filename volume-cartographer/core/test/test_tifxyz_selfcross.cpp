// Regression fixtures for the vc_tifxyz_selfcross census kernel.
//
// The kernel classifies triangle-pair contacts as transverse, coplanar or
// grazing, and the distinctions are the point: two sheets pressed flat are
// coplanar, an exact edge-on meeting is a touch, and only interiors passing
// through each other is transverse. Synthetic fixtures are full of exactly
// those coincidences -- a sheet whose vertex row lands on the other's plane
// TOUCHES it, one at exactly a vertex column meets it along mesh edges --
// so each fixture below states which trap it exists to pin down.
//
// The grids follow the stitch trick: two sub-sheets separated by a band of
// invalid rows wider than the adjacency exclusion, so every contact between
// them is non-adjacent by construction.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "../../apps/src/vc_tifxyz_selfcross_impl.hpp"

#include <opencv2/core/mat.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using vc_selfcross::Contact;
using vc_selfcross::CensusCounts;
using vc_selfcross::census;
using vc_selfcross::COPLANAR;
using vc_selfcross::GRAZING;
using vc_selfcross::NONE;
using vc_selfcross::TRANSVERSE;
using vc_selfcross::Tri;
using vc_selfcross::Vec3;
using vc_selfcross::tri_tri;

namespace {

constexpr int GAP = 4;   // invalid rows between sub-sheets; exclusion is 1

const cv::Vec3f INVALID(-1.f, -1.f, -1.f);

// A flat sheet in the plane z = z0, rows*cols, pitch 4.
void put_plane(cv::Mat_<cv::Vec3f>& P, int row0, int rows, int cols,
               double z0)
{
    for (int v = 0; v < rows; ++v)
        for (int u = 0; u < cols; ++u)
            P(row0 + v, u) = cv::Vec3f(4.f * u, 4.f * v, (float)z0);
}

cv::Mat_<cv::Vec3f> blank(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> P(rows, cols);
    P.setTo(INVALID);
    return P;
}

struct Run {
    CensusCounts counts;
    std::vector<Contact> contacts;
};

Run run(const cv::Mat_<cv::Vec3f>& P, int diagonal, double cell = 40.0,
        int exclude = 1, double maxedge = 60.0, int threads = 2)
{
    Run r;
    r.counts = census(P, diagonal, cell, exclude, maxedge, threads,
                      r.contacts);
    return r;
}

}  // namespace

TEST_CASE("a flat sheet censuses clean under both triangulations")
{
    cv::Mat_<cv::Vec3f> P = blank(12, 16);
    put_plane(P, 0, 12, 16, 5.0);
    for (int d = 0; d < 2; ++d) {
        Run r = run(P, d);
        CHECK(r.counts.triangles == 11u * 15u * 2u);
        CHECK(r.counts.transverse == 0u);
        CHECK(r.counts.coplanar == 0u);
        CHECK(r.counts.grazing == 0u);
    }
}

TEST_CASE("a sheet driven through a plane is transverse, and the site lies "
          "on the crossing")
{
    // The second sub-sheet is TILTED and OFFSET. An axis-aligned sheet at a
    // whole multiple of the grid pitch meets the plane exactly along mesh
    // edges, which is a touch, not a penetration -- the fixture that taught
    // this is the grazing test below.
    cv::Mat_<cv::Vec3f> P = blank(10 + GAP + 10, 12);
    put_plane(P, 0, 10, 12, 5.0);
    for (int v = 0; v < 10; ++v)
        for (int u = 0; u < 12; ++u)
            P(10 + GAP + v, u) = cv::Vec3f(
                4.f * u + 0.3f * v + 1.7f,          // tilted in x
                4.f * v,
                (float)(4.0 * (u - 5) + 2.0 + 5.0)); // crosses z = 5 off-grid
    for (int d = 0; d < 2; ++d) {
        Run r = run(P, d);
        CHECK(r.counts.transverse > 0u);
        for (const Contact& c : r.contacts) {
            if (c.verdict != TRANSVERSE) continue;
            // one end on each side of the invalid band
            const bool v1_lower = c.v1 < 10, v2_lower = c.v2 < 10;
            CHECK(v1_lower != v2_lower);
            // the reported site sits on the plane sub-sheet (z = 5) to
            // within the contact's own penetration depth -- a centroid
            // midpoint would not satisfy this for skew pairs
            CHECK(std::fabs(c.site.z - 5.0) <= c.penetration + 1e-6);
            CHECK(c.penetration > 0.0f);
        }
    }
}

TEST_CASE("two sheets pressed flat are coplanar, never transverse")
{
    cv::Mat_<cv::Vec3f> P = blank(8 + GAP + 8, 10);
    put_plane(P, 0, 8, 10, 5.0);
    put_plane(P, 8 + GAP, 8, 10, 5.0);   // the same geometry again
    for (int d = 0; d < 2; ++d) {
        Run r = run(P, d);
        CHECK(r.counts.transverse == 0u);
        CHECK(r.counts.coplanar > 0u);
    }
}

TEST_CASE("an exact edge-on meeting is grazing, not transverse")
{
    // A vertical sheet whose bottom vertex ROW lies exactly in the plane
    // z = 5: every meeting is along mesh edges and vertices, where the sign
    // data the interval test relies on is not trustworthy. The kernel must
    // report the uncertainty rather than promote it to a crossing.
    cv::Mat_<cv::Vec3f> P = blank(8 + GAP + 8, 10);
    put_plane(P, 0, 8, 10, 5.0);
    for (int v = 0; v < 8; ++v)
        for (int u = 0; u < 10; ++u)
            P(8 + GAP + v, u) = cv::Vec3f(4.f * u, 4.f * v, 5.f + 4.f * u);
    for (int d = 0; d < 2; ++d) {
        Run r = run(P, d);
        CHECK(r.counts.transverse == 0u);
        CHECK(r.counts.grazing > 0u);
    }
}

TEST_CASE("adjacency exclusion is honoured and reported pairs are canonical")
{
    // Reuse the transverse fixture: no emitted contact may join quads
    // within the Chebyshev exclusion window, and every row must carry the
    // lexicographically smaller endpoint first, so a reader sees exactly
    // one identity per geometric pair.
    cv::Mat_<cv::Vec3f> P = blank(10 + GAP + 10, 12);
    put_plane(P, 0, 10, 12, 5.0);
    for (int v = 0; v < 10; ++v)
        for (int u = 0; u < 12; ++u)
            P(10 + GAP + v, u) = cv::Vec3f(4.f * u + 0.3f * v + 1.7f, 4.f * v,
                                           (float)(4.0 * (u - 5) + 7.0));
    Run r = run(P, 0);
    REQUIRE(!r.contacts.empty());
    for (const Contact& c : r.contacts) {
        const bool adjacent = std::abs(c.v1 - c.v2) <= 1
                           && std::abs(c.u1 - c.u2) <= 1;
        CHECK(!adjacent);
        CHECK(std::make_tuple(c.v1, c.u1, c.t1)
              <= std::make_tuple(c.v2, c.u2, c.t2));
    }
}

TEST_CASE("disjoint triangles near each other's infinite plane are not "
          "grazing")
{
    // Two nearly coplanar right triangles whose bounding boxes overlap at a
    // corner while the triangles themselves stay ~2.7 apart. Every vertex
    // of one lies within plane tolerance of the OTHER's infinite plane, but
    // proximity to an infinite plane says nothing about touching: reporting
    // this as grazing inflates the count with pairs that are simply
    // disjoint. The verdict must be no contact at all.
    Tri a{{0, 0, 5}, {4, 0, 5}, {0, 4, 5}, 0, 0, 0};
    Tri b{{3.9, 3.9, 5.000004}, {8, 3.9, 5.000008}, {3.9, 8, 5.000008},
          10, 10, 0};
    CHECK(tri_tri(a, b).verdict == NONE);

    // ...while the same pair moved into genuine contact IS reported: slide
    // b so its corner vertex sits on a's interior.
    Tri c{{1.0, 1.0, 5.000004}, {8, 1.0, 5.000008}, {1.0, 8, 5.000008},
          10, 10, 0};
    auto r = tri_tri(a, c);
    CHECK(r.verdict != NONE);
}

TEST_CASE("results do not depend on thread count or broad-phase cell size")
{
    // A pair can share several broad-phase cells; it is owned by exactly
    // one, so neither scheduling nor the cell size may move any count.
    // Historically a per-thread dedup produced counts that tracked the
    // thread count, and a missing bbox pre-filter produced grazing counts
    // that tracked the cell size -- this pins both down.
    cv::Mat_<cv::Vec3f> P = blank(10 + GAP + 10, 12);
    put_plane(P, 0, 10, 12, 5.0);
    for (int v = 0; v < 10; ++v)
        for (int u = 0; u < 12; ++u)
            P(10 + GAP + v, u) = cv::Vec3f(4.f * u + 0.3f * v + 1.7f, 4.f * v,
                                           (float)(4.0 * (u - 5) + 7.0));
    Run ref = run(P, 0, 40.0, 1, 60.0, 1);
    REQUIRE(ref.counts.transverse > 0u);
    for (int threads : {2, 5}) {
        Run r = run(P, 0, 40.0, 1, 60.0, threads);
        CHECK(r.contacts.size() == ref.contacts.size());
        for (size_t i = 0; i < r.contacts.size(); ++i) {
            CHECK(r.contacts[i].v1 == ref.contacts[i].v1);
            CHECK(r.contacts[i].u1 == ref.contacts[i].u1);
            CHECK(r.contacts[i].v2 == ref.contacts[i].v2);
            CHECK(r.contacts[i].u2 == ref.contacts[i].u2);
            CHECK(r.contacts[i].verdict == ref.contacts[i].verdict);
        }
    }
    for (double cell : {7.0, 21.0, 160.0}) {
        Run r = run(P, 0, cell);
        CHECK(r.counts.transverse == ref.counts.transverse);
        CHECK(r.counts.coplanar == ref.counts.coplanar);
        CHECK(r.counts.grazing == ref.counts.grazing);
    }
}

TEST_CASE("sub-tolerance contacts are found at every cell size")
{
    // The narrow phase accepts pairs whose boxes come within TOUCH_TOL, so
    // the broad phase must bucket expanded boxes -- otherwise a pair
    // separated by less than the tolerance can straddle a bucket boundary
    // at one cell size and share a bucket at another, and the counts move
    // with --cell. Two flat sheets 5e-5 apart -- inside the plane tolerance
    // at this magnitude, so every pair classifies (as coplanar), and inside
    // the gap a bucket boundary could fall into. The counts must agree
    // across cell sizes, and the fixture is offset in x so geometry sits
    // near bucket boundaries at the default size.
    cv::Mat_<cv::Vec3f> P = blank(8 + GAP + 8, 10);
    for (int v = 0; v < 8; ++v)
        for (int u = 0; u < 10; ++u) {
            P(v, u) = cv::Vec3f(4.f * u + 39.9f, 4.f * v, 5.f);
            P(8 + GAP + v, u) = cv::Vec3f(4.f * u + 39.9f, 4.f * v,
                                          5.f + 5e-5f);
        }
    Run ref = run(P, 0, 40.0);
    CHECK(ref.counts.coplanar + ref.counts.grazing > 0u);
    for (double cell : {5.0, 39.95, 100.0, 400.0}) {
        Run r = run(P, 0, cell);
        CHECK(r.counts.transverse == ref.counts.transverse);
        CHECK(r.counts.coplanar == ref.counts.coplanar);
        CHECK(r.counts.grazing == ref.counts.grazing);
    }
}

TEST_CASE("a cell size far below the surface's extent is refused, not "
          "undefined")
{
    // Positive and finite is not enough: span/cell can overflow the index
    // arithmetic, and each triangle's bucket loop can turn astronomical.
    // The refusal must fire on the DOUBLE quotient, before any integer
    // conversion, so the denormal-smallest cell is the probe that matters.
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 0, 6, 6, 5.0);
    std::vector<Contact> out;
    CHECK_THROWS_AS(census(P, 0, 1e-7, 1, 60.0, 1, out),
                    std::invalid_argument);
    CHECK_THROWS_AS(census(P, 0, std::numeric_limits<double>::min(), 1,
                           60.0, 1, out),
                    std::invalid_argument);
    CHECK_THROWS_AS(census(P, 0, std::numeric_limits<double>::denorm_min(),
                           1, 60.0, 1, out),
                    std::invalid_argument);
}

TEST_CASE("degenerate triangles obey the same distance-gated grazing rule")
{
    // A zero-area triangle cannot be classified by plane signs, but that
    // alone is not contact: two collapsed triangles with overlapping boxes
    // can be far apart. Grazing still requires coming within the touch
    // tolerance; genuine touching still reports.
    const Tri seg_far{{0, 0, 5}, {8, 0, 5}, {4, 0, 5}, 0, 0, 0};
    const Tri seg_near{{0, 4, 5}, {8, 4, 5}, {4, 4, 5}, 9, 9, 0};
    // collapsed vs collapsed, boxes overlap in x/z, 4 apart in y
    CHECK(tri_tri(seg_far, seg_near).verdict == NONE);

    // a collapsed triangle genuinely touching a proper one
    const Tri plane{{0, 0, 5}, {8, 0, 5}, {0, 8, 5}, 0, 0, 0};
    const Tri seg_touch{{1, 1, 5}, {3, 1, 5}, {2, 1, 5}, 9, 9, 0};
    CHECK(tri_tri(plane, seg_touch).verdict == GRAZING);

    // two coincident collapsed triangles
    const Tri seg_a{{0, 0, 5}, {8, 0, 5}, {4, 0, 5}, 0, 0, 0};
    const Tri seg_b{{0, 0, 5}, {8, 0, 5}, {4, 0, 5}, 9, 9, 0};
    CHECK(tri_tri(seg_a, seg_b).verdict == GRAZING);

    // a collapsed triangle beyond the proper one's hypotenuse (x + y = 8):
    // boxes overlap, but the nearest segment point (4,6) sits 1.41 from the
    // triangle -- comfortably outside the touch tolerance
    const Tri seg_off{{4, 6, 5}, {8, 6, 5}, {6, 6, 5}, 9, 9, 0};
    CHECK(tri_tri(plane, seg_off).verdict == NONE);
}

TEST_CASE("quads spanning a grid discontinuity are dropped, not censused")
{
    // Two adjacent valid columns sitting far apart in 3D produce a triangle
    // that spans the gap and crosses everything in its path purely because
    // the mesh has a hole there. maxedge exists to drop it.
    cv::Mat_<cv::Vec3f> P = blank(6 + GAP + 6, 8);
    put_plane(P, 0, 6, 8, 5.0);
    for (int v = 0; v < 6; ++v)
        for (int u = 0; u < 8; ++u)
            P(6 + GAP + v, u) = cv::Vec3f(
                4.f * u + (u >= 4 ? 500.f : 0.f) + 1.7f + 0.3f * v,
                4.f * v, (float)(4.0 * (u - 2) + 7.0));
    Run gated = run(P, 0);
    CHECK(gated.counts.quads_dropped > 0u);
    Run open = run(P, 0, 40.0, 1, /*maxedge=*/0.0);
    CHECK(open.counts.quads_dropped == 0u);
    CHECK(open.counts.triangles > gated.counts.triangles);
}

TEST_CASE("a nonpositive or non-finite cell size is refused")
{
    // cell = 0 divides to infinity whose int conversion is undefined; a
    // negative cell reverses every bucket range so triangles never share a
    // cell and a crossing surface would census CLEAN. Refusing is the only
    // honest answer.
    cv::Mat_<cv::Vec3f> P = blank(4, 4);
    put_plane(P, 0, 4, 4, 5.0);
    std::vector<Contact> out;
    CHECK_THROWS_AS(census(P, 0, 0.0, 1, 60.0, 1, out),
                    std::invalid_argument);
    CHECK_THROWS_AS(census(P, 0, -8.0, 1, 60.0, 1, out),
                    std::invalid_argument);
    CHECK_THROWS_AS(census(P, 0, 40.0, -1, 60.0, 1, out),
                    std::invalid_argument);
}

TEST_CASE("invalid and non-finite points never form quads")
{
    cv::Mat_<cv::Vec3f> P = blank(6, 6);
    put_plane(P, 0, 6, 6, 5.0);
    P(2, 2) = INVALID;
    P(4, 4) = cv::Vec3f(std::nanf(""), 0.f, 5.f);
    Run r = run(P, 0);
    // each poisoned vertex kills its four surrounding quads
    CHECK(r.counts.triangles == (5u * 5u - 8u) * 2u);
    CHECK(r.counts.transverse == 0u);
}
