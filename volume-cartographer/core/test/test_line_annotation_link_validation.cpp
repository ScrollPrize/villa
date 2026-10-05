// Load-time fiber link validation (LineAnnotationFiberLinkValidation.hpp):
// the issue collector VC3D's "Broken branch links" prompt lists, and the
// fixed-point neutralizer that replaces the strict loader's fiber removal.
// The regression pinned here is the 2026-09-30 PHerc0139 cascade: one stale
// direction on one entry must cost exactly one link pair, never a fiber.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

#include "LineAnnotationAdjacentLinks.hpp"
#include "LineAnnotationFiberLinkValidation.hpp"
#include "LineAnnotationFiberLinks.hpp"
#include "LineAnnotationStoredFiber.hpp"

using vc3d::line_annotation::BranchLinkValidationIssue;
using vc3d::line_annotation::FiberBranchRef;
using vc3d::line_annotation::StoredControlPoint;
using vc3d::line_annotation::StoredFiber;

namespace {

const std::filesystem::path kSource{"/proj/fibers"};

// A fiber whose line is a polyline with a visible kink at every control
// point (so a stale direction is detectable), control points every 4 line
// points.
StoredFiber makeFiber(const std::string& name, double xOffset, double yOffset)
{
    StoredFiber fiber;
    fiber.fileName = name;
    fiber.sourceRoot = kSource;
    for (int i = 0; i < 17; ++i) {
        const double t = static_cast<double>(i);
        // Alternating slope every 4 points: a kink at every control point.
        const double kink = ((i / 4) % 2 == 0) ? 0.35 : -0.35;
        fiber.linePoints.emplace_back(xOffset + t * 3.0, yOffset + kink * (t - 4.0 * (i / 4)) + 2.0 * (i / 4), 100.0);
        if (i % 4 == 0) {
            fiber.controlPoints.emplace_back(fiber.linePoints.back());
        }
    }
    return fiber;
}

// Links fiber a's control point ia to fiber b's control point ib the way
// VC3D's link creation does (directions from the loader's own tangent).
void link(StoredFiber& a, int ia, StoredFiber& b, int ib, bool adjacent = false)
{
    using vc3d::line_annotation::endpointTangentFromLinePoints;
    FiberBranchRef ab;
    ab.controlPointIndex = ia;
    ab.branchControlPointIndex = ib;
    ab.branchFileName = b.fileName;
    ab.controlPointPosition = a.controlPoints[static_cast<std::size_t>(ia)];
    ab.branchControlPointPosition = b.controlPoints[static_cast<std::size_t>(ib)];
    ab.controlPointDirection = endpointTangentFromLinePoints(a.linePoints, ab.controlPointPosition);
    ab.branchControlPointDirection =
        endpointTangentFromLinePoints(b.linePoints, ab.branchControlPointPosition);
    ab.adjacent = adjacent;
    FiberBranchRef ba;
    ba.controlPointIndex = ib;
    ba.branchControlPointIndex = ia;
    ba.branchFileName = a.fileName;
    ba.controlPointPosition = ab.branchControlPointPosition;
    ba.branchControlPointPosition = ab.controlPointPosition;
    ba.controlPointDirection = ab.branchControlPointDirection;
    ba.branchControlPointDirection = ab.controlPointDirection;
    ba.adjacent = adjacent;
    a.branches.push_back(ab);
    b.branches.push_back(ba);
}

std::string sourceKey(const StoredFiber& fiber, const std::string& fileName)
{
    return (fiber.sourceRoot / fileName).lexically_normal().string();
}

bool reciprocal(const StoredFiber& fiber, const FiberBranchRef& branch,
                const FiberBranchRef& candidate)
{
    return vc3d::line_annotation::reciprocalBranchRefMatches(
        fiber, branch, candidate,
        [](const cv::Vec3d& a, const cv::Vec3d& b) {
            return vc3d::line_annotation::pointsApproximatelyEqual(a, b);
        },
        [](const cv::Vec3d& a, const cv::Vec3d& b) {
            return vc3d::line_annotation::branchDirectionsCompatible(a, b);
        });
}

std::vector<BranchLinkValidationIssue> collect(const std::vector<StoredFiber>& fibers)
{
    return vc3d::line_annotation::collectFiberBranchIssues(
        fibers, sourceKey,
        [](const StoredFiber& fiber, const std::string& name) { return sourceKey(fiber, name); },
        reciprocal);
}

std::size_t totalEntries(const std::vector<StoredFiber>& fibers)
{
    std::size_t n = 0;
    for (const auto& fiber : fibers) n += fiber.branches.size();
    return n;
}

// A chain a-b-c-d-e plus a cycle c-f-a and an adjacent link d-g: one
// connected component of 7 fibers, 7 link pairs (14 entries).
std::vector<StoredFiber> makeNetwork()
{
    std::vector<StoredFiber> f;
    const char* names[] = {"a.json", "b.json", "c.json", "d.json", "e.json", "f.json", "g.json"};
    for (int i = 0; i < 7; ++i) f.push_back(makeFiber(names[i], 50.0 * i, 10.0 * i));
    link(f[0], 4, f[1], 0);
    link(f[1], 4, f[2], 0);
    link(f[2], 4, f[3], 0);
    link(f[3], 4, f[4], 0);
    link(f[2], 2, f[5], 0);
    link(f[5], 4, f[0], 2);
    link(f[3], 2, f[6], 1, /*adjacent=*/true);
    return f;
}

} // namespace

TEST_CASE("link validation: a consistent network reports nothing")
{
    auto fibers = makeNetwork();
    CHECK(collect(fibers).empty());
    CHECK(totalEntries(fibers) == 14);
}

TEST_CASE("link validation: one stale direction costs one link pair, never a fiber")
{
    auto fibers = makeNetwork();
    // The incident shape: b's entry to c carries a direction from a
    // differently oriented line (rotated well past the 0.26 deg tolerance).
    auto& stale = fibers[1].branches[1];
    REQUIRE(stale.branchFileName == "c.json");
    stale.controlPointDirection = cv::Vec3d{0.0, 0.0, 1.0};

    // Both sides report: b's own entry fails the tangent check, and c's
    // reciprocal no longer finds a partner whose direction agrees.
    const auto issues = collect(fibers);
    REQUIRE(issues.size() == 2);
    CHECK(issues[0].fiberIndex == 1);
    CHECK(issues[0].branchIndex == 1);
    CHECK(issues[0].reason == "branch endpoint direction mismatch");
    CHECK(issues[1].fiberIndex == 2);
    CHECK(issues[1].branchIndex == 0);
    CHECK(issues[1].reason == "missing reciprocal branch");

    const auto result = vc3d::line_annotation::neutralizeFiberBranchIssues(fibers, issues, collect);
    CHECK(fibers.size() == 7);
    CHECK(result.removedEntries == 2);
    CHECK(result.changedFibers == std::vector<std::size_t>{1, 2});
    CHECK(totalEntries(fibers) == 12);
    CHECK(collect(fibers).empty());
    // Every other pair survived, the adjacent one included.
    CHECK(fibers[3].branches.size() == 3);
    CHECK(fibers[6].branches.size() == 1);
    CHECK(fibers[6].branches[0].adjacent);
}

TEST_CASE("link validation: a missing target drops only the dangling entry")
{
    auto fibers = makeNetwork();
    fibers.erase(fibers.begin() + 4);  // e.json is gone from disk
    const auto issues = collect(fibers);
    REQUIRE(issues.size() == 1);
    CHECK(issues[0].reason == "missing linked fiber");
    const auto result = vc3d::line_annotation::neutralizeFiberBranchIssues(fibers, issues, collect);
    CHECK(fibers.size() == 6);
    CHECK(result.removedEntries == 1);
    CHECK(collect(fibers).empty());
}

TEST_CASE("link validation: a one-way link is dropped on the surviving side too")
{
    auto fibers = makeNetwork();
    // d lost its entry back to c (what a one-sided repair on another machine
    // leaves on the shared remote).
    auto& d = fibers[3];
    d.branches.erase(d.branches.begin());
    REQUIRE(collect(fibers).size() == 1);
    const auto result = vc3d::line_annotation::neutralizeFiberBranchIssues(fibers, collect(fibers), collect);
    CHECK(result.removedEntries == 1);
    CHECK(fibers.size() == 7);
    CHECK(collect(fibers).empty());
    CHECK(totalEntries(fibers) == 12);
}

TEST_CASE("link validation: reversed line orientation is what breaks a copied direction")
{
    // The mechanism behind the incident: the same control point, the same
    // stored direction, a reversed line -> the forward-difference tangent
    // now spans the other segment and the kink exceeds the tolerance.
    auto fiber = makeFiber("r.json", 0.0, 0.0);
    const cv::Vec3d point = fiber.controlPoints[2];
    const cv::Vec3d forward = vc3d::line_annotation::endpointTangentFromLinePoints(fiber.linePoints, point);
    std::vector<cv::Vec3d> reversed(fiber.linePoints.rbegin(), fiber.linePoints.rend());
    const cv::Vec3d onReversed = vc3d::line_annotation::endpointTangentFromLinePoints(reversed, point);
    CHECK_FALSE(vc3d::line_annotation::branchDirectionsCompatible(forward, onReversed));
}
