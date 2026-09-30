// Merge / split planning (LineAnnotationStructuralEdits.hpp) checked against
// the loader's own link validation: what the planner + batch canonicalization
// write must load back without a single "Broken branch links" entry, before
// any re-optimization. The 2026-09-30 PHerc0139 merge is the pinned
// regression: both sources reversed, link directions copied verbatim.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/core/types.hpp>

#include "LineAnnotationAdjacentLinks.hpp"
#include "LineAnnotationFiberLinkValidation.hpp"
#include "LineAnnotationFiberLinks.hpp"
#include "LineAnnotationStoredFiber.hpp"
#include "LineAnnotationStructuralEdits.hpp"

using vc3d::line_annotation::BranchLinkValidationIssue;
using vc3d::line_annotation::BranchRedirectSource;
using vc3d::line_annotation::FiberBranchRef;
using vc3d::line_annotation::FiberOptimizationMode;
using vc3d::line_annotation::NewFiberIdentity;
using vc3d::line_annotation::RedirectTarget;
using vc3d::line_annotation::StoredControlPoint;
using vc3d::line_annotation::StoredFiber;

namespace {

const std::filesystem::path kSource{"/proj/fibers"};

// Snapshot type with the one member the batch helpers need.
struct Snapshot {
    StoredFiber fiber;
};

// A fiber with a kink at every control point (4 line points per span), so
// a direction taken from the other orientation is detectably wrong.
StoredFiber makeFiber(uint64_t id, const std::string& name, double x0, double y0, int spans = 4)
{
    StoredFiber fiber;
    fiber.id = id;
    fiber.fileName = name;
    fiber.sourceRoot = kSource;
    const int n = spans * 4 + 1;
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        const double kink = ((i / 4) % 2 == 0) ? 0.35 : -0.35;
        fiber.linePoints.emplace_back(x0 + t * 3.0,
                                      y0 + kink * (t - 4.0 * (i / 4)) + 2.0 * (i / 4),
                                      100.0);
        if (i % 4 == 0) {
            fiber.controlPoints.emplace_back(fiber.linePoints.back());
        }
    }
    return fiber;
}

void link(StoredFiber& a, int ia, StoredFiber& b, int ib, bool adjacent = false)
{
    using vc3d::line_annotation::endpointTangentFromLinePoints;
    FiberBranchRef ab;
    ab.controlPointIndex = ia;
    ab.branchFiberId = b.id;
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
    ba.branchFiberId = a.id;
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

NewFiberIdentity identity(uint64_t id, const std::string& file)
{
    NewFiberIdentity out;
    out.id = id;
    out.username = "tt";
    out.startedAt = "20260930T120000000";
    out.sequence = id;
    out.fileName = file;
    out.sourceRoot = kSource;
    return out;
}

// The controller's phase B in miniature: plan, redirect peer copies, run the
// batch through canonicalize + validate, return the loader's issues over the
// final graph (untouched + batch, originals removed).
struct Committed {
    std::vector<StoredFiber> graph;
    std::vector<BranchLinkValidationIssue> issues;
    std::size_t redirected = 0;
};

Committed commitMerge(const StoredFiber& clicked, int clickedIndex, const StoredFiber& far,
                      int candidateIndex, std::vector<StoredFiber> peers,
                      bool canonicalize = true)
{
    std::string error;
    auto plan = vc3d::line_annotation::planFiberMerge(
        clicked, clickedIndex, far, candidateIndex, FiberOptimizationMode::Lasagna,
        identity(900, "merged.json"), &error);
    REQUIRE_MESSAGE(plan.has_value(), error);
    std::vector<BranchRedirectSource> sources{
        {clicked.id, clicked.fileName, &clicked,
         [&plan](int i) {
             return RedirectTarget{plan->merged.id, plan->merged.fileName, plan->clickedRemap(i)};
         }},
        {far.id, far.fileName, &far,
         [&plan](int i) {
             return RedirectTarget{plan->merged.id, plan->merged.fileName, plan->farRemap(i)};
         }},
    };
    Committed out;
    std::vector<Snapshot> batch;
    batch.push_back({plan->merged});
    for (auto& peer : peers) {
        const auto n = vc3d::line_annotation::redirectBranchRefs(peer.branches, sources, &error);
        REQUIRE_MESSAGE(n.has_value(), error);
        out.redirected += *n;
        batch.push_back({peer});
    }
    if (canonicalize) {
        vc3d::line_annotation::canonicalizeFiberSaveSnapshots(batch);
        vc3d::line_annotation::validateFiberSaveSnapshots(batch);
    }
    for (const auto& snapshot : batch) {
        out.graph.push_back(snapshot.fiber);
    }
    out.issues = collect(out.graph);
    return out;
}

} // namespace

TEST_CASE("merge: every orientation combination writes loader-clean links (incident shape)")
{
    // Two fibers, each with a third-party link at an interior control point
    // and one at the endpoint away from the join; peers hold the reciprocals.
    for (const bool clickAtStart : {false, true}) {
        for (const bool candidateAtEnd : {false, true}) {
            auto a = makeFiber(1, "a.json", 0.0, 0.0);
            auto b = makeFiber(2, "b.json", 100.0, 0.0);
            auto p = makeFiber(3, "p.json", 0.0, 50.0);
            auto q = makeFiber(4, "q.json", 100.0, 50.0);
            link(a, 1, p, 2);
            link(a, clickAtStart ? 4 : 0, p, 4);
            link(b, 3, q, 1);
            link(b, candidateAtEnd ? 0 : 4, q, 3, /*adjacent=*/true);
            const int clickedIndex = clickAtStart ? 0 : 4;
            const int candidateIndex = candidateAtEnd ? 4 : 0;

            CAPTURE(clickAtStart);
            CAPTURE(candidateAtEnd);
            // Without the batch canonicalization (the pre-fix write): a
            // reversed side fails the loader's direction check.
            const auto raw = commitMerge(a, clickedIndex, b, candidateIndex, {p, q}, false);
            if (clickAtStart || candidateAtEnd) {
                CHECK_FALSE(raw.issues.empty());
            }
            const auto fixed = commitMerge(a, clickedIndex, b, candidateIndex, {p, q});
            CHECK(fixed.issues.empty());
            CHECK(fixed.redirected == 4);
            REQUIRE(fixed.graph.size() == 3);
            const auto& merged = fixed.graph[0];
            CHECK(merged.controlPoints.size() == 10);
            CHECK(merged.branches.size() == 4);
            CHECK(merged.coordinateBaseShapeZYX == a.coordinateBaseShapeZYX);
            CHECK(std::find(merged.tags.begin(), merged.tags.end(),
                            vc3d::line_annotation::kNeedsReoptimizationTag) != merged.tags.end());
            // The adjacent link keeps its kind through the redirect.
            std::size_t adjacentCount = 0;
            for (const auto& branch : merged.branches) adjacentCount += branch.adjacent ? 1 : 0;
            CHECK(adjacentCount == 1);
        }
    }
}

TEST_CASE("merge: consumes exactly the endpoint pair link and refuses other shapes")
{
    auto a = makeFiber(1, "a.json", 0.0, 0.0);
    auto b = makeFiber(2, "b.json", 100.0, 0.0);
    std::string error;

    SUBCASE("not endpoints")
    {
        CHECK_FALSE(vc3d::line_annotation::planFiberMerge(
            a, 2, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error));
        CHECK(error.find("endpoints") != std::string::npos);
    }
    SUBCASE("linked elsewhere")
    {
        link(a, 2, b, 1);
        CHECK_FALSE(vc3d::line_annotation::planFiberMerge(
            a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error));
        CHECK(error.find("linked elsewhere") != std::string::npos);
    }
    SUBCASE("endpoint pair link is consumed")
    {
        link(a, 4, b, 0);
        const auto plan = vc3d::line_annotation::planFiberMerge(
            a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error);
        REQUIRE(plan);
        CHECK(plan->merged.branches.empty());
        CHECK(plan->joinControlIndex == 4);
    }
    SUBCASE("adjacent endpoint link refuses")
    {
        link(a, 4, b, 0, /*adjacent=*/true);
        CHECK_FALSE(vc3d::line_annotation::planFiberMerge(
            a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error));
        CHECK(error.find("adjacent") != std::string::npos);
    }
    SUBCASE("coordinate domains must agree")
    {
        a.coordinateBaseShapeZYX = std::array<std::size_t, 3>{10, 20, 30};
        CHECK_FALSE(vc3d::line_annotation::planFiberMerge(
            a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error));
        CHECK(error.find("coordinate") != std::string::npos);
        b.coordinateBaseShapeZYX = a.coordinateBaseShapeZYX;
        const auto plan = vc3d::line_annotation::planFiberMerge(
            a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error);
        REQUIRE(plan);
        CHECK(plan->merged.coordinateBaseShapeZYX == a.coordinateBaseShapeZYX);
    }
    SUBCASE("kollesis termination at the join refuses")
    {
        a.controlPoints.back().tags = {vc3d::line_annotation::kKollesisTerminationTag};
        CHECK_FALSE(vc3d::line_annotation::planFiberMerge(
            a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error));
        CHECK(error.find("kollesis") != std::string::npos);
    }
}

TEST_CASE("split: links on both sides of the cut land on the right half with clean directions")
{
    for (const bool linkHalves : {false, true}) {
        auto parent = makeFiber(1, "parent.json", 0.0, 0.0, 6);  // 7 control points
        auto p = makeFiber(2, "p.json", 0.0, 50.0);
        auto q = makeFiber(3, "q.json", 100.0, 50.0);
        link(parent, 2, p, 1);   // becomes the prefix's LAST control point
        link(parent, 3, q, 1);   // becomes the suffix's FIRST control point
        link(parent, 6, q, 3);   // suffix interior/end

        std::string error;
        auto plan = vc3d::line_annotation::planFiberSplit(
            parent, 2, identity(10, "prefix.json"), identity(11, "suffix.json"), linkHalves, &error);
        REQUIRE_MESSAGE(plan.has_value(), error);
        CHECK(plan->prefix.controlPoints.size() == 3);
        CHECK(plan->suffix.controlPoints.size() == 4);
        CHECK(plan->prefix.coordinateBaseShapeZYX == parent.coordinateBaseShapeZYX);

        std::vector<BranchRedirectSource> sources{
            {parent.id, parent.fileName, &parent,
             [&plan](int i) -> std::optional<RedirectTarget> {
                 const auto remapped =
                     vc3d::line_annotation::remappedSplitControlPointIndex(plan->plan, i);
                 if (!remapped) return std::nullopt;
                 const StoredFiber& half = remapped->first ? plan->suffix : plan->prefix;
                 return RedirectTarget{half.id, half.fileName, remapped->second};
             }},
        };
        std::vector<Snapshot> batch{{plan->prefix}, {plan->suffix}};
        for (auto* peer : {&p, &q}) {
            const auto n = vc3d::line_annotation::redirectBranchRefs(peer->branches, sources, &error);
            REQUIRE_MESSAGE(n.has_value(), error);
            batch.push_back({*peer});
        }
        // Pre-fix state: the prefix's last control point now sits at the end
        // of its line, so the stored (forward) direction of that link no
        // longer matches the loader's backward difference there.
        {
            std::vector<StoredFiber> graph;
            for (const auto& s : batch) graph.push_back(s.fiber);
            CHECK_FALSE(collect(graph).empty());
        }
        vc3d::line_annotation::canonicalizeFiberSaveSnapshots(batch);
        CHECK_NOTHROW(vc3d::line_annotation::validateFiberSaveSnapshots(batch));
        std::vector<StoredFiber> graph;
        for (const auto& s : batch) graph.push_back(s.fiber);
        CAPTURE(linkHalves);
        CHECK(collect(graph).empty());
        CHECK(graph[0].branches.size() == (linkHalves ? 2 : 1));
        CHECK(graph[1].branches.size() == (linkHalves ? 3 : 2));
    }
}

TEST_CASE("redirect: a live ref holding the partner's session index resolves by position")
{
    auto a = makeFiber(1, "a.json", 0.0, 0.0);
    auto b = makeFiber(2, "b.json", 100.0, 0.0);
    auto p = makeFiber(3, "p.json", 0.0, 50.0);
    link(a, 2, p, 1);
    // The peer's open pane synced against a's pane, which had an extra
    // (unsaved) control point before index 2: the ref says index 3.
    std::vector<FiberBranchRef> live = p.branches;
    live[0].branchControlPointIndex = 3;

    std::string error;
    auto plan = vc3d::line_annotation::planFiberMerge(
        a, 0, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error);
    REQUIRE(plan);
    std::vector<BranchRedirectSource> sources{
        {a.id, a.fileName, &a,
         [&plan](int i) {
             return RedirectTarget{plan->merged.id, plan->merged.fileName, plan->clickedRemap(i)};
         }},
    };
    const auto n = vc3d::line_annotation::redirectBranchRefs(live, sources, &error);
    REQUIRE_MESSAGE(n.has_value(), error);
    CHECK(*n == 1);
    CHECK(live[0].branchFileName == "m.json");
    // a was reversed (clicked at index 0): stored index 2 of 5 -> 2.
    CHECK(live[0].branchControlPointIndex == plan->clickedRemap(2));
    CHECK(live[0].controlPointIndex == 1);

    // A ref whose position is not on the original at all refuses without
    // touching anything.
    std::vector<FiberBranchRef> bogus = p.branches;
    bogus[0].branchControlPointPosition = cv::Vec3d{-1.0, -1.0, -1.0};
    bogus[0].branchControlPointIndex = 99;
    CHECK_FALSE(vc3d::line_annotation::redirectBranchRefs(bogus, sources, &error));
    CHECK(bogus[0].branchFileName == "a.json");
}

TEST_CASE("graph check: a dangling reference to a retired original blocks, unrelated defects do not")
{
    auto a = makeFiber(1, "a.json", 0.0, 0.0);
    auto b = makeFiber(2, "b.json", 100.0, 0.0);
    auto p = makeFiber(3, "p.json", 0.0, 50.0);
    auto u = makeFiber(4, "u.json", 300.0, 0.0);
    auto v = makeFiber(5, "v.json", 400.0, 0.0);
    link(a, 2, p, 1);
    link(u, 1, v, 1);
    v.branches[0].controlPointDirection = cv::Vec3d{0.0, 0.0, 1.0};  // unrelated defect

    std::string error;
    auto plan = vc3d::line_annotation::planFiberMerge(
        a, 4, b, 0, FiberOptimizationMode::Lasagna, identity(9, "m.json"), &error);
    REQUIRE(plan);
    // p was NOT redirected (the controller forgot a peer): it still names a.
    std::vector<StoredFiber> graph{plan->merged, p, u, v};
    const std::unordered_set<std::string> batch{sourceKey(a, "m.json")};
    const std::unordered_set<std::string> retired{sourceKey(a, "a.json"), sourceKey(b, "b.json")};
    const auto check = vc3d::line_annotation::checkStructuralEditGraph(
        graph, batch, retired, collect, sourceKey);
    // Both ends of the forgotten redirect block: the merged fiber's entry
    // has no reciprocal under its new name, and p's entry names a retired
    // file. The u-v defect is reported but does not block.
    REQUIRE(check.blocking.size() == 2);
    CHECK(graph[check.blocking[0].fiberIndex].fileName == "m.json");
    CHECK(check.blocking[0].reason == "missing reciprocal branch");
    CHECK(graph[check.blocking[1].fiberIndex].fileName == "p.json");
    CHECK(check.blocking[1].reason == "missing linked fiber");
    CHECK(check.unrelated.size() == 2);
}

TEST_CASE("geometry key: identical geometry under another name is recognised")
{
    const auto a = makeFiber(1, "a.json", 0.0, 0.0);
    auto copy = a;
    copy.fileName = "copy.json";
    CHECK(vc3d::line_annotation::fiberGeometryKey(a.controlPoints, a.linePoints) ==
          vc3d::line_annotation::fiberGeometryKey(copy.controlPoints, copy.linePoints));
    copy.linePoints.back()[0] += 1.0e-9;
    CHECK(vc3d::line_annotation::fiberGeometryKey(a.controlPoints, a.linePoints) !=
          vc3d::line_annotation::fiberGeometryKey(copy.controlPoints, copy.linePoints));
}
