#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/atlas/FiberIntersections.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace {

vc::atlas::FiberPoint p(double x, double y, double z)
{
    return {{x, y, z}, std::nullopt};
}

vc::atlas::FiberPoint pn(double x, double y, double z, cv::Vec3d n)
{
    return {{x, y, z}, n};
}

vc::atlas::FiberPolyline fiber(uint64_t id,
                               uint64_t generation,
                               std::vector<vc::atlas::FiberPoint> points)
{
    return {id, generation, std::move(points)};
}

cv::Mat_<cv::Vec3f> flatStripGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            points(row, col) = {
                static_cast<float>(col),
                static_cast<float>(row),
                0.0f};
        }
    }
    return points;
}

cv::Mat_<cv::Vec3f> foldedOverlappingStripGrid()
{
    cv::Mat_<cv::Vec3f> points(4, 2);
    points(0, 0) = {0.0f, 0.0f, 0.0f};
    points(0, 1) = {1.0f, 0.0f, 0.0f};
    points(1, 0) = {0.0f, 1.0f, 0.0f};
    points(1, 1) = {1.0f, 1.0f, 0.0f};
    points(2, 0) = {0.0f, 0.0f, 10.0f};
    points(2, 1) = {1.0f, 0.0f, 10.0f};
    points(3, 0) = {0.0f, 1.0f, 10.0f};
    points(3, 1) = {1.0f, 1.0f, 10.0f};
    return points;
}

vc::atlas::FiberIntersectionCandidate normalizedCandidateForPair(
    vc::atlas::FiberIntersectionCandidate candidate,
    uint64_t sourceFiberId,
    uint64_t targetFiberId)
{
    if (candidate.sourceFiberId == targetFiberId &&
        candidate.targetFiberId == sourceFiberId) {
        std::swap(candidate.sourceFiberId, candidate.targetFiberId);
        std::swap(candidate.sourceGeneration, candidate.targetGeneration);
        std::swap(candidate.sourceSegmentIndex, candidate.targetSegmentIndex);
        std::swap(candidate.sourceArclength, candidate.targetArclength);
    }
    return candidate;
}

} // namespace

TEST_CASE("Fiber R-tree candidate search uses straight segment distance")
{
    vc::atlas::FiberSpatialIndex index;
    auto a = fiber(1, 1, {p(0, 0, 0), p(10, 0, 0)});
    auto b = fiber(2, 1, {p(5, -1, 0), p(5, 1, 0)});
    auto far = fiber(3, 1, {p(0, 10, 0), p(10, 10, 0)});
    index.upsertCommitted(b);
    index.upsertCommitted(far);

    vc::atlas::FiberIntersectionBroadPhaseOptions options;
    options.maxDistance = 0.25;
    options.maxSampleSpacing = 1.0;
    const auto candidates = index.candidatesForFiber(a, options);

    REQUIRE(candidates.size() == 1);
    CHECK(candidates[0].targetFiberId == 2);
    CHECK(candidates[0].straightDistance == doctest::Approx(0.0));
    CHECK(candidates[0].sourceArclength == doctest::Approx(5.0));
}

TEST_CASE("Fiber R-tree filters stale generations and recent fibers override committed entries")
{
    vc::atlas::FiberSpatialIndex index;
    auto source = fiber(1, 1, {p(0, 0, 0), p(10, 0, 0)});
    auto oldTarget = fiber(2, 1, {p(5, -1, 0), p(5, 1, 0)});
    auto newTarget = fiber(2, 2, {p(0, 8, 0), p(10, 8, 0)});

    index.upsertCommitted(oldTarget);
    index.upsertRecent(newTarget);

    vc::atlas::FiberIntersectionBroadPhaseOptions options;
    options.maxDistance = 0.25;
    options.maxSampleSpacing = 1.0;
    CHECK(index.candidatesForFiber(source, options).empty());
    CHECK(index.generation(2) == 2);
}

TEST_CASE("Fiber R-tree preserves separated local intersections before clustering")
{
    vc::atlas::FiberSpatialIndex index;
    auto source = fiber(1, 1, {
        p(0, 0, 0),
        p(10, 0, 0),
        p(10, 10, 0),
        p(0, 10, 0),
    });
    auto target = fiber(2, 1, {p(5, -1, 0), p(5, 11, 0)});
    index.upsertCommitted(target);

    vc::atlas::FiberIntersectionBroadPhaseOptions options;
    options.maxDistance = 0.25;
    options.maxSampleSpacing = 1.0;
    options.clusterArclength = 2.0;
    const auto candidates = index.candidatesForFiber(source, options);

    REQUIRE(candidates.size() == 2);
    CHECK(candidates[0].sourceSegmentIndex != candidates[1].sourceSegmentIndex);
}

TEST_CASE("Fiber point R-tree indexes interpolated dense samples for long sparse segments")
{
    vc::atlas::FiberSpatialIndex index;
    auto source = fiber(1, 1, {p(250.0 / 3.0, 1.0, 0), p(250.0 / 3.0, 2.0, 0)});
    auto target = fiber(2, 1, {p(0, 0, 0), p(250, 0, 0)});
    index.upsertCommitted(target);

    vc::atlas::FiberIntersectionBroadPhaseOptions options;
    options.maxDistance = 1.1;
    const auto candidates = index.candidatesForFiber(source, options);

    REQUIRE(candidates.size() == 1);
    CHECK(candidates[0].targetFiberId == 2);
    CHECK(candidates[0].targetArclength == doctest::Approx(250.0 / 3.0));
    CHECK(candidates[0].straightDistance == doctest::Approx(1.0));
}

TEST_CASE("Fiber point R-tree direct search converges from offset dense seed")
{
    vc::atlas::FiberSpatialIndex index;
    auto source = fiber(1, 1, {p(0, 0, 0), p(200, 0, 0)});
    auto target = fiber(2, 1, {p(100, -50, 0), p(100, 50, 0)});
    index.upsertCommitted(target);

    vc::atlas::FiberIntersectionBroadPhaseOptions options;
    options.maxDistance = 15.0;
    options.maxSampleSpacing = 10.0;
    options.seedStride = 100;
    options.clusterArclength = 0.1;
    const auto candidates = index.candidatesForFiber(source, options);

    REQUIRE(candidates.size() == 1);
    CHECK(candidates[0].sourceArclength == doctest::Approx(100.0));
    CHECK(candidates[0].targetArclength == doctest::Approx(50.0));
    CHECK(candidates[0].straightDistance == doctest::Approx(0.0));
}

TEST_CASE("Fiber point R-tree coverage suppresses same-target repeated hits only")
{
    auto source = fiber(1, 1, {p(0, 0, 0), p(0, 1, 0)});
    auto targetA = fiber(2, 1, {p(-0.1, 0, 0), p(0.1, 0, 0), p(0.2, 0, 0)});
    auto targetB = fiber(3, 1, {p(0, -0.1, 0), p(0, 0.1, 0), p(0, 0.2, 0)});

    vc::atlas::FiberIntersectionBroadPhaseOptions options;
    options.maxDistance = 0.3;
    options.maxSampleSpacing = 1.0;
    options.clusterArclength = 0.0;

    vc::atlas::FiberSpatialIndex oneTarget;
    oneTarget.upsertCommitted(targetA);
    CHECK(oneTarget.candidatesForFiber(source, options).size() == 1);

    vc::atlas::FiberSpatialIndex twoTargets;
    twoTargets.upsertCommitted(targetA);
    twoTargets.upsertCommitted(targetB);
    const auto candidates = twoTargets.candidatesForFiber(source, options);
    REQUIRE(candidates.size() == 2);
    std::set<uint64_t> targets;
    for (const auto& candidate : candidates) {
        targets.insert(candidate.targetFiberId);
    }
    CHECK(targets == std::set<uint64_t>{2, 3});
}

TEST_CASE("Fiber side strip query intersects sparse fiber segments with strip triangles")
{
    vc::atlas::FiberSpatialIndex index;
    auto crossing = fiber(2, 1, {p(0.5, 0.5, -5), p(0.5, 0.5, 5)});
    auto farCrossing = fiber(3, 1, {p(5, 5, -5), p(5, 5, 5)});
    auto offStrip = fiber(4, 1, {p(0.5, 0.5, 10), p(1.5, 0.5, 10)});
    index.upsertCommitted(crossing);
    index.upsertCommitted(farCrossing);
    index.upsertCommitted(offStrip);

    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);

    const auto intersections = index.sideStripIntersections(options);
    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].fiberId == 2);
    CHECK(intersections[0].point[2] == doctest::Approx(0.0));
    CHECK(intersections[0].stripRow == doctest::Approx(0.5));
    CHECK(intersections[0].stripCol == doctest::Approx(0.5));

    options.excludedFiberIds = {2};
    CHECK(index.sideStripIntersections(options).empty());
}

TEST_CASE("Fiber side strip query can use direct fiber snapshots without building an index")
{
    vc::atlas::FiberSpatialIndex emptyIndex;
    auto crossing = fiber(2, 1, {p(0.5, 0.5, -5), p(0.5, 0.5, 5)});
    auto offStrip = fiber(4, 1, {p(0.5, 0.5, 10), p(1.5, 0.5, 10)});

    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);
    options.queryFibers = {&crossing, &offStrip};

    const auto intersections = emptyIndex.sideStripIntersections(options);
    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].fiberId == 2);
}

TEST_CASE("Fiber side strip query finds long segments and preserves multiple crossings")
{
    vc::atlas::FiberSpatialIndex index;
    auto crossing = fiber(2, 1, {p(1.0, 0.5, -1000), p(1.0, 0.5, 1000)});
    auto multi = fiber(3, 1, {
        p(0.25, 0.5, -1),
        p(0.25, 0.5, 1),
        p(1.75, 0.5, 1),
        p(1.75, 0.5, -1),
    });
    index.upsertCommitted(crossing);
    index.upsertCommitted(multi);

    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 3);
    options.maxResults = 16;

    const auto intersections = index.sideStripIntersections(options);
    REQUIRE(intersections.size() == 3);
    CHECK(std::count_if(intersections.begin(), intersections.end(), [](const auto& hit) {
        return hit.fiberId == 3;
    }) == 2);
    CHECK(std::any_of(intersections.begin(), intersections.end(), [](const auto& hit) {
        return hit.fiberId == 2 && hit.point[2] == doctest::Approx(0.0);
    }));
}

TEST_CASE("Fiber side strip query reports branch link line hits and misses")
{
    vc::atlas::FiberSpatialIndex index;
    index.upsertCommitted(fiber(2, 1, {p(0.25, 0.5, -5), p(0.25, 0.5, 5)}));
    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);
    options.branchLinks = {
        {77, {0.5, 0.5, -5.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -5.0}},
        {88, {5.0, 5.0, -5.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -5.0}},
    };

    std::vector<vc::atlas::FiberSideStripProgressPhase> phases;
    const auto intersections = index.sideStripIntersections(
        options,
        [&phases](vc::atlas::FiberSideStripProgressPhase phase, size_t completed, size_t) {
            if (completed == 0) {
                phases.push_back(phase);
            }
        });
    REQUIRE(intersections.size() == 2);
    CHECK(intersections[0].source == vc::atlas::FiberSideStripIntersectionSource::BranchLink);
    CHECK(intersections[0].fiberId == 77);
    CHECK(intersections[0].point[2] == doctest::Approx(0.0));
    CHECK(intersections[0].connectorStart[2] == doctest::Approx(-5.0));
    CHECK(intersections[1].source == vc::atlas::FiberSideStripIntersectionSource::FiberSegment);
    const auto branchPhase = std::find(phases.begin(),
                                       phases.end(),
                                       vc::atlas::FiberSideStripProgressPhase::BranchLinks);
    const auto fiberPhase = std::find(phases.begin(),
                                      phases.end(),
                                      vc::atlas::FiberSideStripProgressPhase::FiberSegments);
    REQUIRE(branchPhase != phases.end());
    REQUIRE(fiberPhase != phases.end());
    CHECK(branchPhase < fiberPhase);

    options.branchLinks = {
        {88, {5.0, 5.0, -5.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -5.0}},
    };
    options.excludedFiberIds = {2};
    CHECK(index.sideStripIntersections(options).empty());
}

TEST_CASE("Fiber side strip query keeps the projection nearest to the branch target point")
{
    vc::atlas::FiberSpatialIndex index;
    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = foldedOverlappingStripGrid();
    options.maxResults = 0;
    options.branchLinks = {
        {77, {0.5, 0.5, 9.0}, {0.0, 0.0, 1.0}, {0.5, 0.5, -1.0}},
    };

    const auto intersections = index.sideStripIntersections(options);

    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].source == vc::atlas::FiberSideStripIntersectionSource::BranchLink);
    CHECK(intersections[0].fiberId == 77);
    CHECK(intersections[0].point[2] == doctest::Approx(10.0));
    CHECK(intersections[0].connectorStart[2] == doctest::Approx(-1.0));
    CHECK(intersections[0].projectionTarget[2] == doctest::Approx(9.0));
}

TEST_CASE("Fiber side strip branch projection ignores local connector proximity")
{
    vc::atlas::FiberSpatialIndex index;
    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = foldedOverlappingStripGrid();
    options.maxResults = 0;
    options.branchLinks = {
        {77, {0.5, 0.5, -1.0}, {0.0, 0.0, 1.0}, {0.5, 0.5, 9.0}},
    };

    const auto intersections = index.sideStripIntersections(options);

    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].source == vc::atlas::FiberSideStripIntersectionSource::BranchLink);
    CHECK(intersections[0].fiberId == 77);
    CHECK(intersections[0].point[2] == doctest::Approx(0.0));
    CHECK(intersections[0].connectorStart[2] == doctest::Approx(9.0));
    CHECK(intersections[0].projectionTarget[2] == doctest::Approx(-1.0));
}

TEST_CASE("Fiber side strip branch projection emits no connector without a strip hit")
{
    vc::atlas::FiberSpatialIndex index;
    vc::atlas::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);
    options.maxResults = 0;
    options.branchLinks = {
        {77, {5.0, 5.0, -1.0}, {0.0, 0.0, 1.0}, {0.5, 0.5, 0.0}},
    };

    CHECK(index.sideStripIntersections(options).empty());
}

TEST_CASE("Fiber intersection search runs two-sided discovery and preserves distinct minima")
{
    vc::atlas::FiberIntersectionCache cache;
    auto source = fiber(1, 1, {p(0, 0, 0), p(400, 0, 0)});
    auto target = fiber(2, 1, {
        p(100, -20, 0),
        p(100, 20, 0),
        p(300, 20, 0),
        p(300, -20, 0),
    });

    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    broad.maxDistance = 1.0;
    broad.maxSampleSpacing = 1.0;
    broad.clusterArclength = 2.0;
    vc::atlas::FiberIntersectionCeresOptions ceres;
    ceres.deduplicateArclength = 2.0;

    const auto results = vc::atlas::searchFiberIntersections(
        {source, target},
        {1},
        {2},
        &cache,
        broad,
        ceres);

    REQUIRE(results.size() == 2);
    std::vector<double> sourceArclengths;
    for (const auto& result : results) {
        CHECK(result.sourceFiberId == 1);
        CHECK(result.targetFiberId == 2);
        sourceArclengths.push_back(result.sourceArclength);
    }
    std::sort(sourceArclengths.begin(), sourceArclengths.end());
    CHECK(sourceArclengths[0] == doctest::Approx(100.0).epsilon(1e-5));
    CHECK(sourceArclengths[1] == doctest::Approx(300.0).epsilon(1e-5));
}

TEST_CASE("Fiber intersection search matches legacy indexed candidate refinement")
{
    vc::atlas::FiberIntersectionCache cache;
    auto source = fiber(1, 1, {p(0, 0, 0), p(400, 0, 0)});
    auto target = fiber(2, 1, {
        p(100, -20, 0),
        p(100, 20, 0),
        p(300, 20, 0),
        p(300, -20, 0),
    });
    auto distractor = fiber(3, 1, {p(0, 80, 0), p(400, 80, 0)});

    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    broad.maxDistance = 1.0;
    broad.maxSampleSpacing = 1.0;
    broad.clusterArclength = 2.0;
    vc::atlas::FiberIntersectionCeresOptions ceres;
    ceres.deduplicateArclength = 2.0;

    const auto results = vc::atlas::searchFiberIntersections(
        {source, target, distractor},
        {1},
        {2},
        &cache,
        broad,
        ceres);

    vc::atlas::FiberSpatialIndex legacyIndex;
    legacyIndex.upsertCommitted(source);
    legacyIndex.upsertCommitted(target);
    legacyIndex.upsertCommitted(distractor);

    std::vector<vc::atlas::FiberIntersectionCandidate> legacyCandidates;
    for (const auto& candidate : legacyIndex.candidatesForFiber(source, broad)) {
        if (candidate.targetFiberId == target.id) {
            legacyCandidates.push_back(candidate);
        }
    }
    for (const auto& candidate : legacyIndex.candidatesForFiber(target, broad)) {
        if (candidate.targetFiberId == source.id) {
            legacyCandidates.push_back(
                normalizedCandidateForPair(candidate, source.id, target.id));
        }
    }

    std::vector<vc::atlas::FiberIntersectionResult> legacyResults;
    for (const auto& candidate : legacyCandidates) {
        legacyResults.push_back(vc::atlas::refineFiberIntersectionCandidate(source,
                                                                            target,
                                                                            candidate,
                                                                            ceres));
    }
    legacyResults = vc::atlas::deduplicateFiberIntersectionResults(
        std::move(legacyResults),
        ceres.deduplicateArclength);

    REQUIRE(results.size() == legacyResults.size());
    for (size_t i = 0; i < results.size(); ++i) {
        CHECK(results[i].sourceFiberId == legacyResults[i].sourceFiberId);
        CHECK(results[i].targetFiberId == legacyResults[i].targetFiberId);
        CHECK(results[i].sourceArclength == doctest::Approx(legacyResults[i].sourceArclength).epsilon(1e-5));
        CHECK(results[i].targetArclength == doctest::Approx(legacyResults[i].targetArclength).epsilon(1e-5));
        CHECK(results[i].candidateDistance == doctest::Approx(legacyResults[i].candidateDistance).epsilon(1e-5));
    }
}

TEST_CASE("Atlas search phase progress maps five equal phases")
{
    using vc::atlas::AtlasSearchProgressPhase;

    CHECK(vc::atlas::atlasSearchPhaseProgressPercent(
              AtlasSearchProgressPhase::PrepareInputs,
              0,
              4) == 0);
    CHECK(vc::atlas::atlasSearchPhaseProgressPercent(
              AtlasSearchProgressPhase::PrepareInputs,
              4,
              4) == 20);
    CHECK(vc::atlas::atlasSearchPhaseProgressPercent(
              AtlasSearchProgressPhase::BuildSpatialIndex,
              4,
              4) == 40);
    CHECK(vc::atlas::atlasSearchPhaseProgressPercent(
              AtlasSearchProgressPhase::SearchPairs,
              12,
              12) == 60);
    CHECK(vc::atlas::atlasSearchPhaseProgressPercent(
              AtlasSearchProgressPhase::PrepareSigningSurface,
              2,
              2) == 80);
    CHECK(vc::atlas::atlasSearchPhaseProgressPercent(
              AtlasSearchProgressPhase::FinishResults,
              3,
              3) == 100);
}

TEST_CASE("Fiber intersection search reports pair progress and cancels")
{
    vc::atlas::FiberIntersectionCache cache;
    auto sourceA = fiber(1, 1, {p(0, 0, 0), p(10, 0, 0)});
    auto sourceB = fiber(2, 1, {p(0, 10, 0), p(10, 10, 0)});
    auto target = fiber(3, 1, {p(100, 100, 0), p(110, 100, 0)});

    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    broad.maxDistance = 0.25;
    vc::atlas::FiberIntersectionCeresOptions ceres;

    std::vector<size_t> buildCompleted;
    std::vector<size_t> buildTotals;
    std::vector<size_t> completed;
    std::vector<size_t> totals;
    std::vector<vc::atlas::AtlasSearchProgressPhase> phases;
    (void)vc::atlas::searchFiberIntersections(
        {sourceA, sourceB, target},
        {1, 2, 999},
        {3},
        &cache,
        broad,
        ceres,
        nullptr,
        [&](vc::atlas::AtlasSearchProgressPhase phase, size_t done, size_t total) {
            if (phase == vc::atlas::AtlasSearchProgressPhase::BuildSpatialIndex) {
                buildCompleted.push_back(done);
                buildTotals.push_back(total);
                return;
            }
            if (phase != vc::atlas::AtlasSearchProgressPhase::SearchPairs) {
                return;
            }
            phases.push_back(phase);
            completed.push_back(done);
            totals.push_back(total);
        });

    CHECK(buildCompleted == std::vector<size_t>{0, 1, 2, 3});
    CHECK(std::all_of(buildTotals.begin(), buildTotals.end(), [](size_t total) {
        return total == 3;
    }));
    CHECK(phases.size() == 7);
    CHECK(completed == std::vector<size_t>{0, 1, 2, 3, 4, 5, 6});
    CHECK(std::all_of(totals.begin(), totals.end(), [](size_t total) {
        return total == 6;
    }));

    bool cancel = false;
    completed.clear();
    const auto canceledResults = vc::atlas::searchFiberIntersections(
        {sourceA, sourceB, target},
        {1, 2},
        {3},
        &cache,
        broad,
        ceres,
        nullptr,
        [&](vc::atlas::AtlasSearchProgressPhase phase, size_t done, size_t /*total*/) {
            if (phase != vc::atlas::AtlasSearchProgressPhase::SearchPairs) {
                return;
            }
            completed.push_back(done);
            if (done > 0) {
                cancel = true;
            }
        },
        [&cancel]() {
            return cancel;
        });

    CHECK(canceledResults.empty());
    REQUIRE(completed.size() >= 2);
    CHECK(completed.front() == 0);
    CHECK(completed.back() >= 1);
}

TEST_CASE("Fiber intersection search ignores extensions outside outer control points")
{
    vc::atlas::FiberIntersectionCache cache;
    auto source = fiber(1, 1, {p(0, 0, 0), p(10, 0, 0)});
    source.controlPoints = {
        {2.0, 0.0, 0.0},
        {8.0, 0.0, 0.0},
    };
    auto target = fiber(2, 1, {
        p(1, -1, 0),
        p(1, 1, 0),
        p(5, 1, 0),
        p(5, -1, 0),
    });
    target.controlPoints = {
        {1.0, -1.0, 0.0},
        {5.0, -1.0, 0.0},
    };

    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    broad.maxDistance = 0.25;
    broad.maxSampleSpacing = 0.5;
    broad.clusterArclength = 1.0;
    vc::atlas::FiberIntersectionCeresOptions ceres;
    ceres.deduplicateArclength = 1.0;

    const auto results = vc::atlas::searchFiberIntersections(
        {source, target},
        {1},
        {2},
        &cache,
        broad,
        ceres);

    REQUIRE(results.size() == 1);
    CHECK(results[0].sourceArclength == doctest::Approx(5.0).epsilon(1e-5));
    CHECK(results[0].sourcePoint[0] == doctest::Approx(5.0).epsilon(1e-5));
}

TEST_CASE("Fiber Ceres refinement uses one solve and sign-ambiguous normal residuals")
{
    auto source = fiber(1, 1, {
        pn(0, 0.2, 0, {0, 1, 0}),
        pn(10, 0.2, 0, {0, 1, 0}),
    });
    auto target = fiber(2, 1, {
        pn(5, -4, 0, {1, 0, 0}),
        pn(5, 4, 0, {1, 0, 0}),
    });
    vc::atlas::FiberIntersectionCandidate candidate;
    candidate.sourceFiberId = 1;
    candidate.sourceGeneration = 1;
    candidate.sourceArclength = 4.5;
    candidate.targetFiberId = 2;
    candidate.targetGeneration = 1;
    candidate.targetArclength = 3.5;
    candidate.straightDistance = 0.2;

    vc::atlas::FiberIntersectionCeresOptions options;
    auto result = vc::atlas::refineFiberIntersectionCandidate(source, target, candidate, options);
    CHECK(result.ceresSolves == 1);
    CHECK(result.usedNormalResiduals);
    const cv::Vec3d delta = result.sourcePoint - result.targetPoint;
    CHECK(std::sqrt(delta.dot(delta)) < 1.0e-3);

    auto flipped = target;
    for (auto& point : flipped.points) {
        point.normal = -*point.normal;
    }
    auto flippedResult = vc::atlas::refineFiberIntersectionCandidate(source, flipped, candidate, options);
    CHECK(flippedResult.refinedScore == doctest::Approx(result.refinedScore).epsilon(1e-8));
    CHECK(flippedResult.sourceArclength == doctest::Approx(result.sourceArclength).epsilon(1e-5));
}

TEST_CASE("Fiber Ceres results deduplicate converged arclength neighborhoods")
{
    std::vector<vc::atlas::FiberIntersectionResult> results(2);
    results[0].sourceFiberId = 1;
    results[0].targetFiberId = 2;
    results[0].sourceArclength = 5.0;
    results[0].targetArclength = 6.0;
    results[0].refinedScore = 0.2;
    results[1] = results[0];
    results[1].sourceArclength = 5.3;
    results[1].targetArclength = 6.2;
    results[1].refinedScore = 0.1;

    const auto deduped = vc::atlas::deduplicateFiberIntersectionResults(std::move(results), 1.0);
    REQUIRE(deduped.size() == 1);
    CHECK(deduped[0].refinedScore == doctest::Approx(0.1));
}

TEST_CASE("Fiber intersection refresh picks nearest arclength result")
{
    std::vector<vc::atlas::FiberIntersectionResult> results(3);
    results[0].sourceArclength = 10.0;
    results[0].targetArclength = 20.0;
    results[1].sourceArclength = 12.0;
    results[1].targetArclength = 19.0;
    results[2].sourceArclength = 30.0;
    results[2].targetArclength = 40.0;

    const auto nearest = vc::atlas::nearestIntersectionResultByArclength(results, 11.5, 19.2);
    REQUIRE(nearest.has_value());
    CHECK(*nearest == 1);

    CHECK_FALSE(vc::atlas::nearestIntersectionResultByArclength(
                    results,
                    std::numeric_limits<double>::quiet_NaN(),
                    19.2)
                    .has_value());
}

TEST_CASE("Fiber intersection cache keys include pair generations and options")
{
    vc::atlas::FiberIntersectionCache cache;
    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    vc::atlas::FiberIntersectionCeresOptions ceres;
    std::vector<vc::atlas::FiberIntersectionResult> stored(1);
    stored[0].sourceFiberId = 1;
    stored[0].targetFiberId = 2;
    cache.store(1, 3, 2, 4, broad, ceres, stored);

    std::vector<vc::atlas::FiberIntersectionResult> hit;
    CHECK(cache.lookup(2, 4, 1, 3, broad, ceres, hit));
    REQUIRE(hit.size() == 1);
    CHECK(hit[0].cacheHit);
    CHECK(hit[0].ceresSolves == 0);

    std::vector<vc::atlas::FiberIntersectionResult> miss;
    CHECK_FALSE(cache.lookup(1, 3, 2, 5, broad, ceres, miss));
    broad.maxDistance = 3.0;
    CHECK_FALSE(cache.lookup(1, 3, 2, 4, broad, ceres, miss));

    cache.pruneFiber(1);
    CHECK(cache.size() == 0);
}
