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

TEST_CASE("Fiber intersection search runs two-sided discovery and preserves distinct minima")
{
    vc::atlas::FiberSpatialIndex index;
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
        index,
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

TEST_CASE("Fiber intersection search ignores extensions outside outer control points")
{
    vc::atlas::FiberSpatialIndex index;
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
        index,
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
