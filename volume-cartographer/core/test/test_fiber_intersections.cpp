#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/atlas/FiberIntersections.hpp"

#include <algorithm>
#include <cmath>

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
    options.clusterArclength = 2.0;
    const auto candidates = index.candidatesForFiber(source, options);

    REQUIRE(candidates.size() == 2);
    CHECK(candidates[0].sourceSegmentIndex != candidates[1].sourceSegmentIndex);
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
