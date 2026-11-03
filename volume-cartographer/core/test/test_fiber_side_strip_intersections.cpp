// Fiber segments and branch-link lines crossing a side strip, as the line
// annotation side-strip view queries them.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "vc/fiber_tracer/FiberSideStripIntersections.hpp"

#include <algorithm>
#include <vector>

namespace {

// Strips sit above z = 0, where surface vertices are valid.
constexpr double kZ = 100.0;
double z(double value) { return kZ + value; }

cv::Vec3d p(double x, double y, double depth) { return {x, y, z(depth)}; }

vc::fiber_tracer::FiberPolyline fiber(uint64_t id, uint64_t generation, std::vector<cv::Vec3d> points)
{
    return {id, generation, std::move(points)};
}

cv::Mat_<cv::Vec3f> flatStripGrid(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
            points(row, col) = {static_cast<float>(col), static_cast<float>(row), static_cast<float>(z(0))};
    return points;
}

cv::Mat_<cv::Vec3f> foldedOverlappingStripGrid()
{
    cv::Mat_<cv::Vec3f> points(4, 2);
    const auto f = [](double v) { return static_cast<float>(z(v)); };
    points(0, 0) = {0.0f, 0.0f, f(0)};
    points(0, 1) = {1.0f, 0.0f, f(0)};
    points(1, 0) = {0.0f, 1.0f, f(0)};
    points(1, 1) = {1.0f, 1.0f, f(0)};
    points(2, 0) = {0.0f, 0.0f, f(10)};
    points(2, 1) = {1.0f, 0.0f, f(10)};
    points(3, 0) = {0.0f, 1.0f, f(10)};
    points(3, 1) = {1.0f, 1.0f, f(10)};
    return points;
}

} // namespace

TEST_CASE("Fiber side strip query intersects sparse fiber segments with strip triangles")
{
    auto crossing = fiber(2, 1, {p(0.5, 0.5, -5), p(0.5, 0.5, 5)});
    auto farCrossing = fiber(3, 1, {p(5, 5, -5), p(5, 5, 5)});
    auto offStrip = fiber(4, 1, {p(0.5, 0.5, 10), p(1.5, 0.5, 10)});

    vc::fiber_tracer::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);
    options.queryFibers = {&crossing, &farCrossing, &offStrip};

    const auto intersections = vc::fiber_tracer::fiberSideStripIntersections(options);
    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].fiberId == 2);
    CHECK(intersections[0].point[2] == doctest::Approx(z(0.0)));
    CHECK(intersections[0].stripRow == doctest::Approx(0.5));
    CHECK(intersections[0].stripCol == doctest::Approx(0.5));

    options.excludedFiberIds = {2};
    CHECK(vc::fiber_tracer::fiberSideStripIntersections(options).empty());
}

TEST_CASE("Fiber side strip query finds long segments and preserves multiple crossings")
{
    auto crossing = fiber(2, 1, {p(1.0, 0.5, -1000), p(1.0, 0.5, 1000)});
    auto multi = fiber(3, 1, {
        p(0.25, 0.5, -1),
        p(0.25, 0.5, 1),
        p(1.75, 0.5, 1),
        p(1.75, 0.5, -1),
    });

    vc::fiber_tracer::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 3);
    options.queryFibers = {&crossing, &multi};
    options.maxResults = 16;

    const auto intersections = vc::fiber_tracer::fiberSideStripIntersections(options);
    REQUIRE(intersections.size() == 3);
    CHECK(std::count_if(intersections.begin(), intersections.end(), [](const auto& hit) {
        return hit.fiberId == 3;
    }) == 2);
    CHECK(std::any_of(intersections.begin(), intersections.end(), [](const auto& hit) {
        return hit.fiberId == 2 && hit.point[2] == doctest::Approx(z(0.0));
    }));
}

TEST_CASE("Fiber side strip query reports branch link line hits and misses")
{
    const auto crossing = fiber(2, 1, {p(0.25, 0.5, -5), p(0.25, 0.5, 5)});
    vc::fiber_tracer::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);
    options.queryFibers = {&crossing};
    options.branchLinks = {
        {77, {0.5, 0.5, z(-5.0)}, {0.0, 0.0, 1.0}, {0.0, 0.0, z(-5.0)}},
        {88, {5.0, 5.0, z(-5.0)}, {0.0, 0.0, 1.0}, {0.0, 0.0, z(-5.0)}},
    };

    std::vector<vc::fiber_tracer::FiberSideStripProgressPhase> phases;
    const auto intersections = vc::fiber_tracer::fiberSideStripIntersections(
        options,
        [&phases](vc::fiber_tracer::FiberSideStripProgressPhase phase, size_t completed, size_t) {
            if (completed == 0) {
                phases.push_back(phase);
            }
        });
    REQUIRE(intersections.size() == 2);
    CHECK(intersections[0].source == vc::fiber_tracer::FiberSideStripIntersectionSource::BranchLink);
    CHECK(intersections[0].fiberId == 77);
    CHECK(intersections[0].point[2] == doctest::Approx(z(0.0)));
    CHECK(intersections[0].connectorStart[2] == doctest::Approx(z(-5.0)));
    CHECK(intersections[1].source == vc::fiber_tracer::FiberSideStripIntersectionSource::FiberSegment);
    const auto branchPhase = std::find(phases.begin(),
                                       phases.end(),
                                       vc::fiber_tracer::FiberSideStripProgressPhase::BranchLinks);
    const auto fiberPhase = std::find(phases.begin(),
                                      phases.end(),
                                      vc::fiber_tracer::FiberSideStripProgressPhase::FiberSegments);
    REQUIRE(branchPhase != phases.end());
    REQUIRE(fiberPhase != phases.end());
    CHECK(branchPhase < fiberPhase);

    options.branchLinks = {
        {88, {5.0, 5.0, z(-5.0)}, {0.0, 0.0, 1.0}, {0.0, 0.0, z(-5.0)}},
    };
    options.excludedFiberIds = {2};
    CHECK(vc::fiber_tracer::fiberSideStripIntersections(options).empty());
}

TEST_CASE("Fiber side strip query keeps the projection nearest to the branch target point")
{
    vc::fiber_tracer::FiberSideStripQueryOptions options;
    options.stripPoints = foldedOverlappingStripGrid();
    options.maxResults = 0;
    options.branchLinks = {
        {77, {0.5, 0.5, z(9.0)}, {0.0, 0.0, 1.0}, {0.5, 0.5, z(-1.0)}},
    };

    const auto intersections = vc::fiber_tracer::fiberSideStripIntersections(options);

    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].source == vc::fiber_tracer::FiberSideStripIntersectionSource::BranchLink);
    CHECK(intersections[0].fiberId == 77);
    CHECK(intersections[0].point[2] == doctest::Approx(z(10.0)));
    CHECK(intersections[0].connectorStart[2] == doctest::Approx(z(-1.0)));
    CHECK(intersections[0].projectionTarget[2] == doctest::Approx(z(9.0)));
}

TEST_CASE("Fiber side strip branch projection ignores local connector proximity")
{
    vc::fiber_tracer::FiberSideStripQueryOptions options;
    options.stripPoints = foldedOverlappingStripGrid();
    options.maxResults = 0;
    options.branchLinks = {
        {77, {0.5, 0.5, z(-1.0)}, {0.0, 0.0, 1.0}, {0.5, 0.5, z(9.0)}},
    };

    const auto intersections = vc::fiber_tracer::fiberSideStripIntersections(options);

    REQUIRE(intersections.size() == 1);
    CHECK(intersections[0].source == vc::fiber_tracer::FiberSideStripIntersectionSource::BranchLink);
    CHECK(intersections[0].fiberId == 77);
    CHECK(intersections[0].point[2] == doctest::Approx(z(0.0)));
    CHECK(intersections[0].connectorStart[2] == doctest::Approx(z(9.0)));
    CHECK(intersections[0].projectionTarget[2] == doctest::Approx(z(-1.0)));
}

TEST_CASE("Fiber side strip branch projection emits no connector without a strip hit")
{
    vc::fiber_tracer::FiberSideStripQueryOptions options;
    options.stripPoints = flatStripGrid(2, 2);
    options.maxResults = 0;
    options.branchLinks = {
        {77, {5.0, 5.0, z(-1.0)}, {0.0, 0.0, 1.0}, {0.5, 0.5, z(0.0)}},
    };

    CHECK(vc::fiber_tracer::fiberSideStripIntersections(options).empty());
}
