#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/SheetSeed.hpp"
#include <limits>

using namespace vc::sheet;

TEST_CASE("sheet seed stays fixed and produces identical planar grids across modes and workers")
{
    const cv::Vec3f seed(100, 200, 300);
    for (auto mode : {SheetWalkMode::Projection, SheetWalkMode::Streamline}) {
        for (auto integrator : {SheetIntegrator::Midpoint, SheetIntegrator::Heun, SheetIntegrator::RK4}) {
            SeedOptions options;
            options.generations = 3;
            options.spacing = 2;
            options.walk = {0.5, mode, integrator};
            const auto normal = [](const cv::Vec3f&) { return std::optional<cv::Vec3f>({0, 0, -1}); };
            const auto points = growSheetSeed(seed, normal, options);
            CHECK(points.rows == 7);
            CHECK(points.cols == 7);
            for (int r = 0; r < 7; ++r)
                for (int c = 0; c < 7; ++c)
                    CHECK(cv::norm(points(r, c) - (seed + cv::Vec3f(2 * (c - 3), 2 * (r - 3), 0))) < 1e-5);
            options.workers = 4;
            const auto parallel = growSheetSeed(seed, normal, options);
            CHECK(cv::norm(points, parallel, cv::NORM_INF) == 0);
        }
    }
}

TEST_CASE("sheet seed follows a cylinder and is insensitive to normal signs")
{
    SeedOptions options;
    options.generations = 5;
    options.spacing = 2;
    options.walk = {0.25, SheetWalkMode::Streamline, SheetIntegrator::RK4};
    const auto cylinder = [](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
        auto n = cv::Vec3f(p[0], p[1], 0);
        return n / cv::norm(n);
    };
    const auto points = growSheetSeed({100, 0, 100}, cylinder, options);
    const auto flipped = growSheetSeed({100, 0, 100}, [&](const cv::Vec3f& p) {
        auto n = cylinder(p);
        return p[1] < 0 ? n : std::optional<cv::Vec3f>(-*n);
    }, options);
    CHECK(cv::norm(points, flipped, cv::NORM_INF) < 1e-4);
    for (const auto& p : points)
        CHECK(std::abs(std::hypot(p[0], p[1]) - 100) < 1e-3);
    CHECK(cv::norm(points(5, 5) - cv::Vec3f(100, 0, 100)) == 0);
}

TEST_CASE("sheet seed rejects missing normals and invalid sizes")
{
    SeedOptions options;
    CHECK_THROWS(growSheetSeed({10, 10, 10}, {}, options));
    const SheetNormalFn normal = [](const cv::Vec3f&) { return std::optional<cv::Vec3f>({0, 0, 1}); };
    options.generations = 100000;
    CHECK_THROWS(growSheetSeed({10, 10, 10}, normal, options));
    options.generations = 2;
    options.spacing = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(growSheetSeed({10, 10, 10}, normal, options));
    options.spacing = 2;
    CHECK_THROWS(growSheetSeed({10, 10, 10}, [](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
        if (p[0] != 10) throw std::runtime_error("read failed");
        return cv::Vec3f(0, 0, 1);
    }, options));
}

TEST_CASE("sheet seed progress is ordered and complete with multiple workers")
{
    for (auto mode : {SheetWalkMode::Projection, SheetWalkMode::Streamline}) {
        for (double maxStep : {0.0, 0.5}) {
            SeedOptions options;
            options.generations = 3;
            options.spacing = 2;
            options.workers = 4;
            options.walk = {maxStep, mode, SheetIntegrator::RK4};
            const SheetNormalFn normal = [](const cv::Vec3f&) { return std::optional<cv::Vec3f>({0, 0, 1}); };
            struct Event { SeedStage stage; std::size_t completed, total; };
            std::vector<Event> events;
            const auto observed = growSheetSeed({10, 20, 30}, normal, options,
                [&](SeedStage stage, std::size_t completed, std::size_t total) {
                    events.push_back({stage, completed, total});
                });
            const auto unobserved = growSheetSeed({10, 20, 30}, normal, options);
            CHECK(cv::norm(observed, unobserved, cv::NORM_INF) == 0);
            std::size_t index = 0;
            const std::size_t totals[] = {1, 6, 6, 42};
            for (auto stage : {SeedStage::SeedNormal, SeedStage::CentralColumn,
                               SeedStage::RowDirections, SeedStage::Rows}) {
                const auto total = totals[static_cast<int>(stage)];
                for (std::size_t count = 0; count <= total; ++count) {
                    REQUIRE(index < events.size());
                    CHECK(events[index].stage == stage);
                    CHECK(events[index].completed == count);
                    CHECK(events[index].total == total);
                    ++index;
                }
            }
            CHECK(index == events.size());
        }
    }
}

TEST_CASE("sheet seed does not report completed growth after a failed query")
{
    SeedOptions options;
    options.generations = 3;
    options.workers = 4;
    bool rowsCompleted = false;
    const auto normal = [](const cv::Vec3f& p) -> std::optional<cv::Vec3f> {
        if (p[0] != 100) throw std::runtime_error("read failed");
        return cv::Vec3f(0, 0, 1);
    };
    CHECK_THROWS(growSheetSeed({100, 100, 100}, normal, options,
        [&](SeedStage stage, std::size_t completed, std::size_t total) {
            if (stage == SeedStage::Rows && completed == total) rowsCompleted = true;
        }));
    CHECK_FALSE(rowsCompleted);
}
