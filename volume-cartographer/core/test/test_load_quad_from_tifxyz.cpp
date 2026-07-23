// Coverage for the free `load_quad_from_tifxyz` factory in QuadSurface.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <filesystem>
#include <opencv2/core.hpp>
#include <random>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

#ifndef VC_TEST_FIXTURES_DIR
#define VC_TEST_FIXTURES_DIR "core/test/data"
#endif

fs::path fixtureSegment(const std::string& name)
{
    return fs::path(VC_TEST_FIXTURES_DIR) / "segments" / name;
}

} // namespace

TEST_CASE("load_quad_from_tifxyz: returns a valid surface for a real fixture")
{
    auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing at " << seg);
        return;
    }
    auto qs = load_quad_from_tifxyz(seg.string(), 0);
    REQUIRE(qs);
    qs->ensureLoaded();
    CHECK(qs->rawPointsPtr() != nullptr);
    CHECK(qs->rawPointsPtr()->rows == 129);
    CHECK(qs->rawPointsPtr()->cols == 129);
    CHECK(qs->meta["uuid"].get_string() == "20241113070770");
}

TEST_CASE("load_quad_from_tifxyz: missing path throws")
{
    CHECK_THROWS(load_quad_from_tifxyz("/__no__/__here__", 0));
}

TEST_CASE("load_quad_from_tifxyz: works for the other fixtures too")
{
    for (const char* name : {"20241113080880", "20241113090990"}) {
        auto seg = fixtureSegment(name);
        if (!fs::exists(seg / "meta.json")) continue;
        auto qs = load_quad_from_tifxyz(seg.string(), 0);
        REQUIRE(qs);
        qs->ensureLoaded();
        CHECK(qs->rawPointsPtr() != nullptr);
        CHECK(qs->meta["uuid"].get_string() == std::string(name));
    }
}

TEST_CASE("load_quad_from_tifxyz_region: decodes only the requested grid region")
{
    const auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing at " << seg);
        return;
    }

    QuadSurface lazy(seg);
    CHECK_FALSE(lazy.isLoaded());
    CHECK(lazy.gridSize() == cv::Size(129, 129));
    CHECK_FALSE(lazy.isLoaded());

    const cv::Rect requested(7, 11, 23, 31);
    auto full = load_quad_from_tifxyz(seg);
    auto region = load_quad_from_tifxyz_region(seg, requested);
    REQUIRE(full);
    REQUIRE(region);
    REQUIRE(region->rawPointsPtr()->size() == requested.size());
    CHECK(region->path.empty());

    const auto expected = (*full->rawPointsPtr())(requested);
    CHECK(cv::norm(expected, *region->rawPointsPtr(), cv::NORM_INF)
          == doctest::Approx(0.0));
}

TEST_CASE("load_quad_from_tifxyz_region: rejects an out-of-bounds region")
{
    const auto seg = fixtureSegment("20241113070770");
    if (!fs::exists(seg / "meta.json")) {
        MESSAGE("Skipping: fixture missing at " << seg);
        return;
    }
    CHECK_THROWS(load_quad_from_tifxyz_region(seg, cv::Rect(120, 0, 20, 10)));
}
