// Coverage for the free `load_quad_from_tifxyz` factory in QuadSurface.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <filesystem>
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
