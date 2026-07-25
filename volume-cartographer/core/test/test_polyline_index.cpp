#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PolylineIndex.hpp"

TEST_CASE("PolylineIndex queries connected segments by box and category")
{
    PolylineIndex index;
    index.build({
        {7, "fibers", {{0, 0, 0}, {5, 0, 0}, {10, 0, 0}}},
        {9, "tracks", {{0, 4, 0}, {10, 4, 0}}},
    });
    CHECK(index.polylineCount() == 2);
    CHECK(index.segmentCount() == 3);

    auto all = index.query({4, -1, -1}, {6, 5, 1});
    REQUIRE(all.size() == 3);
    auto fibers = index.query({4, -1, -1}, {6, 5, 1}, std::string("fibers"));
    REQUIRE(fibers.size() == 2);
    CHECK(fibers[0].objectId == 7);
    CHECK(fibers[0].segmentIndex == 0);
    CHECK(fibers[1].segmentIndex == 1);
}

TEST_CASE("PolylineIndex applies build padding and validates generations")
{
    PolylineIndex index;
    index.build({{1, "pcl", {{0, 0, 0}, {1, 0, 0}}}}, 2.0f);
    CHECK(index.query({0, 1.5f, 0}, {1, 1.6f, 0}).size() == 1);
    CHECK_THROWS(index.build({{1, "bad", {}}}));
    CHECK_THROWS(index.build({{1, "bad", {{NAN, 0, 0}, {1, 0, 0}}}}));
}
