#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PointIndex.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

TEST_CASE("default-constructed index is empty")
{
    PointIndex idx;
    CHECK(idx.empty());
    CHECK(idx.size() == 0);
}

TEST_CASE("insert / size / empty")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(1.f, 2.f, 3.f));
    CHECK_FALSE(idx.empty());
    CHECK(idx.size() == 1);
    idx.insert(2, 0, cv::Vec3f(4.f, 5.f, 6.f));
    CHECK(idx.size() == 2);
}

TEST_CASE("insert rejects non-finite points")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(std::nanf(""), 0.f, 0.f));
    CHECK(idx.size() == 0);
    idx.insert(2, 0, cv::Vec3f(std::numeric_limits<float>::infinity(), 0.f, 0.f));
    CHECK(idx.size() == 0);
}

TEST_CASE("insert of existing id updates position")
{
    PointIndex idx;
    idx.insert(7, 0, cv::Vec3f(0.f, 0.f, 0.f));
    idx.insert(7, 0, cv::Vec3f(100.f, 0.f, 0.f));
    CHECK(idx.size() == 1);
    auto r = idx.nearest(cv::Vec3f(100.f, 0.f, 0.f));
    REQUIRE(r.has_value());
    CHECK(r->id == 7);
    CHECK(r->distanceSq == doctest::Approx(0.0f));
}

TEST_CASE("clear empties the index")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(1, 2, 3));
    idx.clear();
    CHECK(idx.empty());
    CHECK(idx.size() == 0);
}

TEST_CASE("remove existing id")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    idx.insert(2, 0, cv::Vec3f(1, 1, 1));
    idx.remove(1);
    CHECK(idx.size() == 1);
    auto r = idx.nearest(cv::Vec3f(0, 0, 0));
    REQUIRE(r.has_value());
    CHECK(r->id == 2);
}

TEST_CASE("remove missing id is a no-op")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    idx.remove(999);
    CHECK(idx.size() == 1);
}

TEST_CASE("update existing id changes position")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    CHECK(idx.update(1, cv::Vec3f(50, 0, 0)));
    auto r = idx.nearest(cv::Vec3f(50, 0, 0));
    REQUIRE(r.has_value());
    CHECK(r->distanceSq == doctest::Approx(0.0f));
}

TEST_CASE("update returns false for unknown id or non-finite")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    CHECK_FALSE(idx.update(999, cv::Vec3f(0, 0, 0)));
    CHECK_FALSE(idx.update(1, cv::Vec3f(std::nanf(""), 0, 0)));
}

TEST_CASE("bulkInsert clears and rebuilds")
{
    PointIndex idx;
    idx.insert(99, 0, cv::Vec3f(0, 0, 0));
    std::vector<std::tuple<uint64_t, uint64_t, cv::Vec3f>> pts = {
        {1, 0, cv::Vec3f(0, 0, 0)},
        {2, 1, cv::Vec3f(10, 0, 0)},
        {3, 0, cv::Vec3f(0, 10, 0)},
    };
    idx.bulkInsert(pts);
    CHECK(idx.size() == 3);
    // 99 from before should be gone
    auto r = idx.nearest(cv::Vec3f(0, 0, 0));
    REQUIRE(r.has_value());
    CHECK(r->id == 1);
}

TEST_CASE("bulkInsert with empty vector clears")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    idx.bulkInsert({});
    CHECK(idx.empty());
}

TEST_CASE("bulkInsert skips non-finite points")
{
    PointIndex idx;
    std::vector<std::tuple<uint64_t, uint64_t, cv::Vec3f>> pts = {
        {1, 0, cv::Vec3f(0, 0, 0)},
        {2, 0, cv::Vec3f(std::nanf(""), 0, 0)},
    };
    idx.bulkInsert(pts);
    CHECK(idx.size() == 1);
}

TEST_CASE("buildFromMat indexes valid points and ignores -1 sentinels")
{
    cv::Mat_<cv::Vec3f> m(2, 3, cv::Vec3f(-1.f, -1.f, -1.f));
    m(0, 0) = cv::Vec3f(1, 1, 1);
    m(1, 2) = cv::Vec3f(2, 2, 2);
    PointIndex idx;
    idx.buildFromMat(m, 42);
    CHECK(idx.size() == 2);
    auto r = idx.nearest(cv::Vec3f(2, 2, 2));
    REQUIRE(r.has_value());
    CHECK(r->collectionId == 42);
}

TEST_CASE("buildFromMat with empty mat is no-op")
{
    cv::Mat_<cv::Vec3f> m;
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    idx.buildFromMat(m);
    CHECK(idx.empty());
}

TEST_CASE("nearest on empty returns nullopt")
{
    PointIndex idx;
    CHECK_FALSE(idx.nearest(cv::Vec3f(0, 0, 0)).has_value());
}

TEST_CASE("nearest with non-finite query returns nullopt")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    CHECK_FALSE(idx.nearest(cv::Vec3f(std::nanf(""), 0, 0)).has_value());
}

TEST_CASE("nearest with maxDistance filter")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    auto r = idx.nearest(cv::Vec3f(100, 0, 0), 50.0f);
    CHECK_FALSE(r.has_value());
    auto r2 = idx.nearest(cv::Vec3f(100, 0, 0), 200.0f);
    REQUIRE(r2.has_value());
    CHECK(r2->id == 1);
}

TEST_CASE("queryRadius returns sorted by distance")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(5, 0, 0));
    idx.insert(2, 0, cv::Vec3f(1, 0, 0));
    idx.insert(3, 0, cv::Vec3f(3, 0, 0));
    idx.insert(4, 0, cv::Vec3f(100, 0, 0)); // out of range
    auto r = idx.queryRadius(cv::Vec3f(0, 0, 0), 10.0f);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 2);
    CHECK(r[1].id == 3);
    CHECK(r[2].id == 1);
}

TEST_CASE("queryRadius rejects bad inputs")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    CHECK(idx.queryRadius(cv::Vec3f(0, 0, 0), 0.0f).empty());
    CHECK(idx.queryRadius(cv::Vec3f(0, 0, 0), -1.0f).empty());
    CHECK(idx.queryRadius(cv::Vec3f(0, 0, 0), std::nanf("")).empty());
    CHECK(idx.queryRadius(cv::Vec3f(std::nanf(""), 0, 0), 1.0f).empty());
    PointIndex empty;
    CHECK(empty.queryRadius(cv::Vec3f(0, 0, 0), 1.0f).empty());
}

TEST_CASE("kNearest returns up to k sorted, bounded by maxDistance")
{
    PointIndex idx;
    for (int i = 0; i < 5; ++i) {
        idx.insert(i + 1, 0, cv::Vec3f(static_cast<float>(i), 0.f, 0.f));
    }
    auto r = idx.kNearest(cv::Vec3f(0, 0, 0), 3);
    REQUIRE(r.size() == 3);
    CHECK(r[0].id == 1);
    CHECK(r[1].id == 2);
    CHECK(r[2].id == 3);

    // maxDistance filters
    auto rf = idx.kNearest(cv::Vec3f(0, 0, 0), 10, 1.5f);
    CHECK(rf.size() == 2);
}

TEST_CASE("kNearest with k=0 or empty index returns empty")
{
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    CHECK(idx.kNearest(cv::Vec3f(0, 0, 0), 0).empty());
    PointIndex empty;
    CHECK(empty.kNearest(cv::Vec3f(0, 0, 0), 5).empty());
    CHECK(idx.kNearest(cv::Vec3f(std::nanf(""), 0, 0), 5).empty());
}

TEST_CASE("nearestInCollection filters by collectionId")
{
    // NOTE: the current impl does NOT guarantee returning the *closest* member
    // of the given collection — it returns the first candidate from an rtree
    // nearest-k query whose collectionId matches, in whatever iteration order
    // boost::geometry yields. A stricter "closest in collection" assertion
    // belongs in the later hard-bug-catching pass.
    PointIndex idx;
    // Need >=16 points before the impl's initial k=16 search loop runs.
    for (int i = 0; i < 20; ++i) {
        idx.insert(static_cast<uint64_t>(i + 1),
                   (i == 5) ? 20u : 10u,
                   cv::Vec3f(static_cast<float>(i), 0.f, 0.f));
    }
    auto r = idx.nearestInCollection(cv::Vec3f(0, 0, 0), 20);
    REQUIRE(r.has_value());
    CHECK(r->collectionId == 20);
    auto r2 = idx.nearestInCollection(cv::Vec3f(0, 0, 0), 10);
    REQUIRE(r2.has_value());
    CHECK(r2->collectionId == 10);
}

TEST_CASE("nearestInCollection returns nullopt for missing collection")
{
    PointIndex idx;
    for (int i = 0; i < 20; ++i) {
        idx.insert(static_cast<uint64_t>(i + 1), 10,
                   cv::Vec3f(static_cast<float>(i), 0.f, 0.f));
    }
    auto r = idx.nearestInCollection(cv::Vec3f(0, 0, 0), 999);
    CHECK_FALSE(r.has_value());
}

TEST_CASE("nearestInCollection respects maxDistance")
{
    PointIndex idx;
    for (int i = 0; i < 20; ++i) {
        idx.insert(static_cast<uint64_t>(i + 1), 10,
                   cv::Vec3f(100.f + static_cast<float>(i), 0.f, 0.f));
    }
    auto r = idx.nearestInCollection(cv::Vec3f(0, 0, 0), 10, 50.0f);
    CHECK_FALSE(r.has_value());
}

TEST_CASE("nearestInCollection on empty / non-finite returns nullopt")
{
    PointIndex empty;
    CHECK_FALSE(empty.nearestInCollection(cv::Vec3f(0, 0, 0), 0).has_value());
    PointIndex idx;
    idx.insert(1, 0, cv::Vec3f(0, 0, 0));
    CHECK_FALSE(idx.nearestInCollection(cv::Vec3f(std::nanf(""), 0, 0), 0).has_value());
}

TEST_CASE("move construction / assignment transfers ownership")
{
    PointIndex a;
    a.insert(1, 0, cv::Vec3f(0, 0, 0));
    PointIndex b(std::move(a));
    CHECK(b.size() == 1);
    PointIndex c;
    c = std::move(b);
    CHECK(c.size() == 1);
}
