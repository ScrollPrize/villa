// Data/IO coverage for PointCollections — the Qt-free class split out of
// VCCollection. Exercises mutation, queries, autofill, anchors, and a
// JSON save/load round-trip without any Qt.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/PointCollections.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <unistd.h>  // ::getpid for unique temp file names

namespace fs = std::filesystem;

TEST_CASE("add/query points and collections")
{
    PointCollections c;
    const auto p0 = c.addPoint("alpha", {1, 2, 3});
    const auto p1 = c.addPoint("alpha", {4, 5, 6});

    const uint64_t cid = c.getCollectionId("alpha");
    CHECK(cid != 0);
    CHECK(c.getAllCollections().size() == 1);
    CHECK(c.getAllCollections().at(cid).points.size() == 2);
    CHECK(p0.id != p1.id);
    CHECK(p0.collectionId == cid);

    auto got = c.getPoint(p0.id);
    REQUIRE(got.has_value());
    CHECK(got->p == cv::Vec3f(1, 2, 3));

    CHECK(c.getPoints("alpha").size() == 2);
}

TEST_CASE("update and remove")
{
    PointCollections c;
    auto p = c.addPoint("a", {0, 0, 0});
    p.p = {9, 9, 9};
    c.updatePoint(p);
    CHECK(c.getPoint(p.id)->p == cv::Vec3f(9, 9, 9));

    c.removePoint(p.id);
    CHECK_FALSE(c.getPoint(p.id).has_value());
}

TEST_CASE("rename, clear, clearAll")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    const uint64_t cid = c.getCollectionId("a");
    c.renameCollection(cid, "b");
    CHECK(c.getCollectionId("b") == cid);
    CHECK(c.getCollectionId("a") == 0);

    c.clearCollection(cid);  // removes the collection entirely
    CHECK(c.getAllCollections().count(cid) == 0);

    c.addPoint("x", {1, 1, 1});
    c.clearAll();
    CHECK(c.getAllCollections().empty());
}

TEST_CASE("tags and anchor2d + offset")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    const uint64_t cid = c.getCollectionId("a");

    c.setCollectionTag(cid, "k", "v");
    CHECK(c.getCollectionTag(cid, "k") == std::optional<std::string>("v"));
    c.removeCollectionTag(cid, "k");
    CHECK_FALSE(c.getCollectionTag(cid, "k").has_value());

    c.setCollectionAnchor2d(cid, cv::Vec2f(10, 20));
    REQUIRE(c.getCollectionAnchor2d(cid).has_value());
    c.applyAnchorOffset(5, -5);
    auto a = c.getCollectionAnchor2d(cid);
    REQUIRE(a.has_value());
    CHECK((*a)[0] == doctest::Approx(15));
    CHECK((*a)[1] == doctest::Approx(15));
}

TEST_CASE("addPoints batch + generateNewCollectionName")
{
    PointCollections c;
    c.addPoints("batch", {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}});
    CHECK(c.getPoints("batch").size() == 3);

    // No collections named "col1" yet → first generated name.
    CHECK(c.generateNewCollectionName() == "col1");
    c.addCollection("col1");
    CHECK(c.generateNewCollectionName() == "col2");
}

TEST_CASE("setCollectionMetadata round-trips the flag")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    const uint64_t cid = c.getCollectionId("a");
    CollectionMetadata m;
    m.absolute_winding_number = false;
    c.setCollectionMetadata(cid, m);
    CHECK_FALSE(c.getAllCollections().at(cid).metadata.absolute_winding_number);
}

TEST_CASE("autofill winding modes")
{
    auto windings = [](PointCollections& c, const std::string& name) {
        std::vector<float> w;
        for (const auto& p : c.getPoints(name)) w.push_back(p.winding_annotation);
        std::sort(w.begin(), w.end());
        return w;
    };

    SUBCASE("incremental assigns 1..N by id order")
    {
        PointCollections c;
        c.addPoints("a", {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}});
        const uint64_t cid = c.getCollectionId("a");
        c.autoFillWindingNumbers(cid, PointCollections::WindingFillMode::Incremental);
        CHECK(windings(c, "a") == std::vector<float>{1, 2, 3});
    }
    SUBCASE("decremental assigns N..1")
    {
        PointCollections c;
        c.addPoints("a", {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}});
        const uint64_t cid = c.getCollectionId("a");
        c.autoFillWindingNumbers(cid, PointCollections::WindingFillMode::Decremental);
        CHECK(windings(c, "a") == std::vector<float>{1, 2, 3});  // sorted; values are 3,2,1
    }
    SUBCASE("constant assigns the given value")
    {
        PointCollections c;
        c.addPoints("a", {{0, 0, 0}, {1, 1, 1}});
        const uint64_t cid = c.getCollectionId("a");
        c.autoFillWindingNumbers(cid, PointCollections::WindingFillMode::Constant, 7.5f);
        CHECK(windings(c, "a") == std::vector<float>{7.5f, 7.5f});
    }
}

TEST_CASE("setAutoFillMode + computeAutoFillValue")
{
    PointCollections c;
    c.addPoints("a", {{0, 0, 0}, {1, 1, 1}});
    const uint64_t cid = c.getCollectionId("a");

    c.setAutoFillMode(cid, PointCollections::WindingFillMode::Constant, 4.0f);
    CHECK(c.getAutoFillMode(cid) == PointCollections::WindingFillMode::Constant);
    CHECK(c.getAutoFillConstant(cid) == doctest::Approx(4.0f));
    CHECK(c.computeAutoFillValue(cid) == doctest::Approx(4.0f));

    // Incremental → max existing winding + 1; none set yet → 1.
    c.setAutoFillMode(cid, PointCollections::WindingFillMode::Incremental);
    CHECK(c.computeAutoFillValue(cid) == doctest::Approx(1.0f));
    c.autoFillWindingNumbers(cid, PointCollections::WindingFillMode::Incremental);  // sets 1,2
    CHECK(c.computeAutoFillValue(cid) == doctest::Approx(3.0f));  // max(2)+1
}

TEST_CASE("missing-id queries are safe")
{
    PointCollections c;
    CHECK(c.getCollectionId("nope") == 0);
    CHECK_FALSE(c.getPoint(424242).has_value());
    CHECK(c.getPoints("nope").empty());
    CHECK(c.getAutoFillMode(999) == PointCollections::WindingFillMode::None);
}

TEST_CASE("segment-path round-trip (anchored collections only)")
{
    const fs::path dir = fs::temp_directory_path() /
        ("pc_seg_" + std::to_string(::getpid()));
    fs::create_directories(dir);

    PointCollections c;
    c.addPoint("anchored", {1, 2, 3});
    c.addPoint("plain", {4, 5, 6});
    const uint64_t anchored = c.getCollectionId("anchored");
    c.setCollectionAnchor2d(anchored, cv::Vec2f(8, 9));
    REQUIRE(c.saveToSegmentPath(dir));
    CHECK(fs::exists(dir / "corrections.json"));

    PointCollections loaded;
    REQUIRE(loaded.loadFromSegmentPath(dir));
    // Only the anchored collection is persisted.
    bool has_anchored = false, has_plain = false;
    for (const auto& [id, col] : loaded.getAllCollections()) {
        if (col.name == "anchored") has_anchored = true;
        if (col.name == "plain") has_plain = true;
    }
    CHECK(has_anchored);
    CHECK_FALSE(has_plain);

    fs::remove_all(dir);
}

TEST_CASE("JSON round-trip preserves data")
{
    PointCollections c;
    c.addPoint("alpha", {1, 2, 3});
    c.addPoint("alpha", {4, 5, 6});
    c.addPoint("beta", {7, 8, 9});
    const uint64_t alpha = c.getCollectionId("alpha");
    c.setCollectionColor(alpha, {0.1f, 0.2f, 0.3f});
    c.setCollectionTag(alpha, "kind", "correction");

    const fs::path tmp = fs::temp_directory_path() /
        ("pc_roundtrip_" + std::to_string(::getpid()) + ".json");
    REQUIRE(c.saveToJSON(tmp.string()));

    PointCollections loaded;
    REQUIRE(loaded.loadFromJSON(tmp.string()));
    fs::remove(tmp);

    CHECK(loaded.getAllCollections().size() == c.getAllCollections().size());
    const uint64_t la = loaded.getCollectionId("alpha");
    CHECK(la != 0);
    CHECK(loaded.getPoints("alpha").size() == 2);
    CHECK(loaded.getPoints("beta").size() == 1);
    CHECK(loaded.getCollectionTag(la, "kind") == std::optional<std::string>("correction"));
}

TEST_CASE("JSON round-trip preserves per-point fields")
{
    PointCollections c;
    auto p = c.addPoint("a", {1.5f, 2.5f, 3.5f});
    p.winding_annotation = 4.0f;
    p.fiber_dir = "h";
    c.updatePoint(p);
    auto linked = c.addPoint("a", {9, 9, 9});  // leaves winding_annotation NaN
    p.links.push_back(linked.id);
    c.updatePoint(p);

    const fs::path tmp = fs::temp_directory_path() /
        ("pc_fields_" + std::to_string(::getpid()) + ".json");
    REQUIRE(c.saveToJSON(tmp.string()));
    PointCollections loaded;
    REQUIRE(loaded.loadFromJSON(tmp.string()));
    fs::remove(tmp);

    int with_winding = 0, nan_winding = 0;
    cv::Vec3f seen_pos{};
    bool saw_link = false;
    bool saw_h = false;
    bool saw_unknown_dir = false;
    for (const auto& q : loaded.getPoints("a")) {
        if (std::isnan(q.winding_annotation)) {
            nan_winding++;
            saw_unknown_dir = saw_unknown_dir || q.fiber_dir.empty();
        } else {
            with_winding++;
            seen_pos = q.p;
            saw_link = q.links.size() == 1 && q.links.front() == linked.id;
            saw_h = q.fiber_dir == "h";
        }
    }
    CHECK(with_winding == 1);
    CHECK(nan_winding == 1);
    CHECK(seen_pos == cv::Vec3f(1.5f, 2.5f, 3.5f));  // float position survives
    CHECK(saw_link);
    CHECK(saw_h);
    CHECK(saw_unknown_dir);
}

TEST_CASE("old point JSON defaults links and fiber direction")
{
    const fs::path tmp = fs::temp_directory_path() /
        ("pc_old_fields_" + std::to_string(::getpid()) + ".json");
    std::ofstream out(tmp);
    out << R"({
        "vc_pointcollections_json_version": "1",
        "collections": {
            "1": {
                "name": "old",
                "points": {
                    "2": {"p": [1, 2, 3], "wind_a": null, "creation_time": 0}
                },
                "metadata": {},
                "color": [1, 1, 1]
            }
        }
    })";
    out.close();

    PointCollections loaded;
    REQUIRE(loaded.loadFromJSON(tmp.string()));
    fs::remove(tmp);

    const auto points = loaded.getPoints("old");
    REQUIRE(points.size() == 1);
    CHECK(points.front().links.empty());
    CHECK(points.front().fiber_dir.empty());
    CHECK(loaded.getAllCollections().begin()->second.metadata.absolute_winding_number);
}

TEST_CASE("loadFromJSON rejects bad input and returns false")
{
    const fs::path tmp = fs::temp_directory_path() /
        ("pc_bad_" + std::to_string(::getpid()) + ".json");

    auto write = [&](const std::string& s) {
        std::ofstream o(tmp); o << s; o.close();
    };

    PointCollections c;

    SUBCASE("malformed json")
    {
        write("{ this is not json ");
        CHECK_FALSE(c.loadFromJSON(tmp.string()));
    }
    SUBCASE("missing version key")
    {
        write(R"({"collections": {}})");
        CHECK_FALSE(c.loadFromJSON(tmp.string()));
    }
    SUBCASE("nonexistent file")
    {
        CHECK_FALSE(c.loadFromJSON((tmp.string() + ".missing")));
    }
    fs::remove(tmp);
}

TEST_CASE("saveToJSON fails on an unwritable path")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    CHECK_FALSE(c.saveToJSON("/nonexistent_dir_xyz/corrections.json"));
}

TEST_CASE("ids stay unique across a load + further adds")
{
    PointCollections c;
    c.addPoint("a", {0, 0, 0});
    c.addPoint("a", {1, 1, 1});

    const fs::path tmp = fs::temp_directory_path() /
        ("pc_ids_" + std::to_string(::getpid()) + ".json");
    REQUIRE(c.saveToJSON(tmp.string()));
    PointCollections loaded;
    REQUIRE(loaded.loadFromJSON(tmp.string()));
    fs::remove(tmp);

    auto fresh = loaded.addPoint("a", {2, 2, 2});
    // The fresh id must be exactly one of the three points (no collision with
    // a loaded id, which would have left only two distinct ids / two points).
    int matches = 0;
    for (const auto& q : loaded.getPoints("a")) {
        if (q.id == fresh.id) matches++;
    }
    CHECK(matches == 1);
    CHECK(loaded.getPoints("a").size() == 3);
}
