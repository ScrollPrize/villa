#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/TrackCrossings.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <bit>
#include <cstring>
#include <functional>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include "vc_object_store/Uuid.hpp"
#include <string>
#include <thread>

#include <opencv2/core.hpp>


namespace
{
using namespace vc::objects;
using namespace vc::object_links;
namespace fs = std::filesystem;

struct TempDir {
    fs::path dir;
    explicit TempDir(const std::string& name)
    {
        dir = fs::temp_directory_path() / ("vc_object_links_test_" + name + "_" + vc::objects::util::newUuid());
        fs::create_directories(dir);
    }
    ~TempDir() { fs::remove_all(dir); }
};

ObjectStore makeDb(const fs::path& dir)
{
    OpenOptions o;
    o.author = "links-test";
    o.hostnameOverride = "testhost";
    return ObjectStore::create(dir / "graph.vcobj", o);
}

std::string addObject(ObjectStore& db, const std::string& kind)
{
    auto w = db.write();
    ObjectRecord o;
    o.kind = kind;
    std::string uuid = db.upsertObject(w, o);
    w.commit();
    return uuid;
}

LinkCandidate cand(const std::string& a, const std::string& b)
{
    LinkCandidate c;
    c.aUuid = a;
    c.bUuid = b;
    return c;
}

size_t edgeCount(ObjectStore& db, const std::string& uuid)
{
    auto r = db.read();
    return db.edgesOf(r, uuid).size();
}

} // namespace

TEST_CASE("commitCandidates stamps provenance, pending, dedupes")
{
    TempDir tmp("commit");
    auto db = makeDb(tmp.dir);
    const std::string a = addObject(db, "patch");
    const std::string b = addObject(db, "patch");

    LinkCandidate c = cand(a, b);
    c.score = 0.87;
    c.meta = R"({"evidence":"synthetic"})";
    auto report = commitCandidates(db, kPatchOverlapDetector, 3, {c, c});
    CHECK(report.edgesCreated == 1);
    CHECK(report.edgesSkippedExisting == 1);  // duplicate within the batch

    auto r = db.read();
    const auto edges = db.edgesOf(r, a);
    REQUIRE(edges.size() == 1);
    CHECK(edges[0].approvalState == "pending");
    CHECK(edges[0].meta.find("\"patch-overlap\"") != std::string::npos);
    CHECK(edges[0].meta.find("\"version\":3") != std::string::npos);
    CHECK(edges[0].meta.find("synthetic") != std::string::npos);
    r.commit();

    // Re-committing the same candidates creates nothing new.
    auto again = commitCandidates(db, kPatchOverlapDetector, 3, {c});
    CHECK(again.edgesCreated == 0);
    CHECK(again.edgesSkippedExisting == 1);
    CHECK(again.staleRemoved == 0);
}

TEST_CASE("stale replacement removes only own pending edges")
{
    TempDir tmp("stale");
    auto db = makeDb(tmp.dir);
    const std::string a = addObject(db, "patch");
    const std::string b = addObject(db, "patch");
    const std::string cObj = addObject(db, "patch");

    commitCandidates(db, "det-a", 1, {cand(a, b), cand(a, cObj)});
    commitCandidates(db, "det-b", 1, {cand(b, cObj)});

    // Approve det-a's a-b edge.
    {
        auto r = db.read();
        std::string abUuid;
        for (const auto& e : db.edgesOf(r, a)) {
            if (e.bUuid == b || e.aUuid == b) abUuid = e.uuid;
        }
        r.commit();
        auto w = db.write();
        db.setApproval(w, abUuid, "approved");
        w.commit();
    }

    // det-a re-runs proposing only the (approved) a-b edge: it
    // survives, det-a's pending a-c edge is stale-removed, det-b's
    // edge is untouched.
    auto report = commitCandidates(db, "det-a", 2, {cand(a, b)});
    CHECK(report.staleRemoved == 1);
    CHECK(edgeCount(db, a) == 1);
    CHECK(edgeCount(db, cObj) == 1);

    // Manually created edges (no detector provenance) are never stale.
    {
        auto w = db.write();
        EdgeRecord manual;
        manual.aUuid = a;
        manual.bUuid = cObj;
        manual.kind = "same-surface";
        db.addEdge(w, manual);
        w.commit();
    }
    commitCandidates(db, "det-a", 3, {cand(a, b)});
    CHECK(edgeCount(db, cObj) == 2);
}

TEST_CASE("track crossing kernel: clustered medians, clearance and tie breaking")
{
    namespace tc = vc::track_crossings;
    std::vector<int32_t> xyz;
    for (int track = 0; track < 2; ++track)
        for (int i = 0; i <= 20; ++i) xyz.insert(xyz.end(), {0, track, i});
    const std::vector<int64_t> offsets{0, 21, 42};
    const std::vector<uint64_t> ids{17, uint64_t{1} << 40};
    tc::EventBuffer events{{{0,1,2,2}, {0,1,3,3}, {0,1,8,8}, {0,1,12,12}, {0,1,19,19}}};
    auto result = tc::consolidate_crossing_events(events, xyz, offsets, ids, 2);
    CHECK(events.size() == 0);
    // Clusters [2,3], [8,12], [19]; use upper median 12 in the middle.
    CHECK(result.accepted_events == 3);
    CHECK(result.paired_tracks == 2);
    CHECK(result.offsets == std::vector<int64_t>{0,1,2});
    CHECK(result.partners == std::vector<int32_t>{1,0});
    CHECK(result.self_local == std::vector<int32_t>{12,12});
    CHECK(result.positions == std::vector<double>{12,12});
    CHECK(result.clearances == std::vector<double>{8,8});
    tc::Parameters p;
    p.clusterIndexRadius = 0;
    tc::EventBuffer tied{{{0,1,8,8}, {0,1,12,12}}};
    auto tie = tc::consolidate_crossing_events(tied, xyz, offsets, ids, 1, {}, p);
    CHECK(tie.self_local == result.self_local);
}

TEST_CASE("track crossing radix sort is stable with constrained OpenMP teams")
{
    const std::vector<uint64_t> keys{9,1,9,3,1,(uint64_t{1}<<60)-1,0};
    const std::vector<uint32_t> expected{6,1,4,3,0,2,5};
#ifdef _OPENMP
    const auto dynamic = omp_get_dynamic();
    omp_set_dynamic(1);
#endif
    CHECK(vc::track_crossings::parallel_argsort(keys, 3) == expected);
#ifdef _OPENMP
    omp_set_dynamic(dynamic);
#endif
    CHECK(vc::track_crossings::parallel_argsort(keys, 1) == expected);
}

TEST_CASE("track linking cancellation and invalid parameters never write")
{
    TempDir tmp("track_cancel");
    auto db = makeDb(tmp.dir);
    const auto a = addObject(db, "track"), b = addObject(db, "track");
    auto c = cand(a, b);
    c.kind = "crossing";
    commitCandidates(db, kTrackTrackDetector, 1, {c});
    std::atomic_bool cancel{true};
    auto report = linkTracksToTracks(db, {}, &cancel);
    REQUIRE(report.problems.size() == 1);
    CHECK(report.problems[0].find("cancelled") != std::string::npos);
    CHECK(edgeCount(db, a) == 1);
    TrackTrackParams params;
    params.minAngleDegrees = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(linkTracksToTracks(db, params));
    CHECK(edgeCount(db, a) == 1);
}

TEST_CASE("track crossing kernel rejects invalid array indices")
{
    namespace tc = vc::track_crossings;
    const std::vector<int32_t> coordinates{0,0,0};
    const std::vector<int64_t> offsets{0,1};
    const std::vector<uint64_t> keys{0}, ids{0};
    const std::vector<int8_t> families{0};
    const std::vector<uint32_t> invalidOrder{1};
    CHECK_THROWS(tc::scan_crossing_events(coordinates, offsets, families, keys, invalidOrder, 1));
    tc::EventBuffer invalidEvents{{{0,1,0,0}}};
    CHECK_THROWS(tc::consolidate_crossing_events(invalidEvents, coordinates, offsets, ids, 1));
    const std::vector<uint64_t> oversizedKey{uint64_t{1} << 60};
    CHECK_THROWS(tc::parallel_argsort(oversizedKey, 1));
}

TEST_CASE("caller-owned candidate transaction rolls registration and edges back together")
{
    TempDir tmp("atomic_registration");
    auto db = makeDb(tmp.dir);
    {
        auto write = db.write();
        ObjectRecord object;
        object.kind = "track";
        const auto a = db.upsertObject(write, object);
        const auto b = db.upsertObject(write, object);
        const auto report = commitCandidates(db, write, kTrackTrackDetector, 1, {cand(a, b)});
        CHECK(report.edgesCreated == 1);
        // Deliberately do not commit the outer transaction.
    }
    auto read = db.read();
    CHECK(db.objectsByKind(read, "track").empty());
    CHECK(db.edgesByKind(read, "same-surface").empty());
}

TEST_CASE("pending detector links refresh anchors and evidence while retaining identity and annotations")
{
    TempDir tmp("refresh_candidates");
    auto db = makeDb(tmp.dir);
    const auto a = addObject(db, "fiber"), b = addObject(db, "patch");
    auto c = cand(a, b);
    c.aAnchor.points = {{"control_points", {4}}}; c.aAnchor.position = {1, 2, 3}; c.score = 0.5;
    c.meta = R"({"distance":5})";
    REQUIRE(commitCandidates(db, "detector", 1, {c}).edgesCreated == 1);
    std::string id;
    { auto read = db.read(); id = db.edgesOf(read, a).at(0).uuid; }
    { auto w = db.write(); db.setMeta(w, id, "review_note", "\"retain\""); auto edge = *db.edgeByUuid(w, id); edge.tags = {"user"}; db.updateEdge(w, edge); w.commit(); }
    c.aAnchor.position = {7, 8, 9}; c.score = 0.9; c.meta = R"({"distance":1})";
    CHECK(commitCandidates(db, "detector", 2, {c}).edgesUpdated == 1);
    { auto read = db.read(); const auto edge = db.edgeByUuid(read, id);
      REQUIRE(edge); const auto& anchor = edge->aUuid == a ? edge->aAnchor : edge->bAnchor;
      CHECK(anchor.position == c.aAnchor.position);
      const auto meta = nlohmann::json::parse(edge->meta);
      CHECK(meta["review_note"] == "retain"); CHECK(meta["distance"] == 1); CHECK(meta["detector"]["score"] == 0.9);
      CHECK(edge->tags == std::vector<std::string>{"user"}); }
    CHECK(commitCandidates(db, "detector", 2, {c}).edgesSkippedExisting == 1);
    { auto w = db.write(); db.setApproval(w, id, "approved"); w.commit(); }
    c.aAnchor.position = {10, 11, 12};
    CHECK(commitCandidates(db, "detector", 3, {c}).edgesUpdated == 0);
    { auto read = db.read(); CHECK(db.edgeByUuid(read, id)->approvalState == "approved"); }
    { auto w = db.write(); db.setApproval(w, id, "pending"); w.commit(); }
    // Caller rollback preserves the old content and journal.
    { auto w = db.write(); CHECK(commitCandidates(db, w, "detector", 3, {c}).edgesUpdated == 1); }
    { auto read = db.read(); const auto edge = db.edgeByUuid(read, id);
      CHECK((edge->aUuid == a ? edge->aAnchor : edge->bAnchor).position == std::optional(std::array<double,3>{7,8,9})); }
}

namespace {
template<class T> GeometryArray array(const std::string& name, const std::string& dtype,
    const std::vector<int64_t>& shape, const std::vector<T>& values)
{
    GeometryArray a{name, dtype, shape, {}};
    a.bytes.resize(values.size() * sizeof(T));
    if (!values.empty()) std::memcpy(a.bytes.data(), values.data(), a.bytes.size());
    if constexpr (std::endian::native != std::endian::little)
        for (size_t i = 0; i < a.bytes.size(); i += sizeof(T))
            std::reverse(a.bytes.begin() + i, a.bytes.begin() + i + sizeof(T));
    return a;
}
std::string storeGeometry(ObjectStore& db, const GeometryData& geometry)
{
    auto txn = db.write();
    ObjectRecord object;
    object.kind = geometry.kind;
    object.storageRef = "/does/not/exist";
    const auto uuid = db.upsertObject(txn, object);
    db.putGeometry(txn, uuid, geometry);
    txn.commit();
    return uuid;
}
GeometryData storedPlane(int x, int width)
{
    GeometryData geometry;
    geometry.kind = "patch";
    geometry.attributes = R"({"scale":[1,1]})";
    std::vector<float> xyz;
    for (int row = 0; row < 21; ++row)
        for (int col = 0; col <= width / 2; ++col) {
            xyz.push_back(float(x + 2*col)); xyz.push_back(float(108 + 2*row)); xyz.push_back(128);
        }
    geometry.arrays.push_back(array<float>("points", "<f4", {21, width / 2 + 1, 3}, xyz));
    return geometry;
}
fs::path writeCt(const fs::path& dir, bool valid = true)
{
    fs::create_directories(dir);
    const nlohmann::json meta{{"zarr_format", 2}, {"shape", {256, 256, 512}},
        {"chunks", {256, 256, 512}}, {"dtype", "|u1"}, {"compressor", nullptr},
        {"fill_value", 0}, {"order", "C"}, {"filters", nullptr}};
    std::ofstream(dir / ".zarray") << meta.dump();
    std::ofstream out(dir / "0.0.0", std::ios::binary);
    for (int z = 0; z < 256; ++z) {
        const char value = valid ? char(128 + 100 * std::cos(z * CV_PI / 8)) : 0;
        const std::string slice(256 * 512, value);
        out.write(slice.data(), slice.size());
    }
    return dir;
}

} // namespace

TEST_CASE("stored fibers and tracks use the same patch linker without source paths")
{
    TempDir tmp("stored_lines");
    auto db = makeDb(tmp.dir);
    const auto ct = writeCt(tmp.dir / "ct");
    const auto patch = storeGeometry(db, storedPlane(100, 100));
    GeometryData fiber;
    fiber.kind = "fiber";
    std::vector<double> xyz;
    std::vector<int32_t> trackXyz;
    for (int x = 100; x <= 340; x += 2) {
        xyz.insert(xyz.end(), {double(x), 128, 128});
        trackXyz.insert(trackXyz.end(), {x, 128, 128});
    }
    fiber.arrays = {array<double>("line_points", "<f8", {121, 3}, xyz),
                   array<double>("control_points", "<f8", {2, 3}, {100,128,128,340,128,128})};
    const auto fiberId = storeGeometry(db, fiber);
    GeometryData track;
    track.kind = "track";
    track.arrays = {array<int32_t>("points", "<i4", {121, 3}, trackXyz)};
    const auto trackId = storeGeometry(db, track);
    FiberPatchParams fp;
    fp.stripMarchStep = 20;
    fp.fiberUuids = {fiberId}; fp.patchUuids = {patch};
    const auto fr = linkFibersToPatches(db, ct, fp);
    for (const auto& problem : fr.problems) INFO(problem);
    REQUIRE(fr.problems.empty());
    CHECK(fr.edgesCreated == 1);
    TrackPatchParams tp;
    tp.stripMarchStep = 20;
    tp.trackUuids = {trackId}; tp.patchUuids = {patch};
    const auto tr = linkTracksToPatches(db, ct, tp);
    for (const auto& problem : tr.problems) INFO(problem);
    REQUIRE(tr.problems.empty());
    CHECK(tr.edgesCreated == 1);
    auto read = db.read();
    CHECK_FALSE(db.edgeGeometryState(read, db.edgesOf(read, fiberId).at(0).uuid).stale);
    CHECK_FALSE(db.edgeGeometryState(read, db.edgesOf(read, trackId).at(0).uuid).stale);
    CHECK(db.objectsByKind(read, "fiber").size() == 1);
    CHECK(db.objectsByKind(read, "track").size() == 1);
}

TEST_CASE("discovery refuses to publish links if stored inputs change during computation")
{
    TempDir tmp("stored_conflict");
    auto db = makeDb(tmp.dir);
    const auto a = storeGeometry(db, storedPlane(100, 100));
    const auto b = storeGeometry(db, storedPlane(100, 100));
    PatchOverlapParams params;
    bool edited = false;
    params.progress = [&](const vc::objects::util::ProgressUpdate& progress) {
        if (progress.stage == "checking patch pairs" && !edited) {
            edited = true;
            auto txn = db.write();
            db.putGeometry(txn, b, storedPlane(1000, 100), db.currentGeometry(txn, b)->uuid);
            txn.commit();
        }
    };
    CHECK_THROWS_WITH_AS(linkPatchOverlaps(db, params), doctest::Contains("changed during discovery"), std::runtime_error);
    CHECK(edited);
    CHECK(edgeCount(db, a) == 0);
    params.progress = {};
    CHECK(linkPatchOverlaps(db, params).edgesCreated == 0);
}

TEST_CASE("candidate identity canonicalizes UUID case and endpoint order")
{
    TempDir tmp("canonical_candidates");
    auto db = makeDb(tmp.dir);
    const auto a = addObject(db, "patch"), b = addObject(db, "patch");
    auto upper = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(std::toupper(c)); });
        return value;
    };
    const auto original = cand(a, b), reversed = cand(upper(b), upper(a));
    const auto first = commitCandidates(db, "canonical", 1, {original, reversed});
    CHECK(first.edgesCreated == 1);
    CHECK(first.edgesSkippedExisting == 1);
    const auto again = commitCandidates(db, "canonical", 1, {reversed});
    CHECK(again.edgesCreated == 0);
    CHECK(again.edgesSkippedExisting == 1);
    CHECK(edgeCount(db, a) == 1);
}
