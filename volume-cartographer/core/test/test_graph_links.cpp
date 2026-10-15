#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/graph_links/GraphLinks.hpp"

#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include "vc/core/util/Uuid.hpp"
#include <string>

#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"

namespace
{
using namespace vc::graph;
using namespace vc::graph_links;
namespace fs = std::filesystem;

struct TempDir {
    fs::path dir;
    explicit TempDir(const std::string& name)
    {
        dir = fs::temp_directory_path() / ("vc_graph_links_test_" + name + "_" + vc::util::newUuid());
        fs::create_directories(dir);
    }
    ~TempDir() { fs::remove_all(dir); }
};

GraphDb makeDb(const fs::path& dir)
{
    OpenOptions o;
    o.author = "links-test";
    o.hostnameOverride = "testhost";
    return GraphDb::create(dir / "graph.vcgr", o);
}

std::string addObject(GraphDb& db, const std::string& kind)
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

size_t edgeCount(GraphDb& db, const std::string& uuid)
{
    auto r = db.read();
    return db.edgesOf(r, uuid).size();
}

std::string writePatch(const fs::path& dir, int rows, int cols,
                       const std::function<cv::Vec3f(int row, int col)>& point)
{
    cv::Mat_<cv::Vec3f> points(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            points(r, c) = point(r, c);
    QuadSurface surface(points, cv::Vec2f(1.0f, 1.0f));
    fs::create_directories(dir);
    surface.save(dir, true);
    return QuadSurface(dir).objectUuid();
}

}  // namespace

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

TEST_CASE("placeholder detectors write nothing and say so")
{
    TempDir tmp("placeholder");
    auto db = makeDb(tmp.dir);
    const std::string p = addObject(db, "patch");
    addObject(db, "fiber");
    addObject(db, "track");

    for (const auto& report : {linkFibersToFibers(db), linkTracksToFibers(db),
                               linkTracksToTracks(db), linkTracksToPatches(db)}) {
        CHECK(report.edgesCreated == 0);
        REQUIRE(report.problems.size() == 1);
        CHECK(report.problems[0].find("not implemented") != std::string::npos);
        CHECK(report.objectsConsidered > 0);
    }
    CHECK(edgeCount(db, p) == 0);
}

TEST_CASE("patch overlap: reflected overlap linked, thin strip and disjoint rejected")
{
    TempDir tmp("patch_overlap");
    const auto patchesDir = tmp.dir / "patches";
    // A: 60x40 patch on z=100 (off the tifxyz zero sentinel), xyz=(100+col,100+row,100).
    const std::string a = writePatch(patchesDir / "a_patch", 40, 60,
        [](int row, int col) { return cv::Vec3f(float(100 + col), float(100 + row), 100.0f); });
    // B: same physical sheet, transposed parameterization (reflected pose):
    // xyz=(110+row, 105+col, 100) overlaps A over x in [110,139], y in [105,139].
    const std::string b = writePatch(patchesDir / "b_patch", 30, 40,
        [](int row, int col) { return cv::Vec3f(float(110 + row), float(105 + col), 100.0f); });
    // C: far away, no overlap with anything.
    const std::string c = writePatch(patchesDir / "c_far", 20, 20,
        [](int row, int col) { return cv::Vec3f(float(1000 + col), float(row), 500.0f); });
    // D: two-row strip on A's sheet: geometrically coincident but the
    // overlap is a sliver -> minor ("thickness") spread must reject it.
    const std::string d = writePatch(patchesDir / "d_thin", 2, 50,
        [](int row, int col) { return cv::Vec3f(float(100 + col), float(120 + row), 100.0f); });
    REQUIRE(!a.empty());
    REQUIRE(!b.empty());

    auto db = makeDb(tmp.dir);
    const auto report = linkPatchOverlaps(db, patchesDir);
    for (const auto& problem : report.problems) MESSAGE(problem.c_str());
    CHECK(report.objectsConsidered == 4);
    CHECK(report.edgesCreated == 1);

    // The detector registered the patches itself.
    auto r = db.read();
    CHECK(db.objectsByKind(r, "patch").size() == 4);
    const auto aEdges = db.edgesOf(r, a);
    REQUIRE(aEdges.size() == 1);
    CHECK(aEdges[0].approvalState == "pending");
    CHECK(aEdges[0].kind == "same-surface");
    CHECK(aEdges[0].meta.find("\"patch-overlap\"") != std::string::npos);
    CHECK(aEdges[0].meta.find("\"reflected\":true") != std::string::npos);
    CHECK(db.edgesOf(r, c).empty());
    CHECK(db.edgesOf(r, d).empty());
    CHECK(db.componentOf(r, a).size() == 2);
    r.commit();

    // Re-run: nothing new, nothing removed.
    const auto again = linkPatchOverlaps(db, patchesDir);
    CHECK(again.edgesCreated == 0);
    CHECK(again.staleRemoved == 0);
    CHECK(edgeCount(db, a) == 1);
}

namespace {
// Periodic CT sheets with +Z normal. Real uint8 chunks exercise the
// production volume reader and CT estimator (flat intensities are invalid).
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

std::string writeFiber(const fs::path& dir, bool controls = true)
{
    fs::create_directories(dir);
    const std::string uuid = vc::util::newUuid();
    nlohmann::json line = nlohmann::json::array();
    for (int x = 100; x <= 340; x += 2) line.push_back({x, 128, 128});
    nlohmann::json root{{"uuid", uuid}, {"line_points", line}};
    if (controls) root["control_points"] = {{{"position", {100, 128, 128}}},
        {{"position", {150, 128, 128}}}, {{"position", {240, 128, 128}}}, {{"position", {340, 128, 128}}}};
    std::ofstream(dir / "fiber.json") << root.dump();
    return uuid;
}

std::string planePatch(const fs::path& dir, int x, int width)
{
    return writePatch(dir, 21, width / 2 + 1, [x](int r, int c) {
        return cv::Vec3f(float(x + 2 * c), float(108 + 2 * r), 128);
    });
}

std::vector<EdgeRecord> fiberEdges(GraphDb& db, const std::string& fiber)
{
    auto txn = db.read();
    return db.edgesOf(txn, fiber);
}
// Rerun/metadata tests use one 20-voxel march step on planar CT. The first
// geometry test below separately exercises the default 5-voxel marching.
DetectorReport linkTestFiberPatches(GraphDb& db, const fs::path& fibers, const fs::path& patches,
                                   const fs::path& ct, FiberPatchParams params = {},
                                   const std::atomic_bool* cancel = nullptr)
{
    params.stripMarchStep = 20;
    return linkFibersToPatches(db, fibers, patches, ct, params, cancel);
}
} // namespace

TEST_CASE("fiber-patch: anchors, multi-round growth, strip gate and isolated clusters")
{
    TempDir tmp("fiber_growth");
    auto db = makeDb(tmp.dir);
    const auto fibers = tmp.dir / "fibers", patches = tmp.dir / "patches";
    const auto ctVolume = writeCt(tmp.dir / "ct");
    const auto fiber = writeFiber(fibers);
    const auto a = planePatch(patches / "anchor", 100, 100);
    const auto b = planePatch(patches / "growth1", 175, 60);
    const auto c = planePatch(patches / "growth2", 205, 60);
    const auto isolated = planePatch(patches / "isolated", 310, 30);
    // Both wrong-sheet patches touch the entire fiber in a narrow groove
    // but otherwise lie 8 voxels away. They agree with each other, yet
    // must never bootstrap a second linked cluster.
    for (int start : {100, 110})
        writePatch(patches / ("wrong" + std::to_string(start)), 21, 71, [start](int r, int col) {
            return cv::Vec3f(float(start + 2 * col), float(108 + 2 * r), r == 10 ? 128.0f : 136.0f);
        });
    const auto report = linkFibersToPatches(db, fibers, patches, ctVolume);
    for (const auto& problem : report.problems) INFO(problem);
    CHECK(report.problems.empty());
    CHECK(report.objectsConsidered == 7);
    CHECK(report.edgesCreated == 3);
    const auto edges = fiberEdges(db, fiber);
    REQUIRE(edges.size() == 3);
    for (const auto& edge : edges) {
        const auto target = edge.aUuid == fiber ? edge.bUuid : edge.aUuid;
        const auto& anchor = edge.aUuid == fiber ? edge.aAnchor : edge.bAnchor;
        CHECK(edge.approvalState == "pending");
        CHECK(target != isolated);
        const auto meta = nlohmann::json::parse(edge.meta);
        CHECK(meta.at("detector").at("name") == "fiber-patch");
        CHECK(meta.at("detector").at("version") == 2);
        CHECK(meta.at("normal_method") == "ct-anisotropic-1-24-4-grid3-r48-v4");
        CHECK_FALSE(meta.at("pred_dt_snap").get<bool>());
        CHECK(meta.at("settings").at("strip_rows_each_side") == 1);
        CHECK(meta.at("settings").at("strip_spacing") == 20);
        const auto& evidence = meta.at("fiber_patch");
        CHECK(evidence.at("round") == (target == a ? 0 : target == b ? 1 : 2));
        CHECK(evidence.at("anchor") == (target == a));
        REQUIRE(anchor.index.has_value());
        if (target == a) {
            CHECK(*anchor.index == 1);
            CHECK(evidence.at("contact_length").get<double>() >= 100);
        }
        CHECK((target == a || target == b || target == c));
    }
    const auto repeat = linkTestFiberPatches(db, fibers, patches, ctVolume);
    CHECK(repeat.edgesCreated == 0);
    CHECK(repeat.edgesSkippedExisting == 3);
    FiberPatchParams oneRound;
    oneRound.maxRounds = 1;
    const auto limited = linkTestFiberPatches(db, fibers, patches, ctVolume, oneRound);
    CHECK(limited.staleRemoved == 1);
    CHECK(fiberEdges(db, fiber).size() == 2);
    // Removing all anchors must also remove all growth: no self-seeding.
    oneRound.anchorContact = 1000;
    const auto noAnchor = linkTestFiberPatches(db, fibers, patches, ctVolume, oneRound);
    CHECK(noAnchor.candidatesFound == 0);
    CHECK(noAnchor.staleRemoved == 2);
}

TEST_CASE("fiber-patch: reviewed and unrelated edges survive scoped reruns")
{
    TempDir tmp("fiber_review");
    auto db = makeDb(tmp.dir);
    const auto ctVolume = writeCt(tmp.dir / "ct");
    const auto fibers = tmp.dir / "fibers", patches = tmp.dir / "patches";
    const auto fiber = writeFiber(fibers, false);
    planePatch(patches / "a", 100, 100);
    planePatch(patches / "b", 190, 100);
    planePatch(patches / "c", 240, 100);
    REQUIRE(linkTestFiberPatches(db, fibers, patches, ctVolume).edgesCreated == 3);
    const auto edges = fiberEdges(db, fiber);
    REQUIRE(edges.size() == 3);
    const auto& anchor = edges[0].aUuid == fiber ? edges[0].aAnchor : edges[0].bAnchor;
    CHECK_FALSE(anchor.index.has_value());
    REQUIRE(anchor.position.has_value());
    {
        auto txn = db.write();
        db.setApproval(txn, edges[0].uuid, "approved");
        db.setApproval(txn, edges[1].uuid, "rejected");
        txn.commit();
    }
    const auto otherFiber = addObject(db, "fiber"), otherPatch = addObject(db, "patch");
    commitCandidates(db, kFiberPatchDetector, 1, {cand(otherFiber, otherPatch)}, false);
    std::atomic_bool cancel{true};
    const auto cancelled = linkTestFiberPatches(db, fibers, patches, ctVolume, {}, &cancel);
    CHECK_FALSE(cancelled.problems.empty());
    CHECK(fiberEdges(db, fiber).size() == 3);
    // Flat CT must preserve existing decisions and pending edges.
    const auto invalid = writeCt(tmp.dir / "invalid", false);
    CHECK_FALSE(linkTestFiberPatches(db, fibers, patches, invalid).problems.empty());
    CHECK(fiberEdges(db, fiber).size() == 3);
    FiberPatchParams p;
    p.anchorContact = 1000;
    p.replaceStalePending = false;
    CHECK(linkTestFiberPatches(db, fibers, patches, ctVolume, p).staleRemoved == 0);
    p.replaceStalePending = true;
    CHECK(linkTestFiberPatches(db, fibers, patches, ctVolume, p).staleRemoved == 1);
    CHECK(fiberEdges(db, fiber).size() == 2);
    CHECK(fiberEdges(db, otherFiber).size() == 1);
}

TEST_CASE("fiber-patch: validates parameters and rejects malformed geometry before linking")
{
    TempDir tmp("fiber_invalid");
    auto db = makeDb(tmp.dir);
    const auto ctVolume = writeCt(tmp.dir / "ct");
    const auto fibers = tmp.dir / "fibers", patches = tmp.dir / "patches";
    const auto fiber = writeFiber(fibers);
    planePatch(patches / "a", 100, 100);
    for (int which = 0; which < 5; ++which) {
        FiberPatchParams p;
        if (which == 0) p.stripSpacing = 0;
        if (which == 1) p.stripRowsEachSide = 0;
        if (which == 2) p.proximity = std::numeric_limits<double>::quiet_NaN();
        if (which == 3) p.stripWeights[0] = -1;
        if (which == 4) p.adjacentLow = p.sameSheet;
        CHECK_THROWS_AS(linkTestFiberPatches(db, fibers, patches, ctVolume, p), std::invalid_argument);
    }
    // Invalid control points cannot be silently dropped: anchor indices
    // must continue to refer to the original JSON array.
    std::ofstream(fibers / "fiber.json") << nlohmann::json{{"uuid", fiber},
        {"line_points", {{100, 128, 128}, {200, 128, 128}}},
        {"control_points", {{100, 128, 128}, {"bad"}, {200, 128, 128}}}}.dump();
    const auto report = linkTestFiberPatches(db, fibers, patches, ctVolume);
    CHECK_FALSE(report.problems.empty());
    CHECK(report.edgesCreated == 0);
    auto txn = db.read();
    CHECK_FALSE(db.objectByUuid(txn, fiber).has_value());
}

TEST_CASE("fiber-patch: independent coordinate scales preserve links, source anchors and source geometry")
{
    TempDir tmp("fiber_scales");
    auto db = makeDb(tmp.dir);
    const auto fibers = tmp.dir / "fibers", patches = tmp.dir / "patches";
    const auto fiber = writeFiber(fibers);
    const auto fiberFile = fibers / "fiber.json";
    nlohmann::json root;
    { std::ifstream in(fiberFile); in >> root; }
    for (const auto* key : {"line_points", "control_points"})
        for (auto& point : root[key]) {
            auto& xyz = point.is_object() ? point["position"] : point;
            for (auto& v : xyz) v = v.get<double>() * 4;
        }
    const auto sourceJson = root.dump();
    std::ofstream(fiberFile) << sourceJson;
    const auto patch = writePatch(patches / "a", 21, 51, [](int r, int c) {
        // Invalid vertices must retain their sentinel when scaling.
        if (r == 0) return cv::Vec3f(-1, -1, -1);
        return cv::Vec3f(float(100 + 2 * c) / 2, float(108 + 2 * r) / 2, 64);
    });
    const auto ctVolume = writeCt(tmp.dir / "ct");
    FiberPatchParams p;
    p.fiberScale = 0.25;
    p.patchScale = 2;
    p.workingToBaseScale = 0.125;
    p.ctToBaseScale = 0.125;
    const auto report = linkTestFiberPatches(db, fibers, patches, ctVolume, p);
    CHECK(report.problems.empty());
    REQUIRE(report.edgesCreated == 1);
    const auto edges = fiberEdges(db, fiber);
    const auto& anchor = edges[0].aUuid == fiber ? edges[0].aAnchor : edges[0].bAnchor;
    REQUIRE(anchor.index.has_value());
    CHECK(*anchor.index == 1);
    REQUIRE(anchor.position.has_value());
    CHECK(*anchor.position == std::array<double, 3>{600, 512, 512});
    const auto meta = nlohmann::json::parse(edges[0].meta);
    CHECK(meta["settings"]["fiber_scale"] == 0.25);
    CHECK(meta["settings"]["patch_scale"] == 2);
    CHECK(meta["settings"]["working_to_base_scale"] == 0.125);
    CHECK(meta["settings"]["ct_to_base_scale"] == 0.125);
    CHECK(meta["fiber_patch"]["contact_length"].get<double>() == doctest::Approx(102));
    {
        auto txn = db.read();
        const auto f = db.objectByUuid(txn, fiber);
        const auto patchObj = db.objectByUuid(txn, patch);
        REQUIRE(f->fiber.has_value());
        CHECK(f->fiber->length == doctest::Approx(960));
        REQUIRE(patchObj->patch->bbox.has_value());
        CHECK((*patchObj->patch->bbox)[0] == doctest::Approx(50));
        CHECK((*patchObj->patch->bbox)[3] == doctest::Approx(100));
    }
    nlohmann::json after;
    { std::ifstream in(fiberFile); in >> after; }
    CHECK(after.dump() == sourceJson);
    QuadSurface source(patches / "a");
    source.ensureLoaded();
    CHECK((*source.rawPointsPtr())(1, 0)[0] == doctest::Approx(50));
    CHECK((*source.rawPointsPtr())(0, 0)[0] == doctest::Approx(-1));
    // An incorrect mapping puts the fiber outside the CT; preserve pending links.
    auto outside = p;
    outside.workingToBaseScale = 1;
    CHECK_FALSE(linkTestFiberPatches(db, fibers, patches, ctVolume, outside).problems.empty());
    CHECK(fiberEdges(db, fiber).size() == 1);
    CHECK_THROWS(linkTestFiberPatches(db, fibers, patches, tmp.dir / "missing-ct", p));
    CHECK(fiberEdges(db, fiber).size() == 1);
    // All coordinate scales must be strictly positive and finite, before I/O.
    for (int which = 0; which < 5; ++which) {
        auto invalid = p;
        if (which == 0) invalid.fiberScale = 0;
        if (which == 1) invalid.patchScale = -1;
        if (which == 2) invalid.workingToBaseScale = std::numeric_limits<double>::infinity();
        if (which == 3) invalid.ctToBaseScale = 0;
        if (which == 4) invalid.ctLevel = -1;
        CHECK_THROWS_AS(linkTestFiberPatches(db, fibers, patches, ctVolume, invalid), std::invalid_argument);
    }
}

TEST_CASE("fiber-patch: Lasagna reference uses the same linker with explicit source provenance")
{
    TempDir tmp("fiber_lasagna");
    auto db = makeDb(tmp.dir);
    const auto fibers = tmp.dir / "fibers", patches = tmp.dir / "patches";
    const auto fiber = writeFiber(fibers);
    planePatch(patches / "anchor", 100, 100);
    nlohmann::json groups;
    for (const auto& channel : {"nx", "ny", "grad_mag"}) {
        const auto dir = tmp.dir / channel;
        fs::create_directories(dir);
        const nlohmann::json meta{{"zarr_format", 2}, {"shape", {4,4,4}},
            {"chunks", {4,4,4}}, {"dtype", "|u1"}, {"compressor", nullptr},
            {"fill_value", 0}, {"order", "C"}, {"filters", nullptr}};
        std::ofstream(dir / ".zarray") << meta.dump();
        const std::string bytes(64, std::string(channel) == "grad_mag" ? char(255) : char(128));
        std::ofstream(dir / "0.0.0", std::ios::binary).write(bytes.data(), bytes.size());
        groups[channel] = {{"zarr", channel}, {"scaledown", 0}, {"channels", {channel}}};
    }
    const auto manifest = tmp.dir / "normals.lasagna.json";
    std::ofstream(manifest) << nlohmann::json{{"version", 2}, {"source_to_base", 512},
        {"grad_mag_encode_scale", 255}, {"grad_mag_factor", 1}, {"groups", groups}}.dump();
    FiberPatchParams params;
    params.normalsManifest = manifest.string();
    params.workingToBaseScale = 2;
    const auto report = linkTestFiberPatches(db, fibers, patches, tmp.dir / "no-ct", params);
    for (const auto& problem : report.problems) INFO(problem);
    REQUIRE(report.problems.empty());
    REQUIRE(report.edgesCreated == 1);
    const auto edges = fiberEdges(db, fiber);
    REQUIRE(edges.size() == 1);
    const auto meta = nlohmann::json::parse(edges.front().meta);
    CHECK(meta["normal_method"] == "lasagna-normal-v1");
    CHECK(meta["normals_manifest"] == manifest.string());
    CHECK_FALSE(meta.contains("ct_volume"));
    CHECK(meta["settings"]["working_to_base_scale"] == 2);
    CHECK(meta["fiber_patch"]["anchor"] == true);
    CHECK_FALSE(meta["pred_dt_snap"].get<bool>());
}
