#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/graph_links/GraphLinks.hpp"

#include <filesystem>
#include <functional>
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
        dir = fs::temp_directory_path() / ("vc_graph_links_test_" + name);
        fs::remove_all(dir);
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

    for (const auto& report : {linkFibersToPatches(db), linkFibersToFibers(db), linkTracksToFibers(db),
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
