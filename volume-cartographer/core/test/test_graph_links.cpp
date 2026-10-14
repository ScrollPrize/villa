#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/graph_links/GraphLinks.hpp"

#include <filesystem>
#include <string>

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

    for (const auto& report : {linkPatchOverlaps(db), linkFibersToPatches(db), linkFibersToFibers(db),
                               linkTracksToFibers(db), linkTracksToTracks(db), linkTracksToPatches(db)}) {
        CHECK(report.edgesCreated == 0);
        REQUIRE(report.problems.size() == 1);
        CHECK(report.problems[0].find("not implemented") != std::string::npos);
        CHECK(report.objectsConsidered > 0);
    }
    CHECK(edgeCount(db, p) == 0);
}
