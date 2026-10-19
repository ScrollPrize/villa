#include "vc/core/objects/import/TrackImport.hpp"
#include "vc/core/objects/import/PatchImport.hpp"
#include "vc/core/objects/import/FiberImport.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/objects/ObjectStore.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace
{
using namespace vc::objects;
namespace fs = std::filesystem;

struct TempDir {
    fs::path dir;
    explicit TempDir(const std::string& name)
    {
        dir = fs::temp_directory_path() / ("vc_graph_test_" + name);
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    ~TempDir() { fs::remove_all(dir); }
};

OpenOptions testOptions()
{
    OpenOptions o;
    o.author = "tester";
    o.hostnameOverride = "testhost";
    o.readerPoolSize = 4;
    return o;
}

ObjectRecord makeObject(const std::string& kind, const std::string& name = "")
{
    ObjectRecord o;
    o.kind = kind;
    o.name = name;
    return o;
}

std::string addObject(ObjectStore& db, const std::string& kind, const std::string& name = "")
{
    auto w = db.write();
    std::string uuid = db.upsertObject(w, makeObject(kind, name));
    w.commit();
    return uuid;
}

EdgeRecord makeEdge(const std::string& a, const std::string& b, const std::string& kind = "connected")
{
    EdgeRecord e;
    e.aUuid = a;
    e.bUuid = b;
    e.kind = kind;
    return e;
}

std::string addEdge(ObjectStore& db, const EdgeRecord& e)
{
    auto w = db.write();
    std::string uuid = db.addEdge(w, e);
    w.commit();
    return uuid;
}

ObjectRecord getObject(ObjectStore& db, const std::string& uuid)
{
    auto r = db.read();
    auto obj = db.objectByUuid(r, uuid);
    REQUIRE(obj.has_value());
    return *obj;
}

}  // namespace

TEST_CASE("create, open, and validation")
{
    TempDir tmp("create");
    const auto path = tmp.dir / "graph.vcobj";
    {
        auto db = ObjectStore::create(path, testOptions());
        CHECK(db.hostname() == "testhost");
        CHECK(db.author() == "tester");
        addObject(db, "patch", "p1");
    }
    {
        auto db = ObjectStore::open(path, testOptions());
        auto r = db.read();
        CHECK(db.objectsByKind(r, "patch").size() == 1);
    }
    // Re-create over an existing DB is refused.
    CHECK_THROWS(ObjectStore::create(path, testOptions()));
    // Read-write open without an author is refused.
    OpenOptions noAuthor = testOptions();
    noAuthor.author = "";
    CHECK_THROWS(ObjectStore::open(path, noAuthor));
    // A non-graph file is refused.
    const auto bogus = tmp.dir / "bogus.db";
    { std::ofstream(bogus) << "not a database"; }
    CHECK_THROWS(ObjectStore::open(bogus, testOptions()));
    // WAL side files exist while open; checkpoint leaves a copyable main file.
    {
        auto db = ObjectStore::open(path, testOptions());
        addObject(db, "patch", "p2");
        db.checkpoint();
        const auto copy = tmp.dir / "copy.vcobj";
        fs::copy_file(path, copy);
        OpenOptions ro = testOptions();
        ro.readOnly = true;
        auto rodb = ObjectStore::open(copy, ro);
        auto r = rodb.read();
        CHECK(rodb.objectsByKind(r, "patch").size() == 2);
        CHECK_THROWS(rodb.write());
    }
}

TEST_CASE("changed track measurements invalidate review, journal changes and roll back")
{
    TempDir tmp("track_length");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    auto object = makeObject("track");
    object.track = TrackDetail{"dataset", "1"};
    std::string uuid;
    { auto write = db.write(); uuid = db.upsertObject(write, object);
      db.setApproval(write, uuid, "approved"); write.commit(); }
    const auto before = getObject(db, uuid);
    { auto write = db.write(); db.setTrackLength(write, uuid, 17); write.commit(); }
    const auto after = getObject(db, uuid);
    CHECK(after.track->length == 17);
    CHECK(after.generation == before.generation + 1);
    CHECK(after.changedBy == "tester");
    CHECK(after.changedHost == "testhost");
    CHECK_FALSE(after.approvedBy.has_value());
    CHECK_FALSE(after.approvedAt.has_value());
    CHECK_FALSE(after.approvedHost.has_value());
    CHECK_FALSE(after.approvedGeneration.has_value());
    CHECK(after.approvalState == "pending");
    int64_t seq;
    { auto read = db.read(); const auto journal = db.journalSince(read, 0);
      CHECK(journal.back().changeKind == "track_length"); seq = journal.back().seq; }
    { auto write = db.write(); db.setTrackLength(write, uuid, 17); write.commit(); }
    { auto write = db.write(); db.setTrackLength(write, uuid, 19); } // rollback
    { auto read = db.read(); CHECK(db.journalSince(read, seq).empty()); }
    CHECK(getObject(db, uuid).track->length == 17);
    { auto write = db.write();
      CHECK_THROWS(db.setTrackLength(write, uuid, -1));
      CHECK_THROWS(db.setTrackLength(write, uuid, std::numeric_limits<double>::infinity()));
      CHECK_THROWS(db.setTrackLength(write, uuid, std::numeric_limits<double>::quiet_NaN())); }
}

TEST_CASE("object CRUD stamps, generation, journal, kinds as data")
{
    TempDir tmp("crud");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());

    ObjectRecord fiber = makeObject("fiber", "f1");
    fiber.tags = {"reviewed", "h"};
    FiberDetail fd;
    fd.family = "H";
    fd.controlPointCount = 3;
    fiber.fiber = fd;
    std::string uuid;
    {
        auto w = db.write();
        uuid = db.upsertObject(w, fiber);
        w.commit();
    }
    ObjectRecord got = getObject(db, uuid);
    CHECK(got.kind == "fiber");
    CHECK(got.generation == 1);
    CHECK(got.createdBy == "tester");
    CHECK(got.createdHost == "testhost");
    CHECK(got.changedHost == "testhost");
    CHECK(got.approvalState == "pending");
    CHECK(got.tags.size() == 2);
    REQUIRE(got.fiber.has_value());
    CHECK(got.fiber->family == "H");
    CHECK(got.fiber->controlPointCount == 3);

    // Update bumps generation exactly once and restamps changed_* only.
    got.name = "renamed";
    {
        auto w = db.write();
        db.upsertObject(w, got);
        w.commit();
    }
    ObjectRecord updated = getObject(db, uuid);
    CHECK(updated.generation == 2);
    CHECK(updated.name == "renamed");
    CHECK(updated.createdAt == got.createdAt);

    // Rolled-back transaction changes nothing.
    {
        auto w = db.write();
        ObjectRecord edit = updated;
        edit.name = "never";
        db.upsertObject(w, edit);
    }  // no commit = rollback
    CHECK(getObject(db, uuid).name == "renamed");
    CHECK(getObject(db, uuid).generation == 2);

    // Kinds are data: a made-up kind with no detail table round-trips
    // through the same generic calls.
    const std::string odd = addObject(db, "widget", "w1");
    CHECK(getObject(db, odd).kind == "widget");

    // Empty kind and kind changes are refused.
    {
        auto w = db.write();
        ObjectRecord bad = makeObject("", "x");
        CHECK_THROWS(db.upsertObject(w, bad));
    }
    {
        auto w = db.write();
        ObjectRecord edit = updated;
        edit.kind = "patch";
        CHECK_THROWS(db.upsertObject(w, edit));
    }

    // Journal records the full generation stamps.
    {
        auto r = db.read();
        auto entries = db.journalSince(r, 0);
        REQUIRE(entries.size() >= 3);
        CHECK(entries[0].recordUuid == uuid);
        CHECK(entries[0].author == "tester");
        CHECK(entries[0].hostname == "testhost");
        CHECK(entries[0].generation == 1);
        CHECK(entries[1].generation == 2);
    }
}

TEST_CASE("approval lifecycle and open-ended metadata")
{
    TempDir tmp("approval");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    const std::string uuid = addObject(db, "fiber", "f");

    {
        auto w = db.write();
        db.setApproval(w, uuid, "approved");
        w.commit();
    }
    ObjectRecord got = getObject(db, uuid);
    CHECK(got.approvalState == "approved");
    CHECK(got.approvedBy == "tester");
    CHECK(got.approvedHost == "testhost");
    CHECK(got.approvedGeneration == 1);
    CHECK(got.generation == 1);  // approving does not bump generation

    // A content change (here setMeta) bumps generation and resets approval.
    {
        auto w = db.write();
        db.setMeta(w, uuid, "review.note", "\"looks good\"");
        db.setMeta(w, uuid, "score", "7");
        w.commit();
    }
    got = getObject(db, uuid);
    CHECK(got.approvalState == "pending");
    CHECK(got.approvedBy == std::nullopt);
    CHECK(got.generation == 3);  // one bump per setMeta call
    CHECK(got.meta.find("review.note") != std::string::npos);
    CHECK(got.meta.find("looks good") != std::string::npos);

    // Removing one key leaves the others alone.
    {
        auto w = db.write();
        db.removeMeta(w, uuid, "score");
        w.commit();
    }
    got = getObject(db, uuid);
    CHECK(got.meta.find("score") == std::string::npos);
    CHECK(got.meta.find("review.note") != std::string::npos);

    // Invalid meta JSON value and invalid approval state are refused.
    {
        auto w = db.write();
        CHECK_THROWS(db.setMeta(w, uuid, "k", "not json"));
    }
    {
        auto w = db.write();
        CHECK_THROWS(db.setApproval(w, uuid, "maybe"));
    }
    // The failed setMeta's transaction rolled back: no stray generation bump
    // beyond the removeMeta one (generation was 4 after removeMeta).
    CHECK(getObject(db, uuid).generation == 4);

    // Approval filtering on listing calls.
    {
        auto w = db.write();
        db.setApproval(w, uuid, "rejected");
        w.commit();
    }
    auto r = db.read();
    CHECK(db.objectsByKind(r, "fiber", "rejected").size() == 1);
    CHECK(db.objectsByKind(r, "fiber", "approved").empty());
}

TEST_CASE("edges: canonicalisation, anchors, uniqueness")
{
    TempDir tmp("edges");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    const std::string a = addObject(db, "fiber", "a");
    const std::string b = addObject(db, "patch", "b");
    const std::string c = addObject(db, "fiber", "c");

    // One generic addEdge for every pairing, kind carried as metadata.
    const std::string e1 = addEdge(db, makeEdge(a, b, "same-surface"));
    CHECK(addEdge(db, makeEdge(b, c, "same-surface")) != e1);

    // Duplicate unanchored edge of the same kind is refused in either direction.
    {
        auto w = db.write();
        CHECK_THROWS(db.addEdge(w, makeEdge(b, a, "same-surface")));
    }
    // The same pair with a different kind is a different edge.
    addEdge(db, makeEdge(a, b, "other-relation"));
    // Empty kind refused; self-edge refused; missing endpoint refused.
    {
        auto w = db.write();
        CHECK_THROWS(db.addEdge(w, makeEdge(a, b, "")));
        CHECK_THROWS(db.addEdge(w, makeEdge(a, a)));
        CHECK_THROWS(db.addEdge(w, makeEdge(a, "00000000-0000-4000-8000-000000000000")));
    }

    // Anchors swap along with endpoints under canonicalisation.
    EdgeRecord anchored = makeEdge(c, a, "fiber_branch");
    anchored.aAnchor.index = 5;
    anchored.aAnchor.position = std::array<double, 3>{1, 2, 3};
    anchored.bAnchor.index = 2;
    const std::string eA = addEdge(db, anchored);
    {
        auto r = db.read();
        auto stored = db.edgeByUuid(r, eA);
        REQUIRE(stored.has_value());
        const bool cIsA = stored->aUuid == c;
        const Anchor& cAnchor = cIsA ? stored->aAnchor : stored->bAnchor;
        CHECK(cAnchor.index == 5);
        REQUIRE(cAnchor.position.has_value());
        CHECK((*cAnchor.position)[0] == doctest::Approx(1.0));
    }
    // Same anchored pair again refuses; a different anchor point is fine.
    {
        auto w = db.write();
        CHECK_THROWS(db.addEdge(w, anchored));
    }
    anchored.aAnchor.index = 9;
    addEdge(db, anchored);

    // Deleting an object with edges is refused; after removing its edges it works.
    {
        auto w = db.write();
        CHECK_THROWS(db.deleteObject(w, b));
    }
    {
        auto w = db.write();
        auto edges = db.edgesOf(w, b);
        REQUIRE(edges.size() == 3);
        for (const auto& e : edges) db.removeEdge(w, e.uuid);
        db.deleteObject(w, b);
        w.commit();
    }
    auto r = db.read();
    CHECK(!db.objectByUuid(r, b).has_value());
}

TEST_CASE("traversal crosses object kinds and filters by edge kind")
{
    TempDir tmp("traversal");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    const std::string f1 = addObject(db, "fiber");
    const std::string p1 = addObject(db, "patch");
    const std::string f2 = addObject(db, "fiber");
    const std::string lonely = addObject(db, "track");
    addEdge(db, makeEdge(f1, p1, "same-surface"));
    addEdge(db, makeEdge(p1, f2, "same-surface"));
    addEdge(db, makeEdge(f1, f2, "weak-hint"));  // cycle + differently-kinded edge

    auto r = db.read();
    const auto component = db.componentOf(r, f1);
    CHECK(component.size() == 3);
    CHECK(std::set<std::string>(component.begin(), component.end()).count(p1) == 1);
    CHECK(db.componentOf(r, lonely).size() == 1);
    // Filtering to one edge kind isolates f1<->f2 from the patch path only
    // if we exclude same-surface; with only weak-hint, p1 drops out.
    const auto weak = db.componentOf(r, f1, {"weak-hint"});
    CHECK(weak.size() == 2);
    CHECK(std::set<std::string>(weak.begin(), weak.end()).count(p1) == 0);
    CHECK(db.reachableWithin(r, f1, 0).size() == 1);
    CHECK(db.reachableWithin(r, f1, 1).size() == 3);
    CHECK(db.neighbours(r, p1).size() == 2);
    CHECK(db.neighbours(r, p1, std::nullopt, "fiber").size() == 2);
    CHECK(db.neighbours(r, p1, std::nullopt, "track").empty());
}

TEST_CASE("export and re-import preserve the graph")
{
    TempDir tmp("export");
    auto db = ObjectStore::create(tmp.dir / "a.vcobj", testOptions());
    const std::string a = addObject(db, "fiber", "a");
    const std::string b = addObject(db, "patch", "b");
    const std::string e = addEdge(db, makeEdge(a, b, "same-surface"));
    {
        auto w = db.write();
        db.setApproval(w, e, "approved");
        w.commit();
    }
    std::string json;
    {
        auto r = db.read();
        json = db.exportJson(r);
    }
    auto db2 = ObjectStore::create(tmp.dir / "b.vcobj", testOptions());
    {
        auto w = db2.write();
        db2.importJson(w, json);
        w.commit();
    }
    auto r = db2.read();
    REQUIRE(db2.objectByUuid(r, a).has_value());
    REQUIRE(db2.objectByUuid(r, b).has_value());
    auto edge = db2.edgeByUuid(r, e);
    REQUIRE(edge.has_value());
    CHECK(edge->kind == "same-surface");
    CHECK(edge->approvalState == "approved");
    CHECK(db2.componentOf(r, a).size() == 2);
}

TEST_CASE("snapshot conflicts roll back only the import when caught inside a transaction")
{
    TempDir tmp("import_conflict");
    auto source = ObjectStore::create(tmp.dir / "source.vcobj", testOptions());
    const auto a = addObject(source, "fiber", "first");
    const auto b = addObject(source, "patch", "second");
    std::string snapshot;
    { auto read = source.read(); snapshot = source.exportJson(read); }
    auto target = ObjectStore::create(tmp.dir / "target.vcobj", testOptions());
    auto conflicting = makeObject("patch", "local edit");
    conflicting.uuid = b;
    auto write = target.write();
    target.upsertObject(write, conflicting);
    const auto local = target.upsertObject(write, makeObject("fiber", "unrelated"));
    const auto journalSize = target.journalSince(write, 0).size();
    CHECK_THROWS(target.importJson(write, snapshot));
    CHECK_FALSE(target.objectByUuid(write, a).has_value());
    CHECK(target.objectByUuid(write, b)->name == "local edit");
    CHECK(target.objectByUuid(write, local).has_value());
    CHECK(target.journalSince(write, 0).size() == journalSize);
    write.commit();
    CHECK(getObject(target, b).name == "local edit");
}

TEST_CASE("family-only refresh invalidates approval and unchanged measurements preserve it")
{
    TempDir tmp("family_refresh");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    auto object = makeObject("track");
    object.track = TrackDetail{"dataset", "1"};
    object.track->length = 17;
    std::string uuid;
    { auto w = db.write(); uuid = db.upsertObject(w, object); db.setApproval(w, uuid, "approved"); w.commit(); }
    { auto w = db.write(); db.setTrackLength(w, uuid, 17, "horizontal"); w.commit(); }
    CHECK(getObject(db, uuid).generation == 2);
    CHECK(getObject(db, uuid).approvalState == "pending");
    { auto w = db.write(); db.setApproval(w, uuid, "approved"); w.commit(); }
    size_t journalSize;
    { auto r = db.read(); journalSize = db.journalSince(r, 0).size(); }
    { auto w = db.write(); db.setTrackLength(w, uuid, 17, "horizontal"); w.commit(); }
    CHECK(getObject(db, uuid).generation == 2);
    CHECK(getObject(db, uuid).approvalState == "approved");
    { auto r = db.read(); CHECK(db.journalSince(r, 0).size() == journalSize); }
}

namespace
{
void writeText(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

std::string fiberJson(const std::string& uuid, const std::string& branchEntries)
{
    return R"({"type":"vc3d_fiber","version":4,"uuid":")" + uuid + R"(",
      "optimization_mode":"lasagna","width":10,"tags":["t1"],
      "control_points":[{"position":[0,0,0]},{"position":[1,0,0]},{"position":[2,0,0]}],
      "line_points":[[0,0,0],[1,0,0],[2,0,0]],
      "branches":[)" + branchEntries + "]}\n";
}

std::string branchJson(int local, const std::string& file, int target, bool pending)
{
    return std::string("{\"control_point_index\":") + std::to_string(local) +
           ",\"branch_file\":\"" + file + "\",\"branch_control_point_index\":" + std::to_string(target) +
           ",\"control_point_position\":[0,0,0],\"branch_control_point_position\":[1,0,0]" +
           ",\"control_point_direction\":[1,0,0],\"branch_control_point_direction\":[1,0,0]" +
           (pending ? ",\"pending\":true" : "") + "}";
}
}  // namespace

TEST_CASE("fiber importer: reciprocal, one-sided, unresolved, pending mapping")
{
    TempDir tmp("import_fibers");
    const auto dir = tmp.dir / "fibers";
    const std::string u1 = "11111111-1111-4111-8111-111111111111";
    const std::string u2 = "22222222-2222-4222-8222-222222222222";
    const std::string u3 = "33333333-3333-4333-8333-333333333333";
    // f1 <-> f2 reciprocal (not pending); f1 -> f3 one-sided (pending);
    // f2 -> missing.json unresolved.
    writeText(dir / "f1.json", fiberJson(u1, branchJson(1, "f2.json", 2, false) + "," + branchJson(0, "f3.json", 1, true)));
    writeText(dir / "f2.json", fiberJson(u2, branchJson(2, "f1.json", 1, false) + "," + branchJson(0, "missing.json", 0, false)));
    writeText(dir / "f3.json", fiberJson(u3, ""));
    writeText(dir / "notes.json", R"({"unrelated":true})");

    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    ImportReport report;
    std::set<std::string> stages;
    {
        auto w = db.write();
        report = importFibers(db, w, dir, [&](const vc::util::ProgressUpdate& update) {
            CHECK((update.total == 0 || update.completed <= update.total));
            stages.insert(std::string(update.stage));
        });
        w.commit();
    }
    CHECK(stages.contains("reading fibers"));
    CHECK(stages.contains("registering fibers"));
    CHECK(stages.contains("importing fiber branches"));
    CHECK(report.objectsImported == 3);
    CHECK(report.edgesImported == 2);
    REQUIRE(report.problems.size() == 2);  // one one-sided, one unresolved

    auto r = db.read();
    auto f1Edges = db.edgesOf(r, u1);
    REQUIRE(f1Edges.size() == 2);
    for (const auto& e : f1Edges) {
        CHECK(e.kind == "fiber_branch");
        const bool toF3 = (e.aUuid == u3 || e.bUuid == u3);
        if (toF3) {
            CHECK(e.approvalState == "pending");
        } else {
            // Legacy non-pending branch: approved, but with no fabricated stamps.
            CHECK(e.approvalState == "approved");
            CHECK(e.approvedBy == std::nullopt);
            CHECK(e.approvedGeneration == std::nullopt);
        }
    }
    CHECK(db.componentOf(r, u1).size() == 3);
    r.commit();

    // Re-import is a no-op.
    {
        auto w = db.write();
        ImportReport again = importFibers(db, w, dir);
        w.commit();
        CHECK(again.objectsImported == 0);
        CHECK(again.edgesImported == 0);
    }
}

TEST_CASE("patch importer registers segments and ignores overlapping.json")
{
    TempDir tmp("import_patches");
    const auto segs = tmp.dir / "segments";
    writeText(segs / "seg_a" / "meta.json", R"({"uuid":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa","grid_width":10})");
    writeText(segs / "seg_b" / "meta.json", R"({"uuid":"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"})");
    writeText(segs / "seg_a" / "overlapping.json", R"(["seg_b"])");
    writeText(segs / "seg_b" / "overlapping.json", R"(["seg_a"])");
    writeText(segs / "no_uuid" / "meta.json", R"({"name":"x"})");

    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    ImportReport report;
    {
        auto w = db.write();
        report = importPatches(db, w, segs);
        w.commit();
    }
    CHECK(report.objectsImported == 2);
    CHECK(report.problems.size() == 1);  // the uuid-less segment
    auto r = db.read();
    const auto patches = db.objectsByKind(r, "patch");
    REQUIRE(patches.size() == 2);
    // The overlapping.json files had no effect: no edges exist.
    for (const auto& p : patches)
        CHECK(db.edgesOf(r, p.uuid).empty());
}

TEST_CASE("external geometry resolves by UUID after rename and path reuse")
{
    TempDir tmp("resolve_uuid");
    const auto patches = tmp.dir / "patches";
    const auto fibers = tmp.dir / "fibers";
    const std::string patchUuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    const std::string fiberUuid = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
    writeText(patches / "original" / "meta.json", "{\"uuid\":\"" + patchUuid + "\"}");
    writeText(fibers / "original.json", "{\"uuid\":\"" + fiberUuid + "\",\"line_points\":[]}");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    {
        auto w = db.write();
        REQUIRE(importPatches(db, w, patches).objectsImported == 1);
        REQUIRE(importFibers(db, w, fibers).objectsImported == 1);
        w.commit();
    }
    auto r = db.read();
    CHECK(db.resolveStoragePath(r, patchUuid) == patches / "original");
    CHECK(db.resolveStoragePath(r, fiberUuid) == fibers / "original.json");
    fs::rename(patches / "original", patches / "renamed");
    fs::rename(fibers / "original.json", fibers / "renamed.json");
    // Reusing the old name must not silently substitute a different object.
    writeText(patches / "original" / "meta.json", "{\"uuid\":\"" + fiberUuid + "\"}");
    writeText(fibers / "original.json", "{\"uuid\":\"" + patchUuid + "\",\"line_points\":[]}");
    writeText(fibers / "broken.json", "invalid JSON");
    CHECK(db.resolveStoragePath(r, patchUuid) == patches / "renamed");
    CHECK(db.resolveStoragePath(r, fiberUuid) == fibers / "renamed.json");
    CHECK(db.objectByUuid(r, patchUuid)->storageRef == "patches/original");
    CHECK(db.objectByUuid(r, fiberUuid)->generation == 1);
    // A fallback search refuses ambiguous copies of the same UUID.
    fs::copy_file(fibers / "renamed.json", fibers / "duplicate.json");
    CHECK_THROWS(db.resolveStoragePath(r, fiberUuid));
    fs::remove(fibers / "duplicate.json");
    fs::remove(fibers / "renamed.json");
    CHECK_FALSE(db.resolveStoragePath(r, fiberUuid).has_value());
}

TEST_CASE("external geometry supports a separate dataset root on read-only graphs")
{
    TempDir tmp("resolve_root");
    const auto source = tmp.dir / "source";
    const std::string uuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    writeText(source / "patches" / "old" / "meta.json", "{\"uuid\":\"" + uuid + "\"}");
    {
        auto db = ObjectStore::create(source / "graph.vcobj", testOptions());
        auto w = db.write();
        importPatches(db, w, source / "patches");
        w.commit();
        db.checkpoint();
    }
    fs::rename(source / "graph.vcobj", tmp.dir / "graph.vcobj");
    fs::rename(source, tmp.dir / "dataset");
    const auto root = tmp.dir / "dataset";
    fs::rename(root / "patches" / "old", root / "patches" / "new");
    // A duplicate inside the strip cache must not participate in discovery.
    writeText(root / ".strip-cache" / "copy" / "meta.json", "{\"uuid\":\"" + uuid + "\"}");
    OpenOptions options;
    options.readOnly = true;
    auto db = ObjectStore::open(tmp.dir / "graph.vcobj", options);
    auto r = db.read();
    CHECK(db.resolveStoragePath(r, uuid, root) == root / "patches" / "new");
    CHECK(db.resolveStoragePath(r, uuid, root / "patches" / "new") == root / "patches" / "new");
    CHECK_FALSE(db.resolveStoragePath(r, uuid, tmp.dir / "missing").has_value());
    CHECK_FALSE(db.resolveStoragePath(r, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", root).has_value());
    CHECK(db.objectByUuid(r, uuid)->generation == 1);
}

TEST_CASE("sweep reports missing external data")
{
    TempDir tmp("sweep");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    ObjectRecord o = makeObject("patch", "gone");
    o.storageKind = "file";
    o.storageRef = "nope.tifxyz";
    o.dataGeneration = 1;
    {
        auto w = db.write();
        db.upsertObject(w, o);
        w.commit();
    }
    auto r = db.read();
    const auto problems = db.sweepExternalData(r);
    REQUIRE(problems.size() == 1);
    CHECK(problems[0].find("storage file missing") != std::string::npos);
}

TEST_CASE("reader snapshot is stable across a concurrent commit")
{
    TempDir tmp("snapshot");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    const std::string first = addObject(db, "patch", "first");
    auto r = db.read();
    CHECK(db.objectsByKind(r, "patch").size() == 1);
    addObject(db, "patch", "second");  // writer commits while r is open
    CHECK(db.objectsByKind(r, "patch").size() == 1);  // snapshot unchanged
    r.commit();
    auto r2 = db.read();
    CHECK(db.objectsByKind(r2, "patch").size() == 2);
    (void)first;
}

TEST_CASE("concurrent readers while a writer commits")
{
    TempDir tmp("concurrent");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    const std::string anchorObj = addObject(db, "patch", "anchor");
    std::atomic<int> errors{0};
    std::atomic<bool> done{false};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            while (!done.load()) {
                try {
                    auto r = db.read();
                    const auto comp = db.componentOf(r, anchorObj);
                    if (comp.empty()) ++errors;
                    // A reader must never observe a half-written chain:
                    // every component member must be readable in the same snapshot.
                    for (const auto& u : comp)
                        if (!db.objectByUuid(r, u)) ++errors;
                    r.commit();
                } catch (...) {
                    ++errors;
                }
            }
        });
    }
    std::string prev = anchorObj;
    for (int i = 0; i < 40; ++i) {
        auto w = db.write();
        ObjectRecord o = makeObject("fiber", "chain" + std::to_string(i));
        const std::string uuid = db.upsertObject(w, o);
        EdgeRecord e = makeEdge(prev, uuid, "chain");
        db.addEdge(w, e);
        w.commit();
        prev = uuid;
    }
    done.store(true);
    for (auto& t : readers) t.join();
    CHECK(errors.load() == 0);
    auto r = db.read();
    CHECK(db.componentOf(r, anchorObj).size() == 41);
}

TEST_CASE("traversal cancellation via atomic flag")
{
    TempDir tmp("cancel");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    const std::string a = addObject(db, "fiber");
    const std::string b = addObject(db, "fiber");
    addEdge(db, makeEdge(a, b));

    auto r = db.read();
    std::atomic_bool cancel{true};
    CHECK_THROWS_AS(db.componentOf(r, a, {}, &cancel), std::runtime_error);
    // The connection is usable again afterwards (progress handler cleared).
    cancel.store(false);
    CHECK(db.componentOf(r, a, {}, &cancel).size() == 2);
    cancel.store(true);
    CHECK_THROWS_AS(db.reachableWithin(r, a, 5, {}, &cancel), std::runtime_error);
}

TEST_CASE("track list registration is identity-stable and idempotent")
{
    TempDir tmp("tracks");
    const auto listPath = tmp.dir / "tracks.json";
    writeText(listPath,
              R"([{"dataset_ref":"spiral://scroll1/tracks","external_key":"t-001","family":"H","point_count":42,"length":12.5},
                  {"dataset_ref":"spiral://scroll1/tracks","external_key":"t-002","family":"V"},
                  {"external_key":"no-dataset"},
                  42])");
    auto db = ObjectStore::create(tmp.dir / "graph.vcobj", testOptions());
    ImportReport report;
    {
        auto w = db.write();
        report = importTrackList(db, w, listPath);
        w.commit();
    }
    CHECK(report.objectsImported == 2);
    CHECK(report.problems.size() == 2);

    auto r = db.read();
    const auto tracks = db.objectsByKind(r, "track");
    REQUIRE(tracks.size() == 2);
    const std::string firstUuid = tracks[0].uuid;
    CHECK(tracks[0].track->externalKey == "t-001");
    CHECK(tracks[0].track->pointCount == 42);
    CHECK(tracks[0].generation == 1);
    r.commit();

    // Re-registering the same list keeps UUIDs and generations.
    {
        auto w = db.write();
        ImportReport again = importTrackList(db, w, listPath);
        w.commit();
        CHECK(again.objectsImported == 0);
    }
    auto r2 = db.read();
    const auto tracks2 = db.objectsByKind(r2, "track");
    REQUIRE(tracks2.size() == 2);
    CHECK(tracks2[0].uuid == firstUuid);
    CHECK(tracks2[0].generation == 1);
}
