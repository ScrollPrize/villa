#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc_object_store/objects/ObjectStore.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
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
        dir = fs::temp_directory_path() / ("vc_object_test_" + name);
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

void writeText(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

}  // namespace

TEST_CASE("create, open, and validation")
{
    TempDir tmp("create");
    const auto path = tmp.dir / "objects.vcobj";
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

TEST_CASE("edges: canonicalisation, anchors, uniqueness")
{
    TempDir tmp("edges");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
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
    anchored.aAnchor.points = {{"control_points", {5}}};
    anchored.aAnchor.position = std::array<double, 3>{1, 2, 3};
    anchored.bAnchor.points = {{"control_points", {2}}};
    const std::string eA = addEdge(db, anchored);
    {
        auto r = db.read();
        auto stored = db.edgeByUuid(r, eA);
        REQUIRE(stored.has_value());
        const bool cIsA = stored->aUuid == c;
        const Anchor& cAnchor = cIsA ? stored->aAnchor : stored->bAnchor;
        CHECK(cAnchor.points.at(0).index == std::vector<int64_t>{5});
        REQUIRE(cAnchor.position.has_value());
        CHECK((*cAnchor.position)[0] == doctest::Approx(1.0));
    }
    // Same anchored pair again refuses; a different anchor point is fine.
    {
        auto w = db.write();
        CHECK_THROWS(db.addEdge(w, anchored));
    }
    anchored.aAnchor.points = {{"control_points", {9}}};
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

TEST_CASE("source geometry resolves by UUID after rename and path reuse")
{
    TempDir tmp("resolve_uuid");
    const auto patches = tmp.dir / "patches";
    const auto fibers = tmp.dir / "fibers";
    const std::string patchUuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    const std::string fiberUuid = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
    writeText(patches / "original" / "meta.json", "{\"uuid\":\"" + patchUuid + "\"}");
    writeText(fibers / "original.json", "{\"uuid\":\"" + fiberUuid + "\",\"line_points\":[]}");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    {
        auto w = db.write();
        auto patch = makeObject("patch"); patch.uuid = patchUuid;
        patch.storageKind = "file"; patch.storageRef = "patches/original";
        db.upsertObject(w, patch);
        auto fiber = makeObject("fiber"); fiber.uuid = fiberUuid;
        fiber.storageKind = "file"; fiber.storageRef = "fibers/original.json";
        db.upsertObject(w, fiber);
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

TEST_CASE("source geometry supports a separate dataset root on read-only stores")
{
    TempDir tmp("resolve_root");
    const auto source = tmp.dir / "source";
    const std::string uuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    writeText(source / "patches" / "old" / "meta.json", "{\"uuid\":\"" + uuid + "\"}");
    {
        auto db = ObjectStore::create(source / "objects.vcobj", testOptions());
        auto w = db.write();
        auto patch = makeObject("patch"); patch.uuid = uuid;
        patch.storageKind = "file"; patch.storageRef = "patches/old";
        db.upsertObject(w, patch);
        w.commit();
        db.checkpoint();
    }
    fs::rename(source / "objects.vcobj", tmp.dir / "objects.vcobj");
    fs::rename(source, tmp.dir / "dataset");
    const auto root = tmp.dir / "dataset";
    fs::rename(root / "patches" / "old", root / "patches" / "new");
    // A duplicate inside the strip cache must not participate in discovery.
    writeText(root / ".strip-cache" / "copy" / "meta.json", "{\"uuid\":\"" + uuid + "\"}");
    OpenOptions options;
    options.readOnly = true;
    auto db = ObjectStore::open(tmp.dir / "objects.vcobj", options);
    auto r = db.read();
    CHECK(db.resolveStoragePath(r, uuid, root) == root / "patches" / "new");
    CHECK(db.resolveStoragePath(r, uuid, root / "patches" / "new") == root / "patches" / "new");
    CHECK_FALSE(db.resolveStoragePath(r, uuid, tmp.dir / "missing").has_value());
    CHECK_FALSE(db.resolveStoragePath(r, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", root).has_value());
    CHECK(db.objectByUuid(r, uuid)->generation == 1);
}

TEST_CASE("concurrent readers while a writer commits")
{
    TempDir tmp("concurrent");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
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
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
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


TEST_CASE("all edge mutation paths reject nonfinite anchors without changing records")
{
    TempDir tmp("finite_anchors");
    auto db = ObjectStore::create(tmp.dir / "objects.vcobj", testOptions());
    const auto a = addObject(db, "patch"), b = addObject(db, "patch");
    const auto id = addEdge(db, makeEdge(a, b));
    auto txn = db.write();
    const auto original = *db.edgeByUuid(txn, id);
    const auto journalSize = db.journalSince(txn, 0).size();
    for (double invalid : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto edge = original;
        edge.aAnchor.position = std::array<double, 3>{invalid, 1, 2};
        CHECK_THROWS_AS(db.updateEdge(txn, edge), std::invalid_argument);
        edge.uuid.clear();
        CHECK_THROWS_AS(db.addEdge(txn, edge), std::invalid_argument);
        EdgeGeometryRepair repair;
        repair.edgeUuid = id; repair.expectedGeneration = original.generation;
        repair.aAnchor = edge.aAnchor; repair.bAnchor = edge.bAnchor;
        CHECK_THROWS_AS(db.repairEdgeGeometry(txn, repair), std::invalid_argument);
    }
    CHECK(db.edgeByUuid(txn, id)->generation == original.generation);
    CHECK(db.edgeByUuid(txn, id)->aAnchor == original.aAnchor);
    CHECK(db.journalSince(txn, 0).size() == journalSize);
    db.setMeta(txn, a, "continued", "true");
    txn.commit();
}
