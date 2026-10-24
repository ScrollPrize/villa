#include "TestSupport.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <bit>
#include <iostream>
#include <thread>

using namespace vc::objects;

namespace {
OpenOptions as(const std::string& author) { OpenOptions options; options.author = author; return options; }

std::string meta(ObjectStore& store, const std::string& uuid)
{
    auto txn = store.read();
    const auto object = store.objectByUuid(txn, uuid);
    return nlohmann::json::parse(object ? object->meta : store.edgeByUuid(txn, uuid)->meta).dump();
}

void setNote(ObjectStore& store, const std::string& uuid, const std::string& value)
{
    auto txn = store.write();
    store.setMeta(txn, uuid, "note", nlohmann::json(value).dump());
    txn.commit();
}

GeometryArrayEdit moveX(int64_t point, double x)
{
    const auto bits = std::bit_cast<uint64_t>(x);
    GeometryArrayEdit edit{"control_points", point * 3, std::vector<uint8_t>(8)};
    for (int i = 0; i < 8; ++i) edit.bytes[i] = uint8_t(bits >> (i * 8));
    return edit;
}

void sql(const std::filesystem::path& path, const std::string& statement)
{
    sqlite3* db = nullptr;
    require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "raw open");
    const int rc = sqlite3_exec(db, statement.c_str(), nullptr, nullptr, nullptr);
    sqlite3_close(db);
    require(rc == SQLITE_OK, "raw statement");
}
}

int main() try
{
    TemporaryDirectory directory;
    const auto hubPath = directory.path / "share" / "hub.vcobj";
    std::filesystem::create_directory(directory.path / "share");
    const auto a = util::newUuid(), b = util::newUuid(), c = util::newUuid(), ab = util::newUuid();
    vc::geometry::Fiber fiber;
    fiber.controlPoints = {{0,0,0}, {10,0,0}, {20,0,0}};
    fiber.controlPointIds = {1,2,3};
    {
        auto origin = ObjectStore::create(hubPath, as("curator"));
        auto txn = origin.write();
        for (const auto& uuid : {a, b}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "fiber";
            origin.upsertObject(txn, record);
            origin.putGeometry(txn, uuid, vc::geometry::encode(fiber));
        }
        EdgeRecord edge; edge.uuid = ab; edge.kind = "fiber_branch"; edge.aUuid = a; edge.bUuid = b;
        edge.aAnchor.points = {{"control_points", {0}}}; edge.bAnchor.points = edge.aAnchor.points;
        origin.addEdge(txn, edge);
        txn.commit();
        origin.checkpoint();
    }
    // Ordinary stores never see hosting.
    {
        auto plain = ObjectStore::open(hubPath, as("curator"));
        require(!hubStatus(plain), "plain store has no hub status");
        setNote(plain, a, "0");
    }
    initHub(hubPath, "curator");
    mustFail([&] { ObjectStore::open(hubPath, as("curator")); });
    mustFail([&] { auto readOnly = as("curator"); readOnly.readOnly = true; ObjectStore::open(hubPath, readOnly); });
    mustFail([&] { auto shared = as("curator"); shared.sharedFile = true; ObjectStore::open(directory.path / "absent.vcobj", shared); });

    const auto alicePath = directory.path / "alice.vcobj", bobPath = directory.path / "bob.vcobj";
    int64_t progressCalls = 0;
    cloneHub(hubPath, alicePath, as("alice"), [&](const auto& event) { require(event.completed <= event.total, "progress"); ++progressCalls; });
    cloneHub(hubPath, bobPath, as("bob"));
    require(progressCalls > 0, "clone reports progress");
    mustFail([&] { cloneHub(hubPath, bobPath, as("bob")); });
    auto alice = ObjectStore::open(alicePath, as("alice"));
    auto bob = ObjectStore::open(bobPath, as("bob"));
    {
        auto x = alice.read(); auto y = bob.read();
        require(alice.uuid(x) == bob.uuid(y), "replicas share the store identity");
        require(alice.replicaUuid(x) != bob.replicaUuid(y), "replicas have their own identity");
    }
    require(meta(alice, a) == meta(bob, a), "clones start from the hub");

    // Editing an object on the hub requires its lease.
    mustFail([&] { setNote(alice, a, "1"); });
    require(hubStatus(alice)->pending == 0, "rejected edit leaves nothing pending");
    acquireLeases(alice, {a});
    mustFail([&] { acquireLeases(bob, {a}); });
    setNote(alice, a, "1");
    std::string geometry;
    {
        auto txn = alice.write();
        geometry = alice.editGeometry(txn, a, {moveX(2, 25)}, alice.currentGeometry(txn, a)->uuid);
        txn.commit();
    }
    require(hubStatus(alice)->pending == 1, "edit is pending");
    auto report = hubSync(alice);
    require(report.pushed == 1 && report.conflicts.empty(), "push");
    require(hubStatus(alice)->pending == 0, "push clears pending");
    report = hubPull(bob);
    require(report.pulled == 1 && report.conflicts.empty(), "pull");
    require(meta(bob, a) == meta(alice, a), "pulled metadata");
    {
        auto x = alice.read(); auto y = bob.read();
        require(bob.currentGeometry(y, a)->uuid == geometry, "geometry revision identity is shared");
        require(bob.readGeometryArray(y, geometry, "control_points") == alice.readGeometryArray(x, geometry, "control_points"),
                "pulled geometry bytes");
    }
    require(hubStatus(bob)->pending == 0, "pulled changes are not pending");
    require(hubStatus(bob)->leases.size() == 1 && hubStatus(bob)->leases[0].author == "alice", "lease mirror");

    // New objects need no lease. Links are shared records: linking to a hub
    // object does not need its lease, and changing the same link twice conflicts.
    {
        auto txn = bob.write();
        ObjectRecord record; record.uuid = c; record.kind = "fiber";
        bob.upsertObject(txn, record);
        bob.putGeometry(txn, c, vc::geometry::encode(fiber));
        txn.commit();
    }
    std::string cb;
    {
        auto txn = bob.write();
        EdgeRecord edge; edge.kind = "fiber_branch"; edge.aUuid = c; edge.bUuid = b;
        edge.aAnchor.points = {{"control_points", {1}}}; edge.bAnchor.points = edge.aAnchor.points;
        cb = bob.addEdge(txn, edge);
        txn.commit();
    }
    report = hubSync(bob);
    require(report.pushed == 2 && report.conflicts.empty(), "push creation and link");
    hubPull(alice);
    {
        auto txn = alice.read();
        require(alice.objectByUuid(txn, c).has_value() && alice.edgesOf(txn, c).size() == 1, "creation reached alice");
    }
    mustFail([&] { setNote(alice, c, "mine"); });  // created objects stay leased to their creator
    mustFail([&] { releaseLeases(alice, {a}); setNote(alice, a, "after release"); });
    setNote(alice, cb, "alice");
    setNote(bob, cb, "bob");
    require(hubSync(bob).pushed == 1, "first link change wins");
    report = hubSync(alice);
    require(report.conflicts.size() == 1 && report.conflicts[0].uuid == cb, "second link change conflicts");
    resolveHubConflict(alice, cb, "hub");
    require(meta(alice, cb) == meta(bob, cb), "link conflict resolved");

    // A direct hub edit honours leases too.
    {
        auto shared = as("curator"); shared.sharedFile = true;
        auto hub = ObjectStore::open(hubPath, shared);
        mustFail([&] { setNote(hub, c, "curator"); });
    }

    // Lost bookkeeping after a successful push: the retry recognizes its own record.
    acquireLeases(alice, {a});
    setNote(alice, a, "2");
    require(hubPush(alice).pushed == 1, "push before simulated crash");
    sql(alicePath, "INSERT INTO hub_pending VALUES('" + a + "',7,0);DELETE FROM hub_versions WHERE record_uuid='" + a + "';");
    report = hubPush(alice);
    require(report.pushed == 0 && report.conflicts.empty() && hubStatus(alice)->pending == 0, "idempotent retry");

    // An expired lease taken over by another replica produces a conflict, not an overwrite.
    releaseLeases(alice, {a});
    acquireLeases(alice, {a}, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    hubPull(bob);
    acquireLeases(bob, {a});
    setNote(bob, a, "bob");
    hubSync(bob);
    setNote(alice, a, "alice");  // alice's mirror still shows her lease
    report = hubSync(alice);
    require(report.conflicts.size() == 1 && report.conflicts[0].uuid == a, "conflict reported");
    require(hubStatus(alice)->conflicts.size() == 1, "conflict recorded");
    mustFail([&] { hubPush(alice); });
    resolveHubConflict(alice, a, "hub");
    require(meta(alice, a) == meta(bob, a) && hubStatus(alice)->pending == 0, "kept hub record");

    // Keeping the local record publishes it once the lease is back.
    releaseLeases(bob, {a});
    acquireLeases(alice, {a}, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    acquireLeases(bob, {a});
    setNote(bob, a, "bob again");
    hubSync(bob);
    releaseLeases(bob, {a});
    setNote(alice, a, "alice again");
    require(hubSync(alice).conflicts.size() == 1, "second conflict");
    mustFail([&] { resolveHubConflict(alice, a, "local"); });  // expired lease
    renewLeases(alice, {a});
    report = resolveHubConflict(alice, a, "local");
    require(report.pushed == 1 && report.conflicts.empty(), "kept local record");
    hubPull(bob);
    require(meta(bob, a) == meta(alice, a), "local resolution reached the hub");

    // Concurrent pushes from separate replicas serialize on the hub lock.
    acquireLeases(bob, {b});
    hubPull(alice);
    auto writer = [](ObjectStore& store, const std::string& uuid) {
        for (int i = 0; i < 10; ++i) { setNote(store, uuid, std::to_string(i)); hubSync(store); }
    };
    std::thread first([&] { writer(alice, a); }), second([&] { writer(bob, b); });
    first.join(); second.join();
    hubSync(alice); hubSync(bob);
    require(meta(alice, a) == meta(bob, a) && meta(alice, b) == meta(bob, b), "replicas converge");

    // Detaching leaves an ordinary store.
    { auto closing = std::move(bob); }
    detachHub(bobPath, "bob");
    bob = ObjectStore::open(bobPath, as("bob"));
    require(!hubStatus(bob), "detached");
    setNote(bob, a, "standalone");
    std::cout << "object_store_hub: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
