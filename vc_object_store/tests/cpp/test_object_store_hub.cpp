#include "TestSupport.hpp"
#include "HubTestSupport.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include <map>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <bit>
#include <cstring>
#include <iostream>
#include <thread>

using namespace vc::objects;

namespace {

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

    // Idle publication must not invalidate local readers through data_version.
    // Lease-only changes still need to reach the mirror (no journal entries).
    {
        hubPull(alice);
        sqlite3* observer = nullptr;
        require(sqlite3_open_v2(alicePath.c_str(), &observer, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK, "open observer");
        const auto close = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>(observer, sqlite3_close);
        const auto version = [&] {
            sqlite3_stmt* stmt = nullptr;
            require(sqlite3_prepare_v2(observer, "PRAGMA data_version", -1, &stmt, nullptr) == SQLITE_OK, "data version query");
            require(sqlite3_step(stmt) == SQLITE_ROW, "data version row");
            const auto value = sqlite3_column_int(stmt, 0);
            sqlite3_finalize(stmt);
            return value;
        };
        const auto before = version();
        require(hubPull(alice).pulled == 0 && version() == before, "idle pull preserves local data version");
        acquireLeases(bob, {b});
        require(hubPull(alice).pulled == 0 && version() != before, "lease-only pull changes local data version");
        require(hubStatus(alice)->leases == hubStatus(bob)->leases, "lease-only pull mirrors acquisition");
        const auto leased = version();
        hubPull(alice);
        require(version() == leased, "unchanged leases do not cause a write");
        renewLeases(bob, {b}, 1200);
        hubPull(alice);
        require(hubStatus(alice)->leases == hubStatus(bob)->leases && version() != leased,
                "lease-only pull mirrors renewal of the same object");
        const auto renewed = version();
        releaseLeases(bob, {b});
        hubPull(alice);
        require(hubStatus(alice)->leases.empty() && version() != renewed, "lease-only pull mirrors release");
    }

    // Editing an object on the hub requires its lease.
    mustFail([&] { setNote(alice, a, "1"); });
    require(hubStatus(alice)->pending.empty(), "rejected edit leaves nothing pending");
    acquireLeases(alice, {a});
    mustFail([&] { acquireLeases(bob, {a}); });
    setNote(alice, a, "1");
    std::string geometry;
    {
        auto txn = alice.write();
        geometry = alice.editGeometry(txn, a, {moveX(2, 25)}, alice.currentGeometry(txn, a)->uuid);
        txn.commit();
    }
    require(hubStatus(alice)->pending.size() == 1, "edit is pending");
    auto report = hubSync(alice);
    require(report.pushed == 1 && report.conflicts.empty(), "push");
    require(hubStatus(alice)->pending.empty(), "push clears pending");
    report = hubPull(bob);
    require(report.pulled == 1 && report.conflicts.empty(), "pull");
    require(meta(bob, a) == meta(alice, a), "pulled metadata");
    {
        auto x = alice.read(); auto y = bob.read();
        require(bob.currentGeometry(y, a)->uuid == geometry, "geometry revision identity is shared");
        require(bob.readGeometryArray(y, geometry, "control_points") == alice.readGeometryArray(x, geometry, "control_points"),
                "pulled geometry bytes");
    }
    require(hubStatus(bob)->pending.empty(), "pulled changes are not pending");
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
    require(report.pushed == 0 && report.conflicts.empty() && hubStatus(alice)->pending.empty(), "idempotent retry");

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
    require(meta(alice, a) == meta(bob, a) && hubStatus(alice)->pending.empty(), "kept hub record");

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

    // Every geometry kind travels through the hub, including tiled surfaces with
    // sparse deltas and surface channels, stable-ID annotations and track identity.
    {
        using namespace vc::geometry;
        Surface surface{200, 150, {1, 1}, {}};
        for (int64_t row = 0; row < surface.rows; ++row)
            for (int64_t column = 0; column < surface.columns; ++column)
                surface.points.push_back({float(column + 1), float(row + 1), float(row + column + 1)});
        const std::map<std::string, std::pair<GeometryData, GeometryData>> geometry{
            {"fiber", {encode(Fiber{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{10,20}}),
                       encode(Fiber{{{4,5,7},{8,9,10},{1,2,3}},{{4,5,7},{1,2,3}},{20,30,10}})}},
            {"point_collection", {encode(PointCollection{{10,20},{{1,2,3},{4,5,6}}}),
                                  encode(PointCollection{{20,30,10},{{4,5,7},{8,9,10},{1,2,3}}})}},
            {"track", {encode(Track{{{1,2,3},{4,5,6}}}), encode(Track{{{1,2,3},{4,5,7},{8,9,10}}})}},
            {"surface", {encode(surface), encode(surface)}}};
        std::map<std::string, std::string> objects;
        {
            auto txn = bob.write();
            for (const auto& [kind, data] : geometry) {
                ObjectRecord record; record.uuid = util::newUuid(); record.kind = kind;
                if (kind == "track") { record.track.emplace(); record.track->datasetRef = "hub"; record.track->externalKey = "1"; }
                if (kind == "point_collection") record.meta = R"({"source":{"point_attributes":[{"note":"first"},{"note":"second"}]}})";
                bob.upsertObject(txn, record);
                bob.putGeometry(txn, record.uuid, data.first);
                objects[kind] = record.uuid;
            }
            txn.commit();
        }
        auto same = [&](const std::string& uuid) {
            auto x = alice.read(); auto y = bob.read();
            const auto mine = alice.currentGeometry(x, uuid), theirs = bob.currentGeometry(y, uuid);
            require(mine && theirs && mine->uuid == theirs->uuid && mine->contentHash == theirs->contentHash, "same geometry head");
            for (const auto& array : mine->arrays)
                require(alice.readGeometryArray(x, mine->uuid, array.name) == bob.readGeometryArray(y, theirs->uuid, array.name),
                        "same array bytes");
            const auto history = [](ObjectStore& store, const Txn& txn, std::optional<std::string> revision) {
                std::vector<std::string> chain;
                for (; revision; revision = store.geometryRevision(txn, *revision)->parentUuid) chain.push_back(*revision);
                return chain;
            };
            require(history(alice, x, mine->uuid) == history(bob, y, theirs->uuid), "same geometry history");
            require(nlohmann::json::parse(alice.objectByUuid(x, uuid)->meta) == nlohmann::json::parse(bob.objectByUuid(y, uuid)->meta),
                    "same metadata");
        };
        require(hubSync(bob).pushed == 4, "push every kind");
        hubPull(alice);
        for (const auto& [kind, uuid] : objects) same(uuid);

        std::vector<std::string> uuids;
        for (const auto& [kind, uuid] : objects) uuids.push_back(uuid);
        mustFail([&] { acquireLeases(alice, uuids); });  // pushing new objects leased them to their creator
        releaseLeases(bob, uuids);
        acquireLeases(alice, uuids);
        {
            RevisionSelection expected;
            {
                auto read = alice.read();
                for (const auto& uuid : uuids) expected[uuid] = alice.currentObjectRevision(read, uuid)->uuid;
            }
            auto txn = alice.write(expected);
            for (const auto& [kind, uuid] : objects) {
                auto revision = alice.currentGeometry(txn, uuid)->uuid;
                if (kind != "surface") { updateGeometryArrays(alice, txn, uuid, geometry.at(kind).second, revision); continue; }
                std::vector<GeometryArrayEdit> moved;
                for (int64_t row = 0; row < 2; ++row) {
                    std::vector<uint8_t> bytes(2 * 3 * sizeof(float));
                    for (size_t i = 0; i < 6; ++i) {
                        const float value = 1000.0f + float(row * 6 + int64_t(i));
                        std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
                    }
                    moved.push_back({"points", ((130 + row) * surface.columns + 140) * 3, std::move(bytes)});
                }
                revision = alice.editGeometry(txn, uuid, moved, revision);
                GeometryArray mask{"valid", "|u1", {surface.rows, surface.columns},
                                   std::vector<uint8_t>(size_t(surface.rows * surface.columns), 1)};
                alice.updateSurfaceChannels(txn, uuid, {mask}, {}, revision);
            }
            txn.commit();
        }
        require(hubSync(alice).pushed == 4, "push edits of every kind");
        hubPull(bob);
        for (const auto& [kind, uuid] : objects) same(uuid);
        {
            auto x = alice.read(); auto y = bob.read();
            const auto uuid = objects.at("point_collection");
            require(alice.geometryPointAnnotations(x, alice.currentGeometry(x, uuid)->uuid, "points") ==
                    bob.geometryPointAnnotations(y, bob.currentGeometry(y, uuid)->uuid, "points"), "annotations follow stable IDs");
            require(bob.currentGeometry(y, objects.at("surface"))->arrays.size() == 2, "surface channel arrived");
        }
        mustFail([&] { auto txn = bob.write(); bob.setMeta(txn, objects.at("surface"), "note", "1"); txn.commit(); });
        releaseLeases(alice, uuids);
    }

    // A session syncs its attached replica in the background: callers only hold
    // leases while editing and read the status it reports.
    {
        hubSync(alice); hubSync(bob);
        auto eventually = [](const auto& condition, const char* message) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
            while (!condition()) {
                require(std::chrono::steady_clock::now() < deadline, message);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        };
        ObjectStoreSession session;
        HubSessionOptions fast; fast.syncInterval = std::chrono::milliseconds(200); fast.pushDelay = std::chrono::milliseconds(50);
        const auto attachment = session.attach(alicePath, as("alice"), fast);
        eventually([&] { return session.hubStatus(attachment) && session.hubStatus(attachment)->syncedAtUs; }, "session syncs on attach");
        require(session.artifactDirectory(attachment, b) == std::filesystem::path(hubPath.string() + ".artifacts") / b,
                "replicas share the artifact directory beside the hub");
        mustFail([&] { session.act(attachment, {}, [&](auto& store, auto& txn) { store.setMeta(txn, b, "note", "\"unleased\""); }); });
        releaseLeases(bob, {b});
        {
            auto hold = session.holdLeases(attachment, {b});
            RevisionSelection expected;
            session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
                expected[b] = store.currentObjectRevision(txn, b)->uuid;
            });
            session.act(attachment, expected, [&](auto& store, auto& txn) { store.setMeta(txn, b, "note", "\"session\""); });
            // A waited sync pushes everything committed before it.
            const auto synced = session.syncHubAndWait(attachment, std::chrono::seconds(20));
            require(synced && synced->pending.empty() && synced->error.empty(), "waited sync pushed the action");
            hubPull(bob);
            require(meta(bob, b) == meta(alice, b), "action reached the hub");
            mustFail([&] { acquireLeases(bob, {b}); });  // still held by the session
        }
        eventually([&] {
            try { acquireLeases(bob, {b}); return true; } catch (const HubLeaseRefused&) { return false; }
        }, "lease released once the hold ends");
        bool hubEvent = false;
        for (const auto& event : session.takeEvents()) hubEvent |= event.kind == StoreSessionEvent::Kind::Hub;
        require(hubEvent, "status changes are session events");
        session.detach(attachment);
        require(!session.hubStatus(attachment), "no hub status without an attachment");
        releaseLeases(bob, {b});
        hubSync(alice);
    }

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
