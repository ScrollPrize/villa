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
#include <atomic>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <mutex>
#include <exception>
#include <memory>
#include <thread>
#include <algorithm>

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
        HubSessionOptions fast; fast.syncInterval = std::chrono::milliseconds(200); fast.pushDelay = std::chrono::milliseconds(50);
        // The optional span trace reports the worker's phases and lease waits
        // as strictly nested begin/end pairs on the thread doing the work.
        std::mutex traceMutex;
        std::map<std::thread::id, std::vector<std::string>> traceStacks;
        std::map<std::string, int> traceCounts;
        bool traceNested = true;
        // Parking: the hook blocks the thread entering the named phase until
        // the test lets it go, so the worker can be frozen inside a sync
        // (holding hub_) or just before one.
        std::mutex parkMutex;
        std::condition_variable parkCv;
        std::string parkAt;
        bool parked = false;
        // One begin callback throws: the worker's sync must complete regardless.
        std::atomic<bool> traceThrowOnce{true};
        std::function<void()> onStart;  // runs in the "start" trace (before a worker claims the replica)
        fast.trace = [&](std::string_view name, bool begin) {
            if (begin && name == "start" && onStart) onStart();
            {
                std::lock_guard lock(traceMutex);
                auto& stack = traceStacks[std::this_thread::get_id()];
                if (begin) { stack.emplace_back(name); ++traceCounts[std::string(name)]; }
                else if (stack.empty() || stack.back() != name) traceNested = false;
                else stack.pop_back();
            }
            // The begin was recorded; its end still arrives (the scope ends).
            if (begin && name == "sync.push" && traceThrowOnce.exchange(false)) throw std::runtime_error("trace hook failure");
            if (!begin) return;
            std::unique_lock lock(parkMutex);
            if (parkAt != name) return;
            parked = true;
            parkCv.notify_all();
            parkCv.wait(lock, [&] { return parkAt != name; });
            parked = false;
            parkCv.notify_all();
        };
        auto park = [&](const char* phase) {
            std::unique_lock lock(parkMutex);
            parkAt = phase;
            require(parkCv.wait_for(lock, std::chrono::seconds(20), [&] { return parked; }), "worker parks");
        };
        auto unpark = [&] {
            std::unique_lock lock(parkMutex);
            parkAt.clear();
            parkCv.notify_all();
            parkCv.wait_for(lock, std::chrono::seconds(20), [&] { return !parked; });
        };
        auto count = [&](const char* name) { std::lock_guard lock(traceMutex); return traceCounts[name]; };
        // Helper threads are unparked and joined before they are destroyed;
        // what they throw is rethrown on the test thread by join().
        struct JoinedThread {
            std::function<void()> release; std::exception_ptr error; std::thread thread;
            JoinedThread(std::function<void()> body, std::function<void()> unpark)
                : release(std::move(unpark)),
                  thread([this, body = std::move(body)] { try { body(); } catch (...) { error = std::current_exception(); } }) {}
            void join() { if (thread.joinable()) thread.join(); if (error) std::rethrow_exception(error); }
            ~JoinedThread() { if (thread.joinable()) { release(); thread.join(); } }
        };
        auto hubSeconds = [&](const std::string& object, int64_t seconds) {
            // Shortens this replica's lease on the hub itself: the mirror
            // learns of it at the next pull (direct lease calls on a managed
            // replica are refused, see below).
            const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            sql(hubPath, "UPDATE hub_leases SET expires_us=" + std::to_string(now + seconds * 1000000) +
                         " WHERE object_uuid='" + object + "';");
        };
        ObjectStoreSession session;
        // A failed assertion while the worker is parked must still let the
        // session stop: the guard (declared after the session) unparks before
        // the session is destroyed, and the callback state outlives both.
        struct UnparkOnExit { std::function<void()> unpark; ~UnparkOnExit() { unpark(); } } unparkOnExit{unpark};
        auto attachment = session.attach(alicePath, as("alice"), fast);
        // Reattaching the same replica hands the lease ownership over.
        attachment = session.replace(attachment, alicePath, as("alice"), fast);
        // One worker manages a replica's leases: a second session, also
        // through an alias of the file, and direct lease calls are refused
        // while it lives.
        {
            ObjectStoreSession other;
            mustFail([&] { other.attach(alicePath, as("alice"), fast); }, "a second session on the replica");
            // Refused before anything was published: the session has no
            // attachment to grant holds on without a worker.
            require(other.attachments().empty(), "a refused attachment leaves the session detached");
            // A session attached elsewhere keeps that attachment when the
            // reattachment is refused.
            const auto plainPath = directory.path / "plain.vcobj";
            { auto plain = ObjectStore::create(plainPath, as("alice")); }
            const auto before = other.attach(plainPath, as("alice"));
            mustFail([&] { other.attach(alicePath, as("alice"), fast); }, "a refused reattachment");
            require(other.attachments() == std::vector{before} && other.current(before), "a refused reattachment keeps the previous attachment");
        }
        const auto aliasPath = directory.path / "alias.vcobj";
        std::filesystem::create_symlink(alicePath, aliasPath);
        mustFail([&] { ObjectStoreSession other; other.attach(aliasPath, as("alice"), fast); }, "a second session through an alias");
        mustFail([&] { renewLeases(alice, {b}); }, "a direct lease call on a managed replica");
        mustFail([&] { auto alias = ObjectStore::open(aliasPath, as("alice")); renewLeases(alias, {b}); },
                 "a direct lease call through an alias");
        eventually([&] { return session.hubStatus(attachment) && session.hubStatus(attachment)->syncedAtUs; }, "session syncs on attach");
        require(!traceThrowOnce && session.hubStatus(attachment)->error.empty(), "a throwing trace callback does not fail the sync");
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
        releaseLeases(bob, {b});
        {
            // A hold on an object another live hold keeps leased never waits
            // for a running sync, while one that needs a lease does.
            auto editing = session.holdLeases(attachment, {b});  // acquires b
            park("sync.push");  // the worker now holds hub_ mid-sync
            const auto waits = count("hold.wait"), acquires = count("hold.acquire");
            const auto start = std::chrono::steady_clock::now();
            auto fast = session.holdLeases(attachment, {b});
            require(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500) &&
                    count("hold.wait") == waits && count("hold.acquire") == acquires,
                    "a held object's lease is granted without the sync");
            RevisionSelection expected;
            session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
                expected[b] = store.currentObjectRevision(txn, b)->uuid;
            });
            session.act(attachment, expected, [&](auto& store, auto& txn) { store.setMeta(txn, b, "note", "\"fast\""); });
            mustFail([&] { acquireLeases(bob, {b}); });
            // Mixed request: b qualifies, c is leased to its creator bob. The
            // refusal leaves b's count as it was.
            std::atomic<bool> refused{false};
            JoinedThread mixed{[&] {
                try { session.holdLeases(attachment, {b, c}); } catch (const HubLeaseRefused&) { refused = true; }
            }, unpark};
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            require(!refused, "a request needing a lease waits for the sync");
            unpark();
            mixed.join();
            require(refused, "the unleased object's refusal propagates");
            park("sync.push");
            const auto waitsAfter = count("hold.wait");
            auto still = session.holdLeases(attachment, {b});
            require(count("hold.wait") == waitsAfter, "the editing hold survived the refused request");
            unpark();
            // A lease inside the margin leaves the fast path: the hold waits
            // for the sync that renews it.
            hubSeconds(b, 30);
            park("sync.renew");  // the pull mirrored the 30 s expiry; renewal is next
            std::atomic<bool> returned{false};
            JoinedThread slow{[&] { auto renewed = session.holdLeases(attachment, {b}); returned = true; }, unpark};
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            require(!returned, "a lease inside the margin takes the slow path");
            unpark();
            slow.join();
            require(returned && count("hold.wait") > waitsAfter, "the slow path waited for hub_");
            // A never-pushed object needs no lease; its creator lease arrives
            // with the push and later holds take the fast path.
            const auto d = util::newUuid();
            park("sync");  // frozen before taking hub_
            session.act(attachment, {{d, std::nullopt}}, [&](auto& store, auto& txn) {
                ObjectRecord record; record.uuid = d; record.kind = "fiber";
                store.upsertObject(txn, record);
                store.putGeometry(txn, d, vc::geometry::encode(fiber));
            });
            const auto acquiresBefore = count("hold.acquire");
            auto seed = session.holdLeases(attachment, {d}, /*createdHere=*/true);
            require(count("hold.acquire") == acquiresBefore, "a never-pushed object is held without a lease");
            unpark();
            eventually([&] {
                const auto status = session.hubStatus(attachment);
                return status && std::any_of(status->leases.begin(), status->leases.end(),
                                             [&](const auto& lease) { return lease.objectUuid == d; });
            }, "the push grants the creator lease");
            park("sync.push");
            const auto waitsSeed = count("hold.wait");
            auto seedAgain = session.holdLeases(attachment, {d});
            require(count("hold.wait") == waitsSeed, "the creator lease serves the fast path");
            unpark();
            mustFail([&] { hubPull(bob); acquireLeases(bob, {d}); });
            // Only a fresh identity skips the lease: a supplied one (an
            // import) may exist on the hub already, under another replica's
            // creator lease, and is leased like any other object even while
            // this replica knows no hub version for it.
            const auto e = util::newUuid();
            {
                auto txn = bob.write();
                ObjectRecord record; record.uuid = e; record.kind = "fiber";
                bob.upsertObject(txn, record);
                bob.putGeometry(txn, e, vc::geometry::encode(fiber));
                txn.commit();
            }
            park("sync");  // alice has not pulled e when she asks for it
            hubSync(bob);
            mustFail([&] { session.holdLeases(attachment, {e}); }, "a hold on an identity bob created meanwhile");
            unpark();
            releaseLeases(bob, {e});
            // An uncertain release: the hub drops the lease, the mirror write
            // fails. The mirror still shows the lease but is not trusted, so
            // the next hold acquires again instead of editing on a lost lease.
            seedAgain.reset(); seed.reset(); still.reset(); fast.reset(); editing.reset();
            park("sync.release");
            sqlite3* raw = nullptr;
            require(sqlite3_open(alicePath.c_str(), &raw) == SQLITE_OK, "raw replica connection");
            // Closed (and its transaction rolled back) on every exit path.
            const std::unique_ptr<sqlite3, decltype(&sqlite3_close)> rawGuard(raw, sqlite3_close);
            require(sqlite3_exec(raw, "BEGIN IMMEDIATE;", nullptr, nullptr, nullptr) == SQLITE_OK, "replica write lock");
            unpark();
            eventually([&] { const auto status = session.hubStatus(attachment); return status && !status->error.empty(); },
                       "the failed mirror write is reported");
            park("sync");
            sqlite3_exec(raw, "ROLLBACK;", nullptr, nullptr, nullptr);
            const auto mirror = *hubStatus(alice);
            require(std::any_of(mirror.leases.begin(), mirror.leases.end(),
                                [&](const auto& lease) { return lease.objectUuid == b; }), "the stale mirror still shows the lease");
            // A competitor that took the lease meanwhile is seen: the stale
            // mirror is not trusted, so the hold is refused, not granted.
            hubPull(bob); acquireLeases(bob, {b});
            mustFail([&] { session.holdLeases(attachment, {b}); });
            releaseLeases(bob, {b});
            const auto acquiresUncertain = count("hold.acquire");
            auto again = session.holdLeases(attachment, {b});
            require(count("hold.acquire") == acquiresUncertain + 1, "an uncertain release forces re-acquisition");
            again.reset();
            unpark();
            eventually([&] {
                try { acquireLeases(bob, {b}); return true; } catch (const HubLeaseRefused&) { return false; }
            }, "lease released again once the holds end");
            // A competitor holding the lease after our release refuses the hold.
            mustFail([&] { session.holdLeases(attachment, {b}); });
            releaseLeases(bob, {b});
            // A lease that expired outright (the mirror shows it) leaves the
            // fast path: the hold waits for the sync that renews it.
            auto expiring = session.holdLeases(attachment, {b});
            hubSeconds(b, 2);
            park("sync.renew");  // the pull mirrored the 2 s expiry; renewal is next
            std::this_thread::sleep_for(std::chrono::milliseconds(2200));
            const auto waitsExpired = count("hold.wait");
            JoinedThread expired{[&] { auto reacquired = session.holdLeases(attachment, {b}); }, unpark};
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            require(count("hold.wait") == waitsExpired + 1, "an expired lease leaves the fast path");
            unpark();
            expired.join();
            expiring.reset();
            eventually([&] {
                try { acquireLeases(bob, {b}); return true; } catch (const HubLeaseRefused&) { return false; }
            }, "lease released after the re-acquisition's holds end");
        }
        bool hubEvent = false;
        for (const auto& event : session.takeEvents()) hubEvent |= event.kind == StoreSessionEvent::Kind::Hub;
        require(hubEvent, "status changes are session events");
        // A hold that outlives its worker does not keep the lease: stopping
        // the worker releases the leases of pushed objects.
        releaseLeases(bob, {b});
        auto outliving = session.holdLeases(attachment, {b});
        session.detach(attachment);
        require(!session.hubStatus(attachment), "no hub status without an attachment");
        hubPull(bob); acquireLeases(bob, {b});
        releaseLeases(bob, {b});
        outliving.reset();
        // An inherited mirror is not trusted: a release that reached the hub
        // but not the mirror (here: a lease dropped behind the mirror's back
        // and taken by bob) must not let the first hold of a new worker edit
        // on a lease bob owns.
        acquireLeases(alice, {b});
        sql(hubPath, "DELETE FROM hub_leases WHERE object_uuid='" + b + "';");
        {
            const auto mirror = *hubStatus(alice);
            require(std::any_of(mirror.leases.begin(), mirror.leases.end(),
                                [&](const auto& lease) { return lease.objectUuid == b; }), "the mirror still shows the lease");
        }
        hubPull(bob); acquireLeases(bob, {b});
        { std::lock_guard lock(parkMutex); parkAt = "sync"; }  // the new worker parks before its first pull
        attachment = session.attach(alicePath, as("alice"), fast);
        { std::unique_lock lock(parkMutex); require(parkCv.wait_for(lock, std::chrono::seconds(20), [&] { return parked; }), "worker parks"); }
        const auto pullsBefore = count("hold.pull");
        mustFail([&] { session.holdLeases(attachment, {b}); }, "a hold on an inherited mirror row bob took meanwhile");
        require(count("hold.pull") == pullsBefore + 1, "the first hold verifies the inherited mirror");
        unpark();
        releaseLeases(bob, {b});
        // A refused reattachment of ANOTHER replica leaves the current one
        // untouched, its holds included; a refused reattachment of the SAME
        // replica (whose worker had to stop, releasing its leases) leaves
        // the session detached rather than attached with dead holds.
        {
            ObjectStoreSession blocker;
            const auto blocking = blocker.attach(bobPath, as("bob"), fast);  // bob's replica is managed now
            auto held = session.holdLeases(attachment, {b});
            mustFail([&] { session.attach(bobPath, as("bob"), fast); }, "a reattachment to a managed replica");
            require(session.attachments() == std::vector{attachment} && session.current(attachment), "the previous attachment survives");
            // Alice's hold survived: bob's session is refused the lease (not
            // a direct call, which bob's own worker would refuse regardless).
            bool refused = false;
            try { blocker.holdLeases(blocking, {b}); } catch (const HubLeaseRefused& error) { refused = error.held == std::vector<std::string>{b}; }
            require(refused, "the hold survived too");
            held.reset();
            eventually([&] {
                try { blocker.holdLeases(blocking, {b}); return true; } catch (const HubLeaseRefused&) { return false; }
            }, "released once the hold ends");
            blocker.detach(blocking);
        }
        {
            // Another owner appears while alice's worker is stopped (the
            // "start" trace runs between the old worker's stop and the new
            // worker's claim): the worker lock is a flock on the sidecar
            // beside the replica, taken here by a second description.
            auto held = session.holdLeases(attachment, {b});
            const auto sidecar = std::filesystem::weakly_canonical(alicePath).string() + ".worker-lock";
            const int taker = ::open(sidecar.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
            require(taker >= 0, "sidecar opened");
            require(::flock(taker, LOCK_EX | LOCK_NB) != 0, "alice's worker holds the sidecar lock");
            onStart = [&] { require(::flock(taker, LOCK_EX | LOCK_NB) == 0, "the sidecar is free once the old worker stopped"); };
            mustFail([&] { session.replace(attachment, alicePath, as("alice"), fast); }, "a same-replica reattachment refused by another owner");
            onStart = nullptr;
            require(session.attachments().empty() && !session.current(attachment), "the session is detached, not attached with dead holds");
            ::close(taker);
        }
        hubPull(bob); acquireLeases(bob, {b});  // released when alice's worker stopped
        releaseLeases(bob, {b});
        session.detach(attachment);
        {
            std::lock_guard lock(traceMutex);
            require(traceNested, "trace begin/end pairs nest");
            require(std::all_of(traceStacks.begin(), traceStacks.end(), [](const auto& e) { return e.second.empty(); }),
                    "every traced span ended");
            for (const char* name : {"sync", "sync.wait", "sync.pull", "sync.push", "hold", "hold.wait", "hold.acquire"})
                require(traceCounts[name] > 0, "trace reports every sync and hold phase");
            require(traceCounts["hold"] > traceCounts["hold.wait"], "fast holds never waited");
        }
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
