#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <future>
#include <thread>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <sqlite3.h>

using namespace vc::objects;

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "session-test";
    // Both pre-backport formats are rejected before the new journal is queried.
    for (const int version : {14, 15, 16}) {
        const auto oldPath = directory.path / ("schema-" + std::to_string(version) + ".vcobj");
        { auto db = ObjectStore::create(oldPath, options); }
        sqlite3* raw = nullptr;
        require(sqlite3_open(oldPath.c_str(), &raw) == SQLITE_OK, "open schema fixture");
        const auto sql = "PRAGMA user_version=" + std::to_string(version);
        const int rc = sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, nullptr);
        sqlite3_close(raw);
        require(rc == SQLITE_OK, "set old schema version");
        for (const bool readOnly : {false, true}) {
            auto oldOptions = options; oldOptions.readOnly = readOnly;
            bool rejected = false;
            try { auto db = ObjectStore::open(oldPath, oldOptions); }
            catch (const std::runtime_error& error) {
                rejected = std::string(error.what()).find("unsupported database schema") != std::string::npos;
            }
            require(rejected, "old formats are explicitly rejected in read-only and writable mode");
        }
    }
    const auto path = directory.path / "mixed.vcobj";
    auto external = ObjectStore::create(path, options);
    std::string a, b;
    {
        auto txn = external.write();
        ObjectRecord object; object.kind = "fiber"; a = external.upsertObject(txn, object);
        object.kind = "point_collection"; b = external.upsertObject(txn, object);
        txn.commit();
    }
    ObjectStoreSession session;
    auto identity = session.attach(path, options);
    auto events = session.takeEvents();
    require(events.size() == 1 && events[0].kind == StoreSessionEvent::Kind::Attached, "attachment event");
    mustFail([&] { session.attach(directory.path / "absent.vcobj", options); });
    require(!std::filesystem::exists(directory.path / "absent.vcobj"), "attach must not create");
    mustFail([&] { session.replace(identity, directory.path / "absent.vcobj", options); });
    require(session.current(identity), "failed replacement preserved old attachment");
    mustFail([&] { session.attach(path, options); });
    require(session.attachments().size() == 1, "a store is attached once");
    auto readonly = options; readonly.readOnly = true;
    identity = session.replace(identity, path, readonly);
    require(!session.writable(identity), "read-only attachment reports access mode");
    session.read(identity, [&](const auto& store, const auto& txn, const auto&) {
        require(store.objectByUuid(txn, a).has_value(), "read-only attachment reads objects");
    });
    bool callbackRan = false;
    mustFail([&] { session.act(identity, {}, [&](auto&, auto&) { callbackRan = true; }); });
    require(!callbackRan, "read-only actions rejected before callback");
    identity = session.replace(identity, path, options);
    events = session.takeEvents();
    // Two replacements: read-only, then writable again.
    require(events.size() == 4 && events[2].kind == StoreSessionEvent::Kind::Detached &&
            events[3].kind == StoreSessionEvent::Kind::Attached && events[3].attachment == identity,
            "replacement reports the old attachment leaving before the new one joins");
    require(session.writable(identity), "writable reattachment");

    auto heads = [&] {
        RevisionSelection result;
        session.read(identity, [&](const auto& store, const auto& txn, const auto&) {
            result[a] = store.currentObjectRevision(txn, a)->uuid;
            result[b] = store.currentObjectRevision(txn, b)->uuid;
        });
        return result;
    };
    auto expected = heads();
    std::string edge;
    session.act(identity, expected, [&](auto& store, auto& txn) {
        store.setMeta(txn, a, "note", "1");
        store.setMeta(txn, b, "note", "2");
        EdgeRecord link; link.kind = "mixed"; link.aUuid = a; link.bUuid = b;
        edge = store.addEdge(txn, link);
    });
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes.size() == 3, "mixed action notification");
    require(events[0].attachment == identity && events[0].journalPosition > 0, "event identity and journal");
    for (const auto& change : events[0].changes)
        require(change.kind == (change.recordUuid == edge ? "mixed" : change.recordUuid == a ? "fiber" : "point_collection"), "changes carry record kinds");
    const auto committed = heads();
    mustFail([&] { session.act(identity, expected, [&](auto& store, auto& txn) { store.setMeta(txn, a, "stale", "true"); }); });
    require(heads() == committed && session.takeEvents().empty(), "conflict leaves no changes");
    mustFail([&] { session.act(identity, committed, [&](auto& store, auto& txn) {
        store.setMeta(txn, a, "rollback", "true");
        throw std::runtime_error("abort");
    }); });
    require(heads() == committed && session.takeEvents().empty(), "callback rollback");
    mustFail([&] { session.act(identity, {{a, committed.at(a)}}, [&](auto& store, auto& txn) { store.removeEdge(txn, edge); }); });
    require(heads() == committed, "all relationship endpoints must be checked");

    auto upper = a;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    auto receipt = session.act(identity, {{upper, committed.at(a)}}, [&](auto& store, auto& txn) {
        store.setMeta(txn, a, "receipt", "true");
    });
    require(receipt.at(upper) == heads().at(a), "receipt supports accepted UUID spellings");
    session.takeEvents();

    // External changes to a non-fiber object must be observed.
    { auto txn = external.write(); external.setMeta(txn, b, "external", "true"); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes.size() == 1 && events[0].changes[0].recordUuid == b, "external notification");
    { auto txn = external.write(); external.removeEdge(txn, edge); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes[0].kind == "mixed", "deletions keep their record kind");

    // Same expected head submitted concurrently: exactly one action wins.
    expected = heads();
    std::atomic_int successes{0}, conflicts{0};
    auto writer = [&] {
        try {
            session.act(identity, expected, [&](auto& store, auto& txn) { store.setMeta(txn, a, "race", "true"); });
            ++successes;
        } catch (const std::exception&) { ++conflicts; }
    };
    auto one = std::async(std::launch::async, writer), two = std::async(std::launch::async, writer);
    one.get(); two.get(); session.finalize();
    require(successes == 1 && conflicts == 1, "serialized checked actions");
    session.takeEvents();

    // Detachment cancels and waits for a worker-local snapshot. The snapshot
    // remains consistent while another handle commits.
    std::promise<void> entered, changed;
    auto changeReady = changed.get_future();
    auto reader = std::async(std::launch::async, [&] {
        session.read(identity, [&](const auto& store, const auto& txn, const auto& cancelled) {
            const auto before = store.currentObjectRevision(txn, b)->uuid;
            entered.set_value();
            changeReady.wait();
            require(store.currentObjectRevision(txn, b)->uuid == before, "worker snapshot consistency");
            while (!cancelled.load()) std::this_thread::yield();
        });
    });
    entered.get_future().wait();
    { auto txn = external.write(); external.setMeta(txn, b, "external", "2"); txn.commit(); }
    changed.set_value();
    session.detach(identity); reader.get();
    require(session.attachments().empty(), "detached");
    events = session.takeEvents();
    require(events.size() == 2 && events[0].kind == StoreSessionEvent::Kind::Committed &&
            events[1].kind == StoreSessionEvent::Kind::Detached && events[1].attachment == identity,
            "old commits precede detachment");
    mustFail([&] { session.act(identity, {}, [](auto&, auto&) {}); });
    const auto second = session.attach(path, options);
    require(second.storeUuid == identity.storeUuid && second.generation > identity.generation, "same store invalidates old requests");
    mustFail([&] { session.read(identity, [](const auto&, const auto&, const auto&) {}); });
    // Accepted writes finish before detachment. A writer submitted after the
    // transition begins must not slip onto the old attachment.
    std::promise<void> writing, finishWriting;
    auto finishReady = finishWriting.get_future();
    RevisionSelection latest;
    session.read(second, [&](const auto& store, const auto& txn, const auto&) { latest[a] = store.currentObjectRevision(txn, a)->uuid; });
    auto pending = std::async(std::launch::async, [&] {
        session.act(second, latest, [&](auto& store, auto& txn) {
            writing.set_value(); finishReady.wait(); store.setMeta(txn, a, "drained", "true");
        });
    });
    writing.get_future().wait();
    auto detach = std::async(std::launch::async, [&] { session.detach(second); });
    while (session.current(second)) std::this_thread::yield();
    mustFail([&] { session.act(second, latest, [](auto&, auto&) {}); });
    finishWriting.set_value(); pending.get(); detach.get();
    { auto txn = external.read(); require(external.objectByUuid(txn, a)->meta.find("drained") != std::string::npos, "accepted write drained"); }
    auto replacementPath = directory.path / "replacement.vcobj";
    { auto replacement = ObjectStore::create(replacementPath, options); }
    auto replacement = session.attach(replacementPath, options);
    require(replacement.storeUuid != second.storeUuid && replacement.generation > second.generation, "different-store attachment");
    session.takeEvents();

    // Several stores side by side: a read-only catalog next to a writable store.
    // Each keeps its own access mode and events; detaching one leaves the other.
    auto catalog = session.attach(path, readonly);
    require(session.attachments() == std::vector<AttachmentIdentity>{replacement, catalog}, "attachments in attach order");
    require(session.writable(replacement) && !session.writable(catalog), "access mode is per attachment");
    const auto added = util::newUuid();
    session.act(replacement, {{added, std::nullopt}}, [&](auto& store, auto& txn) {
        ObjectRecord o; o.uuid = added; o.kind = "surface"; store.upsertObject(txn, o);
    });
    { auto txn = external.write(); external.setMeta(txn, a, "catalog", "1"); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 3 && events[0].kind == StoreSessionEvent::Kind::Attached && events[0].attachment == catalog,
            "second attachment joins");
    for (const auto& event : std::vector(events.begin() + 1, events.end()))
        require(event.kind == StoreSessionEvent::Kind::Committed && event.changes.size() == 1 &&
                event.changes[0].recordUuid == (event.attachment == catalog ? a : added), "events are tagged by attachment");
    mustFail([&] { session.act(catalog, {}, [](auto&, auto&) {}); });

    // Copying from the read-only catalog into the writable store makes a new,
    // editable object with the same geometry, channels and point annotations.
    std::string surface, points;
    {
        auto txn = external.write();
        ObjectRecord record; record.kind = "surface"; record.name = "catalog surface"; record.tags = {"paths"};
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":1}})";
        surface = external.upsertObject(txn, record);
        auto data = vc::geometry::encode(vc::geometry::Surface{1, 2, {1, 1}, {{1, 2, 3}, {4, 5, 6}}, {255, 0}});
        data.arrays.push_back({"ink", "<u2", {2, 4}, std::vector<uint8_t>(16, 7)});
        external.putGeometry(txn, surface, data);
        ObjectRecord collection; collection.kind = "point_collection";
        collection.meta = R"({"source":{"point_attributes":[{"note":"first"},{}]}})";
        points = external.upsertObject(txn, collection);
        external.putGeometry(txn, points, vc::geometry::encode(vc::geometry::PointCollection{{7, 3}, {{1, 2, 3}, {4, 5, 6}}}));
        txn.commit();
    }
    for (const auto& original : {surface, points}) {
        const auto copied = session.copyObject(catalog, replacement, original);
        require(copied != original, "copies get new UUIDs");
        GeometryData source, target;
        std::string sourceAnnotations, targetAnnotations;
        session.read(catalog, [&](const auto& store, const auto& txn, const auto&) {
            const auto revision = store.currentGeometry(txn, original)->uuid;
            source = store.readGeometryData(txn, revision);
            if (source.kind != "surface") sourceAnnotations = store.geometryPointAnnotations(txn, revision, "points");
        });
        session.read(replacement, [&](const auto& store, const auto& txn, const auto&) {
            const auto object = store.objectByUuid(txn, copied);
            const auto meta = nlohmann::json::parse(object->meta);
            require(meta.at("copied_from").at("object_uuid") == original &&
                    meta.at("copied_from").at("store_uuid") == catalog.storeUuid, "copies record their source");
            if (object->kind == "surface")
                require(object->name == "catalog surface" && object->tags == std::vector<std::string>{"paths"} &&
                        meta.at("coordinates").at("volume_uuid") == "vol", "copies keep name, tags and binding");
            const auto revision = store.currentGeometry(txn, copied)->uuid;
            target = store.readGeometryData(txn, revision);
            if (target.kind != "surface") targetAnnotations = store.geometryPointAnnotations(txn, revision, "points");
        });
        require(source.arrays.size() == target.arrays.size(), "copies keep every array");
        for (size_t i = 0; i < source.arrays.size(); ++i)
            require(source.arrays[i].name == target.arrays[i].name && source.arrays[i].bytes == target.arrays[i].bytes,
                    "copies keep exact geometry");
        require(sourceAnnotations == targetAnnotations, "copies keep point annotations");
    }
    session.takeEvents();
    session.detach(catalog);
    require(session.current(replacement) && !session.current(catalog), "detaching one keeps the other");
    session.read(replacement, [&](const auto& store, const auto& txn, const auto&) {
        require(store.objectByUuid(txn, added).has_value(), "remaining attachment still reads");
    });
    session.detach(replacement);
    std::cout << "session tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
