#include "TestSupport.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <nlohmann/json.hpp>
#include <future>
#include <thread>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <tuple>
#include <cctype>
#include <sqlite3.h>
#include <fstream>

using namespace vc::objects;

namespace {
void referencedArtifacts(HistoryMode sourceMode, HistoryMode overlayMode)
{
    TemporaryDirectory directory;
    OpenOptions options; options.author="artifact-test";options.history=sourceMode;
    const auto base=directory.path/"base.vcobj", overlay=directory.path/"overlay.vcobj";
    const auto oldFile=directory.path/"old.pt", newFile=directory.path/"new.pt";
    std::ofstream(oldFile)<<"old checkpoint"; std::ofstream(newFile)<<"new checkpoint";
    const auto uuid=util::newUuid();
    {auto db=ObjectStore::create(base,options);auto txn=db.write();ObjectRecord record;
        record.uuid=uuid;record.kind="note";db.upsertObject(txn,record);txn.commit();}
    ObjectVersion original;
    {ObjectStoreSession session;const auto attachment=session.attach(base,options);
        session.publishArtifacts(attachment,uuid,1,{{"model.pt",oldFile}});
        original=session.currentObject(attachment,uuid)->reference();}
    {auto db=ObjectStore::open(base,options);auto txn=db.write();db.seal(txn);txn.commit();}
    options.history=overlayMode;
    {auto collection=ObjectCollection::create(overlay,{base},options);}
    ObjectStoreSession session;const auto attachment=session.attach(overlay,options);
    session.publishArtifacts(attachment,uuid,original.version,{{"model.pt",newFile}});
    const auto latest=session.currentObject(attachment,uuid)->reference();
    require(latest.storeUuid!=original.storeUuid,"artifact edit creates an overlay-owned version");
    const auto expected=session.artifacts(attachment,uuid,{}, {},original);
    require(expected.size()==1,"source artifact manifest survives shadowing");
    session.materializeArtifacts(attachment,uuid,directory.path/"historical",{}, {},original);
    const auto bytes=[](const std::filesystem::path& path){std::ifstream in(path);return std::string(std::istreambuf_iterator<char>(in),{});};
    require(bytes(directory.path/"historical"/"model.pt")=="old checkpoint","exact materialization uses old bytes");
    require(bytes(*session.artifactFile(attachment,uuid,"model.pt",directory.path/"cache"))=="new checkpoint","current artifact uses new bytes");
    session.editCurrent(attachment,{uuid},[&](const StoreReader&,const std::string&){return std::vector<TransactionOperation>{DeleteObjectOp{uuid}};});
    require(bytes(*session.artifactFile(attachment,uuid,"model.pt",directory.path/"cache",{}, {},original))=="old checkpoint",
        "source artifacts remain readable after overlay deletion with either history policy");
    mustFail([&]{session.artifacts(attachment,uuid);});
    mustFail([&]{session.artifacts(attachment,util::newUuid(),{}, {},original);});
    mustFail([&]{session.artifacts(attachment,uuid,original.version,{},original);});
    auto wrong=original;wrong.storeUuid=util::newUuid();mustFail([&]{session.artifacts(attachment,uuid,{}, {},wrong);});
}

void editCurrentLifecycle(HistoryMode history)
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "lifecycle-test"; options.history = history;
    const auto path = directory.path / "lifecycle.vcobj";
    auto store = ObjectStore::create(path, options);
    const auto a = util::newUuid(), b = util::newUuid(), c = util::newUuid();
    auto makeEdge = [](const std::string& first, const std::string& second) {
        EdgeRecord edge; edge.uuid = util::newUuid(); edge.kind = "attachment";
        edge.aUuid = first; edge.bUuid = second;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
        return edge;
    };
    auto ab = makeEdge(a, b), bc = makeEdge(b, c);
    {
        auto txn = store.write();
        for (const auto& uuid : {a, b, c}) {
            ObjectRecord record; record.uuid = uuid; record.kind = "surface";
            store.upsertObject(txn, record);
        }
        store.addEdge(txn, ab); store.addEdge(txn, bc); txn.commit();
    }
    ObjectStoreSession session;
    const auto attachment = session.attach(path, options);
    session.editCurrent(attachment, {a, b}, [&](const StoreReader&, const std::string& uuid) {
        if (uuid == a) return std::vector<TransactionOperation>{RetireObjectOp{a}};
        return std::vector<TransactionOperation>{SetMetaOp{b, "note", "1"}, RetainRelationshipOp{bc.uuid}};
    });
    {
        auto txn = store.read();
        require(store.objectByUuid(txn, a)->retired, "editCurrent retires an object with relationships");
        require(!store.edgeByUuid(txn, ab.uuid), "retirement removes its incident relationship");
        require(store.edgeByUuid(txn, bc.uuid).has_value(), "editCurrent retains unrelated surviving relationships");
    }
    session.editCurrent(attachment, {a}, [&](const StoreReader&, const std::string&) {
        return std::vector<TransactionOperation>{RestoreObjectOp{a, {}}};
    });
    {
        auto txn = store.read();
        require(!store.objectByUuid(txn, a)->retired && !store.edgeByUuid(txn, ab.uuid),
                "restoration does not revive removed relationships");
    }
    // A no-op restore of an active endpoint must still retain its relationships
    // when the same request also edits its metadata.
    session.editCurrent(attachment, {b}, [&](const StoreReader&, const std::string&) {
        return std::vector<TransactionOperation>{RestoreObjectOp{b, {}}, SetMetaOp{b, "note", "2"}, RetainRelationshipOp{bc.uuid}};
    });
    {
        auto txn = store.read();
        require(store.edgeByUuid(txn, bc.uuid).has_value(), "no-op restore does not suppress normal retention");
    }
    if (history == HistoryMode::On) {
        ObjectVersion source;
        { auto txn = store.read(); source = store.versionReference(txn, *store.currentObjectRevision(txn, b)); }
        session.editCurrent(attachment, {b}, [&](const StoreReader&, const std::string&) {
            return std::vector<TransactionOperation>{RestoreObjectOp{b, source}};
        });
        { auto txn = store.read(); require(!store.edgeByUuid(txn, bc.uuid), "historical restore removes old observations without retaining them"); }
        bc = makeEdge(b, c);
        VersionSelection expected;
        { auto txn = store.read(); expected = {{b, store.currentVersion(txn, b)}, {c, store.currentVersion(txn, c)}}; }
        { auto txn = store.write(expected); store.addEdge(txn, bc); txn.commit(); }
    }
    // The standalone request helper must also omit an explicitly removed edge.
    auto removal = session.newRequest(attachment);
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        editCurrentRequest(reader, removal, {b}, {SetMetaOp{b, "note", "3"}, RemoveEdgeOp{bc.uuid}});
    });
    require(removal.operations.size() == 2, "no generated retention for an explicitly removed edge");
    session.submit(attachment, removal);
    { auto txn = store.read(); require(!store.edgeByUuid(txn, bc.uuid), "explicit relationship removal succeeds through editCurrentRequest"); }
    // Explicitly asking to retain a missing edge is still an error, not ignored.
    mustFail([&] {
        session.editCurrent(attachment, {b}, [&](const StoreReader&, const std::string&) {
            return std::vector<TransactionOperation>{RetainRelationshipOp{bc.uuid}};
        });
    });
    // Capturing versions must precede callbacks, even if a callback causes a
    // newer head to become visible (as a hosted reader could).
    mustFail([&] {
        session.editCurrent(attachment, {b}, [&](const StoreReader&, const std::string&) {
            auto txn = store.write(); store.setMeta(txn, b, "external", "true"); txn.commit();
            return std::vector<TransactionOperation>{SetMetaOp{b, "stale_callback", "true"}};
        });
    });
    { auto txn = store.read(); require(store.objectByUuid(txn, b)->meta.find("stale_callback") == std::string::npos,
                                      "callback cannot advance the captured edit expectations"); }
}
}

int main() try
{
    for(const auto source:{HistoryMode::On,HistoryMode::Off})for(const auto overlay:{HistoryMode::On,HistoryMode::Off})
        referencedArtifacts(source,overlay);
    editCurrentLifecycle(HistoryMode::On);
    editCurrentLifecycle(HistoryMode::Off);
    TemporaryDirectory directory;
    OpenOptions options; options.author = "session-test";
    // Older formats are rejected before the new journal is queried; the ones
    // vc_object_convert reads (16 to 20) name it.
    for (const int version : {14, 15, 16, 19, 20}) {
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
                const std::string message = error.what();
                rejected = message.find("unsupported database schema") != std::string::npos &&
                           (version < 16 || message.find("vc_object_convert") != std::string::npos);
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
    session.readStore(identity, [&](const StoreReader& reader, const auto&) {
        require(reader.currentObject(a).has_value(), "read-only attachment reads objects");
    });
    // Meta edits of `values` (object -> JSON) expecting `expected`.
    auto metaRequest = [&](const AttachmentIdentity& attachment, const VersionSelection& expected,
                           const std::vector<std::tuple<std::string, std::string, std::string>>& values) {
        auto request = session.newRequest(attachment);
        request.expectedObjects = expected;
        for (const auto& [uuid, key, value] : values) request.operations.push_back(SetMetaOp{uuid, key, value});
        return request;
    };
    mustFail([&] { session.submit(identity, metaRequest(identity, {{a, 1}}, {{a, "note", "0"}})); });
    identity = session.replace(identity, path, options);
    events = session.takeEvents();
    // Two replacements: read-only, then writable again.
    require(events.size() == 4 && events[2].kind == StoreSessionEvent::Kind::Detached &&
            events[3].kind == StoreSessionEvent::Kind::Attached && events[3].attachment == identity,
            "replacement reports the old attachment leaving before the new one joins");
    require(session.writable(identity), "writable reattachment");

    auto heads = [&](const AttachmentIdentity& attachment) {
        VersionSelection result;
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            result[a] = reader.currentObject(a)->version;
            result[b] = reader.currentObject(b)->version;
        });
        return result;
    };
    auto expected = heads(identity);
    const auto edge = util::newUuid();
    {
        auto request = metaRequest(identity, expected, {{a, "note", "1"}, {b, "note", "2"}});
        EdgeRecord link; link.uuid = edge; link.kind = "mixed"; link.aUuid = a; link.bUuid = b;
        request.operations.push_back(AddEdgeOp{edgeRecordJson(link)});
        session.submit(identity, request);
    }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes.size() == 3, "mixed request notification");
    require(events[0].attachment == identity && events[0].journalPosition > 0, "event identity and journal");
    for (const auto& change : events[0].changes)
        require(change.kind == (change.recordUuid == edge ? "mixed" : change.recordUuid == a ? "fiber" : "point_collection"), "changes carry record kinds");
    const auto committed = heads(identity);
    mustFail([&] { session.submit(identity, metaRequest(identity, expected, {{a, "stale", "true"}})); });
    require(heads(identity) == committed && session.takeEvents().empty(), "conflict leaves no changes");
    {
        // A failing operation rolls back the whole request.
        auto request = metaRequest(identity, committed, {{a, "rollback", "true"}});
        request.operations.push_back(RemoveEdgeOp{util::newUuid()});
        mustFail([&] { session.submit(identity, request); });
    }
    require(heads(identity) == committed && session.takeEvents().empty(), "operation rollback");
    {
        auto request = session.newRequest(identity);
        request.expectedObjects = {{a, committed.at(a)}};
        request.operations.push_back(RemoveEdgeOp{edge});
        mustFail([&] { session.submit(identity, request); });
    }
    require(heads(identity) == committed, "all relationship endpoints must be checked");

    auto upper = a;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    const auto receipt = session.submit(identity, metaRequest(identity, {{upper, committed.at(a)}}, {{a, "receipt", "true"}}));
    require(receipt.objects.at(a) == heads(identity).at(a), "expectations accept other UUID spellings");
    session.takeEvents();

    // External changes to a non-fiber object must be observed.
    { auto txn = external.write(); external.setMeta(txn, b, "external", "true"); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes.size() == 1 && events[0].changes[0].recordUuid == b, "external notification");
    { auto txn = external.write(); external.removeEdge(txn, edge); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes[0].kind == "mixed", "deletions keep their record kind");

    // Same expected version submitted concurrently: exactly one request wins.
    expected = heads(identity);
    std::atomic_int successes{0}, conflicts{0};
    auto writer = [&](TransactionRequest request) {
        try {
            session.submit(identity, request);
            ++successes;
        } catch (const std::exception&) { ++conflicts; }
    };
    auto one = std::async(std::launch::async, writer, metaRequest(identity, {{a, expected.at(a)}}, {{a, "race", "1"}}));
    auto two = std::async(std::launch::async, writer, metaRequest(identity, {{a, expected.at(a)}}, {{a, "race", "2"}}));
    one.get(); two.get(); session.finalize();
    require(successes == 1 && conflicts == 1, "serialized checked requests");
    session.takeEvents();

    // Detachment cancels and waits for a worker-local snapshot. The snapshot
    // remains consistent while another handle commits.
    const auto afterDetach = metaRequest(identity, {{a, heads(identity).at(a)}}, {{a, "late", "true"}});
    std::promise<void> entered, changed;
    auto changeReady = changed.get_future();
    auto reader = std::async(std::launch::async, [&] {
        session.readStore(identity, [&](const StoreReader& reader, const auto& cancelled) {
            const auto before = reader.currentObject(b)->version;
            entered.set_value();
            changeReady.wait();
            require(reader.currentObject(b)->version == before, "worker snapshot consistency");
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
    mustFail([&] { session.submit(identity, afterDetach); });
    const auto second = session.attach(path, options);
    require(second.storeUuid == identity.storeUuid && second.generation > identity.generation, "same store invalidates old requests");
    mustFail([&] { session.readStore(identity, [](const StoreReader&, const auto&) {}); });
    // Accepted work finishes before detachment. A request submitted after the
    // transition begins must not slip onto the old attachment.
    std::promise<void> reading, finishReading;
    auto finishReady = finishReading.get_future();
    const auto late = metaRequest(second, {{a, heads(second).at(a)}}, {{a, "drained", "true"}});
    bool drained = false;
    auto pending = std::async(std::launch::async, [&] {
        session.readStore(second, [&](const StoreReader&, const auto&) {
            reading.set_value(); finishReady.wait(); drained = true;
        });
    });
    reading.get_future().wait();
    auto detach = std::async(std::launch::async, [&] { session.detach(second); });
    while (session.current(second)) std::this_thread::yield();
    mustFail([&] { session.submit(second, late); });
    finishReading.set_value(); pending.get(); detach.get();
    require(drained, "accepted read drained");
    { auto txn = external.read(); require(external.objectByUuid(txn, a)->meta.find("drained") == std::string::npos, "late request refused"); }
    auto replacementPath = directory.path / "replacement.vcobj";
    { auto replacement = ObjectStore::create(replacementPath, options); }
    auto replacement = session.attach(replacementPath, options);
    require(replacement.storeUuid != second.storeUuid && replacement.generation > second.generation, "different-store attachment");
    session.takeEvents();
    auto create = [&](const AttachmentIdentity& attachment, const std::vector<ObjectRecord>& records) {
        auto request = session.newRequest(attachment);
        for (const auto& record : records) {
            request.expectedObjects[record.uuid] = std::nullopt;
            request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
        }
        return request;
    };
    auto exists = [&](const AttachmentIdentity& attachment, const std::string& uuid) {
        bool found = false;
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) { found = reader.currentObject(uuid).has_value(); });
        return found;
    };

    // Several stores side by side: a read-only catalog next to a writable store.
    // Each keeps its own access mode and events; detaching one leaves the other.
    auto catalog = session.attach(path, readonly);
    require(session.attachments() == std::vector<AttachmentIdentity>{replacement, catalog}, "attachments in attach order");
    require(session.writable(replacement) && !session.writable(catalog), "access mode is per attachment");
    const auto added = util::newUuid();
    {
        ObjectRecord o; o.uuid = added; o.kind = "surface";
        session.submit(replacement, create(replacement, {o}));
    }
    { auto txn = external.write(); external.setMeta(txn, a, "catalog", "1"); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 3 && events[0].kind == StoreSessionEvent::Kind::Attached && events[0].attachment == catalog,
            "second attachment joins");
    for (const auto& event : std::vector(events.begin() + 1, events.end()))
        require(event.kind == StoreSessionEvent::Kind::Committed && event.changes.size() == 1 &&
                event.changes[0].recordUuid == (event.attachment == catalog ? a : added), "events are tagged by attachment");
    mustFail([&] { session.submit(catalog, metaRequest(catalog, {{a, heads(catalog).at(a)}}, {{a, "read-only", "true"}})); });

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
        nlohmann::json sourceAnnotations, targetAnnotations;
        session.readStore(catalog, [&](const StoreReader& reader, const auto&) {
            const auto revision = reader.currentGeometry(original)->uuid;
            source = reader.readGeometryData(revision);
            if (source.kind != "surface") sourceAnnotations = reader.geometryPointAnnotations(revision, "points");
        });
        session.readStore(replacement, [&](const StoreReader& reader, const auto&) {
            const auto object = reader.currentObject(copied)->record;
            const auto meta = nlohmann::json::parse(object.meta);
            require(meta.at("copied_from").at("object_uuid") == original &&
                    meta.at("copied_from").at("store_uuid") == catalog.storeUuid, "copies record their source");
            if (object.kind == "surface")
                require(object.name == "catalog surface" && object.tags == std::vector<std::string>{"paths"} &&
                        meta.at("coordinates").at("volume_uuid") == "vol", "copies keep name, tags and binding");
            const auto revision = reader.currentGeometry(copied)->uuid;
            target = reader.readGeometryData(revision);
            if (target.kind != "surface") targetAnnotations = reader.geometryPointAnnotations(revision, "points");
        });
        require(source.arrays.size() == target.arrays.size(), "copies keep every array");
        for (size_t i = 0; i < source.arrays.size(); ++i)
            require(source.arrays[i].name == target.arrays[i].name && source.arrays[i].bytes == target.arrays[i].bytes,
                    "copies keep exact geometry");
        require(sourceAnnotations == targetAnnotations, "copies keep point annotations");
    }
    const auto named = session.copyObject(catalog, replacement, surface, "renamed");
    require(session.currentObject(replacement, named)->record.name == "renamed", "a copy can be named");
    {
        // Editing starts from the version on screen.
        const auto pin = pinOf(replacement, *session.currentObject(replacement, named), session.status(replacement).generation);
        require(pin.stored() && pin.reference() == ObjectVersion{replacement.storeUuid, pin.generation, named, pin.version},
                "pin of the current revision");
        session.requireCurrent(pin);
        auto stale = pin;
        ++stale.version;
        bool refused = false;
        try { session.requireCurrent(stale); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "a stale pin cannot be edited");
        StoreSessionEvent committed;
        committed.kind = StoreSessionEvent::Kind::Committed;
        committed.changes.push_back(JournalEntry{});
        committed.changes.back().kind = "surface";
        StoreSessionEvent attached;
        attached.kind = StoreSessionEvent::Kind::Attached;
        require(committed.touches({"fiber", "surface"}) && !committed.touches({"fiber"}) && attached.touches({"fiber"}),
                "events say which record kinds they may have changed");
    }
    // Importing keeps the object's identity: its source version is its first
    // version in the destination, and restoring that version undoes later edits.
    session.importObject(catalog, replacement, surface);
    std::optional<ObjectRevision> imported;
    session.readStore(replacement, [&](const StoreReader& reader, const auto&) {
        const auto history = reader.objectHistory(surface);
        require(history.size() == 1 && history.front().record.name == "catalog surface",
                "an imported object starts at its source version");
        imported = history.front();
    });
    const auto importedGeometry = *imported->geometryRevision;
    {
        auto record = imported->record;
        record.name = "edited";
        auto request = session.newRequest(replacement);
        request.expectedObjects[surface] = imported->version;
        request.operations.push_back(UpdateObjectOp{objectRecordJson(record)});
        request.operations.push_back(PutGeometryOp{surface,
            vc::geometry::encode(vc::geometry::Surface{1, 2, {1, 1}, {{9, 9, 9}, {4, 5, 6}}, {255, 255}}), {}});
        session.submit(replacement, request);
    }
    const auto edited = session.currentObject(replacement, surface);
    require(*edited->geometryRevision != importedGeometry, "edits make new versions");
    {
        auto request = session.newRequest(replacement);
        request.expectedObjects[surface] = edited->version;
        request.operations.push_back(RestoreObjectOp{surface,
            ObjectVersion{replacement.storeUuid, imported->generation, surface, imported->version}});
        session.submit(replacement, request);
    }
    {
        const auto restored = session.currentObject(replacement, surface);
        require(restored->record.name == "catalog surface" && *restored->geometryRevision == importedGeometry,
                "restoring the first version resets to the source");
    }
    bool refused = false;
    try { session.importObject(catalog, replacement, surface); } catch (const std::exception&) { refused = true; }
    require(refused, "an object is imported once");
    session.takeEvents();
    session.detach(catalog);
    require(session.current(replacement) && !session.current(catalog), "detaching one keeps the other");
    require(exists(replacement, added), "remaining attachment still reads");
    // Edits that keep relationships as recorded: an anchored vertex moves and the
    // relationship stays, pinned to the geometry it was made from.
    {
        const auto sheet = util::newUuid(), marker = util::newUuid(), link = util::newUuid();
        {
            ObjectRecord sheetRecord; sheetRecord.uuid = sheet; sheetRecord.kind = "surface";
            ObjectRecord markerRecord; markerRecord.uuid = marker; markerRecord.kind = "point_collection";
            auto request = create(replacement, {sheetRecord, markerRecord});
            request.operations.push_back(PutGeometryOp{sheet,
                vc::geometry::encode(vc::geometry::Surface{2, 2, {1, 1}, {{1, 1, 1}, {2, 1, 1}, {1, 2, 1}, {2, 2, 1}}}), {}});
            request.operations.push_back(PutGeometryOp{marker, vc::geometry::encode(vc::geometry::PointCollection{{0}, {{1, 1, 1}}}), {}});
            EdgeRecord edgeRecord; edgeRecord.uuid = link; edgeRecord.kind = "point-surface"; edgeRecord.aUuid = sheet; edgeRecord.bUuid = marker;
            edgeRecord.aAnchor.points = {{"points", {0, 0}}};
            edgeRecord.bAnchor.points = {{"point_ids", {0}}};
            request.operations.push_back(AddEdgeOp{edgeRecordJson(edgeRecord)});
            session.submit(replacement, request);
        }
        auto edit = [&](bool keepRecorded) {
            auto request = session.newRequest(replacement);
            session.readStore(replacement, [&](const StoreReader& reader, const auto&) {
                request.expectedObjects[sheet] = reader.currentObject(sheet)->version;
                expectRelationshipEndpoints(reader, request);
                require(request.expectedObjectReferences.at(marker) == reader.currentObject(marker)->reference(),
                        "relationship peers have complete snapshot expectations");
            });
            std::vector<uint8_t> moved(12); const float xyz[3] = {1, 1, 5}; std::memcpy(moved.data(), xyz, 12);
            request.operations.push_back(EditGeometryOp{sheet, {{"points", 0, moved}}, {}, keepRecorded});
            return request;
        };
        // An edit that affects the relationship needs a decision about it.
        mustFail([&] { session.submit(replacement, edit(false)); });
        // The typed form: the edit keeps affected relationships as recorded.
        session.submit(replacement, edit(true));
        auto linked = [&] {
            bool found = false;
            session.readStore(replacement, [&](const StoreReader& reader, const auto&) {
                EdgeSelection edges; edges.uuids = std::vector<std::string>{link};
                found = reader.selectEdges(edges).size() == 1;
            });
            return found;
        };
        require(linked(), "relationships survive kept-as-recorded edits");
        session.readStore(replacement, [&](const StoreReader& reader, const auto&) {
            const auto points = reader.readGeometryArray(reader.currentGeometry(sheet)->uuid, "points");
            float z; std::memcpy(&z, points.data() + 8, 4);
            require(z == 5, "the anchored vertex moved");
        });
        // An object expected absent is created by the request.
        const auto created = util::newUuid();
        {
            ObjectRecord record; record.uuid = created; record.kind = "surface";
            session.submit(replacement, create(replacement, {record}));
        }
        require(exists(replacement, created), "requests create objects expected absent");
        // Tags of current records, kept relationships, and selections by tag sets.
        session.tagObjects(replacement, {sheet, created}, {"approved", "reviewed"});
        session.tagObjects(replacement, {created}, {"defective"}, {"approved"});
        session.editRecords(replacement, {created}, [](ObjectRecord& record) { record.name = "renamed"; });
        require(linked(), "tagging keeps relationships as recorded");
        session.readStore(replacement, [&](const StoreReader& reader, const auto&) {
            const auto record = reader.currentObject(created)->record;
            const auto& tags = record.tags;
            require(record.name == "renamed" &&
                        std::find(tags.begin(), tags.end(), "approved") == tags.end() &&
                        std::find(tags.begin(), tags.end(), "defective") != tags.end(),
                    "tag edits add and remove on the current record");
            ObjectSelection selection;
            selection.kind = "surface";
            selection.tagGroups = {{"approved", "defective"}, {"reviewed"}};
            require(reader.selectObjects(selection).size() == 2, "every tag group must be matched by one tag");
            selection.withoutTags = {"defective"};
            const auto kept = reader.selectObjects(selection);
            require(kept.size() == 1 && kept[0].uuid == sheet, "objects with an excluded tag are left out");
        });
        mustFail([&] { session.tagObjects(replacement, {util::newUuid()}, {"approved"}); });
        session.takeEvents();
    }
    session.detach(replacement);
    // Current-only stores use exactly the same checked request/version API.
    // Their notifications invalidate current state rather than invent a journal.
    {
        auto directOptions = options;
        directOptions.history = HistoryMode::Off;
        const auto directPath = directory.path / "direct.vcobj";
        auto direct = ObjectStore::create(directPath, directOptions);
        auto directAttachment = session.attach(directPath, directOptions);
        session.takeEvents();
        const auto uuid = util::newUuid();
        ObjectRecord record; record.uuid = uuid; record.kind = "fiber";
        const auto firstRequest = create(directAttachment, {record});
        const auto first = session.submit(directAttachment, firstRequest);
        require(first.objects.at(uuid) == 1, "history-off record starts at version one");
        require(first.created == std::vector<std::string>{uuid}, "history-off receipt reports created objects");
        const auto head = session.currentObject(directAttachment, uuid);
        require(head && head->version == 1 && head->uuid.empty(), "current version does not invent an archival revision");
        auto notifications = session.takeEvents();
        require(notifications.size() == 1 && notifications.front().revalidate && notifications.front().changes.empty(),
                "history-off commit requests whole-store revalidation without fabricated journal entries");
        const auto pin = pinOf(directAttachment, *head, first.generation);
        session.requireCurrent(pin);
        auto update = metaRequest(directAttachment, {{uuid, 1}}, {{uuid, "a", "1"}, {uuid, "b", "2"}});
        const auto second = session.submit(directAttachment, update);
        require(second.objects.at(uuid) == 2, "several setters advance one record version once");
        const auto repeated = session.submit(directAttachment, firstRequest);
        require(repeated.replayed && repeated.objects.at(uuid) == 1 && repeated.commitSequence == first.commitSequence,
                "history-off request replay returns the original durable receipt after later edits");
        require(session.currentObject(directAttachment, uuid)->version == 2, "retry does not repeat the original mutation");
        auto mismatched = firstRequest;
        mismatched.operations.push_back(SetMetaOp{uuid, "bad_retry", "true"});
        mustFail([&] { session.submit(directAttachment, mismatched); });
        mustFail([&] { session.requireCurrent(pin); });
        mustFail([&] { session.submit(directAttachment, metaRequest(directAttachment, {{uuid, 1}}, {{uuid, "stale", "true"}})); });
        session.takeEvents();
        { auto txn = direct.write(); direct.setMeta(txn, uuid, "external", "true"); txn.commit(); }
        notifications = session.takeEvents();
        require(notifications.size() == 1 && notifications.front().revalidate && notifications.front().touches({"fiber"}),
                "history-off changes from another handle invalidate session consumers");
        // A checked current-only write must not silently edit an undeclared object.
        mustFail([&] {
            auto write = direct.write(VersionSelection{});
            direct.setMeta(write, uuid, "unguarded", "true");
            write.commit();
        });
        const auto peer = util::newUuid();
        ObjectRecord peerRecord; peerRecord.uuid = peer; peerRecord.kind = "fiber";
        session.submit(directAttachment, create(directAttachment, {peerRecord}));
        auto link = session.newRequest(directAttachment);
        link.expectedObjects = {{uuid, 3}, {peer, 1}};
        EdgeRecord edge; edge.uuid = util::newUuid(); edge.kind = "test_link";
        edge.aUuid = uuid; edge.bUuid = peer;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
        link.operations.push_back(AddEdgeOp{edgeRecordJson(edge)});
        const auto linked = session.submit(directAttachment, link);
        require(linked.objects.at(uuid) == 4 && linked.objects.at(peer) == 2 &&
                linked.relationships.at(edge.uuid) == 1,
                "history-off relationship edits number both endpoints and the relationship");
        require(linked.created == std::vector<std::string>{edge.uuid}, "relationship creation is included in the receipt");
        auto missingEndpoint = session.newRequest(directAttachment);
        missingEndpoint.expectedObjects = {{uuid, 4}};
        missingEndpoint.operations.push_back(RemoveEdgeOp{edge.uuid});
        mustFail([&] { session.submit(directAttachment, missingEndpoint); });
        auto remove = session.newRequest(directAttachment);
        remove.expectedObjects = {{uuid, 4}, {peer, 2}};
        remove.operations.push_back(RemoveEdgeOp{edge.uuid});
        const auto removed = session.submit(directAttachment, remove);
        require(removed.objects.at(uuid) == 5 && removed.objects.at(peer) == 3 &&
                removed.relationships.contains(edge.uuid) && !removed.relationships.at(edge.uuid),
                "history-off relationship deletion reports absence and endpoint versions");
        auto noop = session.newRequest(directAttachment);
        noop.expectedObjects = {{uuid, 5}};
        const auto sequence = session.status(directAttachment).commitSequence;
        const auto nochange = session.submit(directAttachment, noop);
        require(!nochange.commitSequence && nochange.objects.at(uuid) == 5 &&
                session.status(directAttachment).commitSequence == sequence,
                "history-off no-op receipt does not advance the store clock");
        session.detach(directAttachment);
    }
    std::cout << "session tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
