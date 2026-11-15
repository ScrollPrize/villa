// Hosted access: the same requests through the public local API and through
// vc_object_server give the same versions, receipts, conflicts, feed and
// historical reads, queries and listings; receipts recover lost replies and
// expired staging; the service owns its store exclusively.
#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/server/Server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

using namespace vc::objects;

namespace {
template<class F> TransactionConflict::Reason conflictOf(F&& callback)
{
    try { callback(); } catch (const TransactionConflict& conflict) { return conflict.reason(); }
    throw std::runtime_error("expected a transaction conflict");
}

ObjectRecord trackRecord(const std::string& uuid)
{
    ObjectRecord record; record.uuid = uuid; record.kind = "track"; record.track.emplace();
    record.track->datasetRef = "hosted"; record.track->externalKey = uuid;
    return record;
}

std::vector<uint8_t> int32Bytes(int32_t value)
{
    std::vector<uint8_t> bytes(4); std::memcpy(bytes.data(), &value, 4); return bytes;
}

bool sameReceipt(const TransactionReceipt& x, const TransactionReceipt& y)
{
    return x.commitSequence == y.commitSequence && x.objects == y.objects && x.relationships == y.relationships &&
           x.created == y.created;
}

// Feed contents without store-specific identities and timestamps.
std::vector<std::string> feedShape(const ChangePage& page)
{
    std::vector<std::string> result;
    for (const auto& commit : page.commits) {
        std::string line = std::to_string(commit.sequence) + " " + commit.author + ":";
        for (const auto& c : commit.changes)
            line += " " + c.recordKind + "/" + c.uuid + "@" + std::to_string(c.version) + (c.deleted ? "x" : "");
        result.push_back(line);
    }
    return result;
}

server::ServerOptions serverOptions()
{
    server::ServerOptions o;
    o.readWorkers = 2;
    o.stagingLifetime = std::chrono::seconds(3600);
    return o;
}

void retainedHistoryOverReferences(HistoryMode sourceMode)
{
    TemporaryDirectory directory;const auto base=directory.path/"base.vcobj",overlay=directory.path/"overlay.vcobj";
    OpenOptions options;options.author="history-policy";options.history=sourceMode;
    const auto id=util::newUuid();
    {auto store=ObjectStore::create(base,options);auto w=store.write();store.upsertObject(w,trackRecord(id));w.commit();
        auto remove=store.write();store.deleteObject(remove,id);store.seal(remove);remove.commit();}
    options.history=sourceMode==HistoryMode::On?HistoryMode::Off:HistoryMode::On;
    {auto collection=ObjectCollection::create(overlay,{base},options);}
    server::Server service(overlay,serverOptions());service.start();
    const auto endpoint="http://127.0.0.1:"+std::to_string(service.port());HostedStore client(endpoint);
    require(client.objectHistory(id,nullptr,true).size()==(sourceMode==HistoryMode::On?2:0),
        "hosted optional history follows source policy, not root policy");
    if(sourceMode==HistoryMode::Off)mustFail([&]{client.objectHistory(id);});
    ObjectStoreSession session;const auto attachment=session.attachHosted(endpoint);
    session.readStore(attachment,[&](const StoreReader& reader,const auto&){
        require(currentVersionOf(reader,id)==(sourceMode==HistoryMode::On?std::optional<int64_t>(2):std::nullopt),
            "hosted retained version helper follows tombstone owner");
    });
}
} // namespace

int main()
try {
    retainedHistoryOverReferences(HistoryMode::Off);
    retainedHistoryOverReferences(HistoryMode::On);
    TemporaryDirectory directory;
    OpenOptions options; options.author = "hosted-test";
    auto local = ObjectStore::create(directory.path / "local.vcobj", options);
    const auto hostedPath = directory.path / "hosted.vcobj";
    { (void)ObjectStore::create(hostedPath, options); }

    // The service cannot start while a standalone writer has the store open.
    {
        auto writer = ObjectStore::open(hostedPath, options);
        mustFail([&] { server::Server refused(hostedPath, serverOptions()); });
    }

    auto service = std::make_unique<server::Server>(hostedPath, serverOptions());
    service->start();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(service->port());
    HostedStore hosted(endpoint);

    // While hosted, direct writable opens fail; read-only inspection still works.
    mustFail([&] { (void)ObjectStore::open(hostedPath, options); });
    {
        OpenOptions readOnly; readOnly.readOnly = true;
        auto inspect = ObjectStore::open(hostedPath, readOnly);
        auto txn = inspect.read();
        require(inspect.status(txn).storeUuid == hosted.status().storeUuid, "read-only open while hosted");
    }

    auto localStatus = [&] { auto txn = local.read(); return local.status(txn); };
    const auto localInitial = localStatus(), hostedInitial = hosted.status();
    require(hostedInitial.commitSequence == 0 && util::isUuid(hostedInitial.generation), "hosted status");

    // One request shape, addressed to either store.
    struct Pair { TransactionRequest local, hosted; };
    auto request = [&](VersionSelection expected, std::vector<TransactionOperation> operations, std::string author = "alice") {
        TransactionRequest r;
        r.author = std::move(author); r.operations = std::move(operations);
        Pair p{r, r};
        p.local.expectedObjects = testNumberedGuards(local, expected);
        p.hosted.expectedObjects = testNumberedGuards(hosted, expected);
        p.local.requestUuid = util::newUuid(); p.local.storeUuid = localInitial.storeUuid; p.local.generation = localInitial.generation;
        p.hosted.requestUuid = util::newUuid(); p.hosted.storeUuid = hostedInitial.storeUuid; p.hosted.generation = hostedInitial.generation;
        return p;
    };
    auto both = [&](const Pair& p, const char* what) {
        const auto l = local.submit(p.local);
        const auto h = hosted.submit(p.hosted);
        require(sameReceipt(l, h), what);
        require(h.storeUuid == hostedInitial.storeUuid && h.requestUuid == p.hosted.requestUuid, "hosted receipt identity");
        return h;
    };

    const auto a = util::newUuid(), b = util::newUuid();
    vc::geometry::Track two; two.points = {{1, 2, 3}, {4, 5, 6}};
    const auto create = request({{a, std::nullopt}, {b, std::nullopt}},
        {CreateObjectOp{objectRecordJson(trackRecord(a))}, PutGeometryOp{a, vc::geometry::encode(two), {}},
         CreateObjectOp{objectRecordJson(trackRecord(b))}});
    const auto created = both(create, "creation parity");
    require(created.commitSequence == 1 && created.objects.at(a) == 1, "hosted creation");

    // Channel names are arbitrary UTF-8 strings, not URL path syntax.
    {
        const auto uuid = util::newUuid();
        vc::geometry::Surface surface{2, 2, {1, 1}, {{1,1,1}, {2,1,1}, {1,2,1}, {2,2,1}}};
        auto data = vc::geometry::encode(surface);
        const std::vector<std::string> names{"ink prediction", "ink/score?x=1#part", "50%+ink", "ink%2Fscore", "墨", ".", ".."};
        for (const auto& name : names) data.arrays.push_back({name, "|u1", {2,2}, {1,2,3,4}});
        ObjectRecord record; record.uuid = uuid; record.kind = "surface";
        // A separate store keeps the following parity/sequence checks unchanged.
        const auto path = directory.path / "channels.vcobj";
        std::string geometry;
        {
            auto db = ObjectStore::create(path, options);
            auto txn = db.write();
            db.upsertObject(txn, record);
            geometry = db.putGeometry(txn, uuid, data);
            txn.commit();
        }
        auto limits = serverOptions(); limits.maxResponseBytes = 2;
        server::Server server(path, limits); server.start();
        HostedStore client("http://127.0.0.1:" + std::to_string(server.port()));
        for (const auto& name : names) {
            try {
                require(client.readGeometryArray(geometry, name) == std::vector<uint8_t>({1,2,3,4}),
                        "hosted channel names survive URL encoding and ranged retries");
            } catch (const std::exception& e) {
                throw std::runtime_error("channel '" + name + "': " + e.what());
            }
        }
    }

    // A lost reply: the identical request is answered by its receipt.
    const auto again = hosted.submit(create.hosted);
    require(again.replayed && sameReceipt(again, created), "hosted retry replays");
    require(hosted.receipt(create.hosted.requestUuid).has_value(), "receipt lookup");
    require(!hosted.receipt(util::newUuid()).has_value(), "unknown receipt");
    require(local.submit(create.local).replayed, "local retry replays");
    { auto lc = create.local; lc.author = "mallory"; mustFail([&] { local.submit(lc); }); }
    {
        auto lw = request({{a, 1}}, {SetMetaOp{a, "x", "1"}}).local; lw.generation = util::newUuid();
        require(conflictOf([&] { local.submit(lw); }) == TransactionConflict::Reason::Generation, "local generation check");
    }
    auto changed = create.hosted; changed.author = "mallory";
    mustFail([&] { hosted.submit(changed); });

    // Conflicts are structured answers on both paths and write nothing.
    const auto stale = request({{a, 7}}, {SetMetaOp{a, "x", "1"}});
    require(conflictOf([&] { local.submit(stale.local); }) == TransactionConflict::Reason::Version, "local stale");
    require(conflictOf([&] { hosted.submit(stale.hosted); }) == TransactionConflict::Reason::Version, "hosted stale");
    auto wrongGeneration = request({{a, 1}}, {SetMetaOp{a, "x", "1"}}).hosted;
    wrongGeneration.generation = util::newUuid();
    require(conflictOf([&] { hosted.submit(wrongGeneration); }) == TransactionConflict::Reason::Generation,
        "hosted generation check");
    {
        const auto omitted = request({}, {SetMetaOp{a, "x", "1"}}); // omitted precondition
        mustFail([&] { local.submit(omitted.local); });
        mustFail([&] { hosted.submit(omitted.hosted); });
    }
    require(hosted.status().commitSequence == 1, "failed hosted requests commit nothing");

    both(request({{a, 1}}, {SetMetaOp{a, "reviewed", "true"}}, "bob"), "metadata parity");
    {
        auto p = request({{a, 2}}, {SetMetaOp{a, "derived", "1"}}); p.local.readDependencies = testNumberedGuards(local, {{b, 1}}); p.hosted.readDependencies = testNumberedGuards(hosted, {{b, 1}});
        both(p, "read dependency parity");
    }
    const auto edge = util::newUuid();
    EdgeRecord link; link.uuid = edge; link.kind = "declared_reference"; link.aUuid = a; link.bUuid = b;
    link.aAnchor.wholeGeometry = link.bAnchor.wholeGeometry = true;
    const auto linked = both(request({{a, 3}, {b, 1}}, {AddEdgeOp{edgeRecordJson(link)}}), "relationship parity");
    require(linked.relationships.at(edge) == 1, "relationship version");
    const auto edited = both(request({{a, 4}, {b, 2}},
        {EditGeometryOp{a, {{"points", 0, int32Bytes(9)}}, {{edge, "retain", "{}", "{}"}}}}), "geometry edit parity");
    require(edited.objects.at(a) == 5, "edit versions object");

    // The feed and historical reads match the local store.
    {
        auto txn = local.read();
        const auto localPage = local.changesSince(txn, {localInitial.storeUuid, localInitial.generation, 0, 0});
        const auto hostedPage = hosted.changesSince({hostedInitial.storeUuid, hostedInitial.generation, 0, 0});
        require(hostedPage.cursorValid && feedShape(localPage) == feedShape(hostedPage), "feed parity");
        require(hostedPage.next.sequence == hosted.status().commitSequence, "feed cursor at head");
        const auto small = hosted.changesSince({hostedInitial.storeUuid, hostedInitial.generation, 0, 0}, 1);
        require(small.more && small.commits.size() == 1, "feed pages");
        require(!hosted.changesSince({util::newUuid(), hostedInitial.generation, 0, 0}).cursorValid, "foreign cursor");

        for (int64_t version = 1; version <= 5; ++version) {
            const auto l = local.objectAtVersion(txn, {localInitial.storeUuid, localInitial.generation, a, version});
            const auto h = hosted.objectAtVersion({hostedInitial.storeUuid, hostedInitial.generation, a, version});
            require(l && h && l->version == h->version && l->record.meta == h->record.meta &&
                l->record.changedBy == h->record.changedBy, "historical object parity");
            require(h->geometryRevision.has_value() == l->geometryRevision.has_value(), "geometry pin parity");
        }
        require(!hosted.objectAtVersion({hostedInitial.storeUuid, util::newUuid(), a, 1}), "other generation never resolves");
        require(!hosted.objectAtVersion({hostedInitial.storeUuid, hostedInitial.generation, a, 99}), "unknown version");

        // Batched reads at versions: same answers as one-at-a-time reads, in
        // order, heads and historical versions alike, split into batches of 2.
        {
            auto references = [&](const StoreStatus& s) {
                std::vector<ObjectVersion> refs;
                for (int64_t version : {5, 1, 3, 99}) refs.push_back({s.storeUuid, s.generation, a, version});
                refs.push_back({s.storeUuid, s.generation, b, 2});
                refs.push_back({s.storeUuid, util::newUuid(), b, 1});
                refs.push_back({util::newUuid(), s.generation, b, 1});
                refs.push_back({s.storeUuid, s.generation, b, 1});
                return refs;
            };
            const auto localRefs = references(localInitial), hostedRefs = references(hostedInitial);
            const auto l = local.objectsAtVersions(txn, localRefs);
            const auto h = hosted.objectsAtVersions(hostedRefs, 2);
            require(l.size() == localRefs.size() && h.size() == hostedRefs.size(), "batched read lengths");
            for (std::size_t i = 0; i < l.size(); ++i) {
                const auto one = local.objectAtVersion(txn, localRefs[i]);
                require(l[i].has_value() == one.has_value() && h[i].has_value() == one.has_value(), "batched presence");
                if (!one) continue;
                require(l[i]->uuid == one->uuid && l[i]->version == one->version && l[i]->generation == one->generation &&
                    l[i]->commitSequence == one->commitSequence && objectRecordJson(l[i]->record) == objectRecordJson(one->record),
                    "batched local equals single read");
                require(h[i]->objectUuid == one->objectUuid && h[i]->version == one->version &&
                    h[i]->record.meta == one->record.meta && h[i]->commitSequence == one->commitSequence,
                    "batched hosted parity");
            }
            require(!l[3] && !l[5] && !l[6] && l[7] && l[7]->version == 1, "batched misses");
            require(l[0]->version == 5 && l[1]->version == 1 && l[1]->record.meta != l[0]->record.meta, "batched order");
            require(hosted.objectsAtVersions({}).empty(), "empty batch");
        }

        StoreStatus asOf;
        const auto head = hosted.currentObjectRevision(a, &asOf);
        require(head && head->version == 5 && asOf.commitSequence == hosted.status().commitSequence, "current head snapshot");
        require(hosted.currentVersion(b) == local.currentVersion(txn, b), "current version parity");
        for (int64_t version : {1, 5}) {
            const auto l = local.objectAtVersion(txn, {localInitial.storeUuid, localInitial.generation, a, version});
            const auto h = hosted.objectAtVersion({hostedInitial.storeUuid, hostedInitial.generation, a, version});
            const auto lg = local.geometryRevision(txn, *l->geometryRevision);
            const auto hg = hosted.geometryRevision(*h->geometryRevision);
            require(lg && hg && lg->contentHash == hg->contentHash && lg->attributes == hg->attributes &&
                lg->arrays.size() == hg->arrays.size(), "geometry descriptor parity");
            for (const auto& array : lg->arrays)
                require(local.readGeometryArray(txn, lg->uuid, array.name) ==
                    hosted.readGeometryArray(hg->uuid, array.name), "geometry bytes parity");
        }
        require(!hosted.geometryRevision(util::newUuid()), "unknown geometry");

        // Bounded annotation reads check each value's packed size before
        // loading it: one large (chunked) value and the per-row defaults.
        {
            auto local = ObjectStore::create(directory.path / "annotations.vcobj", options);
            const auto annotated = util::newUuid();
            vc::geometry::Fiber geometry; geometry.controlPoints = {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}};
            ObjectRecord record; record.uuid = annotated; record.kind = "fiber";
            const auto status = [&] { auto t = local.read(); return local.status(t); }();
            TransactionRequest create;
            create.requestUuid = util::newUuid(); create.storeUuid = status.storeUuid; create.generation = status.generation;
            create.author = "alice"; create.expectedObjects = {{annotated, std::nullopt}};
            create.operations = {CreateObjectOp{objectRecordJson(record)}, PutGeometryOp{annotated, vc::geometry::encode(geometry), {}}};
            local.submit(create);
            const nlohmann::json values = nlohmann::json::array({nlohmann::json{{"note", std::string(5000, 'x')}},
                                                                  nlohmann::json::object(), nlohmann::json{{"tag", 1}}});
            TransactionRequest annotate;
            annotate.requestUuid = util::newUuid(); annotate.storeUuid = status.storeUuid; annotate.generation = status.generation;
            annotate.author = "alice";
            { auto t = local.read(); annotate.expectedObjects[annotated] = local.currentObjectRevision(t, annotated)->reference(); }
            annotate.operations = {UpdateGeometryArraysOp{annotated, vc::geometry::encode(geometry), {{"control_points", values.dump()}}}};
            local.submit(annotate);
            auto txn = local.read();
            const auto revision = local.currentObjectRevision(txn, annotated)->geometryRevision.value();
            const auto full = local.geometryPointAnnotations(txn, revision, "control_points");
            require(full.size() == 3 && full[0].at("note").get<std::string>().size() == 5000, "annotated fixture");
            require(local.geometryPointAnnotationsWithin(txn, revision, "control_points", 1 << 20) == full,
                    "a bounded read within its budget returns every value");
            require(!local.geometryPointAnnotationsWithin(txn, revision, "control_points", 4000),
                    "a chunked value larger than the budget is refused before it is loaded");
            require(!local.geometryPointAnnotationsWithin(txn, revision, "control_points", 8),
                    "the per-row defaults count against the budget");
        }

        // The batched geometry query answers the same descriptors and bytes as
        // the per-revision reads, in request order, with unknown revisions null.
        {
            std::vector<std::string> revisions;
            for (int64_t version : {1, 5})
                revisions.push_back(*hosted.objectAtVersion({hostedInitial.storeUuid, hostedInitial.generation, a, version})->geometryRevision);
            const auto unknown = util::newUuid();
            revisions.push_back(unknown);
            std::vector<std::string> names;
            // Held in a variable: a range-for over a member of the temporary would outlive it before C++23.
            const auto firstRevision = hosted.geometryRevision(revisions[0]);
            for (const auto& array : firstRevision->arrays) names.push_back(array.name);
            names.push_back("no-such-array");
            const auto batch = hosted.geometryBatch(revisions, names, {});
            require(batch && batch->size() == revisions.size(), "geometry batch answers every revision");
            for (size_t i = 0; i < 2; ++i) {
                const auto& entry = (*batch)[i];
                const auto single = hosted.geometryRevision(revisions[i]);
                require(entry.descriptor && entry.descriptor->uuid == single->uuid &&
                        entry.descriptor->contentHash == single->contentHash, "geometry batch descriptor parity");
                require(!entry.arrays.contains("no-such-array"), "geometry batch skips absent arrays");
                for (const auto& array : single->arrays)
                    require(entry.arrays.at(array.name) == hosted.readGeometryArray(single->uuid, array.name),
                            "geometry batch bytes parity");
            }
            require(!(*batch)[2].descriptor && (*batch)[2].uuid == unknown, "geometry batch unknown revision");
        }
    }

    // Queries answer from one server snapshot and match the local store.
    std::vector<std::string> extra;
    {
        const auto sheetUuid = util::newUuid();
        vc::geometry::Surface sheet{70, 40, {1, 1}, {}};
        for (int r = 0; r < 70; ++r)
            for (int c = 0; c < 40; ++c) sheet.points.push_back({float(c), float(r), float(c + 1)});
        ObjectRecord surfaceRecord; surfaceRecord.uuid = sheetUuid; surfaceRecord.kind = "surface";
        both(request({{sheetUuid, std::nullopt}},
            {CreateObjectOp{objectRecordJson(surfaceRecord)}, PutGeometryOp{sheetUuid, vc::geometry::encode(sheet), {}}}),
            "surface parity");
        VersionSelection absent;
        std::vector<TransactionOperation> creations;
        for (int i = 0; i < 5; ++i) {
            extra.push_back(util::newUuid());
            absent[extra.back()] = std::nullopt;
            creations.push_back(CreateObjectOp{objectRecordJson(trackRecord(extra.back()))});
        }
        both(request(absent, creations), "bulk creation parity");

        auto ids = [](const auto& records) {
            std::vector<std::string> result;
            for (const auto& r : records) result.push_back(r.uuid);
            return result;
        };
        auto txn = local.read();
        ObjectSelection tracks; tracks.kind = "track";
        const auto localTracks = local.selectObjects(txn, tracks);
        mustFail([&] { (void)hosted.selectObjects(tracks); }); // one page needs a limit
        StoreStatus asOf;
        const auto all = hosted.selectAllObjects(tracks, std::nullopt, 2, 3, &asOf);
        require(all.size() == 7 && ids(all) == ids(localTracks), "paged selection parity");
        require(asOf.commitSequence == hosted.status().commitSequence, "listing snapshot");
        for (std::size_t i = 0; i < all.size(); ++i) {
            require(all[i].meta == localTracks[i].meta && all[i].version == localTracks[i].version &&
                all[i].changedBy == localTracks[i].changedBy && all[i].track->externalKey == localTracks[i].track->externalKey,
                "record parity");
            // Listings pin what they returned: the published head's numbered version.
            require(all[i].version && all[i].version == localTracks[i].version &&
                all[i].version == local.currentVersion(txn, localTracks[i].uuid) &&
                all[i].versionGeneration == hostedInitial.generation, "selected record versions");
        }
        auto byLength = tracks; byLength.orderBy = "length_desc";
        require(ids(hosted.selectAllObjects(byLength, std::nullopt, 3)) == ids(local.selectObjects(txn, byLength)),
            "offset-paged selection parity");
        for (const auto offset : {1, 3, 7, 10}) {
            auto afterOffset = tracks;
            afterOffset.offset = offset;
            require(ids(hosted.selectAllObjects(afterOffset, std::nullopt, 2, 3, &asOf)) ==
                        ids(local.selectObjects(txn, afterOffset)), "UUID listing with an initial offset");
            require(asOf.commitSequence == local.status(txn).commitSequence, "offset listing snapshot");
        }
        auto page = tracks; page.limit = 3;
        const std::vector<std::string> metaOnly{"meta"};
        const auto projected = hosted.selectObjects(page, metaOnly);
        const auto localProjected = local.selectObjects(txn, page, metaOnly);
        require(ids(projected) == ids(localProjected) && projected[0].meta == localProjected[0].meta &&
            !projected[0].track && projected[0].createdBy.empty(), "projected selection parity");
        require(projected[0].version == localProjected[0].version && projected[0].version.has_value(),
            "projected selections still carry versions");
        {
            const auto none = hosted.selectObjects(page, std::vector<std::string>{});
            require(none[0].version == localProjected[0].version, "uuid-only selections carry versions");
        }
        {
            auto withTransforms = tracks; withTransforms.limit = 1;
            withTransforms.transforms = std::make_shared<VolumeTransforms>(local.volumeTransforms(txn));
            mustFail([&] { (void)hosted.selectObjects(withTransforms); }); // the store's recorded transforms apply
        }

        EdgeSelection links; links.kind = "declared_reference";
        require(ids(hosted.selectAllEdges(links, std::nullopt, 1)) == ids(local.selectEdges(txn, links)) &&
            ids(local.selectEdges(txn, links)) == std::vector<std::string>{edge}, "edge listing parity");
        {
            const auto hostedEdges = hosted.selectAllEdges(links, std::nullopt, 1);
            require(hostedEdges[0].version && hostedEdges[0].version == local.selectEdges(txn, links)[0].version &&
                hostedEdges[0].version == local.currentRelationshipVersion(txn, edge), "selected relationship versions");
        }

        // Graph traversal, counts and tags run on the host.
        auto sorted = [](std::vector<std::string> v) { std::sort(v.begin(), v.end()); return v; };
        for (const std::vector<std::string>& kinds : {std::vector<std::string>{}, {"declared_reference"}, {"parallel"}})
            require(sorted(hosted.traverse({GraphQuery::Kind::Component, a, kinds})) ==
                    sorted(local.componentOf(txn, a, kinds)), "component parity");
        mustFail([&]{local.componentOf(txn,a,{"unsupported_kind"});});
        mustFail([&]{hosted.traverse({GraphQuery::Kind::Component,a,{"unsupported_kind"}});});
        require(sorted(hosted.traverse({GraphQuery::Kind::Component, a, {"declared_reference"}})).size() == 2, "component reaches b");
        require(hosted.traverse({GraphQuery::Kind::Neighbours, b, {"declared_reference"}, std::string("track")}) ==
                local.neighbours(txn, b, std::string("declared_reference"), std::string("track")) &&
                hosted.traverse({GraphQuery::Kind::Neighbours, b, {"declared_reference"}, std::string("track")}) ==
                std::vector<std::string>{a}, "neighbour parity");
        mustFail([&] { (void)hosted.traverse({GraphQuery::Kind::Neighbours, b, {"x", "y"}}); });
        require(hosted.objectCount("track") == local.objectCount(txn, "track") &&
                hosted.objectCount("track", true) == local.objectCount(txn, "track", true) &&
                hosted.objectTags() == local.objectTags(txn) && hosted.objectTags("surface") == local.objectTags(txn, "surface"),
                "count and tag parity");

        std::vector<int64_t> localVersions, hostedVersions;
        for (const auto& r : local.objectHistory(txn, a)) localVersions.push_back(r.version);
        for (const auto& r : hosted.objectHistory(a)) hostedVersions.push_back(r.version);
        require(localVersions == hostedVersions && hostedVersions.back() == 5, "history parity");
        require(hosted.objectHistory(util::newUuid()).empty(), "unknown history");

        for (auto [va, vb] : {std::pair<int64_t, int64_t>{4, 2}, {5, 3}}) {
            const auto l = local.relationshipsAt(txn, std::vector<ObjectVersion>{
                {localInitial.storeUuid, localInitial.generation, a, va}, {localInitial.storeUuid, localInitial.generation, b, vb}});
            const auto h = hosted.relationshipsAt(
                {{hostedInitial.storeUuid, hostedInitial.generation, a, va}, {hostedInitial.storeUuid, hostedInitial.generation, b, vb}});
            require(l.size() == 1 && h.size() == 1 && h[0].relationship.record.uuid == edge &&
                h[0].relationship.version == l[0].relationship.version && h[0].applicable == l[0].applicable &&
                h[0].relationship.aReference.version == (h[0].relationship.record.aUuid == a ? va : vb) &&
                h[0].relationship.bReference.version == (h[0].relationship.record.bUuid == b ? vb : va), "numbered relationship parity");
            require(h[0].relationship.record.versionStoreUuid == hostedInitial.storeUuid &&
                h[0].relationship.record.versionGeneration == h[0].relationship.generation &&
                h[0].relationship.record.version == h[0].relationship.version,
                "relationship wire preserves complete owner identity");
            require(h[0].relationship.aReference.storeUuid == hostedInitial.storeUuid &&
                h[0].relationship.aReference.objectUuid == h[0].relationship.record.aUuid &&
                h[0].relationship.aReference.generation == hostedInitial.generation &&
                h[0].relationship.bReference.storeUuid == hostedInitial.storeUuid &&
                h[0].relationship.bReference.objectUuid == h[0].relationship.record.bUuid &&
                h[0].relationship.bReference.generation == hostedInitial.generation,
                "relationship wire preserves complete selected endpoint references");
        }
        mustFail([&] { (void)hosted.relationshipsAt({{hostedInitial.storeUuid, hostedInitial.generation, a, 99}}); });
        mustFail([&] { (void)hosted.relationshipsAt({{hostedInitial.storeUuid, util::newUuid(), a, 1}}); });

        RegionSlice plane; plane.origin = {0, 0, 31.25}; plane.localBounds = {-1000, -1000, -0.5, 1000, 1000, 0.5};
        RegionOptions surfaces; surfaces.kind = "surface";
        const auto localRegion = local.geometryRegion(txn, plane, surfaces);
        const auto hostedRegion = hosted.geometryRegion(plane, surfaces);
        require(localRegion.complete() && hostedRegion.complete() && localRegion.runs.size() == 1 &&
            hostedRegion.runs.size() == 1 && hostedRegion.runs[0].objectUuid == sheetUuid &&
            hostedRegion.runs[0].points == localRegion.runs[0].points && hostedRegion.runs[0].grid == localRegion.runs[0].grid &&
            hostedRegion.intersectingObjects == localRegion.intersectingObjects, "region parity (exact doubles)");
        require(hostedRegion.runs[0].objectReference == hosted.currentObjectRevision(sheetUuid)->reference() &&
            localRegion.runs[0].objectReference == local.currentObjectRevision(txn, sheetUuid)->reference() &&
            !hostedRegion.runs[0].objectToVolume,
            "region wire retains selected owner state and exact display mapping");
        const auto hostedAnchor = hosted.currentObjectRevision(sheetUuid)->reference();
        const auto hostedSnapshot = hosted.versionsAt(hostedAnchor);
        require(std::find(hostedSnapshot.begin(), hostedSnapshot.end(), hostedAnchor) != hostedSnapshot.end(),
            "hosted export snapshot retains its exact anchor");
        require(!hosted.resolveStoragePath(util::newUuid()), "hosted source path resolver preserves missing-object result");
        surfaces.processingByteBudget = 0;
        require(hosted.geometryRegion(plane, surfaces).stop == RegionStop::ProcessingBudget, "region budget stop");

        SurfaceProximityQuery near; near.point = {10.2, 20, 11.5}; near.tolerance = 2;
        const auto localHits = local.nearestSurfaces(txn, near);
        const auto hostedHits = hosted.nearestSurfaces(near);
        require(localHits.size() == 1 && hostedHits.size() == 1 && hostedHits[0].objectUuid == sheetUuid &&
            hostedHits[0].point == localHits[0].point && hostedHits[0].grid == localHits[0].grid &&
            hostedHits[0].distance == localHits[0].distance, "proximity parity (exact doubles)");
    }

    // Artifacts: the same identities, manifests and bytes locally and hosted;
    // the hosted client never touches the store's files.
    {
        std::string bytes(200000, '\0');
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = char(i * 7 % 253);
        const auto file = directory.path / "artifact.bin";
        std::ofstream(file, std::ios::binary) << bytes;
        const auto hash = hosted.putArtifact(file);
        require(local.putArtifact(file) == hash && hosted.putArtifact(file) == hash, "artifact identity parity");
        require(hosted.artifactLength(hash) == bytes.size() && !hosted.artifactLength(std::string(64, '0')),
                "hosted artifact lengths");
        const auto art = util::newUuid();
        both(request({{art, std::nullopt}}, {CreateObjectOp{objectRecordJson(trackRecord(art))}}), "artifact owner");
        const auto manifest = artifactManifestJson({{"fit/model.bin", hash, bytes.size()}});
        const auto published = both(request({{art, 1}}, {SetMetaOp{art, kArtifactsMetaKey, manifest}}), "manifest parity");
        require(published.objects.at(art) == 2, "manifest versions the object");
        const auto unheld = request({{art, 2}},
            {SetMetaOp{art, kArtifactsMetaKey, artifactManifestJson({{"x.bin", std::string(64, 'e'), 1}})}});
        mustFail([&] { local.submit(unheld.local); });
        mustFail([&] { hosted.submit(unheld.hosted); });
        const auto files = artifactManifest(hosted.currentObjectRevision(art)->record);
        require(files.size() == 1, "hosted manifest");
        const auto fetched = directory.path / "fetched" / files[0].name;
        hosted.fetchArtifact(files[0], fetched);
        std::ifstream in(fetched, std::ios::binary);
        require(std::string(std::istreambuf_iterator<char>(in), {}) == bytes, "fetched artifact bytes");
        const auto part = hosted.readArtifact(hash, 100, 1000);
        require(std::string(part.begin(), part.end()) == bytes.substr(100, 1000), "hosted ranged artifact read");
        auto wrong = files[0]; wrong.length -= 1;
        mustFail([&] { hosted.fetchArtifact(wrong, directory.path / "fetched" / "wrong.bin"); });
        require(!std::filesystem::exists(directory.path / "fetched" / "wrong.bin"), "failed fetch leaves no file");
    }

    // A listing across commits catches up from the change feed: while a writer
    // creates, edits and retires tracks between its pages, every listing is the
    // exact membership and versions as of the feed position it reports.
    std::vector<std::string> churned;
    {
        std::atomic<bool> done{false}, writerFinished{false};
        std::atomic<size_t> createdCount{0};
        std::string writerError;
        std::thread writer([&] {
            try {
                HostedStore client(endpoint);
                // About 200 commits per second: far above interactive editing
                // (a few hundred changes over a whole run), still busy.
                for (int i = 0; !done; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    if (i % 3 == 0) {
                        const auto created = util::newUuid();
                        client.submit(request({{created, std::nullopt}},
                            {CreateObjectOp{objectRecordJson(trackRecord(created))}}).hosted);
                        churned.push_back(created);
                        ++createdCount;
                    } else if (i % 3 == 1) {
                        const auto v = *client.currentVersion(extra[0]);
                        client.submit(request({{extra[0], v}}, {SetMetaOp{extra[0], "tick", std::to_string(i)}}).hosted);
                    } else {
                        const auto& last = churned.back();
                        client.submit(request({{last, 1}}, {RetireObjectOp{last}}).hosted);
                    }
                }
            } catch (const std::exception& e) {
                writerError = e.what();
            }
            writerFinished = true;
        });
        ObjectSelection tracks; tracks.kind = "track";
        std::vector<std::pair<StoreStatus, std::vector<ObjectRecord>>> listings;
        std::string listingError;
        try {
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            const auto deadline = until + std::chrono::seconds(20);
            while (std::chrono::steady_clock::now() < until || listings.size() < 20 || createdCount < 3) {
                if (writerFinished || std::chrono::steady_clock::now() > deadline) break;
                StoreStatus asOf;
                auto listed = hosted.selectAllObjects(tracks, std::vector<std::string>{}, 1, 0, &asOf);
                listings.emplace_back(asOf, std::move(listed));
            }
        } catch (const std::exception& e) {
            listingError = e.what();
        }
        done = true;
        writer.join();
        if (!listingError.empty()) throw std::runtime_error("listing during churn: " + listingError);
        if (!writerError.empty()) throw std::runtime_error("concurrent writer: " + writerError);
        require(churned.size() >= 3, "the writer committed while listings ran");
        // Expected state as of a sequence, from the complete recorded history.
        auto everyTrack = tracks; everyTrack.includeRetired = true;
        const auto all = hosted.selectAllObjects(everyTrack, std::vector<std::string>{"retired"});
        std::map<std::string, std::vector<ObjectRevision>> histories;
        for (const auto& record : all) histories[record.uuid] = hosted.objectHistory(record.uuid);
        int moved = 0;
        for (const auto& [asOf, listed] : listings) {
            std::map<std::string, int64_t> expected;
            for (const auto& [uuid, history] : histories) {
                const ObjectRevision* at = nullptr;
                for (const auto& revision : history)
                    if (revision.commitSequence && *revision.commitSequence <= asOf.commitSequence) at = &revision;
                if (at && !at->deleted && !at->record.retired) expected[uuid] = at->version;
            }
            std::map<std::string, int64_t> got;
            for (const auto& record : listed) got[record.uuid] = record.version.value_or(-1);
            require(got == expected, "a caught-up listing is exact as of its feed position");
            moved += listed.size() != listings.front().second.size();
        }
        std::cout << "catch-up listings during churn: " << listings.size() << " listings, " << churned.size()
                  << " tracks created, membership differed in " << moved << "\n";
        // Quiet store: the 8 tracks from before, plus created tracks not retired.
        const auto active = std::count_if(all.begin(), all.end(), [](const ObjectRecord& r) { return !r.retired; });
        require(all.size() == 8 + churned.size() &&
                int64_t(hosted.selectAllObjects(tracks, std::vector<std::string>{}, 1).size()) == active, "quiet store lists");
    }

    // Inverse edits recheck their mapping in the commit transaction, locally
    // and over the wire, even when a new registration/choice caused the change.
    for (bool remote : {false, true}) {
        ObjectStoreSession session;
        HostedSessionOptions hostedOptions; hostedOptions.author = "mapping-test";
        hostedOptions.pollInterval = std::chrono::hours(1);
        const auto attachment = remote ? session.attachHosted(endpoint, hostedOptions)
                                       : session.attach(local.path(), options);
        const auto uuid = util::newUuid();
        auto create = session.newRequest(attachment);
        create.expectedObjects[uuid] = std::nullopt;
        vc::geometry::Surface grid{1, 2, {1, 1}, {{2, 4, 6}, {8, 10, 12}}};
        create.operations = createSurfaceOperations(uuid, "mapping", {},
            {{"volume_uuid", "mapping-a"}, {"scale_factor", 1}}, {"test", {}, {}}, {}, vc::geometry::encode(grid));
        session.submit(attachment, create);
        const auto original = registerVolumeTransform(session, attachment, "mapping-a", "mapping-b", VolumeAffine::scale(2), "twice");
        auto registrationKey = [&](const std::string& id) {
            std::string key;
            session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
                const auto transforms = reader.volumeTransforms();
                for (const auto& registration : transforms.graph.registrations())
                    if (registration.id == id) key = registration.id + "@" + registration.revision;
            });
            require(!key.empty(), "registration has a revision-qualified path key");
            return key;
        };
        auto edit = [&] {
            auto request = session.newRequest(attachment);
            session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
                const auto head = reader.currentObject(uuid);
                request.expectedObjects[uuid] = head->reference();
                const auto view = GeometrySnapshot(reader).surface(uuid, *head->geometryRevision).viewTransformedAt("mapping-b");
                auto desired = view.readTile(0, 0, 1, 1).points.front();
                desired[0] += 2;
                request.operations = {view.checkMappingOperation(), EditGeometryOp{uuid,
                    view.surfaceEdits(0, 0, 1, 1, {{float(desired[0]), float(desired[1]), float(desired[2])}}), {}, true}};
            });
            TransactionPayloads payloads;
            const auto json = transactionRequestToJson(request, &payloads);
            auto decoded = transactionRequestFromJson(json, payloads);
            require(transactionRequestDigest(decoded) == transactionRequestDigest(request), "mapping guard wire round trip");
            return decoded;
        };
        auto allowed = edit();
        registerVolumeTransform(session, attachment, "unrelated-a", "unrelated-b", VolumeAffine::scale(4), "unrelated");
        const auto saved = session.submit(attachment, allowed);
        require(saved.objects.at(uuid) == 2, "unrelated registration does not block inverse edits");
        auto blocked = edit();
        const auto competing = registerVolumeTransform(session, attachment, "mapping-a", "mapping-b", VolumeAffine::scale(3), "thrice");
        const auto unchanged = session.currentObject(attachment, uuid)->geometryRevision;
        auto refused = [&](const TransactionRequest& request) {
            const auto before = session.status(attachment).commitSequence;
            require(conflictOf([&] { session.submit(attachment, request); }) == TransactionConflict::Reason::ReadDependency,
                    "changed mapping conflicts before inverse edits");
            require(session.status(attachment).commitSequence == before &&
                    session.currentObject(attachment, uuid)->geometryRevision == unchanged &&
                    !session.receipt(attachment, request.requestUuid), "mapping conflict rolls back the whole request");
        };
        refused(blocked); // A newly inserted registration made the route ambiguous.
        chooseVolumeTransform(session, attachment, "mapping-b", "mapping-a", {"inverse:" + registrationKey(original)});
        blocked = edit();
        chooseVolumeTransform(session, attachment, "mapping-a", "mapping-b", {registrationKey(competing)});
        refused(blocked); // A new forward choice overrides the existing reverse choice.
        blocked = edit();
        chooseVolumeTransform(session, attachment, "mapping-a", "mapping-b", {registrationKey(original)});
        refused(blocked); // Changing an existing choice is guarded too.
    }
    // Expired staging: a new request cannot run without its bytes, but the
    // identical retry of a committed one is answered without them.
    const auto later = request({{a, 5}, {b, 3}}, {EditGeometryOp{a, {{"points", 0, int32Bytes(11)}}, {{edge, "retain", "{}", "{}"}}}});
    const auto committed = hosted.submit(later.hosted);
    const auto unstaged = request({{a, 6}, {b, 4}}, {EditGeometryOp{a, {{"points", 0, int32Bytes(12)}}, {{edge, "retain", "{}", "{}"}}}});
    service.reset();
    {
        auto o = serverOptions(); o.stagingLifetime = std::chrono::seconds(0);
        server::Server forgetful(hostedPath, o);
        forgetful.start();
        HostedStore client("http://127.0.0.1:" + std::to_string(forgetful.port()));
        const auto replay = client.submit(later.hosted);
        require(replay.replayed && sameReceipt(replay, committed), "receipt recovery without staged bytes");
        mustFail([&] { client.submit(unstaged.hosted); });
        require(client.status().commitSequence == *committed.commitSequence, "unstaged request did not commit");
    }

    // The shared JSON path behaves the same in process.
    {
        TransactionPayloads payloads;
        const auto local1 = later.local;
        const auto text = transactionRequestToJson(local1, &payloads);
        bool missing = false;
        try { (void)local.submit(text, {}); } catch (const MissingPayloads& e) { missing = e.hashes().size() == 1; }
        require(missing, "missing payloads named");
        const auto first = local.submit(text, [&](const std::string& h) { return std::optional(payloads.at(h)); });
        const auto replay = local.submit(text, {});
        require(replay.replayed && sameReceipt(first, replay), "local receipt without payloads");
    }

    // Once the service stops, standalone writers may open the store again.
    { auto writer = ObjectStore::open(hostedPath, options); }
    std::cout << "object_store_hosted: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object_store_hosted: " << error.what() << "\n";
    return 1;
}
