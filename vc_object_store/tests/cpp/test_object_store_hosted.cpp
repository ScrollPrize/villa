// Hosted access: the same requests through the public local API and through
// vc_object_server give the same versions, receipts, conflicts, feed and
// historical reads, queries and listings; receipts recover lost replies and
// expired staging; the service owns its store exclusively.
#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/server/Server.hpp"

#include <atomic>
#include <cstring>
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
} // namespace

int main()
try {
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
        r.author = std::move(author); r.expectedObjects = std::move(expected); r.operations = std::move(operations);
        Pair p{r, r};
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
        auto p = request({{a, 2}}, {SetMetaOp{a, "derived", "1"}}); p.local.readDependencies = p.hosted.readDependencies = {{b, 1}};
        both(p, "read dependency parity");
    }
    const auto edge = util::newUuid();
    EdgeRecord link; link.uuid = edge; link.kind = "test_link"; link.aUuid = a; link.bUuid = b;
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
            require(all[i].meta == localTracks[i].meta && all[i].generation == localTracks[i].generation &&
                all[i].changedBy == localTracks[i].changedBy && all[i].track->externalKey == localTracks[i].track->externalKey,
                "record parity");
        }
        auto byLength = tracks; byLength.orderBy = "length_desc";
        require(ids(hosted.selectAllObjects(byLength, std::nullopt, 3)) == ids(local.selectObjects(txn, byLength)),
            "offset-paged selection parity");
        auto page = tracks; page.limit = 3;
        const std::vector<std::string> metaOnly{"meta"};
        const auto projected = hosted.selectObjects(page, metaOnly);
        const auto localProjected = local.selectObjects(txn, page, metaOnly);
        require(ids(projected) == ids(localProjected) && projected[0].meta == localProjected[0].meta &&
            !projected[0].track && projected[0].createdBy.empty(), "projected selection parity");
        {
            auto withTransforms = tracks; withTransforms.limit = 1;
            withTransforms.transforms = std::make_shared<VolumeTransforms>(local.volumeTransforms(txn));
            mustFail([&] { (void)hosted.selectObjects(withTransforms); }); // the store's recorded transforms apply
        }

        EdgeSelection links; links.kind = "test_link";
        require(ids(hosted.selectAllEdges(links, std::nullopt, 1)) == ids(local.selectEdges(txn, links)) &&
            ids(local.selectEdges(txn, links)) == std::vector<std::string>{edge}, "edge listing parity");

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
                h[0].relationship.aVersion == va && h[0].relationship.bVersion == vb, "numbered relationship parity");
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
        surfaces.processingByteBudget = 0;
        require(hosted.geometryRegion(plane, surfaces).stop == RegionStop::ProcessingBudget, "region budget stop");

        SurfaceProximityQuery near; near.point = {10.2, 20, 11.5}; near.tolerance = 2;
        const auto localHits = local.nearestSurfaces(txn, near);
        const auto hostedHits = hosted.nearestSurfaces(near);
        require(localHits.size() == 1 && hostedHits.size() == 1 && hostedHits[0].objectUuid == sheetUuid &&
            hostedHits[0].point == localHits[0].point && hostedHits[0].grid == localHits[0].grid &&
            hostedHits[0].distance == localHits[0].distance, "proximity parity (exact doubles)");
    }

    // A listing never combines pages from different snapshots: with commits
    // landing between its pages it restarts, then gives up.
    {
        std::atomic<bool> done{false};
        std::string writerError;
        std::thread writer([&] {
            try {
                HostedStore client(endpoint);
                for (int64_t version = 1; !done; ++version) {
                    auto r = request({{extra[0], version}}, {SetMetaOp{extra[0], "tick", std::to_string(version)}}).hosted;
                    client.submit(r);
                }
            } catch (const std::exception& e) {
                writerError = e.what();
            }
        });
        ObjectSelection tracks; tracks.kind = "track";
        bool busy = false;
        for (int attempt = 0; attempt < 50 && !busy; ++attempt) {
            try { (void)hosted.selectAllObjects(tracks, std::vector<std::string>{}, 1, 0); } catch (const StoreBusy&) { busy = true; }
        }
        done = true;
        writer.join();
        if (!writerError.empty()) throw std::runtime_error("concurrent writer: " + writerError);
        require(busy, "a listing across commits gives up instead of mixing snapshots");
        require(hosted.selectAllObjects(tracks, std::vector<std::string>{}, 1).size() == 7, "quiet store lists");
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
        const auto local1 = request({{a, 5}, {b, 3}}, {EditGeometryOp{a, {{"points", 0, int32Bytes(11)}}, {{edge, "retain", "{}", "{}"}}}}).local;
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
