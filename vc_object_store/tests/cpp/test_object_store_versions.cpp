// Numbered history, checked typed transactions, receipts, the change feed,
// history generations and host-local ownership.
#include "TestSupport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"

#include <cstring>
#include <iostream>

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
    record.track->datasetRef = "versions"; record.track->externalKey = uuid;
    return record;
}

std::vector<uint8_t> int32Bytes(int32_t value)
{
    std::vector<uint8_t> bytes(4); std::memcpy(bytes.data(), &value, 4); return bytes;
}
}

int main()
try {
    TemporaryDirectory directory;
    const auto path = directory.path / "versions.vcobj";
    OpenOptions options; options.author = "versions-test";
    auto db = ObjectStore::create(path, options);
    auto status = [&] { auto txn = db.read(); return db.status(txn); };
    const auto initial = status();
    require(initial.commitSequence == 0 && initial.feedStart == 0 && util::isUuid(initial.generation), "initial status");

    auto request = [&](VersionSelection expected, std::vector<TransactionOperation> operations, std::string author = "alice") {
        TransactionRequest r;
        r.requestUuid = util::newUuid(); r.storeUuid = initial.storeUuid; r.generation = status().generation;
        r.author = std::move(author); r.expectedObjects = std::move(expected); r.operations = std::move(operations);
        return r;
    };

    // Creation expects absence; geometry is created in the same transaction.
    const auto a = util::newUuid(), b = util::newUuid();
    vc::geometry::Track two; two.points = {{1, 2, 3}, {4, 5, 6}};
    auto create = request({{a, std::nullopt}, {b, std::nullopt}},
        {CreateObjectOp{objectRecordJson(trackRecord(a))}, PutGeometryOp{a, vc::geometry::encode(two), {}},
         CreateObjectOp{objectRecordJson(trackRecord(b))}});
    // The request survives its canonical encoding unchanged.
    TransactionPayloads payloads;
    const auto encoded = transactionRequestToJson(create, &payloads);
    require(payloads.size() == 1, "geometry travels as one payload");
    const auto decoded = transactionRequestFromJson(encoded, payloads);
    require(transactionRequestDigest(decoded) == transactionRequestDigest(create), "digest survives round trip");
    require(transactionRequestToJson(decoded) == encoded, "canonical encoding is stable");
    mustFail([&] { (void)transactionRequestFromJson(encoded, {}); });
    const auto created = db.submit(decoded);
    require(created.commitSequence == 1 && !created.replayed, "first commit");
    require(created.objects.at(a) == 1 && created.objects.at(b) == 1, "versions start at 1");
    require(created.created == std::vector<std::string>({a, b}) || created.created == std::vector<std::string>({b, a}),
        "created records");

    // A lost reply: the identical retry returns the saved receipt, even though
    // its absence preconditions no longer hold.
    const auto retry = db.submit(create);
    require(retry.replayed && retry.commitSequence == 1 && retry.objects == created.objects, "receipt recovery");
    require(status().commitSequence == 1, "retry does not commit");
    auto changed = create; changed.author = "mallory";
    mustFail([&] { db.submit(changed); });

    // Stale and missing preconditions write nothing.
    require(conflictOf([&] { db.submit(request({{a, 7}}, {SetMetaOp{a, "x", "1"}})); }) ==
        TransactionConflict::Reason::Version, "stale version");
    mustFail([&] { db.submit(request({}, {SetMetaOp{a, "x", "1"}})); });
    require(status().commitSequence == 1, "failed requests commit nothing");

    // Per-request attribution on one long-lived handle.
    const auto meta = db.submit(request({{a, 1}}, {SetMetaOp{a, "reviewed", "true"}}, "bob"));
    require(meta.commitSequence == 2 && meta.objects.at(a) == 2, "metadata edit versions once");
    {
        auto txn = db.read();
        require(db.objectByUuid(txn, a)->changedBy == "bob", "request author stamps the record");
        const auto page = db.changesSince(txn, {initial.storeUuid, initial.generation, 1, 0});
        require(page.cursorValid && page.commits.size() == 1 && page.commits[0].author == "bob", "commit author");
        require(page.commits[0].changes.size() == 1 && page.commits[0].changes[0].uuid == a &&
            page.commits[0].changes[0].version == 2, "feed change");
        require(db.objectByUuid(txn, b)->changedBy == "alice", "earlier author retained");
    }
    require(db.author() == "versions-test", "handle author unchanged");

    // A successful no-op has a receipt but advances nothing.
    const auto noop = db.submit(request({{a, 2}}, {}));
    require(!noop.commitSequence && noop.objects.at(a) == 2 && status().commitSequence == 2, "no-op");

    // Read dependencies and query guards: checking never versions a record.
    require(conflictOf([&] {
        auto r = request({{a, 2}}, {SetMetaOp{a, "derived", "1"}}); r.readDependencies = {{b, 7}}; db.submit(r);
    }) == TransactionConflict::Reason::ReadDependency, "changed read dependency");
    {
        auto r = request({{a, 2}}, {SetMetaOp{a, "derived", "1"}}); r.readDependencies = {{b, 1}};
        const auto receipt = db.submit(r);
        require(receipt.objects.at(a) == 3 && !receipt.objects.contains(b), "read dependency only checked");
        auto txn = db.read();
        require(db.currentVersion(txn, b) == 1, "dependency kept its version");
    }
    require(conflictOf([&] {
        auto r = request({{a, 3}}, {SetMetaOp{a, "derived", "2"}}); r.expectedCommitSequence = 1; db.submit(r);
    }) == TransactionConflict::Reason::CommitSequence, "query guard");

    // A relationship versions itself and both endpoints in one commit.
    const auto edge = util::newUuid();
    EdgeRecord link; link.uuid = edge; link.kind = "declared_reference"; link.aUuid = a; link.bUuid = b;
    link.aAnchor.wholeGeometry = link.bAnchor.wholeGeometry = true;
    const auto linked = db.submit(request({{a, 3}, {b, 1}}, {AddEdgeOp{edgeRecordJson(link)}}));
    require(linked.objects.at(a) == 4 && linked.objects.at(b) == 2 && linked.relationships.at(edge) == 1, "relationship commit");
    {
        auto txn = db.read();
        const auto rel = db.relationshipAtVersion(txn, {initial.storeUuid, initial.generation, edge, 1});
        require(rel && rel->version == 1 && rel->record.uuid == edge, "relationship reference");
        require((rel->aVersion == 4 && rel->bVersion == 2) || (rel->aVersion == 2 && rel->bVersion == 4), "endpoint pins");
        require(db.currentRelationshipVersion(txn, edge) == 1, "relationship head");
    }

    // Sparse geometry edit through a typed command; the relationship decision
    // takes its observed state from the checked snapshot.
    {
        auto r = request({{a, 4}, {b, 2}}, {EditGeometryOp{a, {{"points", 0, int32Bytes(9)}}, {{edge, "retain", "{}", "{}"}}}});
        r.expectedRelationships = {{edge, 1}};
        const auto receipt = db.submit(r);
        // The retained relationship's new state versions both endpoints once.
        require(receipt.objects.at(a) == 5 && receipt.objects.at(b) == 3 && receipt.relationships.at(edge) == 2,
            "geometry edit versions");
        auto txn = db.read();
        const auto old = db.objectAtVersion(txn, {initial.storeUuid, initial.generation, a, 1});
        const auto now = db.currentObjectRevision(txn, a);
        require(old && now && old->geometryRevision != now->geometryRevision, "historical geometry retained");
        require(db.versionReference(txn, *now) == ObjectVersion{initial.storeUuid, initial.generation, a, 5}, "reference");
    }

    // Removal and deletion are numbered tombstones; recreation continues the
    // numbering, so an old version can never reappear as "new".
    const auto removed = db.submit(request({{a, 5}, {b, 3}}, {RemoveEdgeOp{edge}}));
    require(removed.objects.at(b) == 4 && removed.relationships.at(edge) == std::nullopt, "removal versions");
    const auto deleted = db.submit(request({{b, 4}}, {DeleteObjectOp{b}}));
    require(!deleted.objects.at(b).has_value(), "deleted object has no live version");
    {
        auto txn = db.read();
        require(!db.currentRelationshipVersion(txn, edge) && !db.currentVersion(txn, b), "tombstones");
        require(db.objectHistory(txn, b).back().version == 5 && db.objectHistory(txn, b).back().deleted, "deletion version");
    }
    require(conflictOf([&] { db.submit(request({{b, 4}}, {CreateObjectOp{objectRecordJson(trackRecord(b))}})); }) ==
        TransactionConflict::Reason::Version, "stale base after deletion");
    const auto recreated = db.submit(request({{b, std::nullopt}}, {CreateObjectOp{objectRecordJson(trackRecord(b))}}));
    require(recreated.objects.at(b) == 6 && recreated.created == std::vector<std::string>{b}, "recreation continues numbering");

    // Legacy checked writes produce the same numbered receipts.
    {
        VersionSelection expected;
        { auto txn = db.read(); expected[a] = db.currentVersion(txn, a); }
        auto txn = db.write(expected);
        db.setMeta(txn, a, "legacy", "true");
        txn.commit();
        require(txn.receipt() && txn.receipt()->objects.at(a) == 7, "legacy receipt");
    }

    // Feed pagination keeps commit boundaries and continues oversized commits.
    {
        std::vector<TransactionOperation> operations;
        VersionSelection expected;
        for (int i = 0; i < 5; ++i) {
            const auto uuid = util::newUuid();
            expected[uuid] = std::nullopt;
            operations.push_back(CreateObjectOp{objectRecordJson(trackRecord(uuid))});
        }
        const auto big = db.submit(request(expected, operations));
        auto txn = db.read();
        FeedCursor cursor{initial.storeUuid, initial.generation, *big.commitSequence - 1, 0};
        auto page = db.changesSince(txn, cursor, 2);
        require(page.commits.size() == 1 && !page.commits[0].complete && page.commits[0].changes.size() == 2 && page.more,
            "partial commit page");
        size_t total = 2;
        while (page.more) {
            page = db.changesSince(txn, page.next, 2);
            for (const auto& commit : page.commits) total += commit.changes.size();
        }
        require(total == 5 && page.next.sequence == *big.commitSequence && page.next.offset == 0, "continuation");
        const auto whole = db.changesSince(txn, {initial.storeUuid, initial.generation, 0, 0}, 1000);
        require(whole.commits.size() == size_t(*big.commitSequence) && !whole.more, "whole feed");
        require(!db.changesSince(txn, {initial.storeUuid, initial.generation, 1000, 0}).cursorValid, "future cursor");
        require(!db.changesSince(txn, {initial.storeUuid, util::newUuid(), 1, 0}).cursorValid, "foreign generation");
    }

    // Restore tooling: a new generation for new writes; recorded references
    // keep their generation, and never resolve under another one.
    const auto before = status();
    const auto generation = db.startHistoryGeneration("test restore");
    require(status().generation == generation && status().commitSequence == before.commitSequence + 1, "rotation is a logical provisioning commit");
    {
        auto stale = request({{a, 7}}, {SetMetaOp{a, "after", "1"}});
        stale.generation = before.generation;
        require(conflictOf([&] { db.submit(stale); }) == TransactionConflict::Reason::Generation, "pre-restore request");
    }
    const auto after = db.submit(request({{a, 7}}, {SetMetaOp{a, "after", "1"}}));
    require(after.generation == generation && after.objects.at(a) == 8, "numbering continues in the new generation");
    {
        auto txn = db.read();
        require(db.objectAtVersion(txn, {initial.storeUuid, before.generation, a, 7}).has_value(), "old reference");
        require(!db.objectAtVersion(txn, {initial.storeUuid, generation, a, 7}).has_value(), "no cross-generation match");
        require(!db.objectAtVersion(txn, {initial.storeUuid, before.generation, a, 8}).has_value(), "new version, new generation");
        require(!db.changesSince(txn, {initial.storeUuid, before.generation, before.commitSequence, 0}).cursorValid,
            "pre-restore cursor revalidates");
    }

    // Selections pin what they return: after deletion, recreation and a
    // generation change, every selected record carries its head's version
    // and generation. Rows read inside a write carry none (maybe unpublished).
    {
        auto txn = db.read();
        const auto selected = db.selectObjects(txn, ObjectSelection{});
        require(!selected.empty(), "selected objects");
        for (const auto& record : selected) {
            const auto head = db.currentObjectRevision(txn, record.uuid);
            require(record.version == head->version && record.versionGeneration == head->generation,
                    "selected version is the head version");
        }
    }
    {
        VersionSelection before;
        {
            auto txn = db.read();
            for (const auto& record : db.selectObjects(txn, ObjectSelection{})) before[record.uuid] = record.version;
        }
        auto txn = db.write();
        db.setMeta(txn, a, "unpublished", "1");
        for (const auto& record : db.selectObjects(txn, ObjectSelection{}))
            require(record.version == *before.at(record.uuid) + (record.uuid == a ? 1 : 0),
                    "inline versions reflect pending edits exactly once inside a write");
    }

    // Catalog registrations become versioned records; once recorded they
    // resolve from the store, never from a client's newer catalog copy.
    {
        vcr::catalog::Registration scan{"scan-a-to-b", "catalog-v1", "volume-a", "volume-b", vcr::catalog::Affine::scale(2), "scan"};
        const auto uuid = catalogTransformUuid(scan.id);
        const auto recorded = db.submit(request({{uuid, std::nullopt}}, {RecordCatalogTransformsOp{{scan}}}));
        require(recorded.objects.at(uuid) == 1, "registration recorded");
        const auto unchanged = db.submit(request({{uuid, 1}}, {RecordCatalogTransformsOp{{scan}}}));
        require(!unchanged.commitSequence, "unchanged registration publishes nothing");
        auto newer = std::make_shared<vcr::catalog::RegistrationGraph>();
        auto changedScan = scan; changedScan.revision = "catalog-v2"; changedScan.matrix = vcr::catalog::Affine::scale(3);
        newer->add(changedScan);
        OpenOptions clientOptions; clientOptions.readOnly = true; clientOptions.catalogTransforms = newer;
        auto client = ObjectStore::open(path, clientOptions);
        auto txn = client.read();
        const auto resolution = client.volumeTransforms(txn).resolve("volume-a", "volume-b");
        require(resolution.resolved().forward(0, 0) == 2, "recorded coefficients win over the client catalog");
    }

    // Geometry requests that keep affected relationships as recorded, surface
    // channels, created surfaces, and expectations read through a reader.
    {
        auto txn = db.read();
        const auto dependencies = transformDependencies(LocalStoreReader(db, txn));
        require(!dependencies.empty() && std::all_of(dependencies.begin(), dependencies.end(),
                [](const auto& d) { return d.second && *d.second > 0; }), "transform dependencies");
    }
    {
        auto edits = ObjectStore::create(directory.path / "edits.vcobj", options);
        const auto at = [&] { auto txn = edits.read(); return edits.status(txn); };
        auto make = [&](VersionSelection expected, std::vector<TransactionOperation> operations) {
            TransactionRequest r;
            r.requestUuid = util::newUuid(); r.storeUuid = at().storeUuid; r.generation = at().generation;
            r.author = "carol"; r.expectedObjects = std::move(expected); r.operations = std::move(operations);
            return r;
        };
        const auto surface = util::newUuid(), other = util::newUuid(), relation = util::newUuid();
        vc::geometry::Surface grid{2, 2, {1, 1}, {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}}};
        auto operations = createSurfaceOperations(surface, "made", {"t"}, {{"volume_uuid", "20260411134726"}, {"scale_factor", 1}},
                                                  {"versions-test", {}, {}}, {{"format", "tifxyz"}, {"note", 1}},
                                                  vc::geometry::encode(grid));
        operations.push_back(CreateObjectOp{objectRecordJson(trackRecord(other))});
        edits.submit(make({{surface, std::nullopt}, {other, std::nullopt}}, operations));
        EdgeRecord link; link.uuid = relation; link.kind = "declared_reference"; link.aUuid = surface; link.bUuid = other;
        link.aAnchor.wholeGeometry = link.bAnchor.wholeGeometry = true;
        edits.submit(make({{surface, 1}, {other, 1}}, {AddEdgeOp{edgeRecordJson(link)}}));
        {
            auto txn = edits.read();
            const auto record = edits.objectByUuid(txn, surface);
            require(record->name == "made" && record->tags == std::vector<std::string>{"t"} &&
                    nlohmann::json::parse(record->meta).value("note", 0) == 1 &&
                    !nlohmann::json::parse(record->meta).contains("format"), "created surface record");
        }
        std::vector<uint8_t> moved(12);
        const float xyz[3] = {5, 0, 1};
        std::memcpy(moved.data(), xyz, 12);
        // Without a decision the affected relationship refuses the edit.
        mustFail([&] { edits.submit(make({{surface, 2}, {other, 2}}, {EditGeometryOp{surface, {{"points", 0, moved}}, {}}})); });
        VersionSelection expected{{surface, 2}};
        {
            auto txn = edits.read();
            expectRelationshipEndpoints(LocalStoreReader(edits, txn), expected);
        }
        require(expected == VersionSelection{{surface, 2}, {other, 2}}, "relationship endpoints expected");
        auto keep = make(expected, {EditGeometryOp{surface, {{"points", 0, moved}}, {}, true}});
        const auto plain = transactionRequestToJson(make(expected, {EditGeometryOp{surface, {{"points", 0, moved}}, {}}}));
        TransactionPayloads payloads;
        const auto encoded = transactionRequestToJson(keep, &payloads);
        require(plain.find("keep_affected_recorded") == std::string::npos &&
                encoded.find("\"keep_affected_recorded\":true") != std::string::npos, "keep flag encoded only when set");
        require(transactionRequestDigest(transactionRequestFromJson(encoded, payloads)) == transactionRequestDigest(keep),
                "keep flag survives the round trip");
        const auto kept = edits.submit(keep);
        // Kept as recorded: the relationship and its other end do not change.
        require(kept.objects.at(surface) == 3 && kept.objects.at(other) == 2 && !kept.relationships.contains(relation),
                "affected relationship kept as recorded");
        {
            auto txn = edits.read();
            require(edits.currentRelationshipVersion(txn, relation) == 1, "relationship unchanged");
        }
        // Channels through a typed operation, deciding the relationship the same way.
        const GeometryArray ink{"ink", "|u1", {2, 2}, {1, 2, 3, 4}};
        auto channels = make({{surface, 3}, {other, 2}}, {SetSurfaceChannelsOp{surface, {ink}, {}, {}, true}});
        payloads.clear();
        require(transactionRequestDigest(transactionRequestFromJson(transactionRequestToJson(channels, &payloads), payloads)) ==
                transactionRequestDigest(channels), "channel operation round trip");
        require(transactionJsonPayloadHashes(transactionRequestToJson(channels)).size() == 1, "channel payloads listed");
        edits.submit(channels);
        {
            auto txn = edits.read();
            const auto head = edits.currentObjectRevision(txn, surface);
            require(head->version == 4 && edits.readGeometryArray(txn, *head->geometryRevision, "ink") == ink.bytes, "channel written");
        }

        // Whole-array updates: sparse, identity-keeping, annotations replaced,
        // affected relationships retained or kept as recorded.
        const auto collection = util::newUuid(), marker = util::newUuid(), anchored = util::newUuid();
        ObjectRecord pc; pc.uuid = collection; pc.kind = "point_collection";
        pc.meta = R"({"source":{"point_attributes":[{"n":0},{"n":1}]}})";
        vc::geometry::PointCollection points{{0, 1}, {{1, 1, 1}, {2, 2, 2}}};
        edits.submit(make({{collection, std::nullopt}, {marker, std::nullopt}},
            {CreateObjectOp{objectRecordJson(pc)}, PutGeometryOp{collection, vc::geometry::encode(points), {}, false},
             CreateObjectOp{objectRecordJson(trackRecord(marker))}}));
        EdgeRecord pointLink; pointLink.uuid = anchored; pointLink.kind = "declared_reference"; pointLink.aUuid = collection;
        pointLink.bUuid = marker; pointLink.aAnchor.points = {{"point_ids", {0}}}; pointLink.bAnchor.wholeGeometry = true;
        edits.submit(make({{collection, 1}, {marker, 1}}, {AddEdgeOp{edgeRecordJson(pointLink)}}));
        points.points[0] = {1, 1, 3};
        auto update = make({{collection, 2}, {marker, 2}}, {UpdateGeometryArraysOp{collection, vc::geometry::encode(points),
            {{"points", R"([{"n":5},{"n":6}])"}}, "retain"}});
        payloads.clear();
        const auto updateText = transactionRequestToJson(update, &payloads);
        require(!payloads.empty() && transactionJsonPayloadHashes(updateText).size() == payloads.size(), "update payloads listed");
        require(transactionRequestDigest(transactionRequestFromJson(updateText, payloads)) == transactionRequestDigest(update),
                "update round trip");
        mustFail([&] { edits.submit(make({{collection, 2}, {marker, 2}}, {UpdateGeometryArraysOp{collection,
            vc::geometry::encode(points), {}, "update"}})); });
        const auto retained = edits.submit(update);
        int64_t relationshipVersion = 0;
        {
            auto txn = edits.read();
            const auto head = edits.currentGeometry(txn, collection);
            const auto stored = std::get<vc::geometry::PointCollection>(vc::geometry::decode(edits.readGeometryData(txn, head->uuid)));
            require(stored.ids == points.ids && stored.points == points.points &&
                    edits.geometryPointAnnotations(txn, head->uuid, "points") ==
                        nlohmann::json::parse(R"([{"n":5},{"n":6}])"), "arrays and annotations updated");
            require(retained.objects.at(collection) == 3 && !edits.edgeGeometryState(txn, anchored).stale, "relationship retained");
            relationshipVersion = *edits.currentRelationshipVersion(txn, anchored);
        }
        points.points[0] = {1, 1, 4};
        edits.submit(make({{collection, 3}, {marker, retained.objects.at(marker).value()}},
            {UpdateGeometryArraysOp{collection, vc::geometry::encode(points), {}, "keep_recorded"}}));
        {
            auto txn = edits.read();
            require(edits.currentRelationshipVersion(txn, anchored) == relationshipVersion, "relationship kept as recorded");
            require(edits.geometryPointAnnotations(txn, edits.currentGeometry(txn, collection)->uuid, "points") ==
                        nlohmann::json::parse(R"([{"n":5},{"n":6}])"), "annotations follow stable IDs");
        }
    }

    // Ownership: standalone writers share; the service is exclusive; aliases
    // and handles in the same process are covered.
    {
        OpenOptions service = options; service.serviceOwner = true;
        auto second = ObjectStore::open(path, options);
        mustFail([&] { (void)ObjectStore::open(path, service); });
        const auto alias = directory.path / "alias.vcobj";
        std::filesystem::create_symlink(path, alias);
        mustFail([&] { (void)ObjectStore::open(alias, service); });
    }
    db = ObjectStore::create(directory.path / "other.vcobj", options); // closes the first handle
    {
        OpenOptions service = options; service.serviceOwner = true;
        auto host = ObjectStore::open(path, service);
        mustFail([&] { (void)ObjectStore::open(path, options); });
        mustFail([&] { (void)ObjectStore::open(directory.path / "alias.vcobj", options); });
        OpenOptions reader; reader.readOnly = true;
        auto readOnly = ObjectStore::open(path, reader);
        auto txn = readOnly.read();
        require(readOnly.status(txn).generation == generation, "read-only open while hosted");
    }
    (void)ObjectStore::open(path, options);
    {
        auto provisioned = ObjectStore::create(directory.path / "feed-gap.vcobj", options);
        StoreStatus before;
        { auto read = provisioned.read(); before = provisioned.status(read); }
        { auto write = provisioned.write(); provisioned.seal(write); write.commit(); }
        auto read = provisioned.read();
        const auto after = provisioned.status(read);
        require(after.commitSequence == before.commitSequence + 1, "sealing is a logical provisioning commit");
        const auto page = provisioned.changesSince(read, {before.storeUuid, before.generation, before.commitSequence, 0});
        require(page.cursorValid && page.commits.empty() && !page.more && page.next.sequence == after.commitSequence,
                "empty change feed advances over a provisioning-only sequence gap");
    }
    std::cout << "object store versions: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store versions failed: " << error.what() << "\n";
    return 1;
}
