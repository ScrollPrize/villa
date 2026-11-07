// link_objects: one checked request runs a plan of detectors over a read
// snapshot and publishes their links with the link marker in one commit; a
// commit in between conflicts and writes nothing, an unrelated commit does
// not; an identical retry replays; the hosted service gives the same result.
#include "TestSupport.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/object_links/LinkObjects.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#ifdef VC_LINK_OBJECTS_TEST_HOSTED
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/server/Server.hpp"
#endif

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace vc::objects;
namespace links = vc::object_links;
using Json = nlohmann::json;

namespace {
template<class T>
GeometryArray array(const std::string& name, const std::string& dtype, std::vector<int64_t> shape, const std::vector<T>& values)
{
    GeometryArray result{name, dtype, std::move(shape), std::vector<uint8_t>(values.size() * sizeof(T))};
    std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
    return result;
}

std::string put(ObjectStore& db, ObjectRecord record, const GeometryData& geometry)
{
    record.storageRef = "/missing/source";
    auto txn = db.write();
    const auto uuid = db.upsertObject(txn, record);
    db.putGeometry(txn, uuid, geometry);
    txn.commit();
    return uuid;
}

GeometryData planeGeometry(float offset)
{
    std::vector<float> points;
    for (int y = 0; y < 33; ++y)
        for (int x = 0; x < 33; ++x) points.insert(points.end(), {float(x) + offset, float(y), 20});
    return {"surface", 1, "source_xyz", R"({"scale":[1,1]})", {array("points", "<f4", {33, 33, 3}, points)}};
}

std::string plane(ObjectStore& db, float offset = 0)
{
    ObjectRecord record; record.kind = "surface"; record.name = "surface";
    return put(db, record, planeGeometry(offset));
}

std::string collection(ObjectStore& db)
{
    ObjectRecord record; record.kind = "point_collection";
    record.meta = R"({"source":{"point_attributes":[{"wind_a":3},{"wind_a":4}],"winding_is_absolute":true}})";
    return put(db, record, {"point_collection", 1, "source_xyz", "{}", {
        array<double>("points", "<f8", {2, 3}, {10, 10, 40, 20, 20, 40}),
        array<int64_t>("point_ids", "<i8", {2}, {90, 7})}});
}

std::string track(ObjectStore& db)
{
    ObjectRecord record; record.kind = "track";
    record.track.emplace(); record.track->datasetRef = "dataset"; record.track->externalKey = "0"; record.track->family = "h";
    return put(db, record, {"track", 1, "source_xyz", R"({"family":"h"})", {array<int32_t>("points", "<i4", {3, 3}, {1, 1, 20, 8, 8, 20, 16, 16, 20})}});
}

// Test-only: the real preparer, slowed by `slowPreparationMs` so a hosted
// request can be retried or duplicated while its detection is running.
std::atomic<int> slowPreparationMs{0};
void installSlowPreparer()
{
    links::enableLinkObjects();
    registerOperationPreparer(kLinkObjectsOperation, [](ObjectStore& store, const TransactionOperation& op, const PrepareOptions& options) {
        auto prepared = links::prepareLinkObjects(store, std::get<LinkObjectsOp>(op), options);
        if (slowPreparationMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slowPreparationMs.load()));
        return prepared;
    });
}

// Two overlapping planes and a collection whose points land on them.
struct Fixture { std::string a, b, points; };
Fixture populate(ObjectStore& db)
{
    Fixture f;
    f.a = plane(db); f.b = plane(db); f.points = collection(db);
    return f;
}

links::LinkPlan plan()
{
    links::LinkPlan plan;
    plan.steps.push_back({links::kPointCollectionSurfaceDetector, "",
        Json{{"point_scale", 0.5}, {"window_points", 3}, {"window_min_points", 3}}});
    plan.steps.push_back({links::kSurfaceOverlapDetector, "", Json::object()});
    plan.marker = Json{{"key", "fixture"}, {"selection", {{"surfaces", 2}}}};
    return plan;
}

TransactionRequest request(const StoreStatus& status, const links::LinkPlan& plan)
{
    TransactionRequest r;
    r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation; r.author = "linker";
    r.operations.push_back(links::linkObjectsOperation(plan));
    return r;
}

std::size_t edgeCount(ObjectStore& db, const std::string& kind)
{
    auto txn = db.read();
    EdgeSelection selection; selection.kind = kind;
    return db.selectEdges(txn, selection).size();
}

Json marker(ObjectStore& db)
{
    auto txn = db.read();
    const auto record = db.objectByUuid(txn, links::linkMarkerUuid());
    require(record.has_value() && record->kind == links::kLinkMarkerKind, "the link marker is a record of its kind");
    return Json::parse(record->meta);
}

void checkPublished(ObjectStore& db, const TransactionReceipt& receipt)
{
    require(receipt.commitSequence.has_value(), "a link run commits");
    require(edgeCount(db, "point-surface") == 2 && edgeCount(db, "same-surface") == 1, "both detectors' links are published");
    const auto meta = marker(db);
    require(meta.at("commit_sequence") == *receipt.commitSequence, "the marker names the links' commit");
    require(meta.at("key") == "fixture" && meta.at("selection").at("surfaces") == 2, "the marker keeps the plan's content");
    const auto& detectors = meta.at("detectors");
    require(detectors.size() == 2 && detectors[0].at("detector") == links::kPointCollectionSurfaceDetector &&
            detectors[0].at("report").at("edges_created") == 2 && detectors[1].at("report").at("edges_created") == 1,
            "the marker reports each detector");
    require(receipt.objects.at(links::linkMarkerUuid()).has_value(), "the marker is in the receipt");
}

void localRequests(const TemporaryDirectory& directory)
{
    OpenOptions options; options.author = "links-test";
    auto db = ObjectStore::create(directory.path / "local.vcobj", options);
    const auto fixture = populate(db);
    auto status = [&] { auto txn = db.read(); return db.status(txn); };

    // Detect, publish, mark: one commit.
    const auto first = request(status(), plan());
    const auto receipt = db.submit(first);
    checkPublished(db, receipt);
    require(status().commitSequence == *receipt.commitSequence, "nothing else was committed");
    require(receipt.objects.at(fixture.a).has_value() && receipt.objects.at(fixture.points).has_value(),
            "the receipt carries the versions of the linked objects");

    // An identical retry replays the receipt.
    const auto again = db.submit(first);
    require(again.replayed && again.commitSequence == receipt.commitSequence, "identical retry replays");
    require(status().commitSequence == *receipt.commitSequence, "a replay writes nothing");

    // A rerun with nothing changed: the links are kept, the marker moves on.
    const auto rerun = db.submit(request(status(), plan()));
    require(rerun.commitSequence && *rerun.commitSequence == *receipt.commitSequence + 1, "a rerun commits its marker");
    require(marker(db).at("detectors")[1].at("report").at("edges_skipped_existing") == 1, "a rerun keeps the link");
    require(edgeCount(db, "same-surface") == 1, "no duplicate links");

    // Prepared, then a selected object changes: conflict, nothing written.
    {
        const auto conflicting = request(status(), plan());
        const auto prepared = db.prepare(conflicting);
        require(!prepared.empty() && prepared.expectedObjects().count(fixture.a) && prepared.expectedObjects().count(links::linkMarkerUuid()),
                "the preparation expects what it read");
        {
            auto txn = db.write();
            db.setMeta(txn, fixture.a, "note", R"("moved on")");
            txn.commit();
        }
        const auto before = status().commitSequence;
        bool conflicted = false;
        try { db.submit(conflicting, prepared); } catch (const TransactionConflict& conflict) {
            conflicted = conflict.reason() == TransactionConflict::Reason::Version;
        }
        require(conflicted, "a commit between snapshot and write conflicts");
        require(status().commitSequence == before, "a conflict writes nothing");
        require(!db.receipt(db.read(), conflicting.requestUuid).has_value(), "no receipt for a conflict");
        // Prepared again over the new state, the same request commits.
        const auto retried = db.submit(conflicting, db.prepare(conflicting));
        require(retried.commitSequence && *retried.commitSequence == before + 1, "re-prepared, it commits");
    }

    // Prepared, then an unrelated commit: the links commit without waiting.
    {
        const auto unrelated = request(status(), plan());
        const auto prepared = db.prepare(unrelated);
        plane(db, 100);
        const auto receipt = db.submit(unrelated, prepared);
        require(receipt.commitSequence.has_value(), "an unrelated commit does not conflict");
    }

    // Two link runs over the same snapshot: the second conflicts on the marker.
    {
        const auto one = request(status(), plan()), two = request(status(), plan());
        const auto preparedOne = db.prepare(one), preparedTwo = db.prepare(two);
        require(db.submit(one, preparedOne).commitSequence.has_value(), "first run commits");
        bool conflicted = false;
        try { db.submit(two, preparedTwo); } catch (const TransactionConflict&) { conflicted = true; }
        require(conflicted, "concurrent link runs conflict");
    }

    // A preparation publishes only for its own request, on its own store.
    {
        const auto mine = request(status(), plan());
        auto other = request(status(), plan()); // another UUID: another request
        const auto prepared = db.prepare(mine);
        const auto before = status().commitSequence;
        mustFail([&] { db.submit(other, prepared); });
        require(status().commitSequence == before && !db.receipt(db.read(), other.requestUuid).has_value(),
                "a preparation of another request publishes nothing");
        other.requestUuid = mine.requestUuid;
        other.author = "someone else"; // the same UUID with other contents: another digest
        mustFail([&] { db.submit(other, prepared); });
        require(status().commitSequence == before, "a preparation of other contents publishes nothing");
        auto elsewhere = ObjectStore::create(directory.path / "elsewhere.vcobj", options);
        populate(elsewhere);
        const auto there = [&] { auto txn = elsewhere.read(); return elsewhere.status(txn); }();
        const auto theirs = request(there, plan());
        const auto preparedHere = db.prepare(theirs); // ran over this store's objects
        mustFail([&] { elsewhere.submit(theirs, preparedHere); });
        require(there.commitSequence == [&] { auto txn = elsewhere.read(); return elsewhere.status(txn).commitSequence; }(),
                "a preparation made on another store publishes nothing");
        require(db.submit(mine, prepared).commitSequence.has_value(), "the preparation still serves its own request");
    }

    // A detector that cannot run fails the request: no step publishes and
    // the marker stays.
    {
        track(db);
        const auto before = status().commitSequence;
        const auto markerBefore = marker(db);
        const auto pointEdges = edgeCount(db, "point-surface");
        links::LinkPlan failing = plan();
        failing.steps.push_back({links::kTrackSurfaceDetector, directory.path.string() + "/no-such-ct.zarr", Json::object()});
        const auto failed = request(status(), failing);
        bool refused = false;
        try { db.submit(failed); } catch (const PreparationFailed& error) {
            refused = std::string(error.what()).find(links::kTrackSurfaceDetector) != std::string::npos;
        }
        require(refused, "a failed detector fails the request, naming the detector");
        require(status().commitSequence == before && marker(db) == markerBefore && edgeCount(db, "point-surface") == pointEdges,
                "a failed request writes nothing: no links, the marker unmoved");
        require(!db.receipt(db.read(), failed.requestUuid).has_value(), "no receipt for a failed request");
    }

    // Plan errors are refused before anything runs.
    {
        links::LinkPlan bad = plan();
        bad.steps.push_back({"no-such-detector", "", Json::object()});
        mustFail([&] { db.submit(request(status(), bad)); });
        links::LinkPlan unknownKey = plan();
        unknownKey.steps[0].params["tolerence"] = 1;
        mustFail([&] { db.submit(request(status(), unknownKey)); });
        links::LinkPlan needsNormals = plan();
        needsNormals.steps.push_back({links::kFiberSurfaceDetector, "", Json::object()});
        mustFail([&] { db.submit(request(status(), needsNormals)); });
    }

    // The plan round-trips through its JSON and the request's canonical form.
    {
        const auto r = request(status(), plan());
        const auto text = transactionRequestToJson(r);
        const auto back = transactionRequestFromJson(text, {});
        require(transactionRequestDigest(back) == transactionRequestDigest(r), "link_objects has a canonical encoding");
        require(links::linkPlanJson(links::linkPlanFromJson(links::linkPlanJson(plan()))) == links::linkPlanJson(plan()), "plan JSON round-trips");
    }
}

#ifdef VC_LINK_OBJECTS_TEST_HOSTED
void hostedRequests(const TemporaryDirectory& directory)
{
    OpenOptions options; options.author = "links-test";
    const auto path = directory.path / "hosted.vcobj";
    {
        auto db = ObjectStore::create(path, options);
        populate(db);
    }
    server::ServerOptions serverOptions;
    serverOptions.readWorkers = 2;
    server::Server service(path, serverOptions);
    service.start();
    HostedStore hosted("http://127.0.0.1:" + std::to_string(service.port()));
    const auto first = request(hosted.status(), plan());
    const auto receipt = hosted.submit(first);
    {
        OpenOptions readOnly; readOnly.readOnly = true;
        auto inspect = ObjectStore::open(path, readOnly);
        checkPublished(inspect, receipt);
    }
    const auto again = hosted.submit(first);
    require(again.replayed && again.commitSequence == receipt.commitSequence, "hosted identical retry replays");
    require(hosted.status().commitSequence == *receipt.commitSequence, "hosted replay writes nothing");

    // A client may not point the service at a path of its own.
    links::LinkPlan clientPath = plan();
    clientPath.steps.push_back({links::kFiberSurfaceDetector, "/data/ct.zarr", Json{{"cache_directory", "/home/me/strips"}}});
    mustFail([&] { hosted.submit(request(hosted.status(), clientPath)); });

    // A failed detector fails the hosted request too, and writes nothing.
    {
        OpenOptions readOnly; readOnly.readOnly = true;
        auto inspect = ObjectStore::open(path, readOnly);
        const auto before = hosted.status().commitSequence;
        links::LinkPlan failing = plan();
        failing.steps.push_back({links::kTrackSurfaceDetector, "/no-such-ct.zarr", Json::object()});
        hosted.submit(request(hosted.status(), plan())); // a run to compare the marker against
        const auto markerBefore = marker(inspect);
        bool refused = false;
        try { hosted.submit(request(hosted.status(), failing)); } catch (const PreparationFailed&) { refused = true; }
        require(refused, "a failed hosted detector is reported as such, not retried");
        require(hosted.status().commitSequence == before + 1 && marker(inspect) == markerBefore, "a failed hosted request writes nothing");
    }

    auto preparedCount = [&] { return Json::parse(hosted.serviceReport()).at("link_requests").at("detections").get<uint64_t>(); };

    // The same request sent twice while its detection runs: one detection,
    // one answer for both.
    {
        slowPreparationMs = 1000;
        const auto before = preparedCount();
        const auto same = request(hosted.status(), plan());
        HostedStore second("http://127.0.0.1:" + std::to_string(service.port()));
        std::optional<TransactionReceipt> one, two;
        std::thread first([&] { one = hosted.submit(same); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        two = second.submit(same);
        first.join();
        require(one && two && one->commitSequence == two->commitSequence && one->commitSequence.has_value(),
                "a duplicate of a request in flight gets the running detection's answer");
        require(preparedCount() == before + 1, "the detection ran once");
        // The same UUID with other contents while it runs is refused.
        auto other = same;
        other.author = "impostor";
        slowPreparationMs = 1000;
        const auto again = request(hosted.status(), plan());
        std::thread running([&] { hosted.submit(again); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        other.requestUuid = again.requestUuid;
        bool refused = false;
        try { second.submit(other); } catch (const std::invalid_argument&) { refused = true; }
        running.join();
        require(refused, "a request in flight is not replaced by other contents");
        slowPreparationMs = 0;
    }

    // A client whose link timeout is shorter than the detection retries with
    // the same UUID: the retries wait for the running detection (or replay
    // its receipt) and the detection still runs once.
    {
        slowPreparationMs = 2500;
        const auto before = preparedCount();
        HostedStoreOptions impatient;
        impatient.linkTimeoutS = 1;
        impatient.retries = 5;
        HostedStore client("http://127.0.0.1:" + std::to_string(service.port()), impatient);
        const auto start = std::chrono::steady_clock::now();
        const auto receipt = client.submit(request(client.status(), plan()));
        const auto took = std::chrono::steady_clock::now() - start;
        require(receipt.commitSequence.has_value() && *receipt.commitSequence == hosted.status().commitSequence,
                "an impatient client still gets the receipt");
        require(took >= std::chrono::milliseconds(2500), "the answer came from the one running detection");
        require(preparedCount() == before + 1, "a stalled retry does not start another detection");
        slowPreparationMs = 0;
    }
    service.stop();
}
#endif
} // namespace

int main()
try {
    installSlowPreparer();
    TemporaryDirectory directory;
    localRequests(directory);
#ifdef VC_LINK_OBJECTS_TEST_HOSTED
    hostedRequests(directory);
#endif
    std::cout << "object store link_objects: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store link_objects: " << error.what() << "\n";
    return 1;
}
