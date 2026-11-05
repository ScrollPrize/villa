// Hosted attachments in an ObjectStoreSession: the typed operations answer
// as the store does; read()/act() refuse a hosted attachment; commits arrive
// as events after a check, with their kinds; cached heads are current as of
// the last check; queries combine hosted results, local attachments and
// drafts like local ones; a lost feed position is a revalidation; an
// unreachable host keeps cached data and reports the failure.
#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/server/Server.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>

using namespace vc::objects;
using vc::geometry::Surface;
using vc::geometry::SurfacePoint;

namespace {
const std::string volume = "20260411134726";

Surface tilted(float lift)
{
    Surface surface{30, 20, {1, 1}, {}};
    for (int r = 0; r < 30; ++r)
        for (int c = 0; c < 20; ++c) surface.points.push_back(SurfacePoint{float(c), float(r), float(c + 1) + lift});
    return surface;
}

RegionOptions surfaces()
{
    RegionOptions options;
    options.kind = "surface";
    options.volume = volume;
    return options;
}

RegionSlice plane(double z)
{
    RegionSlice slice;
    slice.origin = {0, 0, z};
    slice.localBounds = {-100, -100, -0.5, 100, 100, 0.5};
    return slice;
}

TransactionRequest request(ObjectStoreSession& session, const AttachmentIdentity& attachment, VersionSelection expected,
                           std::vector<TransactionOperation> operations)
{
    const auto status = session.status(attachment);
    TransactionRequest r;
    r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
    r.author = "session-test"; r.expectedObjects = std::move(expected); r.operations = std::move(operations);
    return r;
}

std::vector<StoreSessionEvent> committedOf(ObjectStoreSession& session, const AttachmentIdentity& attachment)
{
    std::vector<StoreSessionEvent> result;
    for (auto& event : session.takeEvents())
        if (event.kind == StoreSessionEvent::Kind::Committed && event.attachment == attachment) result.push_back(std::move(event));
    return result;
}
} // namespace

int main()
try {
    TemporaryDirectory directory;
    OpenOptions options; options.author = "session-test";
    const auto hostedPath = directory.path / "hosted.vcobj", localPath = directory.path / "local.vcobj";
    std::string surface, track;
    {
        auto store = ObjectStore::create(hostedPath, options);
        auto write = store.write();
        ObjectRecord record; record.kind = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":1}})";
        surface = store.upsertObject(write, record);
        store.putGeometry(write, surface, vc::geometry::encode(tilted(0)));
        ObjectRecord line; line.kind = "track"; line.track.emplace(); line.track->datasetRef = "s"; line.track->externalKey = "k";
        track = store.upsertObject(write, line);
        write.commit();
    }
    std::string localOnly;
    {
        auto store = ObjectStore::create(localPath, options);
        auto write = store.write();
        ObjectRecord record; record.kind = "track"; record.track.emplace(); record.track->datasetRef = "s"; record.track->externalKey = "l";
        record.uuid = "ffffffff-ffff-4fff-bfff-ffffffffffff"; // sorts after the copy below
        localOnly = store.upsertObject(write, record);
        // A local copy of the hosted track (same UUID): the hosted one hides it.
        ObjectRecord copy = record; copy.uuid = track; copy.track->externalKey = "copy";
        store.upsertObject(write, copy);
        write.commit();
    }
    server::ServerOptions serverOptions; serverOptions.readWorkers = 2;
    auto service = std::make_unique<server::Server>(hostedPath, serverOptions);
    service->start();
    const auto port = service->port();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(port);

    ObjectStoreSession session;
    HostedSessionOptions hostedOptions;
    hostedOptions.pollInterval = std::chrono::hours(1); // checks happen through refresh() here
    hostedOptions.client.retries = 1;
    hostedOptions.client.connectTimeoutMs = 1000;
    const auto hosted = session.attachHosted(endpoint, hostedOptions);
    const auto local = session.attach(localPath, options);
    mustFail([&] { (void)session.attachHosted(endpoint, hostedOptions); }); // once per session
    require(session.hosted(hosted) && !session.hosted(local), "backend kinds");
    require(session.freshness(hosted) && session.freshness(hosted)->error.empty() && !session.freshness(local), "freshness");
    (void)session.takeEvents();

    // Local callbacks need a database the session does not have.
    mustFail([&] { session.read(hosted, [](const auto&, const auto&, const auto&) {}); });
    mustFail([&] { (void)session.act(hosted, {}, [](auto&, auto&) {}); });
    mustFail([&] { (void)session.path(hosted); }); // and the attachment can still be detached at the end

    // Typed reads answer as the store does (read-only beside the service).
    {
        OpenOptions readOnly; readOnly.readOnly = true;
        auto direct = ObjectStore::open(hostedPath, readOnly);
        auto txn = direct.read();
        require(session.status(hosted).commitSequence == direct.status(txn).commitSequence, "status");
        const auto head = session.currentObject(hosted, surface);
        require(head && head->uuid == direct.currentObjectRevision(txn, surface)->uuid && head->version == 1, "current object");
        const auto geometry = session.geometryRevision(hosted, *head->geometryRevision);
        require(geometry && geometry->contentHash == direct.geometryRevision(txn, *head->geometryRevision)->contentHash, "geometry");
        require(session.readGeometryArray(hosted, geometry->uuid, "points") ==
                direct.readGeometryArray(txn, geometry->uuid, "points"), "geometry array");
        const auto status = direct.status(txn);
        const auto at = session.objectsAtVersions(hosted, {{status.storeUuid, status.generation, surface, 1},
                                                           {status.storeUuid, status.generation, surface, 2}});
        require(at.size() == 2 && at[0] && at[0]->uuid == head->uuid && !at[1], "objects at versions");
    }

    // A checked edit through the session; its own cache sees it at once, and
    // the next check reports it with its kind.
    const auto receipt = session.submit(hosted, request(session, hosted, {{track, 1}}, {SetMetaOp{track, "note", "\"one\""}}));
    require(receipt.objects.at(track) == 2 && session.currentObject(hosted, track)->version == 2, "own edit visible");
    session.refresh(hosted);
    auto events = committedOf(session, hosted);
    require(events.size() == 1 && events[0].changes.size() == 1 && events[0].changes[0].recordUuid == track &&
            events[0].changes[0].kind == "track" && events[0].touches({"track"}) && !events[0].touches({"surface"}) &&
            events[0].journalPosition == *receipt.commitSequence, "commit event with kind");
    require(session.receipt(hosted, receipt.requestUuid.value())->commitSequence == receipt.commitSequence, "receipt");
    mustFail([&] { (void)session.submit(hosted, request(session, hosted, {{track, 1}}, {SetMetaOp{track, "note", "\"stale\""}})); });

    // Another client's edit: cached heads stay as of the last check.
    require(session.currentObject(hosted, track)->version == 2, "head cached after the check");
    {
        HostedStore other(endpoint);
        const auto status = other.status();
        TransactionRequest r; r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
        r.author = "other"; r.expectedObjects = {{track, 2}}; r.operations = {SetMetaOp{track, "note", "\"two\""}};
        other.submit(r);
    }
    require(session.currentObject(hosted, track)->version == 2, "cached until the next check");
    session.refresh(hosted);
    require(session.currentObject(hosted, track)->version == 3, "invalidated by the feed");
    events = committedOf(session, hosted);
    require(events.size() == 1 && events[0].changes.at(0).author == "other", "external commit event");

    // The background check delivers commits without refresh().
    {
        ObjectStoreSession polled;
        HostedSessionOptions quick = hostedOptions;
        quick.pollInterval = std::chrono::milliseconds(20);
        const auto attachment = polled.attachHosted(endpoint, quick);
        (void)polled.takeEvents();
        polled.submit(attachment, request(polled, attachment, {{track, 3}}, {SetMetaOp{track, "note", "\"polled\""}}));
        bool seen = false;
        for (int i = 0; i < 300 && !seen; ++i) {
            seen = !committedOf(polled, attachment).empty();
            if (!seen) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        require(seen, "polled commit event");
        session.refresh(hosted);
        (void)session.takeEvents();
    }

    // Queries combine the hosted store, a local attachment and drafts.
    const std::vector<AttachmentIdentity> order{hosted, local};
    ObjectSelection tracks; tracks.kind = "track";
    const auto listed = session.selectObjects(order, tracks);
    require(listed.size() == 2 && listed[0].attachment == hosted && listed[0].record.uuid == track &&
            listed[1].attachment == local && listed[1].record.uuid == localOnly, "selection across backends");
    // A limited page that loses its first record to the hosted copy reads on.
    tracks.limit = 1;
    const auto page = session.selectObjects(order, tracks);
    require(page.size() == 2 && page[1].attachment == local && page[1].record.uuid == localOnly, "hidden records refilled");
    tracks.limit.reset();

    // Artifacts through the session, hosted and local alike.
    for (const auto& attachment : order) {
        const auto object = attachment == hosted ? track : localOnly;
        const auto file = directory.path / "weights.bin";
        { std::ofstream(file, std::ios::binary) << "weights for " << object; }
        const auto version = session.currentObject(attachment, object)->version;
        const auto published = session.publishArtifacts(attachment, object, version, {{"model/weights.bin", file}}, "session-test");
        require(published.objects.at(object) == version + 1, "artifacts published");
        if (attachment == hosted) session.refresh(hosted);
        const auto files = session.artifacts(attachment, object);
        require(files.size() == 1 && files[0].name == "model/weights.bin" && session.artifacts(attachment, object, version).empty(),
                "manifest per version");
        const auto out = directory.path / ("materialized-" + object);
        session.materializeArtifacts(attachment, object, out);
        std::ifstream in(out / "model" / "weights.bin", std::ios::binary);
        require(std::string(std::istreambuf_iterator<char>(in), {}) == "weights for " + object, "materialized bytes");
    }
    (void)session.takeEvents();
    const auto committed = session.geometryRegion(order, plane(10.25), surfaces());
    require(!committed.empty() && committed[0].attachment == hosted && !committed[0].result.runs.empty() &&
            committed[0].result.runs[0].objectUuid == surface, "hosted region");
    SurfaceProximityQuery near; near.volume = volume; near.point = {10, 20, 11.2}; near.tolerance = 1;
    require(session.nearestSurfaces(order, near).size() == 1, "hosted proximity");
    // A draft moved off the plane hides the committed surface everywhere.
    session.setDraft(surface, tilted(50), volumeCoordinates(volume, 1), hosted);
    for (const auto& region : session.geometryRegion(order, plane(10.25), surfaces()))
        require(region.result.runs.empty(), "draft replaces the hosted surface");
    require(session.nearestSurfaces(order, near).empty(), "draft replaces the hosted surface near the point");
    session.clearDraft(surface);

    // A lost feed position (the store got a new history generation while the
    // service was down) is a revalidation; the restarted service answers.
    std::vector<uint8_t> cachedBytes = session.readGeometryArray(hosted, *session.currentObject(hosted, surface)->geometryRevision, "points");
    service.reset();
    session.refresh(hosted);
    require(!session.freshness(hosted)->error.empty() && session.freshness(hosted)->status, "unreachable host reported");
    require(session.readGeometryArray(hosted, *session.currentObject(hosted, surface)->geometryRevision, "points") == cachedBytes,
            "cached data readable offline");
    mustFail([&] { (void)session.status(hosted); });
    { auto writer = ObjectStore::open(hostedPath, options); writer.startHistoryGeneration("restored for the test"); }
    serverOptions.port = port;
    service = std::make_unique<server::Server>(hostedPath, serverOptions);
    service->start();
    session.refresh(hosted);
    events = committedOf(session, hosted);
    require(events.size() == 1 && events[0].revalidate && events[0].touches({"anything"}), "revalidation");
    require(session.freshness(hosted)->error.empty(), "reconnected");

    // Detaching stops the poller: nothing arrives after Detached.
    session.detach(hosted);
    const auto last = session.takeEvents();
    require(!last.empty() && last.back().kind == StoreSessionEvent::Kind::Detached, "detached");
    mustFail([&] { (void)session.status(hosted); });
    std::cout << "object store hosted session: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store hosted session failed: " << error.what() << "\n";
    return 1;
}
