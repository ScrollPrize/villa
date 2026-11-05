// Hosted attachments in an ObjectStoreSession: the typed operations answer
// as the store does; read()/act() refuse a hosted attachment; commits arrive
// as events after a check, with their kinds; readStore readers and the
// geometry views over them answer as a local reader does; cached heads are current as of
// the last check; queries combine hosted results, local attachments and
// drafts like local ones; a lost feed position is a revalidation; an
// unreachable host keeps cached data and reports the failure.
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include "vc_object_store/objects/AttachedDocument.hpp"
#include "vc_object_store/objects/Project.hpp"
#include "TestSupport.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/server/Server.hpp"
#include "vc_object_store/store/GeometryViews.hpp"

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
    std::string surface, track, fiber;
    {
        auto store = ObjectStore::create(hostedPath, options);
        auto write = store.write();
        ObjectRecord record; record.kind = "surface"; record.tags = {"probe"};
        record.meta = R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":1}})";
        surface = store.upsertObject(write, record);
        store.putGeometry(write, surface, vc::geometry::encode(tilted(0)));
        ObjectRecord strand; strand.kind = "fiber";
        strand.meta = R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":1}})";
        fiber = store.upsertObject(write, strand);
        vc::geometry::Fiber f; f.controlPoints = {{0, 0, 0}, {3, 4, 0}, {3, 4, 12}};
        store.putGeometry(write, fiber, vc::geometry::encode(f));
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
    hostedOptions.author = "session-test";
    hostedOptions.client.retries = 1;
    hostedOptions.client.connectTimeoutMs = 1000;
    const auto hosted = session.attachHosted(endpoint, hostedOptions);
    const auto local = session.attach(localPath, options);
    mustFail([&] { (void)session.attachHosted(endpoint, hostedOptions); }); // once per session
    require(session.hosted(hosted) && !session.hosted(local), "backend kinds");
    require(session.location(hosted) == endpoint && session.location(local) == session.path(local).string(), "locations");
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

    // readStore: the same reads and geometry views from both backends.
    {
        OpenOptions readOnly; readOnly.readOnly = true;
        auto direct = ObjectStore::open(hostedPath, readOnly);
        auto txn = direct.read();
        const LocalStoreReader expected(direct, txn);
        const GeometrySnapshot expectedViews(expected);
        bool ran = false;
        session.readStore(hosted, [&](const StoreReader& reader, const auto&) {
            const auto head = reader.currentObject(surface);
            require(head && head->uuid == expected.currentObject(surface)->uuid && !reader.currentObject(util::newUuid()),
                    "reader current object");
            require(reader.objectHistory(surface).size() == expected.objectHistory(surface).size(), "reader history");
            require(reader.currentGeometry(surface)->uuid == *head->geometryRevision, "reader current geometry");
            const auto& revision = *head->geometryRevision;
            require(reader.readGeometryRectangle(revision, "points", 2, 3, 5, 4) ==
                    expected.readGeometryRectangle(revision, "points", 2, 3, 5, 4), "reader rectangle");
            require(reader.readGeometryArray(revision, "points", 12, 24) == expected.readGeometryArray(revision, "points", 12, 24),
                    "reader byte range");
            mustFail([&] { (void)reader.readGeometryRectangle(revision, "points", 25, 0, 6, 1); });
            mustFail([&] { (void)reader.readGeometryArray(revision, "points", 0, 1 << 20); });
            ObjectSelection all; all.kind = "surface";
            require(reader.selectObjects(all).size() == 1 && reader.selectEdges({}).empty(), "reader listings");
            const auto state = reader.status();
            const std::vector<ObjectVersion> pair{{state.storeUuid, state.generation, surface, 1}, {state.storeUuid, state.generation, fiber, 1}};
            require(reader.relationshipsAt(pair).size() == expected.relationshipsAt(pair).size(), "reader relationships at versions");
            mustFail([&] { (void)reader.relationshipsAt({{state.storeUuid, state.generation, surface, 9}}); });
            const auto projected = reader.selectObjects(all, std::vector<std::string>{});
            require(projected.size() == 1 && projected[0].uuid == surface && projected[0].meta == "{}" &&
                    projected[0].version == expected.selectObjects(all).front().version, "reader projection");
            require(reader.objectCount("surface") == expected.objectCount("surface") && reader.objectCount("track") == 1 &&
                    reader.objectCount("fiber") == 1, "reader counts");
            const auto strand = reader.currentGeometry(fiber)->uuid;
            const auto packed = reader.readGeometryArrays({strand, strand}, "control_points");
            const auto expectedPacked = expected.readGeometryArrays({strand, strand}, "control_points");
            require(packed.data == expectedPacked.data && packed.offsets == expectedPacked.offsets &&
                    packed.shape == expectedPacked.shape && packed.dtype == expectedPacked.dtype &&
                    packed.offsets == std::vector<uint64_t>{0, 3, 6}, "reader packed arrays");
            mustFail([&] { (void)reader.readGeometryArrays({strand, revision}, "control_points"); });
            require(reader.geometryPointAnnotations(strand, "control_points") ==
                    expected.geometryPointAnnotations(strand, "control_points"), "reader point annotations");
            require(reader.geometryLengthInVolume(fiber, volume, 2) == expected.geometryLengthInVolume(fiber, volume, 2) &&
                    expected.geometryLengthInVolume(fiber, volume, 2) == 34, "reader fiber length");
            require(reader.objectTags() == std::vector<std::string>{"probe"} && reader.objectTags() == expected.objectTags() && reader.objectTags("track") == expected.objectTags("track"),
                    "reader tags");
            const GeometrySnapshot views(reader);
            const auto tile = views.surface(surface).readTile(3, 2, 10, 7);
            const auto expectedTile = expectedViews.surface(surface).readTile(3, 2, 10, 7);
            require(tile.points == expectedTile.points && tile.validity == expectedTile.validity &&
                    tile.scale == expectedTile.scale, "surface view tile");
            const auto shown = views.viewTransformedAt(surface, volume, 2).readTile(0, 0, 30, 20);
            const auto expectedShown = expectedViews.viewTransformedAt(surface, volume, 2).readTile(0, 0, 30, 20);
            require(shown.points == expectedShown.points && shown.validity == expectedShown.validity &&
                    shown.scale == expectedShown.scale && shown.points[21][0] == 2, "transformed view");
            require(views.surface(surface, revision).viewTransformedAt(volume, 2).readTile(0, 0, 1, 1).points ==
                    expectedViews.surface(surface, revision).viewTransformedAt(volume, 2).readTile(0, 0, 1, 1).points,
                    "pinned revision view");
            require(views.surface(surface).availableTransforms().size() ==
                    expectedViews.surface(surface).availableTransforms().size(), "available transforms");
            ran = true;
        });
        session.readStore(local, [&](const StoreReader& reader, const auto&) {
            require(reader.active() && reader.currentObject(localOnly) && reader.status().storeUuid == local.storeUuid,
                    "local reader");
        });
        require(ran, "hosted readStore ran");
    }

    // A checked edit through the session; its own cache sees it at once, and
    // the next check reports it with its kind.
    const auto before = session.status(hosted);
    const auto receipt = session.submit(hosted, request(session, hosted, {{track, 1}}, {SetMetaOp{track, "note", "\"one\""}}));
    require(receipt.objects.at(track) == 2 && session.currentObject(hosted, track)->version == 2, "own edit visible");
    session.readStore(hosted, [&](const StoreReader& reader, const auto&) {
        const FeedCursor since{before.storeUuid, before.generation, before.commitSequence};
        require(changesTouchKinds(reader, since, {"track"}) && !changesTouchKinds(reader, since, {"surface", "fiber"}) &&
                changesTouchKinds(reader, {before.storeUuid, util::newUuid(), 0}, {}), "changed kinds since a position");
        require(reader.unboundObjectCount("fiber", volume) == 0 && reader.unboundObjectCount("track", volume) == 1 &&
                reader.unboundObjectCount("fiber", "elsewhere") == 1, "unbound counts");
    });
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
        // One artifact as a cached local file, kept by content.
        const auto cache = directory.path / ("artifact-cache-" + object);
        const auto copy = session.artifactFile(attachment, object, "model/weights.bin", cache);
        require(copy && *copy == cache / files[0].sha256 / "weights.bin" && std::filesystem::file_size(*copy) == files[0].length &&
                !session.artifactFile(attachment, object, "missing.bin", cache) &&
                !session.artifactFile(attachment, object, "model/weights.bin", cache, version), "artifact file");
        const auto written = std::filesystem::last_write_time(*copy);
        require(session.artifactFile(attachment, object, "model/weights.bin", cache) == copy &&
                std::filesystem::last_write_time(*copy) == written, "cached artifact file kept");
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

    // Typed session edits work on a hosted attachment: record edits keep
    // relationships, a registration and a route choice are checked requests.
    {
        const auto renamed = session.editRecords(hosted, {track}, [](ObjectRecord& record) { record.name = "renamed"; });
        require(renamed && session.currentObject(hosted, track)->record.name == "renamed" &&
                session.currentObject(hosted, track)->version == renamed->objects.at(track), "hosted record edit");
        require(!session.editRecords(hosted, {}, [](ObjectRecord&) {}), "no objects, no request");
        mustFail([&] { (void)session.editRecords(hosted, {util::newUuid()}, [](ObjectRecord&) {}); });
        const auto registration = registerVolumeTransform(session, hosted, "hosted-a", "hosted-b", VolumeAffine{}, "hosted");
        session.refresh(hosted);
        const auto route = session.volumeTransforms(hosted).resolve("hosted-a", "hosted-b").resolved().path;
        const auto choice = chooseVolumeTransform(session, hosted, "hosted-a", "hosted-b", route);
        require(session.currentObject(hosted, registration)->record.kind == "volume_transform" &&
                session.currentObject(hosted, choice)->record.kind == "volume_transform_choice", "hosted transform writes");
        session.refresh(hosted);
        require(chooseVolumeTransform(session, hosted, "hosted-a", "hosted-b", route) == choice &&
                session.currentObject(hosted, choice)->version == 2, "a choice is replaced in place");
        mustFail([&] { (void)chooseVolumeTransform(session, hosted, "hosted-a", "hosted-b", {util::newUuid()}); });
    }

    // Point collections, attached collections, object copies and project
    // records are checked requests too.
    {
        const VolumeVoxels voxels{"20260411134726", 1};
        AttachedCollection marks;
        marks.uuid = util::newUuid(); marks.name = "marks";
        marks.points = {{0, 1}, {{1, 2, 3}, {4, 5, 6}}};
        marks.annotations = nlohmann::json::array({{{"n", 0}}, {{"n", 1}}});
        writeVolumeCollections(session, hosted, voxels, {marks}, {});
        auto listed = volumeCollections(session, hosted, voxels);
        require(listed.size() == 1 && listed[0].points.points == marks.points.points &&
                listed[0].annotations == marks.annotations, "hosted volume collection created");
        marks.points.points[1] = {4, 5, 7}; marks.name = "moved";
        writeVolumeCollections(session, hosted, voxels, {marks}, {});
        listed = volumeCollections(session, hosted, voxels);
        require(listed.size() == 1 && listed[0].name == "moved" && listed[0].points.points == marks.points.points &&
                listed[0].points.ids == marks.points.ids && listed[0].annotations == marks.annotations &&
                session.currentObject(hosted, marks.uuid)->version == 2, "hosted volume collection revised");
        writeVolumeCollections(session, hosted, voxels, {}, {marks.uuid});
        require(volumeCollections(session, hosted, voxels).empty(), "hosted volume collection retired");
        AttachedCollection fix;
        fix.uuid = util::newUuid(); fix.name = "fix";
        fix.points = {{0}, {{0, 0, 1}}};
        fix.annotations = nlohmann::json::array({nlohmann::json::object()});
        replaceAttachedCollections(session, hosted, surface, "corrections", {fix}, volumeCoordinates(voxels));
        require(attachedCollections(session, hosted, surface, "corrections").size() == 1, "hosted attached collection");
        replaceAttachedCollections(session, hosted, surface, "corrections", {}, volumeCoordinates(voxels));
        require(attachedCollections(session, hosted, surface, "corrections").empty(), "hosted attached collection removed");
        const auto copied = session.copyObject(hosted, hosted, fiber, "copied");
        const auto copy = session.currentObject(hosted, copied);
        const auto from = nlohmann::json::parse(copy->record.meta).at("copied_from");
        require(copy->record.name == "copied" && copy->geometryRevision && from.at("object_uuid") == fiber &&
                from.at("version") == session.currentObject(hosted, fiber)->version, "hosted copy");
        mustFail([&] { session.importObject(hosted, hosted, fiber); }); // already there
        require(addProjectResource(session, hosted, kVolumeResource, "https://example.org/v.zarr/") &&
                removeProjectResource(session, hosted, projectResourceUuid(kVolumeResource, "https://example.org/v.zarr/")),
                "hosted project resource");
    }

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
