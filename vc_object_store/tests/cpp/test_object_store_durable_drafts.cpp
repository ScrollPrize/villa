// Durable session drafts: checkpoints restore the last completed checkpoint
// (not later edits), a failed checkpoint keeps the previous one, a directory
// serves one session, durable saves clear only what they saved, conflicts
// keep drafts without resending, and a hosted save with an unknown outcome
// stays pending across a restart until resolved.
#include "TestSupport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/server/Server.hpp"

#include <iostream>

using namespace vc::objects;
using vc::geometry::Surface;
using vc::geometry::SurfacePoint;

namespace {
const std::string volume = "20260411134726";

Surface flat(float z)
{
    Surface surface{20, 20, {1, 1}, {}};
    for (int r = 0; r < 20; ++r)
        for (int c = 0; c < 20; ++c) surface.points.push_back(SurfacePoint{float(c), float(r), z});
    return surface;
}

// Where a draft lies, through the session's draft-aware proximity query.
std::optional<double> draftHeight(ObjectStoreSession& session, const std::string& uuid)
{
    SurfaceProximityQuery near; near.volume = volume; near.point = {10, 10, 0}; near.tolerance = 1000;
    near.uuids = std::vector<std::string>{uuid};
    const auto hits = session.nearestSurfaces({}, near);
    if (hits.empty()) return std::nullopt;
    return hits.front().hit.point[2];
}

TransactionRequest save(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& uuid,
                        std::optional<int64_t> expected, float z)
{
    const auto status = session.status(attachment);
    TransactionRequest r;
    r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
    r.author = "draft-test"; r.expectedObjects = {{uuid, expected}};
    r.operations = {PutGeometryOp{uuid, vc::geometry::encode(flat(z)), {}}};
    return r;
}

std::string createSurface(const std::filesystem::path& path, const OpenOptions& options)
{
    auto store = ObjectStore::create(path, options);
    auto write = store.write();
    ObjectRecord record; record.kind = "surface";
    record.meta = R"({"coordinates":{"volume_uuid":"20260411134726","scale_factor":1}})";
    const auto uuid = store.upsertObject(write, record);
    store.putGeometry(write, uuid, vc::geometry::encode(flat(1)));
    write.commit();
    return uuid;
}

std::size_t filesIn(const std::filesystem::path& directory, const std::string& prefix)
{
    std::size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        n += entry.path().filename().string().rfind(prefix, 0) == 0;
    return n;
}
} // namespace

int main()
try {
    TemporaryDirectory directory;
    OpenOptions options; options.author = "draft-test";
    const auto storePath = directory.path / "local.vcobj", drafts = directory.path / "drafts";
    const auto surface = createSurface(storePath, options);
    const auto coordinates = volumeCoordinates(volume, 1);
    ObjectVersion base;

    // Checkpoint, edit afterwards, "crash": the checkpoint comes back.
    {
        ObjectStoreSession session;
        const auto attachment = session.attach(storePath, options);
        const auto status = session.status(attachment);
        base = {status.storeUuid, status.generation, surface, 1};
        require(session.useDraftDirectory(drafts).checkpoint == 0, "empty directory");
        mustFail([&] { (void)session.useDraftDirectory(drafts); }); // once
        session.setDraft(surface, flat(5), coordinates, attachment, base);
        require(!session.draftState(surface)->checkpointed, "not yet checkpointed");
        require(session.checkpointDrafts() == 1 && session.draftState(surface)->checkpointed, "checkpointed");
        const std::vector<SurfacePoint> moved(400, SurfacePoint{1, 1, 9});
        session.updateDraft(surface, 0, 0, 20, 20, moved);
        require(!session.draftState(surface)->checkpointed && *draftHeight(session, surface) == 9, "edited after the checkpoint");
        ObjectStoreSession other;
        mustFail([&] { (void)other.useDraftDirectory(drafts); }); // one session per directory
    }
    {
        ObjectStoreSession session;
        const auto attachment = session.attach(storePath, options);
        const auto recovery = session.useDraftDirectory(drafts);
        require(recovery.checkpoint == 1 && recovery.drafts == std::vector<std::string>{surface} && recovery.pending.empty(),
                "recovered");
        const auto state = session.draftState(surface);
        require(state && state->base == base && state->checkpointed && state->conflict.empty(), "recovered state");
        require(*draftHeight(session, surface) == 5, "the checkpointed grid, not the later edit");

        // A checkpoint that cannot be written keeps the previous one.
        session.setDraft(surface, flat(7), coordinates, attachment);
        std::filesystem::permissions(drafts, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
        mustFail([&] { (void)session.checkpointDrafts(); });
        std::filesystem::permissions(drafts, std::filesystem::perms::owner_all);
        require(!session.draftCheckpointError().empty() && !session.draftState(surface)->checkpointed, "failure reported");
    }
    {
        ObjectStoreSession session;
        const auto attachment = session.attach(storePath, options);
        require(session.useDraftDirectory(drafts).checkpoint == 1 && *draftHeight(session, surface) == 5, "previous checkpoint kept");

        // A conflicting save keeps the draft and records it; nothing pending.
        mustFail([&] { (void)session.submitDurably(attachment, save(session, attachment, surface, std::nullopt, 5), {surface}); });
        auto state = session.draftState(surface);
        require(state && !state->conflict.empty() && !state->pendingRequest && session.pendingSubmissions().empty(), "conflict kept");
        // Reapplying against the latest version clears it; the save clears the draft.
        session.setDraft(surface, flat(5), coordinates, attachment, base);
        require(session.draftState(surface)->conflict.empty(), "rebased");
        const auto receipt = session.submitDurably(attachment, save(session, attachment, surface, 1, 5), {surface});
        require(receipt.objects.at(surface) == 2 && !session.hasDraft(surface), "saved draft cleared");
        require(filesIn(drafts, "draft-") == 0 && filesIn(drafts, "request-") == 0, "checkpoint holds nothing");
    }

    // Hosted: an unreachable host leaves the outcome unknown; it stays pending
    // across a restart and resolves to the identical request once it answers.
    const auto hostedPath = directory.path / "hosted.vcobj", hostedDrafts = directory.path / "hosted-drafts";
    const auto hostedSurface = createSurface(hostedPath, options);
    server::ServerOptions serverOptions;
    auto service = std::make_unique<server::Server>(hostedPath, serverOptions);
    service->start();
    serverOptions.port = service->port();
    const auto endpoint = "http://127.0.0.1:" + std::to_string(serverOptions.port);
    HostedSessionOptions hostedOptions;
    hostedOptions.pollInterval = std::chrono::hours(1);
    hostedOptions.client.retries = 0;
    hostedOptions.client.connectTimeoutMs = 500;
    std::string pending;
    {
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(endpoint, hostedOptions);
        session.useDraftDirectory(hostedDrafts);
        const auto status = session.status(attachment);
        session.setDraft(hostedSurface, flat(3), coordinates, attachment, ObjectVersion{status.storeUuid, status.generation, hostedSurface, 1});
        const auto request = save(session, attachment, hostedSurface, 1, 3);
        pending = request.requestUuid;
        service.reset();
        mustFail([&] { (void)session.submitDurably(attachment, request, {hostedSurface}); });
        require(session.pendingSubmissions().size() == 1 && session.draftState(hostedSurface)->pendingRequest == pending,
                "unknown outcome stays pending");
    }
    service = std::make_unique<server::Server>(hostedPath, serverOptions);
    service->start();
    {
        ObjectStoreSession session;
        const auto attachment = session.attachHosted(endpoint, hostedOptions);
        const auto recovery = session.useDraftDirectory(hostedDrafts);
        require(recovery.pending.size() == 1 && recovery.pending[0].requestUuid == pending &&
                recovery.pending[0].drafts == std::vector<std::string>{hostedSurface}, "pending recovered");
        // Edited after the save: resolving commits the save, and the edit
        // stays as a draft on the committed version.
        const std::vector<SurfacePoint> moved(400, SurfacePoint{1, 1, 4});
        session.updateDraft(hostedSurface, 0, 0, 20, 20, moved);
        const auto receipt = session.resolvePending(attachment, pending);
        require(receipt.objects.at(hostedSurface) == 2 && session.pendingSubmissions().empty(), "resolved");
        const auto state = session.draftState(hostedSurface);
        require(state && state->base && state->base->version == 2 && !state->pendingRequest, "later edit kept on the new base");
        // Resolving again finds nothing pending; the saved receipt replays.
        mustFail([&] { (void)session.resolvePending(attachment, pending); });
        require(session.receipt(attachment, pending)->commitSequence == receipt.commitSequence, "receipt kept");
    }
    std::cout << "object store durable drafts: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store durable drafts failed: " << error.what() << "\n";
    return 1;
}
