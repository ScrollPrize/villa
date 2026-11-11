#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>
#include <vc_object_store/objects/ObjectCollection.hpp>
#include <utility>
// The VC3D surface catalog over a real object store: rows of the current
// volume's surfaces kept current from commit events, opening a surface in the
// volume's voxels, and record edits (tags, name, retire).
#include "SurfaceCatalog.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/StoreSurfaces.hpp"
#include "segmentation/SurfaceCorrections.hpp"
#include "segmentation/SurfaceStoreEdit.hpp"
#include "SessionDrafts.hpp"
#ifdef VC3D_TEST_HOSTED
#include <vc_object_store/server/Server.hpp>
#endif
#include "vc/core/PointCollections.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_reader/catalog.hpp>
#include <vc_reader/registrations.hpp>
#include "VolumeRegistrations.hpp"
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <fstream>
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <set>

namespace fs = std::filesystem;
using namespace vc::objects;

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
// A new surface `record` (its uuid expected absent) holding `geometry`, in one checked request.
void storeSurface(ObjectStoreSession& session, const AttachmentIdentity& attachment, const ObjectRecord& record,
                  const GeometryData& geometry)
{
    auto request = session.newRequest(attachment);
    request.expectedObjects = {{record.uuid, std::nullopt}};
    request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
    request.operations.push_back(PutGeometryOp{record.uuid, geometry, {}});
    session.submit(attachment, request);
}

void checkCorrectionSnapshots(const fs::path& directory)
{
    SurfaceCatalog::Crossings crossings;
    crossings.requireComplete();
    auto rejectsIncomplete = [&] {
        bool rejected = false;
        try { crossings.requireComplete(); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "partial/remote surface queries cannot be used for persistent annotations");
    };
    crossings.complete = false;
    rejectsIncomplete();
    crossings.complete = true;
    crossings.remote.emplace_back(AttachmentIdentity{}, util::newUuid());
    rejectsIncomplete();
    for (const bool history : {false, true}) for (const bool referenced : {false, true}) {
        std::string phase = "loaded corrections";
        try {
        OpenOptions options; options.author = "correction-test"; options.history = history ? HistoryMode::On : HistoryMode::Off;
        const auto path = directory / (std::string("corrections-") + std::to_string(history) + std::to_string(referenced) + ".vcobj");
        { auto store = ObjectStore::create(path, options); }
        ObjectStoreSession session;
        auto attachment = session.attach(path, options);
        ObjectRecord surface; surface.uuid = util::newUuid(); surface.kind = "surface";
        surface.meta = nlohmann::json{{"coordinates", volumeCoordinates("volume", 1)}}.dump();
        auto request = session.newRequest(attachment);
        request.expectedObjects[surface.uuid] = std::nullopt;
        request.operations.push_back(CreateObjectOp{objectRecordJson(surface)});
        request.operations.push_back(PutGeometryOp{surface.uuid,
            vc::geometry::encode(vc::geometry::Surface{2, 2, {1, 1}, {{0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}}}), {}});
        session.submit(attachment, request);
        auto first = vc3d::corrections::read(session, attachment, surface.uuid);
        std::map<uint64_t, nlohmann::json> data{{1, {{"uuid", util::newUuid()}, {"name", "fix"},
            {"points", {{"7", {{"p", {1., 2., 3.}}}}}}}}};
        vc3d::corrections::write(session, attachment, first, data, {"volume"});
        if (referenced) {
            session.detach(attachment);
            { auto base = ObjectStore::open(path, options); auto txn = base.write(); base.seal(txn); txn.commit(); }
            const auto overlay = fs::path(path.string() + ".overlay");
            { auto collection = ObjectCollection::create(overlay, {path}, options); }
            attachment = session.attach(overlay, options);
        }
        first = vc3d::corrections::read(session, attachment, surface.uuid);
        const auto unchangedOwner = first.baseline.owner;
        const auto unchangedCommit = first.baseline.snapshot.commitSequence;
        vc3d::corrections::write(session, attachment, first, first.collections, {"volume"});
        require(first.baseline.owner == unchangedOwner && first.baseline.snapshot.commitSequence == unchangedCommit,
                "no-op correction save preserves ownership and the loaded commit baseline");
        auto pending = first.collections;
        pending.at(1)["points"]["7"]["p"] = {2., 3., 4.};
        bool failedSave = false;
        try { vc3d::corrections::write(session, attachment, first, pending, {"volume", 0}); }
        catch (const std::exception&) { failedSave = true; }
        require(failedSave && first.baseline.owner == unchangedOwner && first.collections != pending,
                "failed correction persistence retains the loaded baseline for retry");
        // Editing restart must save successfully before reloading this data.
        vc3d::corrections::write(session, attachment, first, pending, {"volume"});
        first = vc3d::corrections::read(session, attachment, surface.uuid);
        require(first.collections == pending, "successful retry then reload preserves previously unsaved correction edits");
        auto stale = vc3d::corrections::read(session, attachment, surface.uuid);
        data.at(1)["points"]["7"]["p"] = {4., 5., 6.};
        vc3d::corrections::write(session, attachment, first, data, {"volume"});
        const auto staleOwner = stale.baseline.owner;
        const auto staleCommit = stale.baseline.snapshot.commitSequence;
        const auto savedCollections = vc3d::corrections::read(session, attachment, surface.uuid).collections;
        vc3d::corrections::write(session, attachment, stale, stale.collections, {"volume"});
        require(stale.baseline.owner == staleOwner && stale.baseline.snapshot.commitSequence == staleCommit &&
                vc3d::corrections::read(session, attachment, surface.uuid).collections == savedCollections,
                "unchanged stale UI requests no write and does not refresh its loaded baseline");
        auto staleEdit = stale.collections;
        staleEdit.at(1)["points"]["7"]["p"] = {7., 8., 9.};
        bool rejected = false;
        try { vc3d::corrections::write(session, attachment, stale, staleEdit, {"volume"}); }
        catch (const TransactionConflict&) { rejected = true; }
        require(rejected, "JSON correction editing retains the loaded object references");
        stale = first;
        data.emplace(2, nlohmann::json{{"uuid", util::newUuid()}, {"name", "new"}, {"points", nlohmann::json::object()}});
        vc3d::corrections::write(session, attachment, first, data, {"volume"});
        rejected = false;
        try { vc3d::corrections::write(session, attachment, stale, {}, {"volume"}); }
        catch (const TransactionConflict&) { rejected = true; }
        const auto remaining = vc3d::corrections::read(session, attachment, surface.uuid).collections.size();
        if (!rejected || remaining != 2)
            throw std::runtime_error("stale correction membership guard: history=" + std::to_string(history) +
                " referenced=" + std::to_string(referenced) + " rejected=" + std::to_string(rejected) + " remaining=" + std::to_string(remaining));
        stale = first;
        vc3d::corrections::write(session, attachment, first, {}, {"volume"});
        staleEdit = stale.collections;
        staleEdit.at(1)["points"]["7"]["p"] = {7., 8., 9.};
        rejected = false;
        try { vc3d::corrections::write(session, attachment, stale, staleEdit, {"volume"}); }
        catch (const TransactionConflict&) { rejected = true; }
        require(rejected, "a stale editor cannot resurrect removed corrections");
        require(vc3d::corrections::read(session, attachment, surface.uuid).collections.empty(), "receipt baseline supports subsequent saves");

        // The two editors and growth evidence share one loaded baseline: own
        // writes advance it; an unrelated external write must not be hidden.
        auto sharedSession = std::shared_ptr<ObjectStoreSession>(&session, [](auto*) {});
        phase = "coordinated surface save";
        const auto opened = SurfaceCatalog::open(sharedSession, attachment, {"volume", 1}, surface.uuid);
        SurfaceStoreEdit editor(sharedSession, opened.pin, opened.view);
        auto coordinated = vc3d::corrections::read(session, attachment, surface.uuid);
        editor.watchAttachedCollections(coordinated.baseline);
        (*opened.surface->rawPointsPtr())(0,0)[2] += 1;
        editor.save(*opened.surface, {0,0,2,2});
        data = {{1, {{"uuid", util::newUuid()}, {"name", "coordinated"},
            {"points", {{"7", {{"p", {1., 2., 3.}}}}, {"8", {{"p", {4., 5., 6.}}}}}}}}};
        vc3d::corrections::write(session, attachment, coordinated, data, {"volume"}, &editor);
        require(editor.pin().reference() == coordinated.baseline.owner, "correction write repins the surface editor");
        phase = "coordinated correction removal";
        data.at(1)["points"].erase("8");
        vc3d::corrections::write(session, attachment, coordinated, data, {"volume"}, &editor);
        (*opened.surface->rawPointsPtr())(0,0)[2] += 1;
        phase = "surface save after corrections";
        editor.save(*opened.surface, {0,0,2,2});
        phase = "coordinated growth evidence";
        vc3d::corrections::GrowthAnnotation annotation; annotation.surfaceUuid = surface.uuid;
        vc3d::corrections::writeGrowthAnnotation(session, attachment, annotation, {"volume"}, &editor);
        data.at(1)["points"]["7"]["p"] = {2., 3., 4.};
        vc3d::corrections::write(session, attachment, coordinated, data, {"volume"}, &editor);

        const auto reloaded = vc3d::corrections::read(session, attachment, surface.uuid);
        phase = "retained point append";
        PointCollections reopened;
        auto unanchored = reloaded.collections.at(1);
        require(reopened.replaceUnanchoredCollections({unanchored}), "reopen point collection");
        for (const auto& collection : reloaded.baseline.collections) reopened.reservePointIds(collection.nextPointId);
        reopened.addPoint("coordinated", {7,8,9});
        const auto& points = reopened.getAllCollections().begin()->second.points;
        const auto newId = std::max_element(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.first < b.first; })->first;
        require(newId > 8, "reopening after deletion does not reuse a retained structural ID");
        auto appended = reloaded;
        auto appendData = reloaded.collections;
        appendData.at(1)["points"][std::to_string(newId)] = {{"p", {7.,8.,9.}}};
        vc3d::corrections::write(session, attachment, appended, appendData, {"volume"});
        data.at(1)["points"]["7"]["p"] = {3., 4., 5.};
        rejected = false;
        try { vc3d::corrections::write(session, attachment, coordinated, data, {"volume"}, &editor); }
        catch (const TransactionConflict&) { rejected = true; }
        require(rejected, "coordinated editor never refreshes away another writer's edit");
        if (history) {
            phase = "historical evidence";
            attachDocument(session, attachment, surface.uuid, {{"records", {{{"p", {1.,2.,3.}}, {"value", 1}}}}},
                {"result", "records", "p"}, "results", volumeCoordinates("volume", 1));
            auto evidence = attachedCollections(session, attachment, surface.uuid, "result");
            const auto oldOwner = evidence.owner;
            evidence.collections.front().annotations[0]["value"] = 2;
            replaceAttachedCollections(session, attachment, evidence, evidence.collections, volumeCoordinates("volume", 1));
            require(attachedDocument(session, attachment, oldOwner, "result")->at("records")[0]["value"] == 1,
                    "historical owner resolves historical attached evidence");
            const auto currentEvidence = attachedDocument(session, attachment, evidence.owner, "result");
            const auto actual = attachedCollections(session, attachment, surface.uuid, "result");
            if (!currentEvidence || currentEvidence->at("records")[0]["value"] != 2)
                throw std::runtime_error("current owner resolves updated attached evidence (old owner v" + std::to_string(oldOwner.version) +
                    ", new v" + std::to_string(evidence.owner.version) + ", actual=" + actual.collections.front().annotations.dump() +
                    ", document=" + (currentEvidence ? currentEvidence->dump() : "missing") + ")");
            require(attachedDocument(session, attachment, surface.uuid, "result") == currentEvidence,
                    "UUID-only document reads keep their current-head semantics");
        }
        phase = "coordinate dialog baseline";
        ObjectRecord prompt; prompt.uuid = util::newUuid(); prompt.kind = "surface";
        auto creation = session.newRequest(attachment);
        creation.expectedObjects[prompt.uuid] = std::nullopt;
        creation.operations = {CreateObjectOp{objectRecordJson(prompt)}};
        session.submit(attachment, creation);
        const auto shown = *session.currentObject(attachment, prompt.uuid);
        session.editCurrent(attachment, {prompt.uuid}, [](const StoreReader&, const std::string& uuid) {
            return std::vector<TransactionOperation>{SetMetaOp{uuid, "coordinates", volumeCoordinates("volume", 2).dump()}};
        });
        auto binding = session.newRequest(attachment);
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            editCurrentRequest(reader, binding, {prompt.uuid}, {bindCoordinatesOperation(shown.record, "volume", 3)});
        });
        expectObject(binding, shown.reference());
        rejected = false;
        try { session.submit(attachment, binding); } catch (const TransactionConflict&) { rejected = true; }
        require(rejected, "coordinate binding rejects a changed declaration after the dialog's selection");
        } catch (const std::exception& error) {
            throw std::runtime_error("history=" + std::to_string(history) + " referenced=" + std::to_string(referenced) +
                " phase=" + phase + ": " + error.what());
        }
    }
}

void checkHistoryGenerationSaves(const fs::path& directory)
{
    const auto path = directory / "generation.vcobj";
    OpenOptions options; options.author = "generation-test";
    auto store = ObjectStore::create(path, options);
    auto session = std::make_shared<ObjectStoreSession>();
    const auto attachment = session->attach(path, options);
    cv::Mat_<cv::Vec3f> points(2, 2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c) points(r, c) = {float(c), float(r), 1.f};
    QuadSurface grid(points, {1.f, 1.f});
    grid.setChannel("review", cv::Mat_<uint8_t>(2, 2, uint8_t{1}));
    ObjectRecord record; record.uuid = util::newUuid(); record.kind = "surface";
    record.meta = nlohmann::json{{"coordinates", volumeCoordinates("vol", 1)}}.dump();
    storeSurface(*session, attachment, record, quad_surface_geometry(grid));
    auto opened = SurfaceCatalog::open(session, attachment, {"vol", 1}, record.uuid);
    SurfaceStoreEdit stale(session, opened.pin);
    store.startHistoryGeneration("restored backup");
    (*opened.surface->rawPointsPtr())(0, 0)[2] = 2;
    bool rejected = false;
    try { stale.save(*opened.surface, {0, 0, 2, 2}, {}); }
    catch (const TransactionConflict&) { rejected = true; }
    require(rejected, "surface opened before reset rejects edits");

    opened = SurfaceCatalog::open(session, attachment, {"vol", 1}, record.uuid);
    require(opened.pin.reference().generation != opened.pin.writeGeneration, "reopened surface retains its publication generation");
    SurfaceStoreEdit current(session, opened.pin);
    const auto publication = opened.pin.reference();
    // An explicit unchanged channel submits a no-op transaction and exercises repinning.
    current.save(*opened.surface, {0, 0, 2, 2}, {"review"});
    require(current.pin().reference() == publication, "no-op save preserves the historical reference");
    (*opened.surface->rawPointsPtr())(0, 0)[2] = 3;
    current.save(*opened.surface, {0, 0, 2, 2}, {});
    require(current.pin().reference().version == publication.version + 1 &&
            current.pin().reference().generation == current.pin().writeGeneration, "reopened surface saves in the active generation");
    (*opened.surface->rawPointsPtr())(0, 0)[2] = 4;
    current.save(*opened.surface, {0, 0, 2, 2}, {});
    require(current.pin().reference().version == publication.version + 2, "subsequent surface saves use the new pin");

    StoreSurfaces tool(path, "generation-test");
    auto loaded = tool.load(record.uuid);
    store.startHistoryGeneration("restored again");
    (*loaded->rawPointsPtr())(0, 0)[2] = 5;
    rejected = false;
    try { tool.update(record.uuid, *loaded); }
    catch (const TransactionConflict&) { rejected = true; }
    require(rejected, "tool input opened before reset rejects edits");
    loaded = tool.load(record.uuid);
    (*loaded->rawPointsPtr())(0, 0)[2] = 6;
    require(tool.update(record.uuid, *loaded) == publication.version + 3,
            "reloaded tool input saves in the active generation");
}

void checkToolMappingGuards(const fs::path& directory)
{
    for (bool history : {false, true}) for (bool referenced : {false, true}) {
        OpenOptions options; options.author = "tool-mapping"; options.history = history ? HistoryMode::On : HistoryMode::Off;
        const auto base = directory / ("tool-mapping-" + std::to_string(history) + std::to_string(referenced) + ".vcobj");
        ObjectRecord surface; surface.uuid = util::newUuid(); surface.kind = "surface";
        surface.meta = nlohmann::json{{"coordinates", volumeCoordinates("source", 1)}}.dump();
        {
            auto source = ObjectStore::create(base, options);
            auto transaction = source.write();
            source.upsertObject(transaction, surface);
            source.putGeometry(transaction, surface.uuid, vc::geometry::encode(vc::geometry::Surface{2, 2, {1, 1},
                {{1,1,1},{2,1,1},{1,2,1},{2,2,1}}}));
            if (referenced) source.seal(transaction);
            transaction.commit();
        }
        auto path = base;
        if (referenced) {
            path = directory / (base.filename().string() + ".overlay");
            auto overlay = ObjectCollection::create(path, {base}, options);
        }
        StoreSurfaces tool(path, "mapping-test");
        const auto attachment = tool.objects().attachment();
        const auto registration = registerVolumeTransform(*tool.objects().session(), attachment,
            "source", "target", VolumeAffine::identity(), "identity");
        auto loaded = tool.load(surface.uuid, true, VolumeVoxels{"target", 1});
        (*loaded->rawPointsPtr())(0,0)[2] += 1;
        tool.update(surface.uuid, *loaded);
        loaded = tool.load(surface.uuid, true, VolumeVoxels{"target", 1});
        (*loaded->rawPointsPtr())(0,0)[2] += 1;
        VolumeAffine moved; moved(0,3) = 10;
        tool.objects().session()->editCurrent(attachment, {registration}, [&](const StoreReader& reader, const std::string& uuid) {
            return std::vector<TransactionOperation>{UpdateObjectOp{objectRecordJson(
                volumeTransformRecord(reader.currentObject(uuid)->record, "source", "target", moved, "moved"))}};
        });
        bool rejected = false;
        try { tool.update(surface.uuid, *loaded); } catch (const TransactionConflict&) { rejected = true; }
        require(rejected, "tool edits check the mapping actually used to load the grid");
        rejected = false;
        try { tool.objects().publishArtifacts(surface.uuid, {}); } catch (const TransactionConflict&) { rejected = true; }
        require(rejected, "artifact results retain the same loaded mapping guard as geometry edits");
        loaded = tool.load(surface.uuid, true, VolumeVoxels{"target", 1});
        tool.objects().publishArtifacts(surface.uuid, {});
        auto stored = tool.load(surface.uuid);
        require((*stored->rawPointsPtr())(0,0)[2] == 2, "mapping conflict leaves surface geometry unchanged");
    }
}

void checkReferencedSurfaceSelections(const fs::path& directory)
{
    OpenOptions options; options.author = "surface-reference-test"; options.history = HistoryMode::Off;
    const auto base = directory / "surface-base.vcobj", overlay = directory / "surface-overlay.vcobj";
    ObjectVersion original, other;
    {
        auto store = ObjectStore::create(base, options);
        auto txn = store.write();
        for (auto* ref : {&original, &other}) {
            ObjectRecord record; record.kind = "surface"; record.name = "original";
            record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":1}})";
            const auto uuid = store.upsertObject(txn, record);
            vc::geometry::Surface grid{1, 1, {1, 1}, {{2, 4, 6}}};
            store.putGeometry(txn, uuid, vc::geometry::encode(grid));
            ref->objectUuid = uuid;
        }
        txn.commit();
        { auto read = store.read(); for (auto* ref : {&original, &other}) *ref = store.currentObjectRevision(read, ref->objectUuid)->reference(); }
        auto seal = store.write(); store.seal(seal); seal.commit();
    }
    { auto collection = ObjectCollection::create(overlay, {base}, options); }
    auto session = std::make_shared<ObjectStoreSession>();
    const auto attachment = session->attach(overlay, options);
    VolumeAffine shifted;
    shifted(0,3)=-3; shifted(2,3)=-7;
    registerVolumeTransform(*session,attachment,"vol","negative",shifted,"negative target coordinates");
    const auto translated = SurfaceCatalog::open(session,attachment,{"negative",1},original.objectUuid,original);
    require((*translated.surface->rawPointsPtr())(0,0)==cv::Vec3f(-1,4,-1) &&
            translated.surface->validMask()(0,0),
            "native materialization preserves valid negative target coordinates");
    const auto open = [&](const ObjectVersion& ref) { return SurfaceCatalog::open(session, attachment, {"vol", 1}, original.objectUuid, ref); };
    require(open(original).editable, "exact current inherited surface remains editable");
    bool rejected = false;
    try { open(other); } catch (const std::exception&) { rejected = true; }
    require(rejected, "same-number reference for a different object is rejected");
    auto wrongOwner = original; wrongOwner.storeUuid = util::newUuid(); rejected = false;
    try { open(wrongOwner); } catch (const std::exception&) { rejected = true; }
    require(rejected, "same-number reference for a different owner is rejected");
    {
        auto store = ObjectStore::open(overlay, options); auto txn = store.write();
        store.setMeta(txn, original.objectUuid, "coordinates", R"({"volume_uuid":"vol","scale_factor":2})"); txn.commit();
        auto deletion = store.write(); store.deleteObject(deletion, original.objectUuid); deletion.commit();
    }
    const auto past = open(original);
    require(!past.current && !past.editable && past.pin.reference() == original &&
            (*past.surface->rawPointsPtr())(0, 0) == cv::Vec3f(2, 4, 6),
            "deleted overlays retain exact source geometry and metadata as a read-only view");
}

// Optional real-data check: copy one stored surface, checkpoint a mask edit,
// recover it in a new session, then verify that saving preserves every XYZ.
void checkDraftRecovery(const fs::path& sourcePath, const fs::path& directory)
{
    OpenOptions writable; writable.author = "draft-recovery-test";
    OpenOptions readOnly; readOnly.readOnly = true;
    const auto path = directory / "recovery.vcobj", checkpoint = directory / "recovery-drafts";
    { auto store = ObjectStore::create(path, writable); }
    std::string uuid;
    vc::geometry::Surface original;
    std::size_t masked = 0;
    {
        auto session = std::make_shared<ObjectStoreSession>();
        SessionDrafts drafts(session);
        require(drafts.start(checkpoint, 0), "real draft directory");
        const auto source = session->attach(sourcePath, readOnly), target = session->attach(path, writable);
        std::string sourceUuid;
        session->readStore(source, [&](const StoreReader& reader, const auto&) {
            ObjectSelection selection; selection.kind = "surface"; selection.limit = 1;
            const auto records = reader.selectObjects(selection);
            require(!records.empty(), "real store contains a surface");
            sourceUuid = records.front().uuid;
        });
        uuid = session->copyObject(source, target, sourceUuid);
        session->readStore(target, [&](const StoreReader& reader, const auto&) {
            original = GeometrySnapshot(reader).surface(uuid).load();
            while (masked < original.points.size() && !original.valid(masked / original.columns, masked % original.columns)) ++masked;
            require(masked < original.points.size(), "real surface has a valid vertex");
            auto surface = load_quad_from_store(reader, uuid);
            const int row = int(masked / original.columns), column = int(masked % original.columns);
            auto mask = surface->validMask().clone();
            mask(row, column) = 0;
            surface->setChannel("mask", mask);
            (*surface->rawPointsPtr())(row, column) = {-1, -1, -1};
            const auto head = reader.currentObject(uuid);
            session->setDraft(uuid, quad_surface_grid(*surface),
                nlohmann::json::parse(head->record.meta).value("coordinates", nlohmann::json::object()),
                target, pinOf(target, *head, reader.status().generation).reference());
        });
        require(drafts.checkpointNow(), "real mask edit checkpointed");
    }
    auto session = std::make_shared<ObjectStoreSession>();
    SessionDrafts drafts(session);
    require(drafts.start(checkpoint, 0) && drafts.recovered(uuid), "real mask edit recovered");
    const auto target = session->attach(path, writable);
    std::unique_ptr<QuadSurface> surface;
    const auto pin = pinOf(target, *session->currentObject(target, uuid), session->status(target).generation);
    session->readStore(target, [&](const StoreReader& reader, const auto&) { surface = load_quad_from_store(reader, uuid); });
    const auto contents = session->draftContents(uuid);
    SessionDrafts::restoreSurface(*surface, contents->surface);
    SurfaceStoreEdit edit(session, pin);
    edit.save(*surface, cv::Rect(0, 0, int(original.columns), int(original.rows)), {"mask"}, contents->revision);
    session->readStore(target, [&](const StoreReader& reader, const auto&) {
        const auto saved = GeometrySnapshot(reader).surface(uuid).load();
        require(saved.points == original.points && !saved.valid(masked / saved.columns, masked % saved.columns),
                "real recovered mask preserves all stored coordinates");
    });
    std::cout << "real draft recovery: " << sourcePath << ", " << original.points.size() << " vertices preserved\n";
}
}

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir dir;
        checkHistoryGenerationSaves(fs::path(dir.path().toStdString()));
        checkReferencedSurfaceSelections(fs::path(dir.path().toStdString()));
        checkCorrectionSnapshots(fs::path(dir.path().toStdString()));
        checkToolMappingGuards(fs::path(dir.path().toStdString()));
        if (argc > 1) checkDraftRecovery(argv[1], fs::path(dir.path().toStdString()));
        const std::filesystem::path volumeRoot1 = fs::path(dir.path().toStdString()) / "ct.zarr";
        const vcr::zarr::CreateOptions levelZero1{.shape = {4, 4, 4}, .chunks = {4, 4, 4}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot1 / "0", levelZero1);
        std::ofstream(volumeRoot1 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "vol1"}, {"name", "vol1"}, {"format", "zarr"}, {"width", 4}, {"height", 4}, {"slices", 4}, {"voxelsize", 1.0}}.dump();
        const auto volume = Volume::New(volumeRoot1);
        const auto path = fs::path(dir.path().toStdString()) / "objects.vcobj";
        OpenOptions options; options.author = "surface-catalog-test";
        { auto created = ObjectStore::create(path, options); }
        auto session = std::make_shared<ObjectStoreSession>();
        const auto attachment = session->attach(path, options);

        cv::Mat_<cv::Vec3f> points(3, 3);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 1.f};
        QuadSurface grid(points, {1.f, 1.f});
        auto add = [&](const std::string& name, const std::optional<nlohmann::json>& coordinates) {
            const auto uuid = util::newUuid();
            ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.name = name; record.tags = {"approved"};
            auto meta = nlohmann::json::object();
            if (coordinates) meta.update(*coordinates);
            record.meta = meta.dump();
            storeSurface(*session, attachment, record, quad_surface_geometry(grid));
            return uuid;
        };
        const auto native = add("native", nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 1)}});
        const auto half = add("half", nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 0.5)}});
        const auto unbound = add("unbound", std::nullopt);
        const auto other = add("other", nlohmann::json{{"coordinates", geometryCoordinates({{"volume_uuid", "vol2"}, {"scale_factor", 1}})}});

        SurfaceCatalog catalog(session);
        std::vector<std::string> committed;
        QObject::connect(&catalog, &SurfaceCatalog::surfacesCommitted,
                         [&](const std::vector<std::string>& uuids) { committed.insert(committed.end(), uuids.begin(), uuids.end()); });
        auto pump = [&](const std::function<bool()>& done, const char* message) {
            QElapsedTimer timer; timer.start();
            while (!done() && timer.elapsed() < 10000) {
                for (const auto& event : session->takeEvents()) catalog.handle(event);
                QCoreApplication::processEvents();
                QThread::msleep(1);
            }
            if (!done()) std::cerr << "rows=" << catalog.rows().size() << " loading=" << catalog.loading()
                                   << " error=" << catalog.error().toStdString() << '\n';
            require(done(), message);
        };
        session->takeEvents();
        catalog.setSource(attachment, volume, {volume->id()});
        pump([&] { return !catalog.loading() && catalog.rows().size() == 4; }, "lists the volume's surfaces and those not bound to it");
        require(catalog.row(native)->mapping == VolumeAffine::identity() && catalog.row(half)->mapping == VolumeAffine::scale(2.0) &&
                !catalog.row(unbound)->bound() && !catalog.row(other)->bound(),
                "rows carry the mapping to the volume; other volumes' surfaces are unbound here");
        require(catalog.row(native)->toolInput() && !catalog.row(half)->toolInput(),
                "tools take surfaces stored in the volume's voxels");
        require(catalog.row(native)->record.name == "native" && catalog.row(native)->record.tags == std::vector<std::string>{"approved"},
                "rows carry name and tags");

        const auto opened = catalog.open(native);
        require(opened.editable && opened.surface->id == native && opened.pin.geometryRevision, "native surfaces open editable and pinned");
        require((*opened.surface->rawPointsPtr())(2, 1) == cv::Vec3f(1, 2, 1), "native points are stored points");
        const auto scaled = catalog.open(half);
        require(scaled.editable && scaled.view.mapping == VolumeAffine::scale(2.0), "store views of scaled surfaces support inverse edits");
        require((*scaled.surface->rawPointsPtr())(2, 1) == cv::Vec3f(2, 4, 2), "scaled points are in volume voxels");
        {
            SurfaceStoreEdit edit(session,scaled.pin,scaled.view);
            cv::Mat_<cv::Vec3f> grown(4,3,cv::Vec3f(-1,-1,-1));
            scaled.surface->rawPointsPtr()->copyTo(grown(cv::Rect(0,0,3,3)));
            grown(3,1)={2,6,2};
            QuadSurface result(grown,{0.5f,0.5f});
            edit.save(result,cv::Rect(0,0,3,4));
            session->readStore(attachment,[&](const vc::objects::StoreReader& reader,const auto&) {
                require(GeometrySnapshot(reader).surface(half).vertex(3,1)==vc::geometry::SurfacePoint{1,3,1},
                        "growth keeps the opened view binding even when its output has no metadata");
            });
        }
        bool refused = false;
        try { catalog.open(unbound); } catch (const std::exception&) { refused = true; }
        require(refused, "unbound surfaces do not open");

        // Editing an opened surface commits to its store object and re-pins.
        {
            auto edited = catalog.open(native);
            SurfaceStoreEdit edit(session, edited.pin);
            session->requireCurrent(edited.pin);
            auto& grid = *edited.surface->rawPointsPtr();
            grid(1, 1) = {5.f, 5.f, 5.f};
            edit.save(*edited.surface, cv::Rect(1, 1, 1, 1));
            require(edit.pin().reference().version == edited.pin.reference().version + 1 && edit.pin().geometryRevision != edited.pin.geometryRevision,
                    "saves re-pin from the receipt");
            require((*catalog.open(native).surface->rawPointsPtr())(1, 1) == cv::Vec3f(5, 5, 5), "vertex edits reach the store");
            // The mask brush: validity 0 and the sentinel in memory; stored coordinates survive.
            cv::Mat_<uint8_t> mask(3, 3, uint8_t(255));
            mask(0, 0) = 0;
            grid(0, 0) = {-1.f, -1.f, -1.f};
            edited.surface->setChannel("mask", mask);
            edit.save(*edited.surface, cv::Rect(0, 0, 3, 3), {"mask"});
            require((*catalog.open(native).surface->rawPointsPtr())(0, 0) == cv::Vec3f(-1, -1, -1), "masked vertices load masked");
            session->readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
                const auto unmasked = load_quad_from_store(reader, native, std::nullopt, false);
                require((*unmasked->rawPointsPtr())(0, 0) == cv::Vec3f(0, 0, 1), "masking keeps stored coordinates");
            });
            // Growth: another grid size replaces the geometry.
            cv::Mat_<cv::Vec3f> grown(4, 3, cv::Vec3f(-1.f, -1.f, -1.f));
            grid.copyTo(grown(cv::Rect(0, 0, 3, 3)));
            grown(3, 1) = {1.f, 3.f, 1.f};
            QuadSurface larger(grown, {1.f, 1.f});
            edit.save(larger, cv::Rect(0, 0, 3, 4));
            require(catalog.open(native).surface->rawPointsPtr()->size() == cv::Size(3, 4), "growth replaces the stored grid");
            bool stale = false;
            try { session->requireCurrent(edited.pin); } catch (const std::exception&) { stale = true; }
            require(stale, "an edit from a superseded pin is refused");
        }

        // Correction points live with their surface as linked point collections.
        {
            PointCollections points;
            points.addPoint("fix", {1, 2, 3});
            const auto fix = points.getCollectionId("fix").value();
            points.setCollectionAnchor2d(fix, cv::Vec2f(1.5f, 0.5f));
            const auto saved = points.anchoredCollections();
            auto correctionSnapshot = vc3d::corrections::read(*session, attachment, half);
            vc3d::corrections::write(*session, attachment, correctionSnapshot, saved, {volume->id()});
            auto stored = vc3d::corrections::read(*session, attachment, half);
            require(stored.collections.size() == 1 && stored.collections.begin()->first == fix, "corrections round-trip by collection id");
            session->readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
                const auto uuid = reader.neighbours(half, std::string(vc::objects::kAttachmentEdgeKind)).at(0);
                require(nlohmann::json::parse(reader.currentObject(uuid)->record.meta)["role"] == "corrections" &&
                            reader.neighbours(half).empty(),
                        "corrections are attached data, not graph links");
            });
            PointCollections reloaded;
            require(reloaded.replaceAnchoredCollections(stored.collections), "stored corrections load");
            const auto& collection = reloaded.getAllCollections().at(fix);
            require(collection.uuid == points.getAllCollections().at(fix).uuid && collection.anchor2d == cv::Vec2f(1.5f, 0.5f) &&
                    collection.points.size() == 1 && collection.points.begin()->second.p == cv::Vec3f(1, 2, 3),
                    "corrections keep identity, anchor and points");
            points.addPoint("fix", {4, 5, 6});
            vc3d::corrections::write(*session, attachment, correctionSnapshot, points.anchoredCollections(), {volume->id()});
            require(vc3d::corrections::read(*session, attachment, half).collections.at(fix).at("points").size() == 2, "corrections update in place");
            {
                const auto firstPoint = points.getAllCollections().at(fix).points.begin()->second;
                auto movedPoint = firstPoint;
                movedPoint.p = {7, 8, 9};
                points.updatePoint(movedPoint);
                vc3d::corrections::write(*session, attachment, correctionSnapshot, points.anchoredCollections(), {volume->id()});
                const auto stored = vc3d::corrections::read(*session, attachment, half).collections.at(fix);
                const auto& moved = stored.at("points").at(std::to_string(firstPoint.id)).at("p");
                require(moved[0].get<double>() == 7 && moved[2].get<double>() == 9, "moved correction points keep their id");
            }
            {
                // Through a view at half scale: points read halved, unchanged ones
                // keep their stored values on save, moved ones map back.
                const vc::objects::ViewFrame coarse{"vol1", 0.5, VolumeAffine::scale(0.5)};
                auto shown = vc3d::corrections::read(*session, attachment, half, coarse);
                const auto& ordered = points.getAllCollections().at(fix).points;
                const auto firstId = std::to_string(ordered.begin()->first), secondId = std::to_string(std::next(ordered.begin())->first);
                require(shown.collections.at(fix).at("points").at(firstId).at("p")[0].get<double>() == 3.5, "corrections read through the view");
                shown.collections.at(fix)["points"][secondId]["p"] = nlohmann::json::parse("[1.0, 1.0, 1.0]");
                vc3d::corrections::write(*session, attachment, shown, shown.collections, {volume->id()});
                const auto stored = vc3d::corrections::read(*session, attachment, half).collections.at(fix).at("points");
                require(stored.at(firstId).at("p")[0].get<double>() == 7 && stored.at(secondId).at("p")[0].get<double>() == 2,
                        "corrections save back to their stored frame");
            }
            correctionSnapshot = vc3d::corrections::read(*session, attachment, half);
            vc3d::corrections::write(*session, attachment, correctionSnapshot, {}, {volume->id()});
            require(vc3d::corrections::read(*session, attachment, half).collections.empty(), "dropped corrections are retired");
            require(vc3d::corrections::read(*session, attachment, native).collections.empty(), "corrections belong to one surface");
            vc3d::corrections::GrowthAnnotation annotation;
            annotation.surfaceUuid = half;
            annotation.beforeGeometry = "before";
            annotation.afterGeometry = "after";
            annotation.points = {{fix, 2, "fix", {1, 2, 3}, 20}, {fix, 1, "fix", {4, 5, 6}, 10}};
            const auto annotationUuid = vc3d::corrections::writeGrowthAnnotation(*session, attachment, annotation, {volume->id()});
            session->readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
                const auto object = reader.currentObject(annotationUuid);
                require(object && nlohmann::json::parse(object->record.meta)["source"]["growth_annotation"]["after_geometry"] == "after",
                        "growth annotations record the revisions");
                const auto points = std::get<vc::geometry::PointCollection>(
                    vc::geometry::decode(reader.readGeometryData(reader.currentGeometry(annotationUuid)->uuid)));
                require(points.ids == std::vector<int64_t>{1, 2}, "annotation points keep placement order");
            });
            require(vc3d::corrections::read(*session, attachment, half).collections.empty(), "annotations are not corrections");
        }

        session->tagObjects(attachment, {native}, {"reviewed"}, {"approved"});
        session->editRecords(attachment, {native}, [](ObjectRecord& record) { record.name = "renamed"; });
        pump([&] {
            const auto* row = catalog.row(native);
            return row && row->record.name == "renamed" && row->record.tags == std::vector<std::string>{"reviewed"};
        }, "record edits reach the rows through commit events");
        require(std::count(committed.begin(), committed.end(), native) >= 1, "commits name the edited surface");
        (void)session->editCurrent(attachment, {half}, [](const vc::objects::StoreReader&, const std::string& uuid) {
            return std::vector<vc::objects::TransactionOperation>{vc::objects::RetireObjectOp{uuid}};
        });
        pump([&] { return !catalog.row(half) && catalog.rows().size() == 3; }, "retired surfaces leave the listing");

        // The store supplies registered views; the application only adapts their tiles.
        VolumeAffine affine;
        affine(0,0)=2; affine(0,1)=0.5; affine(0,3)=10; affine(2,3)=20;
        vc::objects::registerVolumeTransform(*session,attachment,"vol2","vol1",affine,"registered scan");
        pump([&]{return catalog.row(other)&&catalog.row(other)->mapping.has_value();},"registration commits refresh compatible surfaces");
        {
            // Area: the store's, in stored units (vol2: the 2x2 unit grid); the
            // open volume, where the shear doubles x, sees twice it.
            require(catalog.row(other)->record.surface&&catalog.row(other)->record.surface->area==4.0,
                    "the store keeps the area in stored units");
            require(std::abs(([&] { const auto opened = catalog.open(other); const auto& points = *std::as_const(*opened.surface).rawPointsPtr(); return vc::geometry::surfaceGridArea(points.rows, points.cols,
            [&](int64_t r, int64_t c) -> const cv::Vec3f& { return points(int(r), int(c)); },
            [&](int64_t r, int64_t c) { return vc::geometry::validSurfacePoint(points(int(r), int(c))); }); }())-8.0)<1e-9,
                    "the open volume sees the sheared area");
        }
        auto registered=catalog.open(other);
        require((*registered.surface->rawPointsPtr())(2,1)==cv::Vec3f(13,2,21),"surface uses the store's affine view");
        SurfaceStoreEdit registeredEdit(session,registered.pin,registered.view);
        (*registered.surface->rawPointsPtr())(2,1)={15,2,21};
        registeredEdit.save(*registered.surface,cv::Rect(1,2,1,1));
        session->readStore(attachment,[&](const vc::objects::StoreReader& reader,const auto&) {
            require(GeometrySnapshot(reader).surface(other).vertex(2,1)==vc::geometry::SurfacePoint{2,2,1},
                    "registered view edits remain in canonical coordinates");
            require(nlohmann::json::parse(reader.currentObject(other)->record.meta)["coordinates"]["volume_uuid"]=="vol2",
                    "view edits preserve the paired volume");
        });
        vc::objects::registerVolumeTransform(*session,attachment,"vol2","vol1",VolumeAffine::scale(3),"alternative");
        bool staleMapping=false;
        try {registeredEdit.save(*registered.surface,cv::Rect(1,2,1,1));}catch(const std::exception&){staleMapping=true;}
        require(staleMapping,"registration conflicts invalidate writable views");

        // What VC3D does with registrations, through the store: list the related
        // volumes, choose a route for an ambiguous pair, register a new affine.
        {
            namespace reg = vc3d::registrations;
            auto relatedTo = [&](const std::string& volumeId) {
                std::vector<VolumeTransforms::Related> result;
                result = session->volumeTransforms(attachment).related(volumeId);
                return result;
            };
            auto related = relatedTo("vol1");
            require(related.size() == 1 && related[0].volume == "vol2" &&
                    related[0].status == vcr::catalog::TransformStatus::Ambiguous && related[0].routes.size() == 2 &&
                    !related[0].chosen, "two registrations of vol2 make an ambiguous pair");
            const auto chosenPath = related[0].routes[1].path;
            chooseVolumeTransform(*session, attachment, "vol2", "vol1", chosenPath);
            related = relatedTo("vol1");
            require(related[0].status == vcr::catalog::TransformStatus::Resolved && related[0].chosen == chosenPath,
                    "a chosen route resolves the pair");
            chooseVolumeTransform(*session, attachment, "vol2", "vol1", related[0].routes[0].path);
            require(relatedTo("vol1")[0].chosen == related[0].routes[0].path, "choosing again replaces the choice");
            const auto opened = catalog.open(other);
            require(opened.view.mapping == related[0].routes[0].forward, "views follow the chosen route");

            require(reg::parseAffine("[[2, 0, 0, 1], [0, 2, 0, 0], [0, 0, 2, 0]]") ==
                        VolumeAffine::fromJson(nlohmann::json::parse("[[2,0,0,1],[0,2,0,0],[0,0,2,0]]")),
                    "a pasted JSON matrix parses");
            require(reg::parseAffine("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1") == VolumeAffine::identity(), "4x4 rows parse");
            bool rejected = false;
            try { reg::parseAffine("1 0 0\n0 1 0"); } catch (const std::exception&) { rejected = true; }
            require(rejected, "a matrix of the wrong size is refused");
            rejected = false;
            try { reg::parseAffine("0 0 0 0 0 0 0 0 0 0 0 0"); } catch (const std::exception&) { rejected = true; }
            require(rejected, "a singular matrix is refused");

            registerVolumeTransform(*session, attachment, "vol3", "vol1", VolumeAffine::scale(4), "test scan");
            const auto withNew = relatedTo("vol1");
            const auto vol3 = std::find_if(withNew.begin(), withNew.end(), [](const auto& r) { return r.volume == "vol3"; });
            require(vol3 != withNew.end() && vol3->status == vcr::catalog::TransformStatus::Resolved &&
                    vol3->routes.front().forward == VolumeAffine::scale(4), "a registered affine relates its volume");
            require(session->volumeTransforms(attachment).graph.describe(vol3->routes.front().path).find("test scan") != std::string::npos,
                    "routes are described by registration name");
        }

        // The volume at working level 1 is the same volume in half its voxels:
        // surfaces are listed at that scale, and those made there are kept in
        // the volume's own voxels at it, seen at full scale in the full volume.
        const std::filesystem::path volumeRoot2 = fs::path(dir.path().toStdString()) / "level.zarr";
        const vcr::zarr::CreateOptions levelZero2{.shape = {2, 2, 2}, .chunks = {2, 2, 2}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot2 / "0", levelZero2);
        std::ofstream(volumeRoot2 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "vol1"}, {"name", "vol1"}, {"format", "zarr"}, {"width", 2}, {"height", 2}, {"slices", 2}, {"voxelsize", 1.0}}.dump();
        const auto level = Volume::New(volumeRoot2);
        const VolumeVoxels atHalf{volume->id(), 0.5};
        const auto atLevel = util::newUuid();
        {
            ObjectRecord record; record.uuid = atLevel; record.kind = "surface"; record.name = "at-level";
            record.meta = nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(atHalf)}}.dump();
            storeSurface(*session, attachment, record, quad_surface_geometry(grid));
        }
        catalog.setSource(attachment, level, atHalf);
        pump([&] { const auto* row = catalog.row(native); return !catalog.loading() && row && row->mapping; },
             "the level lists the surfaces of its volume");
        require(*catalog.row(native)->mapping == VolumeAffine::scale(0.5), "seen at the level's scale");
        require(catalog.row(atLevel)->toolInput(), "tools at the level take surfaces made there");
        catalog.setSource(attachment, volume, {volume->id()});
        pump([&] { const auto* row = catalog.row(atLevel); return !catalog.loading() && row && row->mapping; },
             "the full volume lists surfaces made at the level");
        require(*catalog.row(atLevel)->mapping == VolumeAffine::scale(2), "seen at the full volume's scale");

        catalog.setSource(std::nullopt, volume, {volume->id()});
        require(catalog.rows().empty(), "no attachment lists nothing");

        // A cold intersection query waits for every selected placeholder and
        // returns their crossings together, without fetching unrelated surfaces.
        {
            const auto remotePath = fs::path(dir.path().toStdString()) / "remote.vcobj";
            const std::vector<std::string> ids{util::newUuid(), util::newUuid(), util::newUuid(), util::newUuid()};
            {
                auto db = ObjectStore::create(remotePath, options);
                auto txn = db.write();
                for (const auto& uuid : ids) {
                    ObjectRecord record;
                    record.uuid = uuid;
                    record.kind = "surface";
                    record.storageKind = "file";
                    record.storageRef = uuid;
                    record.meta = nlohmann::json{{"coordinates", volumeCoordinates(volume->id(), 1)},
                                                 {"published", true}}.dump();
                    record.surface.emplace();
                    record.surface->bbox = std::array<double, 6>{0, 0, 1, 2, 2, 1};
                    db.upsertObject(txn, record);
                }
                txn.commit();
            }
            // The session owns the fetch thread; destroy it before its captures.
            std::atomic_int fetchCount{0};
            ObjectStoreSession remoteSession;
            remoteSession.setGeometryFetcher("published",
                [&](ObjectStore& store, const std::string& uuid, const ObjectStoreSession::FetchWrite& write) {
                    if (uuid == ids[2]) throw std::runtime_error("test fetch failure");
                    const auto version = store.currentVersion(store.read(), uuid);
                    ++fetchCount;
                    write({{uuid, version}}, [&](auto& db, auto& txn) {
                        db.putGeometry(txn, uuid, quad_surface_geometry(grid));
                    });
                });
            const auto remote = remoteSession.attach(remotePath, options);
            RegionSlice slice;
            slice.origin = {0, 1, 0};
            slice.u = {1, 0, 0};
            slice.v = {0, 0, 1};
            slice.normal = {0, 1, 0};
            slice.localBounds = {-1, -1, -1, 3, 3, 1};
            RegionOptions query;
            query.uuids = std::vector<std::string>{ids[0], ids[1]};
            const auto cold = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            std::set<std::string> reached;
            for (const auto& [attachment, uuid] : cold.remote) reached.insert(uuid);
            require(cold.runs.empty() && reached == std::set<std::string>{ids[0], ids[1]},
                    "a cold query lists the selected placeholders it reached");
            for (const auto& [attachment, uuid] : cold.remote) remoteSession.ensureGeometry(attachment, uuid);
            const auto crossings = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            std::set<std::string> crossed;
            for (const auto& run : crossings.runs) crossed.insert(run.objectUuid);
            require(crossings.complete && crossings.remote.empty() && crossed == std::set<std::string>{ids[0], ids[1]} &&
                        fetchCount == 2,
                    "fetched placeholders are read, and only the selected surfaces were fetched");
            const auto warm = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            require(warm.complete && warm.runs.size() == crossings.runs.size() && fetchCount == 2,
                    "warm queries reuse fetched geometry");
            query.uuids = std::vector<std::string>{ids[2]};
            const auto failing = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query);
            bool failed = false;
            try { for (const auto& [attachment, uuid] : failing.remote) remoteSession.ensureGeometry(attachment, uuid); }
            catch (const GeometryUnavailable&) { failed = true; }
            require(failing.remote.size() == 1 && failed, "fetch errors are reported to the fetcher");
            query.uuids = std::vector<std::string>{ids[3]};
            const auto cancelled = SurfaceCatalog::region(remoteSession, {remote}, {volume->id()}, slice, query,
                                                          [] { return true; });
            require(!cancelled.complete && cancelled.runs.empty() && fetchCount == 2,
                    "cancelled queries neither publish intersections nor fetch more surfaces");
        }
        // Durable saves keep the editor's draft (marked saved, not
        // checkpointed); an unsaved later edit is checkpointed and recovered.
        {
            const auto draftDirectory = fs::path(dir.path().toStdString()) / "drafts";
            cv::Mat_<cv::Vec3f> stored(3, 3);
            session->readStore(attachment, [&](const StoreReader& reader, const auto&) {
                const auto view = GeometrySnapshot(reader).surface(native);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) {
                        const auto p = view.vertex(r, c);
                        stored(r, c) = {p[0], p[1], p[2]};
                    }
            });
            ObjectVersion saved;
            {
                auto durable = std::make_shared<ObjectStoreSession>();
                SessionDrafts drafts(durable);
                require(drafts.start(draftDirectory, 0) && drafts.recoveredDrafts().empty(), "draft directory in use");
                const auto store = durable->attach(path, options);
                const auto pin = pinOf(store, *durable->currentObject(store, native), durable->status(store).generation);
                QuadSurface surface(stored, {1.f, 1.f});
                surface.id = native;
                const auto coordinates = vc::objects::volumeCoordinates(volume->id(), 1);
                durable->setDraft(native, quad_surface_grid(surface), coordinates, store, pin.reference());
                durable->markDraftSaved(native, durable->draftRevision(native));
                const auto drafted = [&](int row, int column, cv::Vec3f point) {
                    (*surface.rawPointsPtr())(row, column) = point;
                    const vc::geometry::SurfacePoint p{point[0], point[1], point[2]};
                    durable->updateDraft(native, row, column, 1, 1, std::span(&p, 1));
                };
                drafted(0, 1, {7.f, 7.f, 7.f});
                SurfaceStoreEdit edit(durable, pin);
                edit.save(surface, cv::Rect(1, 0, 1, 1), {}, durable->draftRevision(native));
                saved = edit.pin().reference();
                const auto state = durable->draftState(native);
                require(saved.version == pin.reference().version + 1 && state && state->saved && state->base == saved &&
                        drafts.checkpointNow(), "the save keeps the draft as saved");
                const auto files = [&] {
                    std::size_t n = 0;
                    for (const auto& entry : fs::directory_iterator(draftDirectory))
                        n += entry.path().filename().string().rfind("draft-", 0) == 0;
                    return n;
                };
                require(files() == 0, "a saved draft is not checkpointed");
                drafted(2, 2, {8.f, 8.f, 8.f});
                cv::Mat_<uint8_t> mask(3, 3, uint8_t(255));
                mask(0, 2) = 0;
                surface.setChannel("mask", mask);
                (*surface.rawPointsPtr())(0, 2) = {-1, -1, -1};
                durable->setDraft(native, quad_surface_grid(surface), coordinates, store, saved);
                require(drafts.checkpointNow() && files() == 1, "an unsaved edit is checkpointed");
            }
            auto restarted = std::make_shared<ObjectStoreSession>();
            SessionDrafts drafts(restarted);
            require(drafts.start(draftDirectory, 0) && drafts.recovered(native), "the unsaved edit is recovered");
            const auto contents = restarted->draftContents(native);
            require(contents && contents->surface.points[8] == vc::geometry::SurfacePoint{8, 8, 8} &&
                    contents->surface.points[1] == vc::geometry::SurfacePoint{7, 7, 7} &&
                    restarted->draftState(native)->base == saved, "recovered grid and base");
            const auto store = restarted->attach(path, options);
            auto opened = SurfaceCatalog::open(restarted, store, {volume->id()}, native);
            SessionDrafts::restoreSurface(*opened.surface, contents->surface);
            require(opened.surface->channel("mask", SURF_CHANNEL_NORESIZE).at<uint8_t>(0, 2) == 0,
                    "restoring a draft replaces the open surface's validity mask");
            SurfaceStoreEdit edit(restarted, opened.pin, opened.view);
            edit.save(*opened.surface, cv::Rect(0, 0, 3, 3), {"mask"}, contents->revision);
            restarted->readStore(store, [&](const StoreReader& reader, const auto&) {
                const auto view = GeometrySnapshot(reader).surface(native);
                const auto tile = view.readTile(0, 0, 3, 3);
                require(!tile.validity[2] && tile.points[2] == vc::geometry::SurfacePoint{stored(0, 2)[0], stored(0, 2)[1], stored(0, 2)[2]},
                        "saving a recovered mask keeps the original stored coordinates");
                require(tile.points[8] == vc::geometry::SurfacePoint{8, 8, 8}, "recovered vertex edits are saved too");
            });
            restarted->clearDraft(native);
            drafts.forgetRecovered(native);
            require(!drafts.recovered(native), "discarded");
        }
#ifdef VC3D_TEST_HOSTED
        // A hosted store: the catalog lists and opens its surfaces, edits save
        // through the service, and an outage is reported and fails saves cleanly.
        {
            const auto hostedPath = fs::path(dir.path().toStdString()) / "hosted.vcobj";
            std::string uuid;
            {
                auto store = ObjectStore::create(hostedPath, options);
                auto write = store.write();
                ObjectRecord record; record.kind = "surface"; record.name = "hosted";
                record.meta = nlohmann::json{{"coordinates", vc::objects::volumeCoordinates(volume->id(), 1)}}.dump();
                uuid = store.upsertObject(write, record);
                store.putGeometry(write, uuid, quad_surface_geometry(grid));
                write.commit();
            }
            server::ServerOptions serverOptions;
            auto service = std::make_unique<server::Server>(hostedPath, serverOptions);
            service->start();
            serverOptions.port = service->port();
            const auto endpoint = "http://127.0.0.1:" + std::to_string(serverOptions.port);
            auto hostedSession = std::make_shared<ObjectStoreSession>();
            HostedSessionOptions hostedOptions;
            hostedOptions.author = "surface-catalog-test";
            hostedOptions.pollInterval = std::chrono::hours(1);
            hostedOptions.client.retries = 0;
            hostedOptions.client.connectTimeoutMs = 500;
            const auto hosted = hostedSession->attachHosted(endpoint, hostedOptions);
            require(hostedSession->location(hosted) == endpoint && hostedSession->freshness(hosted)->error.empty(),
                    "hosted attachment");
            SurfaceCatalog hostedCatalog(hostedSession);
            hostedCatalog.setSource(hosted, volume, {volume->id()});
            QElapsedTimer timer; timer.start();
            while ((hostedCatalog.loading() || hostedCatalog.rows().size() != 1) && timer.elapsed() < 10000) {
                for (const auto& event : hostedSession->takeEvents()) hostedCatalog.handle(event);
                QCoreApplication::processEvents();
                QThread::msleep(1);
            }
            require(hostedCatalog.rows().size() == 1 && hostedCatalog.row(uuid), "hosted surfaces are listed");
            auto opened = hostedCatalog.open(uuid);
            require(opened.editable && opened.pin.reference().version == 1, "hosted surfaces open editable");
            SurfaceStoreEdit edit(hostedSession, opened.pin);
            auto& points = *opened.surface->rawPointsPtr();
            const auto vertexAt = [&](int row, int column) {
                cv::Vec3f value;
                hostedSession->readStore(hosted, [&](const StoreReader& reader, const auto&) {
                    const auto p = GeometrySnapshot(reader).surface(uuid).vertex(row, column);
                    value = {p[0], p[1], p[2]};
                });
                return value;
            };
            points(1, 1) = {4.f, 4.f, 4.f};
            edit.save(*opened.surface, cv::Rect(1, 1, 1, 1));
            require(edit.pin().reference().version == 2 && vertexAt(1, 1) == cv::Vec3f(4, 4, 4), "hosted save");

            // The service goes away: the check reports it, a save fails before
            // sending anything (its reads need the service), and nothing is pending.
            service.reset();
            hostedSession->refresh(hosted);
            require(!hostedSession->freshness(hosted)->error.empty(), "an unanswered check is reported");
            points(0, 0) = {6.f, 6.f, 6.f};
            bool failed = false;
            try { edit.save(*opened.surface, cv::Rect(0, 0, 1, 1)); } catch (const std::exception&) { failed = true; }
            require(failed && hostedSession->pendingSubmissions().empty() && edit.pin().reference().version == 2,
                    "a save without the service fails cleanly");
            service = std::make_unique<server::Server>(hostedPath, serverOptions);
            service->start();
            hostedSession->refresh(hosted);
            require(hostedSession->freshness(hosted)->error.empty(), "current again");
            points(2, 2) = {9.f, 9.f, 9.f};
            edit.save(*opened.surface, cv::Rect(0, 0, 3, 3));
            require(edit.pin().reference().version == 3 && vertexAt(0, 0) == cv::Vec3f(6, 6, 6) && vertexAt(2, 2) == cv::Vec3f(9, 9, 9),
                    "the next save carries the edits");
            hostedSession->detach(hosted);
        }
#endif
        session->detach(attachment);
        std::cout << "surface catalog tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface catalog test: " << error.what() << '\n';
        return 1;
    }
}
