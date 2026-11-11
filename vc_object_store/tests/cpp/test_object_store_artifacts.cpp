// Artifacts: content-addressed durable bytes beside the database, and object
// manifests that may only name bytes the store holds, on every write path.
#include "TestSupport.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/ToolObjects.hpp"

#include <fstream>
#include <iostream>

using namespace vc::objects;

namespace {
ObjectRecord trackRecord(const std::string& uuid)
{
    ObjectRecord record; record.uuid = uuid; record.kind = "track"; record.track.emplace();
    record.track->datasetRef = "artifacts"; record.track->externalKey = uuid;
    return record;
}

void writeFile(const std::filesystem::path& path, const std::string& bytes)
{
    std::ofstream(path, std::ios::binary) << bytes;
}

void referencedArtifacts()
{
    for(const auto baseHistory:{HistoryMode::Off,HistoryMode::On})
    for(const auto overlayHistory:{HistoryMode::Off,HistoryMode::On}) {
        TemporaryDirectory directory;
        const auto basePath=directory.path/"base",middlePath=directory.path/"middle",overlayPath=directory.path/"overlay";
        const auto id=util::newUuid();std::string hash;
        OpenOptions options;options.author="artifact-references";options.history=baseHistory;
        writeFile(directory.path/"bytes","reference artifact bytes");
        {
            auto base=ObjectStore::create(basePath,options);
            hash=base.putArtifact(directory.path/"bytes");
            auto w=base.write();base.upsertObject(w,trackRecord(id));
            base.setMeta(w,id,kArtifactsMetaKey,artifactManifestJson({{"bytes",hash,24}}));
            base.seal(w);w.commit();
        }
        options.history=overlayHistory;
        {
            auto middle=ObjectCollection::create(middlePath,{basePath},options);
            auto w=middle.write();middle.setMeta(w,{{},id},"note","true");middle.seal(w);w.commit();
        }
        auto overlay=ObjectCollection::create(overlayPath,{middlePath},options);
        {
            auto w=overlay.write();overlay.setMeta(w,{{},id},"label","3");w.commit();
            auto r=overlay.read();
            require(overlay.artifactLength(r,hash)==24,"artifact size resolves through nested references");
            const auto bytes=overlay.readArtifact(r,hash,10,8);
            require(std::string(bytes.begin(),bytes.end())=="artifact","artifact range resolves through nested references");
        }
        require(!std::filesystem::exists(artifactBlobDirectory(overlayPath))&&!std::filesystem::exists(artifactBlobDirectory(middlePath)),
            "metadata overrides never copy artifact payloads");
        mustFail([&]{auto w=overlay.write();overlay.setMeta(w,{{},id},kArtifactsMetaKey,
            artifactManifestJson({{"missing",std::string(64,'e'),5}}));w.commit();});
        {
            auto r=overlay.read();require(artifactManifest(overlay.object(r,{{},id})->record).front().sha256==hash,
                "invalid artifact replacement rolls back");
        }
        // Direct history-off creates are checked too, not only edits.
        options.history=HistoryMode::Off;
        auto plain=ObjectStore::create(directory.path/"plain",options);
        mustFail([&]{auto w=plain.write();auto record=trackRecord(util::newUuid());
            record.meta=nlohmann::json{{kArtifactsMetaKey,nlohmann::json::parse(artifactManifestJson({{"missing",hash,24}}))}}.dump();
            plain.upsertObject(w,record);w.commit();});
    }
}
} // namespace

int main()
try {
    referencedArtifacts();
    TemporaryDirectory directory;
    const auto path = directory.path / "artifacts.vcobj";
    OpenOptions options; options.author = "artifacts-test";
    auto db = ObjectStore::create(path, options);
    const auto status = [&] { auto txn = db.read(); return db.status(txn); }();
    auto request = [&](VersionSelection expected, std::vector<TransactionOperation> operations) {
        TransactionRequest r;
        r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
        r.author = "alice"; r.expectedObjects = testNumberedGuards(db, expected); r.operations = std::move(operations);
        return r;
    };

    // Bytes are copied in under their SHA-256, read-only, once.
    std::string model(3 << 20, '\0');
    for (std::size_t i = 0; i < model.size(); ++i) model[i] = char(i * 131 % 251);
    writeFile(directory.path / "model.pt", model);
    const auto hash = db.putArtifact(directory.path / "model.pt");
    require(hash == payloadHash(std::vector<uint8_t>(model.begin(), model.end())), "artifact hash");
    require(std::filesystem::exists(directory.path / "model.pt"), "copy keeps the source");
    const auto blob = artifactBlobPath(path, hash);
    require(std::filesystem::file_size(blob) == model.size(), "bytes beside the database");
    require((std::filesystem::status(blob).permissions() & std::filesystem::perms::owner_write) ==
                std::filesystem::perms::none, "artifact bytes are read-only");
    require(db.putArtifact(directory.path / "model.pt") == hash, "same content, same artifact");
    require(db.artifactLength(hash) == model.size() && !db.artifactLength(std::string(64, '0')), "artifact lengths");
    {
        const auto part = db.readArtifact(hash, 1000, 5000);
        require(std::string(part.begin(), part.end()) == model.substr(1000, 5000), "ranged artifact read");
        const auto whole = db.readArtifact(hash);
        require(std::string(whole.begin(), whole.end()) == model, "whole artifact read");
        mustFail([&] { (void)db.readArtifact(hash, model.size() - 2, 3); });
        mustFail([&] { (void)db.readArtifact(std::string(64, '0')); });
    }
    std::size_t blobs = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(artifactBlobDirectory(path)))
        blobs += entry.is_regular_file();
    require(blobs == 1, "no duplicate or leftover files");

    // Moving takes the file; an empty artifact is still an artifact.
    writeFile(directory.path / "log.txt", "fit log\n");
    const auto logHash = db.putArtifact(directory.path / "log.txt", true);
    require(!std::filesystem::exists(directory.path / "log.txt") && db.artifactLength(logHash) == 8u, "move");
    writeFile(directory.path / "empty", "");
    const auto emptyHash = db.putArtifact(directory.path / "empty");
    require(db.artifactLength(emptyHash) == 0u && db.readArtifact(emptyHash).empty(), "empty artifact");

    // Manifests: canonical, validated names, versioned with the object.
    const auto object = util::newUuid();
    db.submit(request({{object, std::nullopt}}, {CreateObjectOp{objectRecordJson(trackRecord(object))}}));
    const std::vector<ArtifactFile> files{{"model/weights.pt", hash, model.size()}, {"log.txt", logHash, 8}};
    const auto manifest = artifactManifestJson(files);
    require(manifest == artifactManifestJson({files[1], files[0]}), "manifest order is canonical");
    for (const auto* bad : {"", "/abs", "a/../b", "a//b", "./a", "a\\b"})
        mustFail([&] { (void)artifactManifestJson({{bad, hash, model.size()}}); });
    mustFail([&] { (void)artifactManifestJson({{"x", "ABC", 1}}); });
    mustFail([&] { (void)artifactManifestJson({files[0], files[0]}); });

    const auto published = db.submit(request({{object, 1}}, {SetMetaOp{object, kArtifactsMetaKey, manifest}}));
    require(published.objects.at(object) == 2, "manifest publication versions the object");
    {
        auto txn = db.read();
        const auto head = db.currentObjectRevision(txn, object);
        auto sorted = files;
        std::sort(sorted.begin(), sorted.end(), [](const auto& x, const auto& y) { return x.name < y.name; });
        require(artifactManifest(head->record) == sorted, "manifest read back");
        const auto first = db.objectAtVersion(txn, {status.storeUuid, status.generation, object, 1});
        require(first && artifactManifest(first->record).empty(), "earlier version has no manifest");
    }

    // A manifest naming bytes the store does not hold is refused, through
    // checked requests and the direct mutation API alike; nothing commits.
    const auto missing = artifactManifestJson({{"other.bin", std::string(64, 'e'), 4}});
    const auto wrongLength = artifactManifestJson({{"log.txt", logHash, 9}});
    for (const auto& value : {missing, wrongLength, std::string(R"([{"name":"../x","sha256":")") + hash + "\",\"length\":1}]",
                              std::string(R"({"not":"an array"})")}) {
        mustFail([&] { db.submit(request({{object, 2}}, {SetMetaOp{object, kArtifactsMetaKey, value}})); });
        mustFail([&] {
            auto txn = db.write();
            db.setMeta(txn, object, kArtifactsMetaKey, value);
            txn.commit();
        });
    }
    {
        auto txn = db.read();
        require(db.status(txn).commitSequence == *published.commitSequence, "refused manifests commit nothing");
    }
    // Other edits of an object with a manifest are not re-checked.
    const auto other = db.submit(request({{object, 2}}, {SetMetaOp{object, "note", "\"ok\""}}));
    require(other.objects.at(object) == 3, "unrelated edit");
    {
        auto txn = db.write();
        db.setMeta(txn, object, kArtifactsMetaKey, artifactManifestJson({files[1]}));
        txn.commit();
        auto read = db.read();
        require(db.currentObjectRevision(read, object)->version == 4, "direct API publishes held manifests");
    }

    // Read-only handles read artifacts but cannot add them.
    {
        OpenOptions readOnly; readOnly.readOnly = true;
        auto inspect = ObjectStore::open(path, readOnly);
        require(inspect.artifactLength(hash) == model.size(), "read-only artifact read");
        writeFile(directory.path / "more", "more");
        mustFail([&] { (void)inspect.putArtifact(directory.path / "more"); });
    }

    // A reset changes the write lineage, not the generations of existing heads.
    {
        ObjectStoreSession session;
        const auto attachment = session.attach(path, options);
        const auto before = session.currentObject(attachment, object);
        auto wrongOwner = before->reference(); wrongOwner.storeUuid = util::newUuid();
        mustFail([&] { session.publishArtifacts(attachment, wrongOwner, {}); });
        const auto oldPin = pinOf(attachment, *before, session.status(attachment).generation);
        ToolObjects tool(path, "loaded-artifact-test");
        tool.noteRead(pinOf(tool.attachment(), *tool.session()->currentObject(tool.attachment(), object),
                          tool.session()->status(tool.attachment()).generation), {}, true);
        const auto generation = db.startHistoryGeneration("restored backup");
        mustFail([&] { session.requireCurrent(oldPin); });
        mustFail([&] { session.publishArtifacts(oldPin, {}); });
        mustFail([&] { tool.publishArtifacts(object, {}); });
        const auto reopened = pinOf(attachment, *session.currentObject(attachment, object), generation);
        require(reopened.reference() == oldPin.reference() && reopened.writeGeneration != oldPin.writeGeneration,
                "reopening preserves the historical reference and observes the new write generation");
        session.requireCurrent(reopened);
        require(session.artifacts(attachment, object, before->version, before->generation) ==
                    std::vector<ArtifactFile>{files[1]}, "historical artifacts after reset");
        mustFail([&] { (void)session.artifacts(attachment, object, before->version, generation); });
        const auto receipt = session.publishArtifacts(attachment, before->reference(), {{"more.txt", directory.path / "more"}});
        require(receipt.generation == generation && session.artifacts(attachment, object).size() == 2,
                "publication preserves the old manifest across generations");
        session.materializeArtifacts(attachment, object, directory.path / "old", before->version, before->generation);
        require(std::filesystem::file_size(directory.path / "old/log.txt") == 8, "materialize old-generation artifacts");
        require(session.artifactFile(attachment, object, "log.txt", directory.path / "cache", before->version,
                                     before->generation).has_value(), "cache old-generation artifact");
        bool conflict = false;
        try { session.publishArtifacts(attachment, before->reference(), {}); }
        catch (const TransactionConflict& e) { conflict = e.reason() == TransactionConflict::Reason::Version; }
        require(conflict && session.artifacts(attachment, object).size() == 2, "stale artifact publisher cannot overwrite manifest");
    }

    std::cout << "object_store_artifacts: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object_store_artifacts: " << error.what() << "\n";
    return 1;
}
