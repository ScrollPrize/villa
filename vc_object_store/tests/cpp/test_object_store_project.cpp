#include "TestSupport.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/Project.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
OpenOptions as(const std::string& author)
{
    OpenOptions options;
    options.author = author;
    return options;
}

std::vector<ProjectResource> resources(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                       const std::string& kind, const std::optional<std::string>& viewer = std::nullopt)
{
    std::vector<ProjectResource> result;
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
        result = projectResources(reader, kind, viewer);
    });
    return result;
}

Json view(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& author)
{
    Json result;
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) { result = projectView(reader, author); });
    return result;
}

void ownerHistoryPolicy(HistoryMode sourceMode,HistoryMode overlayMode)
{
    TemporaryDirectory dir;auto sourceOptions=as("test");sourceOptions.history=sourceMode;
    const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto deleted=util::newUuid();
    const auto resource=projectResourceRecord(kVolumeResource,"/volumes/test");
    {auto store=ObjectStore::create(base,sourceOptions);auto w=store.write();
        store.upsertObject(w,resource);
        for(int i=0;i<2;++i){ObjectRecord r;r.kind=kProjectKind;store.upsertObject(w,r);}
        ObjectRecord r;r.uuid=deleted;r.kind="note";store.upsertObject(w,r);w.commit();
        auto remove=store.write();store.deleteObject(remove,deleted);store.seal(remove);remove.commit();}
    auto overlayOptions=as("test");overlayOptions.history=overlayMode;
    {auto collection=ObjectCollection::create(overlay,{base},overlayOptions);}
    auto store=ObjectStore::open(overlay,overlayOptions);
    const auto local=projectResourceRecord(kVolumeResource,"/volumes/local");
    if(sourceMode!=overlayMode){
        auto w=store.write();store.upsertObject(w,local);w.commit();
    }
    {
        auto txn=store.read();LocalStoreReader reader(store,txn);
        const auto listed=projectResources(reader,kVolumeResource);
        if(sourceMode!=overlayMode)
            require(listed.size()==2 && listed[0].uuid==resource.uuid && listed[1].uuid==local.uuid,
                    "mixed history modes order resources by original creation");
        else require(listed.size()==1,"resource discovery uses owner history policy");
        require(projectRecord(reader).has_value(),"project ordering uses owner history policy");
        require(reader.objectHistory(deleted,true).size()==(sourceMode==HistoryMode::On?2:0),"optional history follows winning tombstone owner");
        require(currentVersionOf(reader,deleted)==(sourceMode==HistoryMode::On?std::optional<int64_t>(2):std::nullopt),
            "retained tombstone version does not depend on overlay history policy");
        if(sourceMode==HistoryMode::Off)mustFail([&]{reader.objectHistory(resource.uuid);});
    }
    if(sourceMode!=overlayMode) {
        auto w=store.write();
        auto override=resource;override.meta=R"({"reviewed":true})";
        store.upsertObject(w,override);w.commit();
        auto r=store.read();
        require(projectResources(LocalStoreReader(store,r),kVolumeResource).front().uuid==resource.uuid,
                "metadata override does not change original creation order");
    }
}

void copyCollectionProject(bool packed,bool sharded)
{
    TemporaryDirectory dir;const auto sourceDir=dir.path/"source",destinationDir=dir.path/"destination";
    std::filesystem::create_directories(sourceDir);std::filesystem::create_directories(destinationDir);
    auto options=as("copy-test");options.history=HistoryMode::Off;options.spatialIndex=false;
    std::filesystem::path source=sourceDir/"root.vcobj";
    std::string artifactOwner;const auto payload=sourceDir/"payload.txt";std::ofstream(payload)<<"retained payload";
    if(sharded){
        const auto owner=util::newUuid(),lineage=util::newUuid();std::vector<std::filesystem::path> shards;
        for(int i=0;i<2;++i){
            auto opt=options;opt.creationStoreUuid=owner;opt.creationGeneration=lineage;
            opt.objectIdRange={1+i*100,100+i*100};opt.edgeIdRange=opt.objectIdRange;
            const auto path=sourceDir/("shard"+std::to_string(i)+".vcobj");shards.push_back(path);
            auto store=ObjectStore::create(path,opt);auto w=store.write();
            store.upsertObject(w,projectResourceRecord(kVolumeResource,"/volume/"+std::to_string(i)));w.commit();
            auto seal=store.write();store.seal(seal);seal.commit();
        }
        auto collection=ObjectCollection::publishShards(source,shards,options);
    }else{
        const auto base=sourceDir/"base.vcobj";createProject(base,"source project",Json::object(),options);
        {ObjectStoreSession session;const auto attachment=session.attach(base,options);
            addProjectResource(session,attachment,kVolumeResource,"/volume/base");
            session.readStore(attachment,[&](const StoreReader& reader,const auto&){artifactOwner=projectRecord(reader)->uuid;});
            session.publishArtifacts(attachment,artifactOwner,session.currentObject(attachment,artifactOwner)->version,{{"note.txt",payload}},"copy-test");}
        {auto store=ObjectStore::open(base,options);auto w=store.write();store.seal(w);w.commit();}
        {auto collection=ObjectCollection::create(source,{base},options);}
        {ObjectStoreSession session;const auto attachment=session.attach(source,options);
            addProjectResource(session,attachment,kVolumeResource,"/volume/overlay");}
    }
    if(packed){
        if(!sharded){auto store=ObjectStore::open(source,options);auto w=store.write();store.seal(w);w.commit();}
        const auto output=sourceDir/"packed.vcobj";packObjectCollection(source,output);source=output;
    }
    const auto destination=destinationDir/"copy.vcobj";
    {ObjectStoreSession session;auto readOptions=options;readOptions.readOnly=true;
        const auto attachment=session.attach(source,readOptions);copyProject(session,attachment,destination,"independent",options);}
    // The copied closure must no longer rely on either original paths or a
    // source packed container. Renaming the temporary source tree makes that explicit.
    std::filesystem::rename(sourceDir,dir.path/"source-unavailable");
    {ObjectStoreSession session;const auto attachment=session.attach(destination,options);
        require(resources(session,attachment,kVolumeResource).size()==2,"copied collection retains both resources");
        session.readStore(attachment,[&](const StoreReader& reader,const auto&){
            require(projectRecord(reader)->name=="independent","copied collection has a mutable independent project");});
        if(!artifactOwner.empty()){
            const auto files=session.artifacts(attachment,artifactOwner);
            require(files.size()==1,"inherited artifact manifest survives copy");
            const auto local=session.artifactFile(attachment,artifactOwner,"note.txt",dir.path/"cache");
            require(local&&std::filesystem::file_size(*local)==16,"inherited artifact content survives copy");
        }
        setProjectName(session,attachment,"renamed again");}
    {ObjectStoreSession session;const auto attachment=session.attach(destination,options);
        require(resources(session,attachment,kVolumeResource).size()==2,"copy remains readable after another root commit");}
}

void missingArtifactExport(HistoryMode history)
{
    TemporaryDirectory dir;const auto source=dir.path/"source.vcobj",payload=dir.path/"payload";
    std::ofstream(payload)<<"retained bytes";
    auto options=as("artifact-copy");options.history=history;
    std::string hash;
    {auto store=ObjectStore::create(source,options);hash=store.putArtifact(payload);
        auto w=store.write();ObjectRecord record;record.uuid=util::newUuid();record.kind="note";
        record.meta=Json{{kArtifactsMetaKey,Json::parse(artifactManifestJson({{"note.txt",hash,14}}))}}.dump();
        store.upsertObject(w,record);w.commit();
        auto finish=store.write();
        if(history==HistoryMode::On)store.removeMeta(finish,record.uuid,kArtifactsMetaKey);
        store.seal(finish);finish.commit();}
    // History-on: only an older revision still needs the blob. History-off:
    // the current manifest does. Neither export may silently drop the bytes.
    std::filesystem::remove(artifactBlobPath(source,hash));
    for(const bool packed:{false,true}){
        const auto target=dir.path/(packed?"packed.vcobj":"copy.vcobj");
        mustFail([&]{if(packed)packObjectCollection(source,target);else copyObjectCollection(source,target,options);});
        for(const auto* suffix:{"",".artifacts",".stores"})
            require(!std::filesystem::exists(target.string()+suffix),"failed artifact export removes owned output and sidecars");
    }
}
} // namespace

int main() try
{
    for (auto history : {HistoryMode::On, HistoryMode::Off}) {
        TemporaryDirectory dir;
        auto options = as("copy-test"); options.history = history;
        auto store = ObjectStore::create(dir.path / "copy.vcobj", options);
        ObjectCopy copy;
        copy.record.uuid = util::newUuid(); copy.record.kind = "fiber";
        copy.geometry.emplace(); copy.geometry->kind = "surface"; // invalid for this target
        auto write = store.write();
        mustFail([&] { writeObjectCopy(store, write, copy); });
        write.commit(); // catching a failure must not publish partial metadata
        auto read = store.read();
        require(!store.objectByUuid(read, copy.record.uuid), "failed composite copy rolled back");
    }
    for(const auto history:{HistoryMode::Off,HistoryMode::On})missingArtifactExport(history);
    for(const auto source:{HistoryMode::Off,HistoryMode::On})for(const auto overlay:{HistoryMode::Off,HistoryMode::On})
        ownerHistoryPolicy(source,overlay);
    for(const bool packed:{false,true})for(const bool sharded:{false,true})copyCollectionProject(packed,sharded);
    TemporaryDirectory directory;
    const auto path = directory.path / "project.vcobj";

    {
        const auto sourcePath = directory.path / "extensions.vcobj";
        const auto targetPath = directory.path / "extensions-copy.vcobj";
        createProject(sourcePath, "extension source", Json::object(), as("alice"));
        const auto owner = vc::objects::util::newUuid();
        ExtensionTable immutable;
        immutable.name = "ext_copy_result"; immutable.extension = "test.copy";
        immutable.columns = {{"owner", ExtensionType::Blob}, {"score", ExtensionType::Real}};
        immutable.primaryKey = {"owner"}; immutable.keySpace = "objects";
        auto manual = immutable; manual.name = "ext_copy_manual"; manual.versioned = true;
        const std::vector<ExtensionRow> first{{extensionUuid(owner), 1.25}}, second{{extensionUuid(owner), 2.5}};
        ObjectVersion sourceFirst;
        std::string sourceTable;
        {
            auto store = ObjectStore::open(sourcePath, as("alice"));
            {
                auto txn = store.write();
                store.createExtensionTable(txn, immutable); store.createExtensionTable(txn, manual);
                ObjectRecord object; object.uuid = owner; object.kind = "custom_annotation";
                store.upsertObject(txn, object);
                store.appendExtensionRows(txn, immutable.name, first);
                store.replaceExtensionRows(txn, manual.name, owner, first); txn.commit();
            }
            {
                auto txn = store.read();
                sourceFirst = store.currentObjectRevision(txn, owner)->reference();
                for (const auto& info : store.extensionTables(txn)) if (info.name == manual.name) sourceTable = info.tableUuid;
                const auto status = store.status(txn);
                require(status.extensionsPresent && storeStatusFromJson(storeStatusToJson(status)).extensionsPresent,
                    "extension presence survives the hosted-reader status protocol");
                // The object-level copier cannot know which immutable result
                // rows belong in a new independent object, so it must refuse.
                const auto project = projectRecord(LocalStoreReader(store, txn));
                mustFail([&] { readObjectCopy(LocalStoreReader(store, txn), project->uuid); });
            }
            {
                auto txn = store.write(VersionSelection{{owner, sourceFirst.version}});
                store.replaceExtensionRows(txn, manual.name, owner, second); txn.commit();
            }
        }
        ObjectStoreSession session;
        const auto attachment = session.attach(sourcePath, as("alice"));
        copyProject(session, attachment, targetPath, "extension copy", as("alice"));
        auto copied = ObjectStore::open(targetPath, as("alice")); auto txn = copied.read();
        std::vector<ExtensionRow> results, current, historical;
        copied.scanExtensionRows(txn, immutable.name, [&](const auto& row) { results.push_back(row); return true; });
        copied.scanExtensionRows(txn, manual.name, [&](const auto& row) { current.push_back(row); return true; });
        auto old = sourceFirst; old.storeUuid = copied.status(txn).storeUuid;
        copied.scanExtensionRowsAtVersion(txn, manual.name, old, [&](const auto& row) { historical.push_back(row); return true; });
        require(results == first && current == second && historical == first,
            "database fork preserves immutable results and all manual field versions");
        for (const auto& info : copied.extensionTables(txn)) if (info.name == manual.name)
            require(info.tableUuid != sourceTable, "fork remaps physical extension table UUID without breaking old snapshots");
    }

    {
        auto directOptions=as("alice"); directOptions.history=HistoryMode::Off;
        const auto directPath=directory.path/"direct.vcobj", copiedPath=directory.path/"direct-copy.vcobj";
        createProject(directPath,"direct",Json::object(),directOptions);
        ObjectStoreSession direct;
        const auto attached=direct.attach(directPath,directOptions);
        setProjectName(direct,attached,"renamed direct");
        require(addProjectResource(direct,attached,kVolumeResource,"/datasets/volume"),"direct resource created");
        const auto uuid=projectResourceUuid(kVolumeResource,"/datasets/volume");
        updateProjectResourceMeta(direct,attached,uuid,{{"quality","checked"}});
        require(resources(direct,attached,kVolumeResource).front().meta.at("quality")=="checked","direct resource edited");
        setProjectView(direct,attached,"alice",{{"volume","one"}});
        require(view(direct,attached,"alice").at("volume")=="one","direct project view edited");
        DatabaseIdentity sourceId;
        { auto store=ObjectStore::open(directPath,directOptions); auto txn=store.read(); sourceId=store.databaseIdentity(txn); }
        auto sourceRequest = direct.newRequest(attached);
        direct.readStore(attached, [&](const StoreReader& reader, const auto&) {
            const auto record = projectRecord(reader);
            sourceRequest.expectedObjects[record->uuid] = record->version;
            sourceRequest.operations.push_back(SetMetaOp{record->uuid, "copy_test", "true"});
        });
        direct.submit(attached, sourceRequest);
        const auto sameNamePath = directory.path / "same-name-copy.vcobj";
        const auto sourceSequence = direct.status(attached).commitSequence;
        copyProject(direct, attached, sameNamePath, "renamed direct", directOptions);
        {
            auto copied = ObjectStore::open(sameNamePath, directOptions);
            { auto read = copied.read(); require(copied.status(read).commitSequence == sourceSequence + 1,
                "independent copy publishes its identity even without a rename"); }
            { auto read = copied.read(); require(!copied.receipt(read, sourceRequest.requestUuid),
                "independent copy cannot replay source-store request receipts"); }
            mustFail([&] { copied.submit(sourceRequest); });
            { auto write = copied.write(); write.commit(); }
            { auto read = copied.read(); require(copied.status(read).commitSequence == sourceSequence + 1,
                "copy finalization leaves no dirty state for the next no-op"); }
        }
        copyProject(direct,attached,copiedPath,"copy direct",directOptions);
        {
            auto copied=ObjectStore::open(copiedPath,directOptions); auto read=copied.read();
            const auto copyId=copied.databaseIdentity(read);
            require(copyId.uuid!=sourceId.uuid,"copied project has independent physical identity");
            for(const auto& table:sourceId.tables)
                for(const auto& other:copyId.tables)
                    if(table.name==other.name) require(table.uuid!=other.uuid,"copied project has independent table identity");
            require(copied.provisioning(read).history==HistoryMode::Off && projectRecord(LocalStoreReader(copied,read))->name=="copy direct",
                    "copied direct project remains history-off and can be renamed");
        }
        require(removeProjectResource(direct,attached,uuid),"direct resource retired");
        require(resources(direct,attached,kVolumeResource).empty(),"direct retired resource hidden");
        require(addProjectResource(direct,attached,kVolumeResource,"/datasets/volume"),"direct resource restored");
        int64_t sequence=0;
        {
            auto store=ObjectStore::open(directPath,directOptions);
            { auto txn=store.read(); sequence=store.provisioning(txn).commitSequence; }
            auto txn=store.write(); store.seal(txn); txn.commit();
        }
        const auto sealedCopy=directory.path/"sealed-copy.vcobj";
        copyProject(direct,attached,sealedCopy,"mutable copy",directOptions);
        {
            auto copied=ObjectStore::open(sealedCopy,directOptions); auto read=copied.read();
            require(!copied.provisioning(read).sealed && projectRecord(LocalStoreReader(copied,read))->name=="mutable copy","sealed project copies into a mutable independent store");
        }
        const auto packedPath=directory.path/"packed-project.vcobj";
        {
            PackedStoreBuilder packed(packedPath);
            packed.importDatabase(sourceId.uuid,directPath); packed.finalize(sourceId.uuid);
        }
        auto readOnly=directOptions;readOnly.readOnly=true;
        ObjectStoreSession packedSession;
        const auto packedAttachment=packedSession.attach(packedPath,readOnly);
        const auto packedCopy=directory.path/"packed-copy.vcobj";
        copyProject(packedSession,packedAttachment,packedCopy,"unpacked copy",directOptions);
        {
            auto copied=ObjectStore::open(packedCopy,directOptions); auto read=copied.read();
            require(!copied.provisioning(read).sealed && projectRecord(LocalStoreReader(copied,read))->name=="unpacked copy","packed project copies through the native VFS into a mutable store");
        }
        {
            auto store=ObjectStore::open(directPath,readOnly); auto txn=store.read();
            require(store.provisioning(txn).sealed && store.provisioning(txn).commitSequence==sequence+1 && store.databaseIdentity(txn).uuid==sourceId.uuid,"copy never unseals or modifies original project");
        }
    }

    // A new project names itself; any other store is an unnamed project.
    createProject(path, "PHerc", {{"open_data", {{"sample", "PHerc0139"}}}}, as("alice"));
    mustFail([&] { createProject(path, "again", Json::object(), as("alice")); });
    ObjectStoreSession session;
    const auto project = session.attach(path, as("alice"));
    session.readStore(project, [&](const vc::objects::StoreReader& reader, const auto&) {
        const auto record = projectRecord(reader);
        require(record && record->name == "PHerc" &&
                Json::parse(record->meta).at("open_data").at("sample") == "PHerc0139", "project record");
    });
    setProjectName(session, project, "renamed");
    {
        auto plain = ObjectStore::create(directory.path / "plain.vcobj", as("alice"));
        auto txn = plain.read();
        require(!projectRecord(LocalStoreReader(plain, txn)), "a plain store has no project record");
    }
    const auto plain = session.attach(directory.path / "plain.vcobj", as("alice"));
    setProjectName(session, plain, "named later");
    session.readStore(plain, [&](const vc::objects::StoreReader& reader, const auto&) {
        require(projectRecord(reader)->name == "named later", "naming creates the record");
    });

    // Resources: one per (kind, location), in the order added; removal retires.
    const std::string zarr = "s3://bucket/scan.zarr/", local = "/data/predictions.zarr";
    require(addProjectResource(session, project, kVolumeResource, zarr, {"vc-remote-auth:anonymous"}), "added");
    require(!addProjectResource(session, project, kVolumeResource, zarr, {"vc-remote-auth:anonymous"}), "unchanged");
    require(addProjectResource(session, project, kVolumeResource, local), "second volume");
    require(addProjectResource(session, project, kNormalGridResource, "/data/normals"), "normal grid");
    mustFail([&] { addProjectResource(session, project, "surface", zarr); });
    mustFail([&] { addProjectResource(session, project, kVolumeResource, ""); });
    auto volumes = resources(session, project, kVolumeResource);
    require(volumes.size() == 2 && volumes[0].location == zarr && volumes[1].location == local, "volumes in order");
    require(volumes[0].tags == std::vector<std::string>{"vc-remote-auth:anonymous"}, "tags kept");
    require(volumes[0].uuid == projectResourceUuid(kVolumeResource, zarr), "named UUID");
    require(resources(session, project, kNormalGridResource).size() == 1, "kinds are separate");
    updateProjectResourceMeta(session, project, volumes[0].uuid, {{"volume_id", "scan-L0"}});
    require(resources(session, project, kVolumeResource)[0].meta.at("volume_id") == "scan-L0", "meta merged");
    require(removeProjectResource(session, project, volumes[1].uuid), "removed");
    require(!removeProjectResource(session, project, volumes[1].uuid), "already removed");
    require(resources(session, project, kVolumeResource).size() == 1, "removed resources are not listed");
    require(addProjectResource(session, project, kVolumeResource, local), "re-added");
    require(resources(session, project, kVolumeResource).size() == 2, "re-added resource listed");

    const auto legacyPath = directory.path / "legacy-resource.vcobj";
    auto legacy = projectResourceRecord(kVolumeResource, "/legacy");
    legacy.uuid = util::nameUuid("vc_project_resource/volume//legacy");
    {
        auto store = ObjectStore::create(legacyPath, as("alice"));
        auto w = store.write();
        store.upsertObject(w, legacy);
        w.commit();
    }
    const auto legacyProject = session.attach(legacyPath, as("alice"));
    const auto newKey = projectResourceUuid(kVolumeResource, "/legacy");
    updateProjectResourceMeta(session, legacyProject, newKey, {{"checked", true}});
    require(resources(session, legacyProject, kVolumeResource).front().uuid == legacy.uuid, "legacy UUID retained");
    require(removeProjectResource(session, legacyProject, newKey), "new deterministic key addresses legacy resource");

    // A resource of one owner (a path on their disk) is theirs alone, even at a path another uses.
    const std::string onDisk = "/home/user/predictions.zarr";
    require(addProjectResource(session, project, kVolumeResource, onDisk, {}, Json::object(), "alice"), "alice's own");
    require(addProjectResource(session, project, kVolumeResource, onDisk, {}, Json::object(), "bob"), "bob's own");
    require(projectResourceUuid(kVolumeResource, onDisk, "alice") != projectResourceUuid(kVolumeResource, onDisk, "bob"),
            "owners name their resources apart");
    mustFail([&] { updateProjectResourceMeta(session, project, projectResourceUuid(kVolumeResource, onDisk, "alice"),
                                             {{"owner", ""}}); });
    const auto seenBy = [&](const std::string& viewer) {
        std::vector<std::string> owners;
        for (const auto& resource : resources(session, project, kVolumeResource, viewer))
            if (resource.location == onDisk) owners.push_back(resource.owner);
        return owners;
    };
    require(seenBy("alice") == std::vector<std::string>{"alice"} && seenBy("bob") == std::vector<std::string>{"bob"},
            "each owner sees only their own");
    require(seenBy("carol").empty() && resources(session, project, kVolumeResource, "carol").size() == 2,
            "others see the shared resources");
    require(removeProjectResource(session, project, projectResourceUuid(kVolumeResource, onDisk, "bob")), "bob removes his");
    require(seenBy("alice") == std::vector<std::string>{"alice"}, "removing one's own leaves the others'");

    // Each author has a view of their own.
    setProjectView(session, project, "alice", {{"current_volume", "scan-L0"}});
    setProjectView(session, project, "bob", {{"current_volume", "predictions"}});
    require(view(session, project, "alice").at("current_volume") == "scan-L0", "alice's view");
    require(view(session, project, "bob").at("current_volume") == "predictions", "bob's view");
    require(view(session, project, "carol").empty(), "no view yet");
    // A copy is a project of its own: same records, another store identity.
    copyProject(session, project, directory.path / "copy.vcobj", "copy", as("alice"));
    mustFail([&] { copyProject(session, project, directory.path / "copy.vcobj", "again", as("alice")); });
    const auto copy = session.attach(directory.path / "copy.vcobj", as("alice"));
    require(copy.storeUuid != project.storeUuid, "the copy has its own identity");
    require(resources(session, copy, kVolumeResource, "carol").size() == 2 && view(session, copy, "bob").at("current_volume") == "predictions",
            "the copy keeps the resources and views");
    session.readStore(copy, [&](const vc::objects::StoreReader& reader, const auto&) {
        require(projectRecord(reader)->name == "copy", "the copy is renamed");
    });
    session.detach(copy);
    session.detach(plain);
    session.detach(project);

    // Several local writers share one project; each keeps their own view.
    {
        ObjectStoreSession aliceSession, bobSession;
        const auto alice = aliceSession.attach(path, as("alice"));
        const auto bob = bobSession.attach(path, as("bob"));
        const std::string shared = "https://example.org/shared.zarr/";
        require(addProjectResource(aliceSession, alice, kVolumeResource, shared), "alice adds a volume");
        setProjectView(aliceSession, alice, "alice", {{"current_volume", "shared"}});
        require(addProjectResource(aliceSession, alice, kVolumeResource, "/home/alice/scan.zarr", {}, Json::object(), "alice"),
                "alice adds a volume on her disk");
        require(resources(bobSession, bob, kVolumeResource, "bob").size() == 3, "bob sees alice's shared volume, not her disk's");
        updateProjectResourceMeta(bobSession, bob, projectResourceUuid(kVolumeResource, shared), {{"volume_id", "shared"}});
        setProjectView(bobSession, bob, "bob", {{"current_volume", "shared"}});
        const auto listed = resources(aliceSession, alice, kVolumeResource);
        require(std::any_of(listed.begin(), listed.end(), [](const auto& r) {
                    return r.meta.contains("volume_id") && r.meta.at("volume_id") == "shared"; }),
                "alice sees bob's resource edit");
        require(view(aliceSession, alice, "alice").at("current_volume") == "shared" &&
                view(aliceSession, alice, "bob").at("current_volume") == "shared", "views kept per author");
        // A copy carries the project's artifacts.
        const auto file = directory.path / "notes.txt";
        std::ofstream(file) << "project notes";
        std::string recordUuid;
        int64_t version = 0;
        aliceSession.readStore(alice, [&](const vc::objects::StoreReader& reader, const auto&) {
            recordUuid = projectRecord(reader)->uuid;
            version = reader.currentObject(recordUuid)->version;
        });
        aliceSession.publishArtifacts(alice, recordUuid, version, {{"notes.txt", file}});
        const auto copyPath = directory.path / "with-artifacts.vcobj";
        copyProject(aliceSession, alice, copyPath, "with artifacts", as("alice"));
        ObjectStoreSession copied;
        const auto copy = copied.attach(copyPath, as("alice"));
        require(resources(copied, copy, kVolumeResource, "alice").size() == 5, "records copied");
        const auto artifacts = copied.artifacts(copy, recordUuid);
        require(artifacts.size() == 1 && std::filesystem::is_regular_file(artifactBlobPath(copyPath, artifacts.front().sha256)),
                "artifact bytes copied");
    }
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
