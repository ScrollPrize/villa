#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/Database.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/GeometryDelta.hpp"
#include "sync/Apply.hpp"
#include "sync/GeometryTransfer.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;

namespace {
using Db=std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)>;
OpenOptions options(){OpenOptions o;o.author="geometry-sync";o.spatialIndex=false;return o;}
ObjectRecord track(const std::string& uuid){ObjectRecord r;r.uuid=uuid;r.kind="track";r.track.emplace();return r;}
GeometryData shape(){return vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,6},{7,8,9}}});}
int64_t chunks(sqlite3* db){Statement q(db,"SELECT count(*) FROM geometry_chunks");q.row();return q.i64(0);}

void borrowedChunks()
{
    TemporaryDirectory dir;
    const auto basePath=dir.path/"base",sourcePath=dir.path/"source",targetPath=dir.path/"linked-target";
    const auto owner=util::newUuid(),id=util::newUuid();
    const auto geometry=shape();
    {
        auto base=ObjectStore::create(basePath,options());auto w=base.write();
        base.upsertObject(w,track(owner));base.putGeometry(w,owner,geometry);base.seal(w);w.commit();
    }
    std::string revision;
    {
        auto source=ObjectCollection::create(sourcePath,{basePath},options());auto w=source.write();
        const auto key=source.upsertObject(w,track(id));
        revision=source.putGeometry(w,key,geometry);source.seal(w);w.commit();
    }
    Db source(openConnection(sourcePath,SQLITE_OPEN_READONLY,{},true),sqlite3_close_v2);
    exec(source.get(),"BEGIN;");
    auto sourceLinks=attachCollectionChunkResolver(source.get(),sourcePath);
    require(chunks(source.get())==0,"source borrows payloads without local registrations");
    {
        auto target=ObjectStore::create(dir.path/"standalone-target",options());auto w=target.write();
        target.upsertObject(w,track(id));auto* sql=ObjectStoreImplAccess::conn(w);
        GeometryStage staged;staged.read(source.get(),sql,id);staged.install(sql,id);
        require(target.readGeometryArray(w,revision,"points")==geometry.arrays.front().bytes,
            "transfer resolves absent local payloads through linked source");
        require(chunks(sql)>0,"standalone destination owns copied payloads");
        w.commit();
    }
    {
        auto target=ObjectCollection::create(targetPath,{basePath},options());auto w=target.write();
        target.upsertObject(w,track(id));w.commit();
    }
    // Exercise the same internal transfer with a destination resolver. A
    // linked destination must not duplicate payloads already owned by its base.
    Db target(openConnection(targetPath,SQLITE_OPEN_READWRITE,{},false),sqlite3_close_v2);
    auto targetLinks=attachCollectionChunkResolver(target.get(),targetPath);
    exec(target.get(),"BEGIN IMMEDIATE;");
    GeometryStage staged;staged.read(source.get(),target.get(),id);staged.install(target.get(),id);
    require(chunks(target.get())==0,"linked transfer does not copy unchanged payloads");
    require(GeometryReader(target.get(),revision).read("points",0)==geometry.arrays.front().bytes,
        "linked destination resolves transferred descriptors");
    exec(target.get(),"ROLLBACK;");
}

void selectedGeometryPointer()
{
    TemporaryDirectory dir;
    auto source=ObjectStore::create(dir.path/"source",options());
    auto target=ObjectStore::create(dir.path/"target",options());
    const auto id=util::newUuid();const auto geometry=shape();
    std::string revision;
    {auto w=source.write();source.upsertObject(w,track(id));revision=source.putGeometry(w,id,geometry);w.commit();}
    auto r=source.read();
    Json record=syncRecord(source,r,id);
    record["storage_ref"]=""; // Legacy metadata stores geometry selection separately.
    const Json empty=Json::object(),merged={{id,record}};
    {
        auto w=target.write();applyObjects(target,w,empty,merged);
        transferGeometry(ObjectStoreImplAccess::conn(r),ObjectStoreImplAccess::conn(w),id);
        applyHeadsAndEdges(target,w,empty,merged);
        require(target.currentGeometry(w,id)->uuid==revision,"selected revision wins over legacy empty storage_ref");
        require(target.readGeometryArray(w,revision,"points")==geometry.arrays.front().bytes,"selected geometry remains readable");
        w.commit();
    }
    {
        auto w=target.write();const Json before={{id,syncRecord(target,w,id)}};
        auto removed=before;removed[id].erase("geometry");removed[id].erase("data_checksum");
        removed[id]["storage_kind"]="";removed[id]["storage_ref"]="";removed[id]["data_generation"]=0;
        applyObjects(target,w,before,removed);applyHeadsAndEdges(target,w,before,removed);
        require(!target.currentGeometry(w,id),"sync clearing geometry removes the current descriptor");
        require(target.geometryRevision(w,revision).has_value(),"clearing current geometry retains history");
        w.commit();
    }
}

void currentDescriptor(HistoryMode history)
{
    TemporaryDirectory dir;auto config=options();config.history=history;
    auto store=ObjectStore::create(dir.path/"descriptor",config);
    const auto id=util::newUuid();const auto geometry=shape();
    auto w=store.write();store.upsertObject(w,track(id));
    const auto revision=store.putGeometry(w,id,geometry);
    auto* db=ObjectStoreImplAccess::conn(w);
    Statement current(db,"SELECT storage_ref,data_checksum,geometry_manifest FROM objects WHERE uuid=?",parseUuid(id));
    require(current.row(),"current geometry is stored directly on the object");
    require(current.text(0)==revision,"object points directly at its current geometry");
    const auto descriptor=current.text(2),checksum=current.text(1);
    require(store.currentGeometry(w,id)->contentHash==checksum,"current descriptor checksum matches object");
    require(GeometryReader(db,revision).read("points",0)==geometry.arrays.front().bytes,"current descriptor reads in either history mode");

    // The object descriptor is authoritative even when an intact archived
    // revision would otherwise allow a reader to silently ignore corruption.
    execute(db,"UPDATE objects SET geometry_manifest='{}' WHERE uuid=?",parseUuid(id));
    mustFail([&]{store.currentGeometry(w,id);});
    mustFail([&]{GeometryReader reader(db,revision);});
    execute(db,"UPDATE objects SET geometry_manifest=? WHERE uuid=?",descriptor,parseUuid(id));
    if(history==HistoryMode::On) {
        auto different=Json::parse(descriptor);different["attributes"]["unexpected"]=true;
        const auto text=different.dump();
        execute(db,"UPDATE objects SET geometry_manifest=?,data_checksum=? WHERE uuid=?",text,hashText(text),parseUuid(id));
        mustFail([&]{store.currentGeometry(w,id);});
        mustFail([&]{GeometryReader reader(db,revision);});
        execute(db,"UPDATE objects SET geometry_manifest=?,data_checksum=? WHERE uuid=?",descriptor,checksum,parseUuid(id));
    }
    require(GeometryReader(db,revision).read("points",0)==geometry.arrays.front().bytes,"restored descriptor remains readable");
    w.commit();
}

void restoreArchivedGeometry()
{
    TemporaryDirectory dir;auto store=ObjectStore::create(dir.path/"restore",options());
    const auto id=util::newUuid();const auto geometry=shape();
    {auto w=store.write();store.upsertObject(w,track(id));w.commit();}
    const auto head=[&]{auto r=store.read();return store.currentObjectRevision(r,id)->uuid;};
    const auto metadataRevision=head();
    std::string geometryToken;
    {auto w=store.write();geometryToken=store.putGeometry(w,id,geometry);w.commit();}
    const auto geometryRevision=head();
    const auto restore=[&](const std::string& revision) {
        RevisionSelection selected;selected[id]=head();
        auto w=store.write(selected);store.restoreObject(w,id,revision);w.commit();
    };
    const auto check=[&] {
        auto r=store.read();require(store.currentGeometry(r,id)->uuid==geometryToken,"restoration selects archived geometry directly");
        require(store.readGeometryArray(r,geometryToken,"points")==geometry.arrays.front().bytes,"restored archive payload reads correctly");
    };
    restore(metadataRevision);
    {auto r=store.read();require(!store.currentGeometry(r,id),"restored metadata-only revision has no current geometry");}
    restore(geometryRevision);check();
    {auto w=store.write();store.deleteObject(w,id);w.commit();}
    {auto w=store.write();store.upsertObject(w,track(id));w.commit();}
    {auto r=store.read();require(!store.currentGeometry(r,id),"revival remains metadata-only until explicit restore");}
    restore(geometryRevision);check();
}
}
int main()try{borrowedChunks();selectedGeometryPointer();currentDescriptor(HistoryMode::Off);currentDescriptor(HistoryMode::On);restoreArchivedGeometry();std::cout<<"geometry sync tests passed\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
