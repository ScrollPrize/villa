#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/Database.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/GeometryDelta.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;

namespace {
using Json=nlohmann::json;
using Db=std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)>;
OpenOptions options(){OpenOptions o;o.author="geometry-descriptors";o.spatialIndex=false;return o;}
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
        auto copied=geometry;
        copied.arrays.front().bytes=GeometryReader(source.get(),revision).read("points",0);
        const auto selected=target.putGeometry(w,id,copied);
        require(target.readGeometryArray(w,selected,"points")==geometry.arrays.front().bytes,
            "copy resolves absent local payloads through linked source");
        require(chunks(sql)>0,"standalone destination owns copied payloads");
        w.commit();
    }
    {
        auto target=ObjectCollection::create(targetPath,{basePath},options());auto w=target.write();
        const auto key=target.upsertObject(w,track(id));
        target.putGeometry(w,key,geometry);
        require(target.readGeometryArray(w,key,"points")==geometry.arrays.front().bytes,
            "linked destination resolves shared geometry descriptors");
        w.commit();
    }
    Db target(openConnection(targetPath,SQLITE_OPEN_READONLY,{},true),sqlite3_close_v2);
    require(chunks(target.get())==0,"linked geometry creation does not copy unchanged payloads");
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
    const auto head=[&]{auto r=store.read();return store.versionReference(r,*store.currentObjectRevision(r,id));};
    const auto metadataRevision=head();
    std::string geometryToken;
    {auto w=store.write();geometryToken=store.putGeometry(w,id,geometry);w.commit();}
    const auto geometryRevision=head();
    const auto restore=[&](const ObjectVersion& revision) {
        const auto current=head();
        auto w=store.write(VersionSelection{{id,current.version}});store.restoreObject(w,id,revision);w.commit();
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

void unpublishedRelationship()
{
    TemporaryDirectory dir;auto store=ObjectStore::create(dir.path/"new-relationship",options());
    const auto a=util::newUuid(),b=util::newUuid();
    std::string before;
    {auto w=store.write();
     for(const auto& id:{a,b}){store.upsertObject(w,track(id));const auto token=store.putGeometry(w,id,shape());if(id==a)before=token;}
     w.commit();}
    auto w=store.write();
    EdgeRecord edge;edge.kind="related";edge.aUuid=a;edge.bUuid=b;
    edge.aAnchor.points=edge.bAnchor.points={{"points",{0}}};
    const auto id=store.addEdge(w,edge);
    mustFail([&]{store.keepRecordedRelationship(w,id);});
    const std::array<int32_t,1> changed{40};
    const std::vector<GeometryArrayEdit> edits{{"points",0,detail::scalarBytes<int32_t>(changed)}};
    auto decision=[&](const std::vector<EdgeRecord>& affected,const std::string& action){
        require(affected.size()==1&&affected.front().uuid==id,"new relationship is affected by its endpoint edit");
        GeometryEdgeEdit result;result.edgeUuid=id;result.expectedVersion=affected.front().version.value();result.action=action;
        result.aRevision=store.currentGeometry(w,affected.front().aUuid)->uuid;
        result.bRevision=store.currentGeometry(w,affected.front().bUuid)->uuid;
        return std::vector<GeometryEdgeEdit>{result};
    };
    mustFail([&]{store.editGeometry(w,a,edits,before,[&](const auto& affected){return decision(affected,"keep_recorded");});});
    require(store.currentGeometry(w,a)->uuid==before,"failed keep-recorded rolls back the entire geometry edit");
    store.editGeometry(w,a,edits,before,[&](const auto& affected){return decision(affected,"retain");});
    w.commit();
    auto read=store.read();
    require(!store.edgeGeometryState(read,id).stale,"explicit retain binds the new relationship to its final endpoints");
}

void changedRelationshipKeepsPublishedGeometry(bool retainFirst)
{
    TemporaryDirectory dir;auto store=ObjectStore::create(dir.path/"changed-relationship",options());
    const auto a=util::newUuid(),b=util::newUuid();std::string edge,before;
    {auto w=store.write();
     for(const auto& id:{a,b}){store.upsertObject(w,track(id));const auto token=store.putGeometry(w,id,shape());if(id==a)before=token;}
     EdgeRecord record;record.kind="related";record.aUuid=a;record.bUuid=b;
     record.aAnchor.points=record.bAnchor.points={{"points",{0}}};edge=store.addEdge(w,record);w.commit();}
    const std::array<int32_t,1> changed{40};
    {auto w=store.write();
     auto record=*store.edgeByUuid(w,edge);record.meta=R"({"review":"keep old observation"})";store.updateEdge(w,record);
     store.editGeometry(w,a,{{"points",0,detail::scalarBytes<int32_t>(changed)}},before,[&](const auto& affected){
        require(affected.size()==1,"changed edge remains visible during endpoint edit");
        GeometryEdgeEdit decision;decision.edgeUuid=edge;decision.expectedVersion=affected.front().version.value();
        decision.action=retainFirst?"retain":"keep_recorded";
        decision.aRevision=store.currentGeometry(w,affected.front().aUuid)->uuid;
        decision.bRevision=store.currentGeometry(w,affected.front().bUuid)->uuid;
        return std::vector<GeometryEdgeEdit>{decision};
     });
     if(retainFirst)store.keepRecordedRelationship(w,edge);
     w.commit();}
    auto read=store.read();const auto state=store.edgeGeometryState(read,edge);
    require(state.stale,"editing relationship metadata must not silently revalidate kept old endpoint geometry");
    const auto record=*store.edgeByUuid(read,edge);
    require((record.aUuid==a?state.aRevision:state.bRevision)==before,"keep-recorded preserves the published endpoint geometry");
}
}
int main()try{borrowedChunks();currentDescriptor(HistoryMode::Off);currentDescriptor(HistoryMode::On);restoreArchivedGeometry();unpublishedRelationship();changedRelationshipKeepsPublishedGeometry(false);changedRelationshipKeepsPublishedGeometry(true);std::cout<<"geometry descriptor tests passed\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
