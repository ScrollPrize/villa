#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "vc_object_store/objects/Selections.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/SurfaceGrid.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "internal/Database.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include <fstream>
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
OpenOptions options(HistoryMode mode){OpenOptions o;o.author="coherent-api";o.history=mode;o.spatialIndex=false;return o;}
ObjectRecord track(const std::string& id){ObjectRecord r;r.uuid=id;r.kind="track";r.track.emplace();r.meta=R"({"quality":7})";return r;}
GeometryData shape(){return vc::geometry::encode(vc::geometry::Track{{{1,2,3},{4,5,6},{7,8,9}}});}
ExtensionTable extension(){
    ExtensionTable spec;spec.name="ext_manual_edges";spec.extension="manual";spec.keySpace="edges";spec.versioned=true;
    spec.columns={{"owner",ExtensionType::Blob},{"value",ExtensionType::Integer}};spec.primaryKey={"owner"};return spec;
}
struct Fixture {
    TemporaryDirectory dir;
    std::filesystem::path base=dir.path/"base"/"base.vcobj",overlay=dir.path/"overlay"/"overlay.vcobj";
    std::string a=util::newUuid(),b=util::newUuid(),file=util::newUuid(),edge,geometry;
    ObjectVersion av,bv;
    int64_t edgeVersion=0;
    Fixture(HistoryMode baseHistory,HistoryMode overlayHistory,bool extensions=false){
        std::filesystem::create_directories(base.parent_path());std::filesystem::create_directories(overlay.parent_path());
        {
            auto store=ObjectStore::create(base,options(baseHistory));
            {auto w=store.write();
                for(const auto& id:{a,b}){auto r=track(id);r.tags={"base"};store.upsertObject(w,r);const auto g=store.putGeometry(w,id,shape());if(id==a)geometry=g;}
                ObjectRecord source;source.uuid=file;source.kind="fiber";source.storageKind="file";source.storageRef="fiber.json";
                store.upsertObject(w,source);
                std::ofstream(base.parent_path()/"fiber.json")<<"{\"uuid\":\""<<file<<"\",\"line_points\":[]}";
                EdgeRecord e;e.kind="related";e.aUuid=a;e.bUuid=b;e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;
                e.aAnchor.metadataKeys=e.bAnchor.metadataKeys={"quality"};edge=store.addEdge(w,e);
                if(extensions)store.createExtensionTable(w,extension());w.commit();
            }
            if(extensions)for(int64_t value=1;value<=4;++value){auto w=store.write();
                const std::array<ExtensionRow,1> rows{ExtensionRow{extensionUuid(edge),value}};
                store.replaceExtensionRows(w,extension().name,edge,rows);w.commit();}
            {auto r=store.read();av=store.versionReference(r,*store.currentObjectRevision(r,a));
                bv=store.versionReference(r,*store.currentObjectRevision(r,b));edgeVersion=*store.currentRelationshipVersion(r,edge);}
            {auto w=store.write();store.seal(w);w.commit();}
        }
        {auto store=ObjectCollection::create(overlay,{base},options(overlayHistory));}
    }
};

void readsAndOverrides(HistoryMode baseHistory,HistoryMode overlayHistory){
    Fixture f(baseHistory,overlayHistory,true);
    auto store=ObjectStore::open(f.overlay,options(overlayHistory));
    {
        auto r=store.read();
        require(store.objectCount(r,"track")==2,"collection counts use visible records");
        if(baseHistory==HistoryMode::On)require(store.geometryHistory(r,f.a).size()==1,"geometry history belongs to the inherited owner");
        else mustFail([&]{store.geometryHistory(r,f.a);});
        require(store.objectsByTag(r,"base").size()==2,"tag queries include inherited objects");
        require(store.componentOf(r,f.a).size()==2,"traversal resolves inherited endpoints");
        auto uppercase=f.a;std::transform(uppercase.begin(),uppercase.end(),uppercase.begin(),[](unsigned char c){return std::toupper(c);});
        require(store.neighbours(r,uppercase)==std::vector{f.b},"neighbours canonicalize endpoint UUID spelling");
        const auto e=*store.edgeByUuid(r,f.edge);
        require(e.otherEndpoint(uppercase)==f.b&&e.otherEndpoint(f.b)==f.a,"opposite endpoint is shared and symmetric");
        mustFail([&]{e.otherEndpoint(f.file);});
        require(store.graphReadView(r).objectUuids.size()==3,"ordinary graph includes inherited objects");
        const auto graph=store.graphReadViewAt(r,std::vector<ObjectVersion>{f.av,f.bv});
        require(graph.objectUuids.size()==2&&graph.edgeUuids.size()==1,"exact graphs work for current-only and archived sources");
        require(store.resolveStoragePath(r,f.file)==f.base.parent_path()/"fiber.json","relative sources use their owner's path");
        std::vector<ExtensionRow> rows;store.scanExtensionRows(r,extension().name,[&](const auto& row){rows.push_back(row);return true;});
        require(rows.size()==1&&std::get<int64_t>(rows[0][1])==4,"ordinary extension reader resolves inherited snapshots");
    }
    // A no-op must not materialize an override or advance the store sequence.
    int64_t sequence;{auto r=store.read();sequence=store.status(r).commitSequence;}
    EdgeRecord unchanged;EdgeGeometryRepair repair;
    {auto r=store.read();unchanged=*store.edgeByUuid(r,f.edge);
        repair.edgeUuid=f.edge;repair.expectedVersion=*unchanged.version;
        repair.aRevision=store.currentGeometry(r,unchanged.aUuid)->uuid;repair.bRevision=store.currentGeometry(r,unchanged.bUuid)->uuid;
        repair.aAnchor=unchanged.aAnchor;repair.bAnchor=unchanged.bAnchor;}
    {auto w=store.write();store.updateEdge(w,unchanged);store.repairEdgeGeometry(w,repair);w.commit();}
    {auto r=store.read();require(store.edgeByUuid(r,f.edge)->versionStoreUuid==unchanged.versionStoreUuid,
        "unchanged edge update/repair preserves inherited ownership");
        require(store.status(r).commitSequence==sequence,"unchanged edge update/repair does not publish");}
    {auto w=store.write();const std::array<ExtensionRow,1> rows{ExtensionRow{extensionUuid(f.edge),int64_t(4)}};
        store.replaceExtensionRows(w,extension().name,f.edge,rows);w.commit();}
    {auto r=store.read();require(store.status(r).commitSequence==sequence,"inherited no-op remains a no-op");}
    {auto w=store.write();const std::array<ExtensionRow,1> rows{ExtensionRow{extensionUuid(f.edge),int64_t(9)}};
        store.replaceExtensionRows(w,extension().name,f.edge,rows);w.commit();}
    {auto r=store.read();require(store.currentRelationshipVersion(r,f.edge)==f.edgeVersion+1,"edge override continues inherited version");}
    {auto w=store.write(VersionSelection{{f.a,f.av.version}});store.setMeta(w,f.a,"note","true");w.commit();}
    {
        auto r=store.read();const auto exact=store.objectAtVersion(r,f.av);require(bool(exact),"exact source version survives shadowing");
        const auto ref=store.versionReference(r,*exact);
        require(ref.storeUuid==f.av.storeUuid&&ref.generation==f.av.generation&&ref.version==f.av.version,"reference reconstruction preserves original source identity");
        require(store.currentGeometry(r,f.a)->uuid==f.geometry,"metadata override retains geometry identity");
        Statement count(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM geometry_chunks");count.row();
        require(count.i64(0)>0,"changed extension owns its snapshot payload");
    }
    mustFail([&]{auto w=store.write();store.setMeta(w,f.a,"quality","8");w.commit();});
}

void edgeIdentity(HistoryMode mode){
    Fixture f(mode,mode);
    auto store=ObjectStore::open(f.overlay,options(mode));EdgeRecord edge;
    {auto r=store.read();edge=*store.edgeByUuid(r,f.edge);}
    mustFail([&]{auto w=store.write();auto replacement=edge;replacement.kind="different";store.addEdge(w,replacement);w.commit();});
    {auto r=store.read();require(store.edgeByUuid(r,f.edge)->kind==edge.kind,"add cannot replace an inherited live edge");}
    {auto w=store.write();store.removeEdge(w,f.edge);store.seal(w);w.commit();}
    int64_t deletedVersion;
    {auto r=store.read();Statement row(ObjectStoreImplAccess::conn(r),"SELECT version FROM edges WHERE uuid=?",parseUuid(f.edge));row.row();deletedVersion=row.i64(0);}
    const auto next=f.dir.path/"next.vcobj";
    {auto collection=ObjectCollection::create(next,{f.overlay},options(mode));}
    auto child=ObjectStore::open(next,options(mode));
    {auto w=child.write();child.addEdge(w,edge);w.commit();}
    {auto r=child.read();require(child.edgeByUuid(r,f.edge)->version==deletedVersion+1,"reviving an inherited tombstone continues its version");}
}

void geometryEdit(HistoryMode baseHistory,HistoryMode overlayHistory,const std::string& action){
    Fixture f(baseHistory,overlayHistory);
    auto store=ObjectStore::open(f.overlay,options(overlayHistory));
    GeometryEdgeEdit decision;decision.edgeUuid=f.edge;decision.expectedVersion=f.edgeVersion;decision.action=action;
    {auto r=store.read();const auto edge=*store.edgeByUuid(r,f.edge);
        decision.aRevision=store.currentGeometry(r,edge.aUuid)->uuid;decision.bRevision=store.currentGeometry(r,edge.bUuid)->uuid;}
    const std::array<int32_t,3> xyz{11,12,13};
    const std::vector<GeometryArrayEdit> edits{{"points",0,scalarBytes<int32_t>(xyz)}};
    // Failure must not leave a partial override in the transaction or on disk.
    mustFail([&]{auto w=store.write();store.editGeometry(w,f.a,edits,f.geometry);w.commit();});
    {auto w=store.write(VersionSelection{{f.a,f.av.version},{f.b,f.bv.version}});
        store.editGeometry(w,f.a,edits,f.geometry,{decision});w.commit();}
    {auto r=store.read();
        require(store.currentGeometry(r,f.a)->uuid!=f.geometry,"ordinary geometry edit overrides inherited descriptor");
        require(store.readGeometryArray(r,f.geometry,"points")==shape().arrays.front().bytes,"old geometry remains readable without copying it");
        require(store.edgeGeometryState(r,f.edge).stale==(action=="keep_recorded"),"shared bindings distinguish revalidation from recorded observations");
        EdgeSelection q;q.applicableOnly=true;
        require(store.selectEdges(r,q).size()==(action=="keep_recorded"?0:1),"graph applicability agrees with geometry validation");
    }
}

void queries(){
    Fixture f(HistoryMode::Off,HistoryMode::Off);
    auto store=ObjectStore::open(f.overlay,options(HistoryMode::Off));
    {auto w=store.write();auto a=*store.objectByUuid(w,f.a);a.tags={"overlay"};store.upsertObject(w,a);w.commit();}
    {auto r=store.read();ObjectSelection q;q.tag="base";require(store.selectObjects(r,q).size()==1,"filtered base records cannot leak through overrides");
        q={};q.connectedTo=f.a;require(store.selectObjects(r,q).size()==2,"connected selection crosses physical stores");
        q={};q.limit=1;const auto first=store.selectObjects(r,q);q.offset=1;const auto second=store.selectObjects(r,q);
        require(first.size()==1&&second.size()==1&&first[0].uuid<second[0].uuid,"pagination is global");
        const auto json=nlohmann::json::parse(store.exportJson(r));
        require(json.at("objects").size()==3&&json.at("edges").size()==1,"export follows same visible graph");}
    {auto w=store.write();store.removeEdge(w,f.edge);store.deleteObject(w,f.a);w.commit();}
    {auto r=store.read();require(store.objectCount(r,"track")==1&&store.selectEdges(r,{}).empty(),"tombstones suppress inherited records across APIs");}
}

void packing(){
    TemporaryDirectory dir;const auto source=dir.path/"source.vcobj",packed=dir.path/"packed.vcobj",repacked=dir.path/"repacked.vcobj";
    const std::string payload="artifact bytes";std::ofstream(dir.path/"payload",std::ios::binary)<<payload;std::string hash;
    {auto store=ObjectStore::create(source,options(HistoryMode::Off));hash=store.putArtifact(dir.path/"payload");
        auto w=store.write();auto r=track(util::newUuid());r.meta=nlohmann::json{{kArtifactsMetaKey,nlohmann::json::parse(artifactManifestJson({{"sample",hash,payload.size()}}))}}.dump();
        store.upsertObject(w,r);store.seal(w);w.commit();}
    packObjectCollection(source,packed);compactPackedStore(packed,repacked);
    for(const auto& path:{packed,repacked}){auto store=ObjectStore::open(path,OpenOptions{.readOnly=true});auto r=store.read();
        const auto bytes=store.readArtifact(hash);require(std::string(bytes.begin(),bytes.end())==payload,"packing and repacking preserve artifact bytes");}
    // Corrupt payloads prevent publication and clean up the new sidecar.
    std::filesystem::permissions(artifactBlobPath(source,hash),std::filesystem::perms::owner_write,std::filesystem::perm_options::add);
    std::ofstream(artifactBlobPath(source,hash),std::ios::binary)<<"corruption";
    const auto failed=dir.path/"failed.vcobj";
    mustFail([&]{packObjectCollection(source,failed);});
    require(!std::filesystem::exists(failed)&&!std::filesystem::exists(artifactBlobDirectory(failed).parent_path()),"failed packing leaves no database or partial artifact sidecar");
}

void spatialQueries(){
    TemporaryDirectory dir;auto opt=options(HistoryMode::Off);opt.spatialIndex=true;
    const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    std::vector<std::string> fibers;std::string surface;
    {auto store=ObjectStore::create(base,opt);auto w=store.write();
        for(int i=0;i<3;++i){ObjectRecord r;r.kind="fiber";r.fiber.emplace();r.fiber->length=20;
            r.meta=R"({"coordinates":{"volume_uuid":"test-volume","scale_factor":1}})";
            fibers.push_back(store.upsertObject(w,r));
            vc::geometry::Fiber line;line.linePoints={{-10,0,1},{10,0,1}};store.putGeometry(w,fibers.back(),vc::geometry::encode(line));}
        ObjectRecord r;r.kind="surface";surface=store.upsertObject(w,r);
        vc::geometry::Surface mesh{2,2,{1,1},{{0,0,1},{2,0,3},{0,2,1},{2,2,3}}};
        store.putGeometry(w,surface,vc::geometry::encode(mesh));store.seal(w);w.commit();}
    {auto collection=ObjectCollection::create(overlay,{base},opt);}
    auto store=ObjectStore::open(overlay,opt);
    const auto check=[&](size_t count){auto r=store.read();ObjectSelection selection;selection.kind="fiber";selection.bbox=std::array<double,6>{-20,-2,0,20,2,4};
        require(store.selectObjects(r,selection).size()==count,"spatial selection sees inherited and overridden objects once");
        require(store.spatialRegions(r,selection).size()==count,"spatial regions follow visible owners");
        ObjectSelection page;page.kind="fiber";page.volume="test-volume";page.orderBy="length_desc";page.limit=1;
        std::set<std::string> paged;
        for(;;){const auto next=store.selectObjects(r,page);if(next.empty())break;
            require(paged.insert(next[0].uuid).second,"cross-store cursor does not repeat");
            page.afterUuid=next[0].uuid;page.afterLength=next[0].volumeLength;}
        require(paged.size()==count,"length cursor need not exist in each physical source");
        RegionSlice slice;slice.origin={0,0,1};RegionOptions region;region.volume="test-volume";
        auto result=store.geometryRegion(r,slice,region);require(result.complete()&&result.intersectingObjects==count,"viewport sees referenced volume frames");
        region.selected=fibers.back();region.maximumObjects=1;
        result=store.geometryRegion(r,slice,region);require(result.intersectingObjects==1&&result.runs[0].objectUuid==fibers.back(),"viewport selected object precedes global UUID merge");
        region.maximumObjects=0;region.uuids=std::vector{fibers.back()};
        require(store.geometryRegion(r,slice,region).intersectingObjects==1,"viewport UUID filter is independent of local row IDs");
        SurfaceProximityQuery near;near.point={1,1,2};near.tolerance=.1;
        require(store.nearestSurfaces(r,near).size()==1,"nearest surface reads referenced mesh");
        RegionOptions surfaces;surfaces.kind="surface";slice.origin={0,0,2};slice.localBounds={-3,-3,-.1,3,3,.1};
        require(store.geometryRegion(r,slice,surfaces).intersectingObjects==1,"surface cut reads referenced tiles");
    };
    check(3);
    {auto w=store.write();store.setMeta(w,fibers[0],"note","true");store.setMeta(w,surface,"note","true");store.deleteObject(w,fibers[1]);w.commit();}
    check(2);
    {auto w=store.write();store.deleteObject(w,surface);w.commit();}
    {auto r=store.read();SurfaceProximityQuery near;near.point={1,1,2};near.tolerance=.1;
        require(store.nearestSurfaces(r,near).empty(),"spatial tombstone hides source mesh");}
}

void followupRegressions(){
    TemporaryDirectory dir;const auto opt=options(HistoryMode::On);const auto basePath=dir.path/"base.vcobj";
    std::string a,b,retired,edge,point,oldGeometry,newGeometry;ObjectVersion oldVersion,newVersion;
    vc::geometry::Fiber fiber;fiber.linePoints={{0,0,0},{10,0,0}};
    const auto original=vc::geometry::encode(fiber);
    {auto store=ObjectStore::create(basePath,opt);
        {auto w=store.write();ObjectRecord r;r.kind="fiber";a=store.upsertObject(w,r);b=store.upsertObject(w,r);retired=store.upsertObject(w,r);
            oldGeometry=store.putGeometry(w,a,original);
            ObjectRecord p;p.kind="point_collection";p.meta=R"({"source":{"origin":"audit-source","source_collection_id":"7"}})";
            point=store.upsertObject(w,p);vc::geometry::PointCollection points;points.ids={42};points.points={{1,2,3}};
            store.putGeometry(w,point,vc::geometry::encode(points));w.commit();}
        {auto r=store.read();oldVersion=store.currentObjectRevision(r,a)->reference();}
        {auto w=store.write();const std::array<double,6> xyz{0,0,0,20,0,0};
            newGeometry=store.editGeometry(w,a,{{"line_points",0,scalarBytes<double>(xyz)}},oldGeometry);w.commit();}
        {auto r=store.read();newVersion=store.currentObjectRevision(r,a)->reference();}
        {auto w=store.write(VersionSelection{{retired,1},{a,newVersion.version},{b,1}});store.retireObject(w,retired);
            EdgeRecord e;e.kind="related";e.aUuid=a;e.bUuid=b;e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;
            edge=store.addEdge(w,e);store.seal(w);w.commit();}
    }
    const auto overlay=[&](const char* name){const auto path=dir.path/name;
        {auto collection=ObjectCollection::create(path,{basePath},opt);}return ObjectStore::open(path,opt);};
    {auto store=overlay("select.vcobj");
        {auto w=store.write();store.upsertObject(w,track(util::newUuid()));w.commit();}
        auto r=store.read();const auto record=recordSelection(store,r,{});
        require(record.objects.size()==4,"selection includes local and inherited objects");
        require(decodeSelectionRecord(encodeSelectionRecord(record))==record,"mixed-owner selection round trips");
        for(const auto& ref:record.objects)require(bool(store.objectAtVersion(r,ref)),"selection references actual owner");
        require(currentFiberNetworkJson(store,r,{a},1024*1024).size()==1,"fiber export resolves inherited exact state");
    }
    {auto store=overlay("collision.vcobj");
        mustFail([&]{auto w=store.write();auto r=track(edge);store.upsertObject(w,r);w.commit();});
        mustFail([&]{auto w=store.write();EdgeRecord e;e.uuid=point;e.kind="related";e.aUuid=a;e.bUuid=b;
            e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;store.addEdge(w,e);w.commit();});
        {auto w=store.write();store.removeEdge(w,edge);w.commit();}
        mustFail([&]{auto w=store.write();store.upsertObject(w,track(edge));w.commit();});
    }
    {auto store=overlay("source.vcobj");
        const auto check=[&](bool found){auto r=store.read();auto* db=ObjectStoreImplAccess::conn(r);
            const auto p=sourcePoint(db,"audit-source",42);const auto c=sourceCollection(db,"audit-source","7");
            require(bool(p)==found&&bool(c)==found,"source lookup follows visible owner and tombstones");
            if(found)require(p->object==point&&*c==point,"source lookup resolves correct object");};
        check(true);
        for(bool sameCollection:{true,false})mustFail([&]{auto w=store.write();auto p=*store.objectByUuid(w,point);p.uuid=util::newUuid();p.storageKind="";p.storageRef="";
            if(!sameCollection)p.meta=R"({"source":{"origin":"audit-source","source_collection_id":"8"}})";
            const auto id=store.upsertObject(w,p);vc::geometry::PointCollection points;points.ids={42};points.points={{1,2,3}};
            store.putGeometry(w,id,vc::geometry::encode(points));
            auto* db=ObjectStoreImplAccess::conn(w);refreshSourceIdentityIndex(db);validateSourceIdentities(db,id);w.commit();});
        {auto w=store.write();store.setMeta(w,point,"note","true");w.commit();}check(true);
        {auto w=store.write();store.deleteObject(w,point);w.commit();}check(false);
    }
    for(bool retire:{false,true}){auto store=overlay(retire?"restore-retired.vcobj":"restore-active.vcobj");
        if(retire){int64_t current;{auto r=store.read();current=*store.currentVersion(r,a);}
            auto w=store.write(VersionSelection{{a,current}});store.retireObject(w,a);w.commit();}
        int64_t version;{auto r=store.read();version=*store.currentVersion(r,a);}
        {auto w=store.write(VersionSelection{{a,version}});store.restoreObject(w,a,oldVersion);w.commit();}
        auto r=store.read();const auto current=store.currentObjectRevision(r,a);
        require(!current->record.retired&&current->version==version+1,"restore publishes once, preserving version continuity");
        require(store.currentGeometry(r,a)->uuid==oldGeometry,"restore installs requested inherited archived geometry");
        require(store.readGeometryArray(r,oldGeometry,"line_points")==scalarBytes<double>(std::array<double,6>{0,0,0,10,0,0}),"restored bytes match old geometry");
        require(store.objectAtVersion(r,newVersion)->geometryRevision==newGeometry,"restore preserves old source version");
        require(store.edgesOf(r,a,true).empty(),"restore does not revive outdated relationships");
    }
    {auto store=overlay("restore-default.vcobj");int64_t version;
        {auto r=store.read();version=*store.currentVersion(r,retired);}
        {auto w=store.write(VersionSelection{{retired,version}});store.restoreObject(w,retired);w.commit();}
        auto r=store.read();require(!store.objectByUuid(r,retired)->retired,"default restore finds inherited owner's last active revision");
    }
    {const auto path=dir.path/"policy.vcobj";
        {auto store=overlay("policy.vcobj");auto w=store.write();store.setMeta(w,a,"note","true");w.commit();}
        for(auto mode:{HistoryMode::Off,HistoryMode::On,HistoryMode::Off}){
            changeHistoryMode(path,mode);auto store=ObjectStore::open(path,options(mode));auto r=store.read();
            Statement count(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM geometry_chunks");count.row();
            require(count.i64(0)==0,"history policy must not copy borrowed delta geometry");
            require(store.currentGeometry(r,a)->uuid==newGeometry,"policy preserves borrowed geometry identity");
            const auto bytes=store.readGeometryArray(r,newGeometry,"line_points");
            require(bytes==scalarBytes<double>(std::array<double,6>{0,0,0,20,0,0}),"borrowed geometry readable after each history transition");
        }
    }
}

void restoreFileReference(){
    Fixture f(HistoryMode::On,HistoryMode::On);
    auto store=ObjectStore::open(f.overlay,options(HistoryMode::On));ObjectVersion original;
    {auto r=store.read();original=store.currentObjectRevision(r,f.file)->reference();}
    {auto w=store.write(VersionSelection{{f.file,original.version}});store.restoreObject(w,f.file,original);w.commit();}
    auto r=store.read();require(store.resolveStoragePath(r,f.file)==f.base.parent_path()/"fiber.json",
        "restore rebases metadata file references through the ordinary override helper");
}

void deletedAndExactReads(HistoryMode baseMode,HistoryMode overlayMode){
    TemporaryDirectory dir;const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto live=util::newUuid(),deleted=util::newUuid();ObjectVersion original;std::string geometry;
    vc::geometry::Surface shape;shape.rows=shape.columns=2;shape.points={{1,1,1},{2,1,1},{1,2,1},{2,2,1}};
    {
        auto store=ObjectStore::create(base,options(baseMode));
        {auto w=store.write();for(const auto& id:{live,deleted}){ObjectRecord r;r.uuid=id;r.kind="surface";
            r.meta=R"({"coordinates":{"volume_uuid":"test-volume","scale_factor":1}})";
            store.upsertObject(w,r);const auto g=store.putGeometry(w,id,vc::geometry::encode(shape));if(id==live)geometry=g;}w.commit();}
        {auto r=store.read();original=store.currentObjectRevision(r,live)->reference();}
        {auto w=store.write();store.deleteObject(w,deleted);store.seal(w);w.commit();}
    }
    {auto collection=ObjectCollection::create(overlay,{base},options(overlayMode));}
    auto store=ObjectStore::open(overlay,options(overlayMode));
    if(baseMode==HistoryMode::On){auto r=store.read();const auto history=store.objectHistory(r,deleted);
        require(history.size()==2&&history.back().deleted,"deleted inherited record retains its own history");}
    else mustFail([&]{auto r=store.read();(void)store.objectHistory(r,deleted);});
    mustFail([&]{auto w=store.write();GeometrySource p;p.object.uuid=deleted;p.object.kind="surface";
        p.geometry=vc::geometry::encode(shape);p.source=nlohmann::json::object();ingestGeometryBatch(store,w,{p});w.commit();});
    {auto r=store.read();require(!store.objectByUuid(r,deleted),"rejected import must not resurrect inherited tombstone");}
    {auto w=store.write();store.setMeta(w,live,"changed","true");w.commit();}
    {auto r=store.read();LocalStoreReader reader(store,r);const auto current=reader.currentObject(live)->reference();
        require(current.storeUuid!=original.storeUuid,"override has new owning store");
        require(bool(reader.objectsAtVersions({current}).front()),"new owner reference resolves after edit");
        const auto grid=readSurfaceGrid(reader,original,"test-volume");
        require(grid.record.reference().storeUuid==original.storeUuid&&!nlohmann::json::parse(grid.record.meta).contains("changed"),
            "exact grid reads original metadata and geometry despite newer overlay");}
    {auto w=store.write();store.deleteObject(w,live);w.commit();}
    {auto r=store.read();LocalStoreReader reader(store,r);GeometrySnapshot snapshot(reader);
        require(snapshot.surface(live,geometry).rows()==2,"exact typed geometry survives current deletion");
        require(readSurfaceGrid(reader,original,"test-volume").rows==2,"exact grid survives current deletion");
        mustFail([&]{snapshot.surface(deleted,geometry);});}
}
}
int main()try{
    for(const auto mode:{HistoryMode::Off,HistoryMode::On})edgeIdentity(mode);
    for(const auto base:{HistoryMode::Off,HistoryMode::On})for(const auto overlay:{HistoryMode::Off,HistoryMode::On}){
        readsAndOverrides(base,overlay);geometryEdit(base,overlay,"retain");
        deletedAndExactReads(base,overlay);
        if(overlay==HistoryMode::On)geometryEdit(base,overlay,"keep_recorded");
    }
    queries();packing();spatialQueries();followupRegressions();restoreFileReference();std::cout<<"coherent store API tests passed\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
