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
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include "vc_object_store/objects/GeometryExport.hpp"
#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/ToolObjects.hpp"
#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/objects/RelationshipInputs.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "internal/Database.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include <fstream>
#include <iostream>
#include <type_traits>

using namespace vc::objects;
using namespace vc::objects::detail;
static_assert(!std::is_constructible_v<RecordExpectation, int64_t>);
static_assert(!std::is_constructible_v<RecordExpectation, std::optional<int64_t>>);
static_assert(!std::is_convertible_v<VersionSelection, RecordSelection>);
namespace {
OpenOptions options(HistoryMode mode){OpenOptions o;o.author="coherent-api";o.history=mode;o.spatialIndex=false;return o;}
ObjectRecord track(const std::string& id){ObjectRecord r;r.uuid=id;r.kind="track";r.track.emplace();r.track->externalKey="0";r.track->datasetRef=id;r.meta=R"({"quality":7})";return r;}
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
            {auto r=store.read();av=store.currentObjectRevision(r,a)->reference();
                bv=store.currentObjectRevision(r,b)->reference();edgeVersion=*store.currentRelationshipVersion(r,edge);}
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
    {auto w=store.write();store.updateEdge(w,unchanged);store.repairEdgeGeometry(w,repair);store.retainRelationship(w,f.edge);store.keepRecordedRelationship(w,f.edge);w.commit();}
    {auto r=store.read();require(store.edgeByUuid(r,f.edge)->versionStoreUuid==unchanged.versionStoreUuid,
        "unchanged edge update/repair/retain/keep preserves inherited ownership");
        require(store.status(r).commitSequence==sequence,"unchanged edge update/repair/retain/keep does not publish");}
    TransactionRequest noop;noop.requestUuid=util::newUuid();noop.author="no-op";
    {auto r=store.read();const auto state=store.status(r);noop.storeUuid=state.storeUuid;noop.generation=state.generation;
        buildCurrentEditRequest(LocalStoreReader(store,r),noop,{f.a},[]{return std::vector<TransactionOperation>{};});}
    require(!store.submit(noop).commitSequence,"default edit builder keeps unchanged inherited relationships without publication");
    {auto w=store.write();const std::array<ExtensionRow,1> rows{ExtensionRow{extensionUuid(f.edge),int64_t(4)}};
        store.replaceExtensionRows(w,extension().name,f.edge,rows);w.commit();}
    {auto r=store.read();require(store.status(r).commitSequence==sequence,"inherited no-op remains a no-op");}
    {auto w=store.write();const std::array<ExtensionRow,1> rows{ExtensionRow{extensionUuid(f.edge),int64_t(9)}};
        store.replaceExtensionRows(w,extension().name,f.edge,rows);w.commit();}
    {auto r=store.read();require(store.currentRelationshipVersion(r,f.edge)==f.edgeVersion+1,"edge override continues inherited version");}
    {auto w=store.writeNumbered(VersionSelection{{f.a,f.av.version}});store.setMeta(w,f.a,"note","true");w.commit();}
    {
        auto r=store.read();const auto exact=store.objectAtVersion(r,f.av);require(bool(exact),"exact source version survives shadowing");
        const auto ref=exact->reference();
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
    {auto w=store.writeNumbered(VersionSelection{{f.a,f.av.version},{f.b,f.bv.version}});
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
    VersionSelection expected;
    {auto r=store.read();for(const auto& id:{f.a,f.b})expected[id]=store.currentVersion(r,id);}
    {auto w=store.writeNumbered(expected);store.removeEdge(w,f.edge);store.deleteObject(w,f.a);w.commit();}
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
        {auto w=store.writeNumbered(VersionSelection{{retired,1},{a,newVersion.version},{b,1}});store.retireObject(w,retired);
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
            auto w=store.writeNumbered(VersionSelection{{a,current}});store.retireObject(w,a);w.commit();}
        int64_t version;{auto r=store.read();version=*store.currentVersion(r,a);}
        {auto w=store.writeNumbered(VersionSelection{{a,version}});store.restoreObject(w,a,oldVersion);w.commit();}
        auto r=store.read();const auto current=store.currentObjectRevision(r,a);
        require(!current->record.retired&&current->version==version+1,"restore publishes once, preserving version continuity");
        require(store.currentGeometry(r,a)->uuid==oldGeometry,"restore installs requested inherited archived geometry");
        require(store.readGeometryArray(r,oldGeometry,"line_points")==scalarBytes<double>(std::array<double,6>{0,0,0,10,0,0}),"restored bytes match old geometry");
        require(store.objectAtVersion(r,newVersion)->geometryRevision==newGeometry,"restore preserves old source version");
        require(store.edgesOf(r,a,true).empty(),"restore does not revive outdated relationships");
    }
    {auto store=overlay("restore-default.vcobj");int64_t version;
        {auto r=store.read();version=*store.currentVersion(r,retired);}
        {auto w=store.writeNumbered(VersionSelection{{retired,version}});store.restoreObject(w,retired);w.commit();}
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
    {auto w=store.writeNumbered(VersionSelection{{f.file,original.version}});store.restoreObject(w,f.file,original);w.commit();}
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
    {auto w=store.write();store.setMeta(w,live,"changed","true");
        store.setMeta(w,live,"coordinates",R"({"volume_uuid":"test-volume","scale_factor":2})");w.commit();}
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
        const auto retained=reader.objectAtGeometry(geometry);
        require(retained&&retained->geometryRevision==geometry,"geometry metadata resolves through deleted overlay");
        if(overlayMode==HistoryMode::Off)require(retained->reference()==original,"current-only overlay resolves retained source metadata");
        const double scale=overlayMode==HistoryMode::On?.5:1;
        require(snapshot.surface(live,geometry).viewTransformedAt("test-volume").readTile(0,0,2,2).points.front()==vc::geometry::Point{scale,scale,scale},
                "retained geometry transforms with its retained source metadata");
        require(snapshot.surface(live,geometry).availableTransforms().size()==reader.availableTransforms(*reader.objectsAtVersions({original}).front()).size(),
                "historical transform listing uses the same selected metadata");
        mustFail([&]{snapshot.surface(deleted,geometry);});}
}
}
void siblingViewWrites(HistoryMode mode){
    TemporaryDirectory dir;
    const auto base=dir.path/"base.vcobj",left=dir.path/"left.vcobj",right=dir.path/"right.vcobj";
    const auto id=util::newUuid();
    {auto store=ObjectStore::create(base,options(mode));auto w=store.write();auto r=track(id);
        r.meta=R"({"coordinates":{"volume_uuid":"source","scale_factor":1}})";
        store.upsertObject(w,r);store.putGeometry(w,id,shape());w.commit();
        auto seal=store.write();store.seal(seal);seal.commit();}
    {auto c=ObjectCollection::create(left,{base},options(mode));}
    {auto c=ObjectCollection::create(right,{base},options(mode));}
    auto a=ObjectStore::open(left,options(mode)),b=ObjectStore::open(right,options(mode));
    {auto w=a.write();a.setMeta(w,id,"branch",R"("left")");w.commit();}
    {auto w=b.write();b.setMeta(w,id,"branch",R"("right")");w.commit();}
    GeometrySnapshot snapshot(a);const auto view=snapshot.viewTransformedAt(id,"source");
    {auto r=b.read();require(view.version()==b.currentVersion(r,id),"sibling counters collide");}
    mustFail([&]{auto w=b.write();view.writeCoordinates(b,w,"points",0,{{9,9,9}});w.commit();});
    {GeometrySnapshot result(b);require(result.track(id).points(0,1).front()==vc::geometry::TrackPoint{1,2,3},
        "rejected sibling write preserves geometry");}
    {auto w=a.write();view.writeCoordinates(a,w,"points",0,{{9,9,9}});w.commit();}
}
void completeReceipts(HistoryMode mode){
    Fixture f(mode,mode);
    auto store=ObjectStore::open(f.overlay,options(mode));
    TransactionRequest request;request.requestUuid=util::newUuid();request.author="receipt-test";
    {auto r=store.read();const auto status=store.status(r);request.storeUuid=status.storeUuid;request.generation=status.generation;}
    request.expectedObjects={{f.a,f.av},{f.b,f.bv}};
    request.operations.push_back(SetMetaOp{f.a,"note","true"});
    const auto receipt=store.submit(request);
    require(receipt.objectReference(f.b)==f.bv,"unchanged inherited receipt identity stays source-owned");
    require(receipt.objectReference(f.a)->storeUuid==receipt.storeUuid,"changed receipt identity belongs to writer");
    {auto r=store.read();require(store.objectAtVersion(r,*receipt.objectReference(f.a)).has_value(),"committed reference resolves");}
    const auto decoded=transactionReceiptFromJson(transactionReceiptToJson(receipt));
    require(decoded.objectReferences==receipt.objectReferences,"receipt transport preserves complete identities");
    auto invalid=nlohmann::json::parse(transactionReceiptToJson(receipt));
    invalid["object_references"][f.a]=nlohmann::json::object();
    mustFail([&]{transactionReceiptFromJson(invalid.dump());});
    invalid.erase("object_references");
    const auto legacy=transactionReceiptFromJson(invalid.dump());
    require(legacy.objects==receipt.objects,"legacy receipts remain readable");
    mustFail([&]{legacy.objectReference(f.a);});
    const auto replay=store.submit(request);
    require(replay.replayed&&replay.objectReferences==receipt.objectReferences,"durable replay preserves reference identities");
    request.requestUuid=util::newUuid();request.expectedObjects.clear(); for (const auto& [id, ref] : receipt.objectReferences) expectObject(request, ref);
    const auto noop=store.submit(request);
    require(!noop.commitSequence&&noop.objectReferences==receipt.objectReferences,"no-op receipt retains both owners");
    VersionSelection expected;
    {auto r=store.read();for(const auto& id:{f.a,f.b})expected[id]=store.currentVersion(r,id);}
    {auto w=store.writeNumbered(expected);store.removeEdge(w,f.edge);store.deleteObject(w,f.a);w.commit();}
    require(store.submit(request).objectReferences==noop.objectReferences,"later deletion cannot change a saved receipt");
}
void retainedPointAllocation(HistoryMode overlayMode){
    TemporaryDirectory dir;const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto id=util::newUuid();ObjectVersion initial;
    vc::geometry::Fiber shape{{{0,0,0},{1,0,0}},{},{0,1}};
    {auto store=ObjectStore::create(base,options(HistoryMode::On));
        {auto w=store.write();ObjectRecord r;r.uuid=id;r.kind="fiber";store.upsertObject(w,r);store.putGeometry(w,id,vc::geometry::encode(shape));w.commit();}
        {auto r=store.read();initial=store.currentObjectRevision(r,id)->reference();}
        {auto w=store.writeNumbered(VersionSelection{{id,initial.version}});auto extended=shape;extended.controlPointIds.push_back(2);extended.controlPoints.push_back({2,0,0});
            updateGeometryArrays(store,w,id,vc::geometry::encode(extended),store.currentGeometry(w,id)->uuid);w.commit();}
        {auto w=store.writeNumbered(VersionSelection{{id,initial.version+1}});store.restoreObject(w,id,initial);store.seal(w);w.commit();}}
    {auto c=ObjectCollection::create(overlay,{base},options(overlayMode));}
    auto store=ObjectStore::open(overlay,options(overlayMode));int64_t version;
    {auto r=store.read();LocalStoreReader reader(store,r);version=*store.currentVersion(r,id);
        require(reader.assignGeometryPointIds(id,"control_points",{0,1,-1,1})==std::vector<int64_t>{0,1,3,4},
            "restored-away IDs stay allocated across store policies");
        require(reader.assignGeometryPointIds(id,"control_points",{1,0})==std::vector<int64_t>{1,0},"reordering preserves IDs");}
    {auto w=store.writeNumbered(VersionSelection{{id,version}});LocalStoreReader reader(store,w);
        shape.controlPointIds=reader.assignGeometryPointIds(id,"control_points",{0,1,-1});shape.controlPoints.push_back({2,0,0});
        updateGeometryArrays(store,w,id,vc::geometry::encode(shape),store.currentGeometry(w,id)->uuid);w.commit();}
    {auto r=store.read();require(store.nextGeometryPointId(r,id,"control_points")==4,"accepted allocation advances retained high-water mark");}
}
void allocationWithoutCurrentGeometry(){
    TemporaryDirectory dir;auto store=ObjectStore::create(dir.path/"metadata.vcobj",options(HistoryMode::On));
    const auto id=util::newUuid();ObjectVersion first;
    {auto w=store.write();ObjectRecord object;object.uuid=id;object.kind="fiber";store.upsertObject(w,object);w.commit();}
    {auto r=store.read();first=store.currentObjectRevision(r,id)->reference();}
    {auto w=store.write();store.putGeometry(w,id,vc::geometry::encode(vc::geometry::Fiber{{{0,0,0}},{},{7}}));w.commit();}
    {auto w=store.writeNumbered(VersionSelection{{id,first.version+1}});store.restoreObject(w,id,first);w.commit();}
    {auto r=store.read();require(!store.currentGeometry(r,id),"restored version predates geometry");
        require(store.nextGeometryPointId(r,id,"control_points")==8,"absent current geometry does not erase retained allocations");}
}
void inventoryCore(HistoryMode baseMode, HistoryMode overlayMode) {
    Fixture f(baseMode, overlayMode, true);
    auto store=ObjectStore::open(f.overlay, options(overlayMode));
    // Registration, scans and reads share the same inherited schema contract.
    {auto w=store.write();auto incompatible=extension();incompatible.columns.back().type=ExtensionType::Text;
        mustFail([&]{store.createExtensionTable(w,incompatible);});w.commit();}
    {auto w=store.write();store.createExtensionTable(w,extension());w.commit();}
    // No-op comparison must precede override creation, not merely roll it back.
    {auto w=store.write();int mutations=0;auto* db=ObjectStoreImplAccess::conn(w);
        sqlite3_update_hook(db,[](void* value,int,const char*,const char* table,sqlite3_int64){
            if(std::string_view(table)=="objects"||std::string_view(table)=="edges")++*static_cast<int*>(value);
        },&mutations);
        store.setMeta(w,f.a,"quality","7");store.removeMeta(w,f.a,"absent");store.setApproval(w,f.a,"pending");
        store.setApproval(w,f.edge,"pending");store.removeMeta(w,f.edge,"absent");
        sqlite3_update_hook(db,nullptr,nullptr);require(mutations==0,"metadata no-ops must not create transient overrides");w.commit();}
    {auto w=store.writeNumbered(VersionSelection{{f.a,f.av.version}});GeometrySequenceEdit edit;edit.array="points";
        require(store.editGeometrySequence(w,f.a,edit,f.geometry)==f.geometry,"empty splice keeps geometry identity");w.commit();}
    {auto r=store.read();require(store.currentObjectRevision(r,f.a)->reference()==f.av,"empty splice keeps source owner");
        const auto exact=store.relationshipAtVersion(r,{store.edgeByUuid(r,f.edge)->versionStoreUuid,
            store.edgeByUuid(r,f.edge)->versionGeneration,f.edge,f.edgeVersion});
        require(exact&& exact->aReference.version>0&&exact->bReference.version>0,"all retention policies return endpoint references");
        require(exact->record.versionStoreUuid==f.av.storeUuid,"relationship history retains its own owner");}
    // Exact borrowed geometry still exports after a metadata override or deletion.
    {auto w=store.write();store.setMeta(w,f.a,"note","true");w.commit();}
#ifdef VC_TEST_NATIVE_EXPORT
    exportGeometry(store,f.a,f.dir.path/"exported",f.av);
#endif
    {auto r=store.read();const auto snapshot=store.versionsAt(r,f.av);
        require(std::find(snapshot.begin(),snapshot.end(),f.av)!=snapshot.end(),"publication selection uses source owner");
        for(const auto& protectedPath:std::vector<std::filesystem::path>{f.base,f.base.parent_path(),f.base.string()+"-wal",f.base.string()+".artifacts"})
            mustFail([&]{store.validateExportDestination(r,protectedPath);});}
    const auto hardlink=f.dir.path/"hardlink";std::filesystem::create_hard_link(f.base,hardlink);
    {auto r=store.read();mustFail([&]{store.validateExportDestination(r,hardlink);});}
#ifdef VC_TEST_NATIVE_EXPORT
    mustFail([&]{exportGeometry(store,f.a,hardlink,f.av,true);});
#endif
    VersionSelection expected;
    {auto r=store.read();for(const auto& id:{f.a,f.b})expected[id]=store.currentVersion(r,id);}
    {auto w=store.writeNumbered(expected);store.removeEdge(w,f.edge);store.deleteObject(w,f.a);w.commit();}
#ifdef VC_TEST_NATIVE_EXPORT
    exportGeometry(store,f.a,f.dir.path/"exported-deleted",f.av);
#endif
    {auto r=store.read();require(store.readGeometryArray(r,f.geometry,"points")==shape().arrays.front().bytes,"protected dependencies stay intact");}
}

void inventoryRestore(HistoryMode baseMode,HistoryMode overlayMode) {
    Fixture f(baseMode,overlayMode);
    auto store=ObjectStore::open(f.overlay,options(overlayMode));
    {auto w=store.writeNumbered(VersionSelection{{f.a,f.av.version},{f.b,f.bv.version}});store.retireObject(w,f.a);w.commit();}
    int64_t version;{auto r=store.read();version=*store.currentVersion(r,f.a);}
    {auto w=store.writeNumbered(VersionSelection{{f.a,version}});store.restoreObject(w,f.a,f.av);w.commit();}
    {auto r=store.read();require(!store.objectByUuid(r,f.a)->retired,"explicit retained source restores independent destination history");}
    // Retired current-only source can be reactivated in a history-on overlay.
    const auto next=f.dir.path/"next.vcobj";
    {auto r=store.read();version=*store.currentVersion(r,f.a);}
    {auto w=store.writeNumbered(VersionSelection{{f.a,version}});store.retireObject(w,f.a);store.seal(w);w.commit();}
    {auto collection=ObjectCollection::create(next,{f.overlay},options(HistoryMode::On));}
    auto child=ObjectStore::open(next,options(HistoryMode::On));
    {auto r=child.read();version=*child.currentVersion(r,f.a);}
    {auto w=child.writeNumbered(VersionSelection{{f.a,version}});child.restoreObject(w,f.a);w.commit();}
    {auto r=child.read();require(!child.objectByUuid(r,f.a)->retired,"default reactivation uses source retention");}
}

void inventoryIdentityAndSources() {
    Fixture f(HistoryMode::Off,HistoryMode::Off);
    {auto store=ObjectStore::open(f.overlay,options(HistoryMode::Off));auto r=store.read();
        const auto copy=readObjectCopy(LocalStoreReader(store,r),f.a);
        require(nlohmann::json::parse(copy.record.meta).at("copied_from").at("store_uuid")==f.av.storeUuid,"copy provenance uses source owner");}
    ObjectStoreSession session;const auto attachment=session.attach(f.overlay,options(HistoryMode::Off));
    PinnedObject pin;
    session.readStore(attachment,[&](const StoreReader& reader,const auto&){pin=pinOf(attachment,*reader.currentObject(f.a),reader.status().generation);});
    session.requireCurrent(pin);
    auto wrong=pin;wrong.selected.storeUuid=util::newUuid();mustFail([&]{session.requireCurrent(wrong);});
    auto request=session.newRequest(attachment);auto other=f.av;other.storeUuid=util::newUuid();expectObject(request,other);
    request.operations.push_back(SetMetaOp{f.a,"note","true"});
    const auto wire=transactionRequestToJson(request);const auto roundtrip=transactionRequestFromJson(wire,{});
    require(roundtrip.expectedObjects==request.expectedObjects,"full write expectations survive transport");
    auto inconsistent=nlohmann::json::parse(wire);inconsistent["expected_objects"][f.a]=other.version+1;
    mustFail([&]{transactionRequestFromJson(inconsistent.dump(),{});});
    inconsistent=nlohmann::json::parse(wire);inconsistent["command_version"]=1;
    mustFail([&]{transactionRequestFromJson(inconsistent.dump(),{});});
    mustFail([&]{session.submit(attachment,roundtrip);});
    // A tool-created output inherits the frame actually loaded, not today's head.
    ToolObjects tool(f.overlay,"test");
    tool.session()->readStore(tool.attachment(),[&](const StoreReader& reader,const auto&){
        tool.noteRead(pinOf(tool.attachment(),*reader.currentObject(f.a),reader.status().generation),{},true);});
    const auto loadedCoordinates=tool.coordinatesFor({f.a},{});
    session.editCurrent(attachment,{f.a},[&](const auto&,const auto&){return std::vector<TransactionOperation>{SetMetaOp{f.a,"coordinates",R"({"volume_uuid":"new-frame","scale_factor":1})"}};});
    require(tool.coordinatesFor({f.a},{})==loadedCoordinates,"derived coordinates follow retained source pin");
}

void inventorySelectedMembership() {
    TemporaryDirectory dir;const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto a=util::newUuid(),b=util::newUuid();std::vector<ObjectVersion> refs;std::string edge;
    {auto store=ObjectStore::create(base,options(HistoryMode::On));
        {auto w=store.write();store.upsertObject(w,track(a));store.upsertObject(w,track(b));
            EdgeRecord e;e.aUuid=a;e.bUuid=b;e.kind="old";edge=store.addEdge(w,e);w.commit();}
        {auto r=store.read();refs={store.currentObjectRevision(r,a)->reference(),store.currentObjectRevision(r,b)->reference()};}
        {auto w=store.write();store.removeEdge(w,edge);EdgeRecord e;e.aUuid=a;e.bUuid=b;e.kind="new";store.addEdge(w,e);store.seal(w);w.commit();}}
    {auto collection=ObjectCollection::create(overlay,{base},options(HistoryMode::Off));}
    auto store=ObjectStore::open(overlay,options(HistoryMode::Off));auto r=store.read();
    const auto graph=LocalStoreReader(store,r).graphReadViewAt(refs);
    require(graph.selectedRelationships->size()==1&&graph.selectedRelationships->front().record.kind=="old",
        "history-off attachment cannot substitute newer source graph memberships");
}

void inventoryRevalidatedPeer(HistoryMode baseMode, HistoryMode overlayMode) {
    Fixture f(baseMode,overlayMode);
    auto store=ObjectStore::open(f.overlay,options(overlayMode));
    {auto w=store.writeNumbered(VersionSelection{{f.a,f.av.version},{f.b,f.bv.version}});
        store.setMeta(w,f.b,"quality","8");store.retainRelationship(w,f.edge);w.commit();}
    auto r=store.read();
    require(store.currentObjectRevision(r,f.a)->reference()==f.av,"revalidation leaves the borrowed endpoint unchanged");
    require(store.currentObjectRevision(r,f.b)->reference()!=f.bv,"revalidated peer has a new local version");
    const auto graph=LocalStoreReader(store,r).graphReadViewAt({f.av,f.bv});
    require(graph.edgeUuids==std::vector<std::string>{f.edge},
        "new binding on one unchanged endpoint must not replace the selected historical peer binding");
    require(graph.selectedRelationships->front().version==f.edgeVersion,
        "the exact old pair retains its original relationship version");
}

void inventoryArtifactAlias() {
    TemporaryDirectory dir;const auto path=dir.path/"store.vcobj",payloads=dir.path/"payloads";
    auto store=ObjectStore::create(path,options(HistoryMode::Off));
    std::filesystem::create_directory(payloads);
    std::ofstream(payloads/"payload")<<"immutable artifact";
    std::filesystem::create_directory_symlink(payloads,path.string()+".artifacts");
    auto r=store.read();
    mustFail([&]{store.validateExportDestination(r,payloads/"payload");});
    mustFail([&]{store.validateExportDestination(r,std::filesystem::path(path.string()+".artifacts")/"payload");});
}

#ifdef VC_TEST_NATIVE_EXPORT
void inventoryMixedOwnerExport() {
    TemporaryDirectory dir;const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto a=util::newUuid(),b=util::newUuid();ObjectVersion selected;
    const auto addFiber=[](auto& store,auto& write,const std::string& uuid){
        ObjectRecord object;object.uuid=uuid;object.kind="fiber";object.formatVersion=4;
        store.upsertObject(write,object);
        store.putGeometry(write,uuid,vc::geometry::encode(vc::geometry::Fiber{
            {{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{0,1}}));
    };
    {auto store=ObjectStore::create(base,options(HistoryMode::Off));auto w=store.write();
        addFiber(store,w,b);store.seal(w);w.commit();}
    {auto collection=ObjectCollection::create(overlay,{base},options(HistoryMode::Off));}
    auto store=ObjectStore::open(overlay,options(HistoryMode::Off));
    {auto w=store.write();addFiber(store,w,a);EdgeRecord edge;edge.kind="fiber_branch";edge.aUuid=a;edge.bUuid=b;
        edge.aAnchor.points={{"control_point_ids",{0}}};edge.bAnchor.points={{"control_point_ids",{1}}};
        edge.aAnchor.direction={1,0,0};edge.bAnchor.direction={0,1,0};
        edge.aAnchor.position={1,2,3};edge.bAnchor.position={4,5,6};store.addEdge(w,edge);w.commit();}
    {auto r=store.read();selected=store.currentObjectRevision(r,a)->reference();}
    const auto currentPath=dir.path/"current.json",exactPath=dir.path/"exact.json";
    exportGeometry(store,a,currentPath);exportGeometry(store,a,exactPath,selected);
    std::ifstream currentFile(currentPath),exactFile(exactPath);
    const auto current=nlohmann::json::parse(currentFile),exact=nlohmann::json::parse(exactFile);
    require(current.at("branches").size()==1,"current export includes the inherited branch peer");
    require(exact.at("branches")==current.at("branches"),"exact current-only export preserves mixed-owner graph closure");
}
#endif

void inventoryPackedAndCoordinates() {
    TemporaryDirectory dir;const auto base=dir.path/"base.vcobj",packed=dir.path/"packed.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto id=util::newUuid(),umb=util::newUuid(),surface=util::newUuid();ObjectVersion ref;
    {auto store=ObjectStore::create(base,options(HistoryMode::Off));
        {auto w=store.write();auto object=track(id);object.track->datasetRef="tracks";store.upsertObject(w,object);
            object={};object.uuid=umb;object.kind="umbilicus";object.meta=R"({"coordinates":{"volume_uuid":"v","scale_factor":1}})";
            store.upsertObject(w,object);store.putGeometry(w,umb,vc::geometry::encode(vc::geometry::Umbilicus{{{1,2,3},{4,5,6}}}));
            object={};object.uuid=surface;object.kind="surface";store.upsertObject(w,object);
            store.putGeometry(w,surface,vc::geometry::encode(vc::geometry::Surface{2,2,{1,1},{{0,0,1},{1,0,1},{0,1,1},{1,1,1}}}));w.commit();}
        {auto r=store.read();ref=store.currentObjectRevision(r,surface)->reference();}
        {auto w=store.write();store.seal(w);w.commit();}}
    packObjectCollection(base,packed);
    std::filesystem::create_directory(dir.path/"nested");
    const auto nested=dir.path/"nested"/"overlay.vcobj";
    {auto collection=ObjectCollection::create(nested,{packed},options(HistoryMode::On));}
    auto store=ObjectStore::open(nested,options(HistoryMode::On));
    {auto r=store.read();mustFail([&]{store.validateExportDestination(r,packed);});
        require(store.resolveStoragePath(r,id)==dir.path/"tracks","track dataset uses physical owner's directory");
        const auto selected=volumeUmbilicus(LocalStoreReader(store,r),{"v",1});
        require(selected&&selected->version.storeUuid==ref.storeUuid,"umbilicus uses selected owner");}
    {auto w=store.write();store.setMeta(w,id,"note","true");w.commit();}
    {auto r=store.read();require(store.resolveStoragePath(r,id)==dir.path/"tracks","track path survives metadata override");}
    std::string geometry;{auto r=store.read();geometry=store.currentGeometry(r,surface)->uuid;}
    {auto w=store.writeNumbered(VersionSelection{{surface,ref.version}});GeometrySurfaceResize resize;resize.rows=resize.columns=2;
        resize.fill={scalarArray<float>("points","<f4",{1,1,3},std::array<float,3>{-1,-1,-1})};
        require(store.resizeSurface(w,surface,resize,geometry)==geometry,"identity resize preserves geometry");w.commit();}
    {auto r=store.read();require(store.currentObjectRevision(r,surface)->reference()==ref,"identity resize preserves owner/version");}
}

void followupAtomicityAndPolicy()
{
    for(const auto mode:{HistoryMode::Off,HistoryMode::On}) {
        Fixture f(HistoryMode::On,mode);
        auto store=ObjectStore::open(f.overlay,options(mode));
        ObjectRecord previous;
        {auto r=store.read();previous=*store.objectByUuid(r,f.a);}
        {auto w=store.write();store.upsertObject(w,previous);w.commit();}
        {auto r=store.read();require(store.currentObjectRevision(r,f.a)->reference()==f.av,"full-record no-op preserves inherited identity");}
        std::optional<std::filesystem::path> originalPath;
        {auto r=store.read();originalPath=store.resolveStoragePath(r,f.a);}
        previous.name="renamed";
        {auto w=store.writeNumbered(VersionSelection{{f.a,f.av.version}});store.upsertObject(w,previous);w.commit();}
        ObjectVersion after;
        {auto r=store.read();require(store.resolveStoragePath(r,f.a)==originalPath,"full-record override keeps the unchanged relative track dataset target");
            previous=*store.objectByUuid(r,f.a);after=store.currentObjectRevision(r,f.a)->reference();}
        {auto w=store.writeNumbered(VersionSelection{{f.a,after.version}});store.upsertObject(w,previous);w.commit();}
        {auto r=store.read();require(store.currentObjectRevision(r,f.a)->reference()==after,"full-record local no-op preserves the version");}
        TransactionRequest request;
        {auto r=store.read();const auto status=store.status(r);
            request.requestUuid=util::newUuid();request.storeUuid=status.storeUuid;request.generation=status.generation;request.author="test";
            buildCurrentEditRequest(LocalStoreReader(store,r),request,{f.a},[&]{return std::vector<TransactionOperation>{SetMetaOp{f.a,"quality","9"}};});}
        store.submit(request);
        {auto r=store.read();EdgeSelection selection;selection.applicableOnly=true;
            require(store.selectEdges(r,selection).empty(),"generic edit preserves recorded evidence regardless of destination history policy");}
        GeometrySource good;good.source=nlohmann::json::object();good.object.uuid=util::newUuid();good.object.kind="umbilicus";
        good.geometry=vc::geometry::encode(vc::geometry::Umbilicus{{{1,2,3}}});
        auto bad=good;bad.object.uuid=util::newUuid();bad.geometry.arrays.clear();
        ObjectRecord unrelated;unrelated.kind="note";
        {auto w=store.write();unrelated.uuid=store.upsertObject(w,unrelated);
            mustFail([&]{ingestGeometryBatch(store,w,{good,bad});});w.commit();}
        {auto r=store.read();require(store.objectByUuid(r,unrelated.uuid).has_value(),"unrelated caller work survives caught batch failure");
            require(!store.objectByUuid(r,good.object.uuid)&&!store.objectByUuid(r,bad.object.uuid),"failed batch rolls back every imported object");}
    }
    for(const auto cache:{size_t(0),size_t(1024*1024)}) {
        TemporaryDirectory dir;auto opt=options(HistoryMode::On);opt.spatialIndex=true;opt.spatialCacheBytes=cache;
        auto store=ObjectStore::create(dir.path/"spatial",opt);std::string retired,deleted;
        {auto w=store.write();for(auto* id:{&retired,&deleted}){ObjectRecord r;r.kind="track";r.track.emplace();
            *id=store.upsertObject(w,r);store.putGeometry(w,*id,shape());}w.commit();}
        {auto w=store.writeNumbered(VersionSelection{{retired,1},{deleted,1}});store.retireObject(w,retired);store.deleteObject(w,deleted);w.commit();}
        auto r=store.read();ObjectSelection selection;selection.bbox=std::array<double,6>{0,0,0,100,100,100};selection.includeRetired=true;
        const auto regions=store.spatialRegions(r,selection);
        require(regions.size()==1&&regions.front().objectUuid==retired,"spatial queries include retired but never deleted objects, with or without cache");
    }
}

void firstInheritedDeletion(HistoryMode baseMode,HistoryMode overlayMode){
    Fixture f(baseMode,overlayMode);auto store=ObjectStore::open(f.overlay,options(overlayMode));
    StoreStatus before;ObjectVersion original;
    {auto r=store.read();before=store.status(r);original=store.currentObjectRevision(r,f.file)->reference();}
    TransactionRequest deletion;deletion.requestUuid=util::newUuid();deletion.author="delete";
    deletion.storeUuid=before.storeUuid;deletion.generation=before.generation;expectObject(deletion,original);
    deletion.operations.push_back(DeleteObjectOp{f.file});
    require(store.submit(deletion).objects.at(f.file)==std::nullopt,"deletion receipt names removed inherited object");
    auto r=store.read();require(!store.objectByUuid(r,f.file),"first inherited deletion hides the object");
    require(bool(store.objectAtVersion(r,original)),"source version remains readable after first deletion");
    const auto feed=store.changesSince(r,{before.storeUuid,before.generation,before.commitSequence,0});
    if(overlayMode==HistoryMode::On){
        require(feed.commits.size()==1&&feed.commits[0].changes.size()==1&&feed.commits[0].changes[0].uuid==f.file&&
            feed.commits[0].changes[0].deleted,"first inherited deletion is published to the feed");
        const auto history=store.objectHistory(r,f.file);
        require(history.size()==1&&history[0].deleted,"first inherited deletion retains one real tombstone, not a fake baseline");
        require((f.overlay.parent_path()/history[0].record.storageRef).lexically_normal()==f.base.parent_path()/"fiber.json",
            "inherited deletion history preserves the external source target");
    }else require(!feed.cursorValid&&store.status(r).commitSequence>before.commitSequence,"history-off deletion invalidates the old feed snapshot");
}

void atomicCompositeWrites(HistoryMode mode){
    TemporaryDirectory dir;auto store=ObjectStore::create(dir.path/"atomic",options(mode));
    ObjectRecord old;old.kind="umbilicus";old.uuid=util::newUuid();
    old.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
    {auto w=store.write();setVolumeUmbilicus(store,w,"a",old,vc::geometry::Umbilicus{{{0,0,0},{0,0,1}}});w.commit();}
    const auto surface=util::newUuid(),replacement=util::newUuid();StoreStatus before;
    {auto r=store.read();before=store.status(r);}
    {auto w=store.write();GeometryData malformed;malformed.kind="surface";
        mustFail([&]{createSurface(store,w,surface,"invalid",{},nlohmann::json::object(),{},nlohmann::json::object(),malformed);});
        auto next=old;next.uuid=replacement;
        mustFail([&]{setVolumeUmbilicus(store,w,"a",next,vc::geometry::Umbilicus{});});
        const vcr::catalog::Registration registration{"id","revision","a","b",VolumeAffine{},"test"};
        mustFail([&]{store.recordCatalogTransforms(w,{registration,registration});});w.commit();}
    auto r=store.read();require(!store.objectByUuid(r,surface)&&!store.objectByUuid(r,replacement),"failed composite writes leave no objects");
    require(!store.objectByUuid(r,old.uuid)->retired,"failed replacement preserves active umbilicus");
    require(store.objectsByKind(r,"volume_transform").empty(),"failed registration batch is atomic");
    require(store.status(r).commitSequence==before.commitSequence,"caught composite failures leave no commit");
}

void measuredLengths(HistoryMode mode){
    TemporaryDirectory dir;auto opt=options(mode);opt.spatialIndex=true;
    const auto base=dir.path/"base",overlay=dir.path/"overlay";
    std::vector<std::string> ids;
    {auto store=ObjectStore::create(base,opt);auto w=store.write();
        for(int i=1;i<=2;++i){ObjectRecord record;record.kind="fiber";record.fiber.emplace();record.fiber->length=i==1?100:1;
            record.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
            ids.push_back(store.upsertObject(w,record));vc::geometry::Fiber fiber;fiber.linePoints={{0,0,0},{double(10*i),0,0}};
            store.putGeometry(w,ids.back(),vc::geometry::encode(fiber));}
        ObjectRecord placeholder;placeholder.kind="fiber";placeholder.fiber.emplace();placeholder.fiber->length=999;
        placeholder.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
        store.upsertObject(w,placeholder);
        store.seal(w);w.commit();}
    {auto collection=ObjectCollection::create(overlay,{base},opt);}
    auto store=ObjectStore::open(overlay,opt);
    const auto check=[&]{auto r=store.read();ObjectSelection all;all.kind="fiber";
        require(store.selectObjects(r,all).size()==3,"ordinary listing includes geometry-less placeholders");
        for(bool volume:{false,true}){
        ObjectSelection selection;selection.kind="fiber";if(volume)selection.volume="a";selection.minLength=15;
        auto selected=store.selectObjects(r,selection);require(selected.size()==1&&selected[0].uuid==ids[1],"length filter measures geometry, not annotation");
        selection.bbox=std::array<double,6>{-1,-1,-1,30,1,1};
        const auto regions=store.spatialRegions(r,selection);require(!regions.empty(),"measured spatial selection retains long fiber");
        for(const auto& region:regions)require(region.objectUuid==ids[1],"spatial length agrees with object length");
        RegionOptions region;region.kind="fiber";region.minLength=15;if(volume)region.volume="a";
        RegionSlice slice;slice.localBounds={-1,-1,-1,30,1,1};
        const auto visible=store.geometryRegion(r,slice,region);require(visible.intersectingObjects==1&&visible.runs[0].objectUuid==ids[1],"viewport length agrees in native and volume frames");
        selection.bbox.reset();selection.minLength.reset();selection.orderBy="length_desc";selection.limit=1;
        const auto first=store.selectObjects(r,selection);require(first.size()==1&&first[0].uuid==ids[1],"geometry length determines order");
        require(first[0].volumeLength==20&&first[0].fiber->length==1,"measured cursor does not mutate selected metadata");
        selection.afterUuid=ids[1];selection.afterLength=20;
        const auto second=store.selectObjects(r,selection);require(second.size()==1&&second[0].uuid==ids[0],"length pagination uses the same measurement");
    }};
    check();{auto w=store.write();store.registerVolumeTransform(w,"unrelated-x","unrelated-y",VolumeAffine{},"unrelated");w.commit();}check();
}

void regionPrecedenceRefill()
{
    TemporaryDirectory dir;auto opts=options(HistoryMode::On);opts.spatialIndex=true;
    const auto path=dir.path/"region";const std::string a="00000000-0000-4000-8000-000000000001",b="00000000-0000-4000-8000-000000000002";
    vc::geometry::Surface grid{2,2,{1,1},{{0,0,1},{2,0,3},{0,2,1},{2,2,3}}};
    {auto store=ObjectStore::create(path,opts);auto w=store.write();for(const auto& id:{a,b}){ObjectRecord record;record.uuid=id;record.kind="surface";
        record.meta=R"({"coordinates":{"volume_uuid":"region","scale_factor":1}})";store.upsertObject(w,record);store.putGeometry(w,id,vc::geometry::encode(grid));}w.commit();}
    ObjectStoreSession session;const auto attachment=session.attach(path,opts);
    for(auto& point:grid.points)point[2]+=100;
    session.setDraft(a,grid,volumeCoordinates("region",1),attachment);
    RegionSlice slice;slice.origin={0,0,2};slice.localBounds={-10,-10,-0.1,10,10,0.1};
    RegionOptions query;query.kind="surface";query.volume="region";query.maximumObjects=1;
    {auto excluded=query;excluded.excludedUuids={a};
        excluded=regionQueryRequestFromJson(regionQueryRequestToJson(slice,excluded)).second;
        auto reader=ObjectStore::open(path,opts);auto txn=reader.read();
        const auto result=reader.geometryRegion(txn,slice,excluded);
        require(result.intersectingObjects==1&&result.runs[0].objectUuid==b,"wire exclusions precede backend admission");}
    const auto rows=session.geometryRegion({attachment},slice,query);
    require(!rows.empty()&&rows[0].result.intersectingObjects==1,"hidden draft cannot consume a visible object slot");
    require(!rows[0].result.runs.empty()&&rows[0].result.runs[0].objectUuid==b,"bounded refill reaches next visible surface");
    require(rows[0].result.stats.outputBytes>rows[0].result.runs[0].points.size()*sizeof(std::array<double,3>),"discarded output remains charged");
    query.processingByteBudget=1;
    const auto bounded=session.geometryRegion({attachment},slice,query);
    require(bounded[0].result.stop==RegionStop::ProcessingBudget&&bounded[0].result.stats.processingBytes<=1,"refill never resets processing budget");
    session.clearDraft(a);
    const auto firstPath=dir.path/"earlier";
    {auto first=ObjectStore::create(firstPath,opts);auto w=first.write();ObjectRecord record;record.uuid=a;record.kind="surface";
        record.meta=R"({"coordinates":{"volume_uuid":"region","scale_factor":1}})";first.upsertObject(w,record);first.putGeometry(w,a,vc::geometry::encode(grid));w.commit();}
    const auto first=session.attach(firstPath,opts);query.processingByteBudget=RegionOptions{}.processingByteBudget;
    const auto stacked=session.geometryRegion({first,attachment},slice,query);
    require(stacked.size()>=2&&stacked[1].result.intersectingObjects==1&&stacked[1].result.runs[0].objectUuid==b,
            "earlier off-screen owner cannot starve later visible results");
}

void duplicateStatementLease()
{
    sqlite3* db=nullptr; require(sqlite3_open(":memory:",&db)==SQLITE_OK,"open statement lease fixture");
    {
        StatementPool pool; activeStatementPools.emplace(db,&pool);
        { Statement first(db,"SELECT 1"); { Statement second(db,"SELECT 1"); require(second.row(),"overlapping statement runs"); } }
        int count=0;for(auto* statement=sqlite3_next_stmt(db,nullptr);statement;statement=sqlite3_next_stmt(db,statement))++count;
        require(count==2&&pool.idle.size()==2,"multimap retains both duplicate SQL handles");
        activeStatementPools.erase(db);
    }
    require(sqlite3_close(db)==SQLITE_OK,"all leased statements released");
}

void foreignPointIdentity(HistoryMode mode, bool referenced)
{
    TemporaryDirectory dir;const auto path=dir.path/"points",base=referenced?dir.path/"base":path;
    const auto target=util::newUuid(),foreign=util::newUuid();std::string original,source;
    {auto store=ObjectStore::create(base,options(mode));auto w=store.write();for(const auto& id:{target,foreign}){ObjectRecord record;record.uuid=id;record.kind="point_collection";
        store.upsertObject(w,record);const auto revision=store.putGeometry(w,id,vc::geometry::encode(vc::geometry::PointCollection{{42},{{1,2,3}}}));
        if(id==target)original=revision;else source=revision;}if(referenced)store.seal(w);w.commit();}
    if(referenced){auto collection=ObjectCollection::create(path,{base},options(mode));}
    auto store=ObjectStore::open(path,options(mode));
    const auto checked=[&]{auto r=store.read();return store.writeNumbered({{target,store.currentVersion(r,target)},{foreign,store.currentVersion(r,foreign)}});};
    std::string empty;{auto w=checked();GeometrySequenceEdit edit;edit.array="points";edit.eraseCount=1;
        empty=store.editGeometrySequence(w,target,edit,original);w.commit();}
    {auto w=checked();GeometrySequenceEdit edit;edit.array="points";GeometrySequencePart part;part.revision=source;part.count=1;edit.parts={part};
        mustFail([&]{store.editGeometrySequence(w,target,edit,empty);});mustFail([&]{w.commit();});}
    {auto r=store.read();require(store.currentGeometry(r,target)->uuid==empty,"foreign identity rejection is atomic");}
    if(mode==HistoryMode::On||referenced){auto w=checked();GeometrySequenceEdit edit;edit.array="points";
        GeometrySequencePart part;part.revision=original;part.count=1;edit.parts={part};store.editGeometrySequence(w,target,edit,empty);w.commit();}
}

void unifiedExpectations(HistoryMode mode)
{
    Fixture f(mode, mode);
    auto store = ObjectStore::open(f.overlay, options(mode));
    auto request = [&] {
        auto r = store.read(); const auto status = store.status(r);
        TransactionRequest q;
        q.requestUuid = util::newUuid(); q.storeUuid = status.storeUuid;
        q.generation = status.generation; q.author = "shared-expectations";
        return q;
    };
    auto q = request(); expectObject(q, f.av); expectRead(q, f.bv);
    RelationshipVersion relationship;
    { auto r = store.read(); relationship = store.relationshipsAt(r, {f.av, f.bv}).front().relationship.reference(); }
    expectRelationship(q, relationship);
    const auto encoded = transactionRequestToJson(q);
    require(nlohmann::json::parse(encoded).at("command_version") == 3, "new dependency guards are version gated");
    const auto decoded = transactionRequestFromJson(encoded, {});
    require(decoded.expectedObjects == q.expectedObjects && decoded.readDependencies == q.readDependencies &&
            decoded.expectedRelationships == q.expectedRelationships, "one handle survives every guard wire path");
    (void)store.submit(q);
    require(store.submit(q).replayed, "complete dependency guards replay through the ordinary saved-receipt path");
    auto wrong = request(); auto other = f.bv; other.storeUuid = util::newUuid(); expectRead(wrong, other);
    mustFail([&] { store.submit(wrong); });
    wrong = request(); auto wrongEdge = relationship; wrongEdge.storeUuid = util::newUuid(); expectRelationship(wrong, wrongEdge);
    mustFail([&] { store.submit(wrong); });
    RecordSelection merged{{f.b, f.bv}};
    mergeExpectations(merged, {{f.b, f.bv}}, "same identity");
    require(merged.at(f.b) == RecordExpectation(f.bv), "merge retains the shared identity");
    mustFail([&] { mergeExpectations(merged, {{f.b, other}}, "different owner"); });
    auto unsupported = nlohmann::json::parse(encoded);
    for (int version : {1, 2}) {
        unsupported["command_version"] = version;
        mustFail([&] { (void)transactionRequestFromJson(unsupported.dump(), {}); });
        mustFail([&] { (void)store.submit(unsupported.dump(), {}); });
    }
    unsupported = nlohmann::json::parse(encoded);
    unsupported["expected_object_references"] = nlohmann::json::array();
    mustFail([&] { (void)transactionRequestFromJson(unsupported.dump(), {}); });
    unsupported = nlohmann::json::parse(encoded);
    unsupported["expected_objects"][f.a] = f.av.version;
    mustFail([&] { (void)transactionRequestFromJson(unsupported.dump(), {}); });
    mustFail([&] { (void)transactionJsonPayloadHashes(unsupported.dump()); });
    mustFail([&] { (void)transactionJsonHasPreparedOperations(unsupported.dump()); });
    mustFail([&] { (void)store.submit(unsupported.dump(), {}); });
    auto withPayload = q;
    withPayload.operations.push_back(EditGeometryOp{f.a, {{"points", 0, {1}}}, {}});
    unsupported = nlohmann::json::parse(transactionRequestToJson(withPayload));
    unsupported["expected_objects"][f.a] = f.av.version;
    bool lookedUpPayload = false;
    mustFail([&] { (void)store.submit(unsupported.dump(), [&](const auto&) -> std::optional<std::vector<uint8_t>> {
        lookedUpPayload = true; return std::nullopt;
    }); });
    require(!lookedUpPayload, "invalid guards fail before payload lookup or saved-receipt recovery");
    auto upper = f.a;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    auto duplicate = nlohmann::json::parse(encoded);
    duplicate["expected_objects"][upper] = duplicate["expected_objects"][f.a];
    mustFail([&] { (void)transactionRequestFromJson(duplicate.dump(), {}); });
    { auto r = store.read(); LocalStoreReader reader(store, r);
        mustFail([&] { (void)selectCurrentExpectations(reader, {{upper, f.av.version}, {f.a, std::nullopt}}); }); }
    const auto id = util::newUuid();
    { auto w = store.write(); ObjectRecord record; record.uuid = id; record.kind = "note"; store.upsertObject(w, record); w.commit(); }
    { auto w = store.write(); store.deleteObject(w, id); w.commit(); }
    ObjectRevision deleted;
    { auto r = store.read(); deleted = *store.currentObjectRevision(r, id); }
    require(deleted.deleted, "selected tombstone fixture");
    { auto r = store.read(); LocalStoreReader reader(store, r);
        require(!reader.currentObject(id) && reader.currentObject(id, true)->reference() == deleted.reference() &&
                currentObjectExpectation(reader, id).deleted, "ordinary reads hide tombstones; explicit guards retain their identity"); }
    auto absent = request(); absent.expectedObjects[id] = std::nullopt; (void)store.submit(absent);
    auto exact = request(); expectObject(exact, deleted);
    exact = transactionRequestFromJson(transactionRequestToJson(exact), {}); (void)store.submit(exact);
    auto live = request(); expectObject(live, deleted.reference()); mustFail([&] { store.submit(live); });
    auto bad = request(); bad.expectedObjects[f.a] = relationship;
    mustFail([&] { (void)transactionRequestToJson(bad); });
    { auto w = store.write(); store.setMeta(w, f.b, "unrelated", "1"); w.commit(); }
    require(store.submit(q).replayed, "saved complete-guard request replays even after its selected inputs change");
    const auto basePath = f.dir.path / "deleted-base", overlayPath = f.dir.path / "deleted-overlay";
    const auto inheritedId = util::newUuid();
    { auto base = ObjectStore::create(basePath, options(mode));
        { auto w = base.write(); ObjectRecord record; record.uuid = inheritedId; record.kind = "note";
            base.upsertObject(w, record); w.commit(); }
        { auto w = base.write(); base.deleteObject(w, inheritedId); base.seal(w); w.commit(); } }
    { auto collection = ObjectCollection::create(overlayPath, {basePath}, options(mode)); }
    auto overlay = ObjectStore::open(overlayPath, options(mode));
    TransactionContext context;
    { auto r = overlay.read(); LocalStoreReader reader(overlay, r);
        context.expectedObjects[inheritedId] = currentObjectExpectation(reader, inheritedId);
        require(context.expectedObjects.at(inheritedId).deleted && !reader.currentObject(inheritedId),
                "inherited tombstone retains the winning owner without a live-object fallback"); }
    { auto w = overlay.write(context); w.commit(); }
}

void fittingRelationshipPolicy()
{
    const std::map<std::string, std::string> kinds{{"s", "surface"}, {"p", "point_collection"},
        {"f", "fiber"}, {"g", "fiber"}, {"t", "track"}, {"u", "track"}};
    EdgeRecord edge; edge.aUuid = "p"; edge.bUuid = "s"; edge.kind = "point-surface"; edge.approvalState = "pending";
    auto decision = fittingRelationshipInput(edge, true, kinds, false);
    require(decision.family == "pcl_patch" && decision.reversed && decision.exclusion.empty(), "shared kind ordering");
    require(fittingRelationshipInput(edge, true, kinds, true).exclusion == "pending (approved-only)", "shared approval gate");
    edge.approvalState = "rejected";
    require(fittingRelationshipInput(edge, true, kinds, false).exclusion == "rejected", "rejected input is never consumed");
    require(fittingRelationshipInput(edge, false, kinds, false).exclusion == "inapplicable", "inapplicability precedes approval");
    edge.aUuid = "missing";
    require(fittingRelationshipInput(edge, false, kinds, false).exclusion == "endpoint outside selection", "selection precedes applicability");
    edge.approvalState = "approved";
    for (const auto& example : std::vector<std::array<std::string, 4>>{
        {"same-surface", "s", "f", "fiber_patch"}, {"same-surface", "s", "s", "patch_overlap"},
        {"fiber_branch", "g", "f", "fiber_branch"}, {"crossing", "t", "u", "track_crossing"}}) {
        edge.kind = example[0]; edge.aUuid = example[1]; edge.bUuid = example[2];
        decision = fittingRelationshipInput(edge, true, kinds, true);
        require(decision.exclusion.empty() && decision.family == example[3], "shared supported families");
    }
    edge.kind = "same-surface";
    require(fittingRelationshipInput(edge, true, kinds, false).exclusion == "unused kind same-surface", "unsupported endpoint pair excluded");
}

int main()try{
    fittingRelationshipPolicy();
    for (const auto mode : {HistoryMode::Off, HistoryMode::On}) unifiedExpectations(mode);
    duplicateStatementLease();
    regionPrecedenceRefill();
    for(const auto mode:{HistoryMode::Off,HistoryMode::On})for(bool referenced:{false,true})foreignPointIdentity(mode,referenced);
    for(const auto mode:{HistoryMode::Off,HistoryMode::On}){atomicCompositeWrites(mode);measuredLengths(mode);}
    followupAtomicityAndPolicy();
    inventoryIdentityAndSources();
    inventorySelectedMembership();inventoryPackedAndCoordinates();inventoryArtifactAlias();
#ifdef VC_TEST_NATIVE_EXPORT
    inventoryMixedOwnerExport();
#endif
    for(const auto base:{HistoryMode::Off,HistoryMode::On})for(const auto overlay:{HistoryMode::Off,HistoryMode::On}){
        firstInheritedDeletion(base,overlay);
        inventoryCore(base,overlay);inventoryRestore(base,overlay);inventoryRevalidatedPeer(base,overlay);
    }
    allocationWithoutCurrentGeometry();
    for(const auto mode:{HistoryMode::Off,HistoryMode::On}){siblingViewWrites(mode);completeReceipts(mode);retainedPointAllocation(mode);}
    for(const auto mode:{HistoryMode::Off,HistoryMode::On})edgeIdentity(mode);
    for(const auto base:{HistoryMode::Off,HistoryMode::On})for(const auto overlay:{HistoryMode::Off,HistoryMode::On}){
        readsAndOverrides(base,overlay);geometryEdit(base,overlay,"retain");
        deletedAndExactReads(base,overlay);
        geometryEdit(base,overlay,"keep_recorded");
    }
    queries();packing();spatialQueries();followupRegressions();restoreFileReference();std::cout<<"coherent store API tests passed\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
