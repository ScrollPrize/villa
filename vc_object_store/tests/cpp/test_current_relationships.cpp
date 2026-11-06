#include "TestSupport.hpp"
#include "internal/RecordSql.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include <nlohmann/json.hpp>
#include <cstring>
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
using Json=nlohmann::json;
namespace {
OpenOptions options(){OpenOptions o;o.author="current-relations";o.history=HistoryMode::Off;o.spatialIndex=false;return o;}
ObjectRecord fiber(){ObjectRecord r;r.kind="fiber";r.fiber.emplace();r.meta=R"({"quality":1})";return r;}
GeometryData shape(){vc::geometry::Fiber f;f.linePoints={{1,2,3},{4,5,6}};return vc::geometry::encode(f);}
EdgeRecord relation(const std::string& a,const std::string& b){
    EdgeRecord e;e.kind="perpendicular";e.aUuid=a;e.bUuid=b;
    e.aAnchor.points=e.bAnchor.points={{"line_points",{0}}};
    e.aAnchor.metadataKeys=e.bAnchor.metadataKeys={"quality"};return e;
}
GeometryArrayEdit edit(int64_t offset,double value){std::vector<uint8_t> bytes(sizeof value);
    std::memcpy(bytes.data(),&value,sizeof value);return {"line_points",offset,std::move(bytes)};}
void noPersistentBindings(ObjectStore& db,const Txn& t){
    Statement sql(ObjectStoreImplAccess::conn(t),"SELECT name FROM sqlite_schema WHERE name IN "
        "('edge_geometry_bindings','edge_lifecycle_bindings','current_edge_metadata')");
    require(!sql.row(),"no persistent current binding tables or compatibility views");
}
struct Fixture {
    ObjectStore db;std::string a,b,e;
    Fixture(const std::filesystem::path& path):db(ObjectStore::create(path,options())){
        auto w=db.write();a=db.upsertObject(w,fiber());b=db.upsertObject(w,fiber());
        db.putGeometry(w,a,shape());db.putGeometry(w,b,shape());e=db.addEdge(w,relation(a,b));w.commit();
    }
    int quality(){auto r=db.read();return Json::parse(db.objectByUuid(r,a)->meta).at("quality");}
    bool hasEdge(){auto r=db.read();return db.edgeByUuid(r,e).has_value();}
};
void metadataAndSavepoints(){
    TemporaryDirectory dir;Fixture f(dir.path/"current");
    {auto r=f.db.read();noPersistentBindings(f.db,r);}
    {auto w=f.db.write();f.db.setMeta(w,f.a,"unrelated","true");w.commit();}
    mustFail([&]{auto w=f.db.write();f.db.setMeta(w,f.a,"quality","2");w.commit();});
    require(f.quality()==1&&f.hasEdge(),"unreviewed metadata edit rolls back with relationship intact");
    {auto w=f.db.write();f.db.setMeta(w,f.a,"quality","2");f.db.retainRelationship(w,f.e);w.commit();}
    require(f.quality()==2&&f.hasEdge(),"explicit metadata revalidation publishes current relationship");
    mustFail([&]{auto w=f.db.write();f.db.setMeta(w,f.a,"quality","3");f.db.retainRelationship(w,f.e);
        f.db.setMeta(w,f.a,"quality","4");w.commit();});
    require(f.quality()==2,"later edit invalidates earlier validation in same transaction");
    {auto w=f.db.write();{Savepoint nested(ObjectStoreImplAccess::conn(w));f.db.setMeta(w,f.a,"quality","3");}
        w.commit();}
    require(f.quality()==2&&f.hasEdge(),"rolled-back nested edit leaves no stale validation requirement");
    mustFail([&]{auto w=f.db.write();{Savepoint nested(ObjectStoreImplAccess::conn(w));
        f.db.setMeta(w,f.a,"quality","3");f.db.retainRelationship(w,f.e);}
        f.db.setMeta(w,f.a,"quality","4");w.commit();});
    require(f.quality()==2,"rolled-back revalidation cannot approve a later edit");
    {auto w=f.db.write();auto record=*f.db.objectByUuid(w,f.a);
        record.meta=R"({"quality":3})";f.db.upsertObject(w,record);
        record.meta=R"({"quality":4})";f.db.upsertObject(w,record);
        f.db.retainRelationship(w,f.e);w.commit();}
    require(f.quality()==4&&f.hasEdge(),"repeated outer upserts preserve one baseline snapshot and final revalidation");
    {auto w=f.db.write();f.db.setMeta(w,f.a,"quality","5");f.db.removeEdge(w,f.e);w.commit();}
    require(f.quality()==5&&!f.hasEdge(),"explicit removal permits anchored metadata change");
    {auto r=f.db.read();noPersistentBindings(f.db,r);}
}
void geometryAndLifecycle(){
    TemporaryDirectory dir;Fixture f(dir.path/"current");std::string before;
    {auto w=f.db.write();mustFail([&]{f.db.deleteObject(w,f.a);});w.commit();}
    {auto r=f.db.read();require(f.db.objectByUuid(r,f.a)&&f.db.edgeByUuid(r,f.e),
        "deletion refuses referenced objects without objects-only endpoint foreign keys");}
    {auto r=f.db.read();before=f.db.currentGeometry(r,f.a)->uuid;}
    mustFail([&]{auto w=f.db.write();f.db.editGeometry(w,f.a,{edit(0,9)},before);w.commit();});
    {auto r=f.db.read();require(f.db.currentGeometry(r,f.a)->uuid==before,"unreviewed geometry rollback");}
    std::string changed;
    {auto w=f.db.write();changed=f.db.editGeometry(w,f.a,{edit(0,9)},before,[&](const auto& affected){
        std::vector<GeometryEdgeEdit> choices;for(const auto& edge:affected){GeometryEdgeEdit c;
            c.edgeUuid=edge.uuid;c.expectedGeneration=edge.generation;c.action="retain";
            c.aRevision=f.db.currentGeometry(w,edge.aUuid)->uuid;c.bRevision=f.db.currentGeometry(w,edge.bUuid)->uuid;
            choices.push_back(c);}return choices;});w.commit();}
    require(f.hasEdge(),"explicit geometry retain leaves current edge");
    {auto w=f.db.write();f.db.editGeometry(w,f.a,{edit(3,12)},changed);w.commit();}
    require(f.hasEdge(),"unanchored point edit does not require review");
    {auto w=f.db.write();{Savepoint nested(ObjectStoreImplAccess::conn(w));f.db.retireObject(w,f.a);}
        w.commit();}
    require(f.hasEdge(),"retirement savepoint rollback restores incident relationship");
    {auto w=f.db.write();f.db.retireObject(w,f.a);w.commit();}
    require(!f.hasEdge(),"retirement transactionally removes current incident edges");
    {auto w=f.db.write();mustFail([&]{f.db.addEdge(w,relation(f.a,f.b));});w.commit();}
    {auto w=f.db.write();f.db.restoreObject(w,f.a);w.commit();}
    require(!f.hasEdge(),"restoration never revives removed relationships");
    {auto w=f.db.write();f.e=f.db.addEdge(w,relation(f.a,f.b));w.commit();}
    require(f.hasEdge(),"explicit relation construction after restoration works");
    {auto r=f.db.read();noPersistentBindings(f.db,r);}
}
void immutableInitial(){
    TemporaryDirectory dir;InitialImportOptions opt;opt.store=options();
    auto result=importInitialCollection(dir.path/"initial",opt,[](size_t,InitialShardImport& shard){
        InitialObjectRecord a{fiber(),{},shape()},b{fiber(),{},shape()};const std::array records{a,b};
        auto ids=shard.appendObjects(records);bool done=false;
        shard.importEdges([&](auto& batch){if(done)return false;done=true;
            batch.push_back({ids[0].key,ids[1].key,relation(ids[0].key.objectUuid,ids[1].key.objectUuid)});return true;});
    });
    auto db=ObjectStore::open(result.path,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=db.read();noPersistentBindings(db,r);
    auto historicalOptions=options();historicalOptions.history=HistoryMode::On;
    auto historical=ObjectStore::create(dir.path/"history",historicalOptions);auto h=historical.read();
    noPersistentBindings(historical,h);
}
void currentGeometryTokens(){
    TemporaryDirectory dir;auto o=options();o.spatialIndex=true;
    auto db=ObjectStore::create(dir.path/"tokens",o);std::string id,token;
    auto x=[](const std::vector<uint8_t>& bytes){double value;std::memcpy(&value,bytes.data(),sizeof value);return value;};
    {auto w=db.write();auto record=fiber();record.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
        id=db.upsertObject(w,record);token=db.putGeometry(w,id,shape());
        require(x(db.readGeometryArray(w,token,"line_points"))==1,"first current payload read");
        const auto second=db.editGeometry(w,id,{edit(0,8)},token);
        require(second==token,"multiple geometry writes share publication token");
        require(x(db.readGeometryArray(w,token,"line_points"))==8,"same-token current read sees latest bytes");
        require(x(db.readGeometryArrays(w,{token,token},"line_points").data)==8,"same-token batched read sees latest bytes");
        db.registerVolumeTransform(w,"a","b",VolumeAffine::scale(2),"test");w.commit();}
    {auto r=db.read();ObjectSelection selected;selected.kind="fiber";
        const auto graph=db.graphReadView(r,selected,{});
        require(graph.geometryRevisions.size()==1&&graph.geometryRevisions[0]==token,"graph preserves current token");
        selected.volume="b";selected.orderBy="length_desc";
        require(db.selectObjects(r,selected).size()==1,"registered length query accepts current token");}
    {auto w=db.write();const auto next=db.editGeometry(w,id,{edit(0,1)},token);
        require(next!=token,"next publication changes geometry token");
        const auto beforeLength=db.geometryLengthInVolume(w,id,"b");
        db.editGeometry(w,id,{edit(0,12)},next);
        require(db.geometryLengthInVolume(w,id,"b")>beforeLength,"same-token edit invalidates transformed length cache");
        w.commit();}
    {auto r=db.read();require(!db.geometryRevision(r,token),"superseded current token is unavailable");}
    std::string sheet,revision;
    {auto w=db.write();ObjectRecord record;record.kind="surface";sheet=db.upsertObject(w,record);
        vc::geometry::Surface surface{2,2,{1,1},{{0,0,10},{1,0,10},{0,1,10},{1,1,10}}};
        revision=db.putGeometry(w,sheet,vc::geometry::encode(surface));w.commit();}
    {auto r=db.read();SurfaceProximityQuery query;query.point={.5,.5,11};query.tolerance=2;
        const auto hits=db.nearestSurfaces(r,query);
        require(hits.size()==1&&hits[0].objectUuid==sheet,"surface proximity accepts current geometry token");}
}
void overlay(HistoryMode history){
    TemporaryDirectory dir;const auto basePath=dir.path/"base";ObjectKey a,b,old;std::string baseGeometry;
    {Fixture f(basePath);auto w=f.db.write();const auto owner=f.db.uuid(w);a={owner,f.a};b={owner,f.b};old={owner,f.e};
        baseGeometry=f.db.currentGeometry(w,f.a)->uuid;f.db.seal(w);w.commit();}
    auto overlayOptions=options();overlayOptions.history=history;
    auto db=ObjectCollection::create(dir.path/"overlay",{basePath},overlayOptions);
    mustFail([&]{auto w=db.write();db.setMeta(w,a,"quality","2");w.commit();});
    {auto r=db.read();require(Json::parse(db.object(r,a)->record.meta).at("quality")==1,"rejected inherited metadata unchanged");}
    ObjectKey edge;
    {auto w=db.write();db.removeEdge(w,old);db.setMeta(w,a,"quality","2");
        edge=db.addEdge(w,a,b,relation(a.objectUuid,b.objectUuid));w.commit();}
    mustFail([&]{auto w=db.write();db.setMeta(w,b,"quality","2");w.commit();});
    {auto r=db.read();require(Json::parse(db.object(r,b)->record.meta).at("quality")==1,
        "first promoted metadata override still requires relationship revalidation");}
    std::string before;{auto r=db.read();before=db.currentGeometry(r,a)->uuid;require(!db.edge(r,old)&&db.edge(r,edge),"inherited edge replaced locally");}
    mustFail([&]{auto w=db.write();db.editGeometry(w,a,{edit(0,9)},before);w.commit();});
    {auto w=db.write();const auto current=*db.edge(w,edge);GeometryEdgeEdit c;c.edgeUuid=edge.objectUuid;
        c.expectedGeneration=current.record.generation;c.action="retain";
        c.aRevision=db.currentGeometry(w,current.a)->uuid;c.bRevision=db.currentGeometry(w,current.b)->uuid;
        db.editGeometry(w,a,{edit(0,9)},before,{c});w.commit();}
    // The second endpoint is still a compact external key. Its promotion
    // reverses native ID ordering; decisions and anchors must follow identity.
    {auto r=db.read();before=db.currentGeometry(r,b)->uuid;}
    mustFail([&]{auto w=db.write();db.editGeometry(w,b,{edit(0,10)},before);w.commit();});
    {auto w=db.write();const auto current=*db.edge(w,edge);GeometryEdgeEdit c;c.edgeUuid=edge.objectUuid;
        c.expectedGeneration=current.record.generation;c.action="retain";
        c.aRevision=db.currentGeometry(w,current.a)->uuid;c.bRevision=db.currentGeometry(w,current.b)->uuid;
        db.editGeometry(w,b,{edit(0,10)},before,{c});
        const auto promoted=*db.edge(w,edge);
        require((promoted.a==a&&promoted.b==b)||(promoted.a==b&&promoted.b==a),"promoted edge retains logical endpoints");
        require(promoted.record.aAnchor.points.size()==1&&promoted.record.bAnchor.points.size()==1,"promotion retains indexed anchors");
        w.commit();}
    {auto base=ObjectStore::open(basePath,OpenOptions{.readOnly=true});auto r=base.read();
        require(Json::parse(base.objectByUuid(r,a.objectUuid)->meta).at("quality")==1,"wrapper leaves readonly metadata unchanged");
        require(base.edgeByUuid(r,old.objectUuid).has_value(),"wrapper leaves readonly relationships unchanged");
        require(base.currentGeometry(r,a.objectUuid)->uuid==baseGeometry,"wrapper leaves readonly geometry unchanged");}
}
}
int main(){try{metadataAndSavepoints();geometryAndLifecycle();immutableInitial();currentGeometryTokens();overlay(HistoryMode::Off);overlay(HistoryMode::On);
    std::cout<<"current-only relationship tests passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
