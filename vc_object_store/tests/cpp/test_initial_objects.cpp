#include "TestSupport.hpp"
#include "internal/InitialObjects.hpp"
#include "internal/RecordSql.hpp"
#include "internal/RecordJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
int64_t scalar(sqlite3* db,const std::string& sql){Statement s(db,sql);require(s.row(),"scalar row");return s.i64(0);}
ObjectRecord record(std::string kind="note") {ObjectRecord r;r.uuid=util::newUuid();r.kind=std::move(kind);r.name="native";r.tags={"b","a","b",""};return r;}
nlohmann::json comparable(const ObjectRecord& r){auto j=objectToJson(r);for(const auto* k:{"created_at","changed_at"})j.erase(k);return j;}
void parity(bool spatial){
    TemporaryDirectory dir;OpenOptions o;o.author="test";o.history=HistoryMode::Off;o.spatialIndex=spatial;
    auto normal=ObjectStore::create(dir.path/"normal",o),initial=ObjectStore::create(dir.path/"initial",o);
    std::vector<InitialObjectRecord> records;
    auto fiber=record("fiber");fiber.fiber=FiberDetail{};fiber.fiber->family="H";fiber.fiber->length=50;
    fiber.fiber->bbox=std::array<double,6>{0,1,2,3,4,5};records.push_back({fiber,20,{}});
    auto track=record("track");track.track=TrackDetail{};track.track->datasetRef="data";track.track->externalKey="key";
    track.meta=R"({"coordinates":{"volume_uuid":"volume","scale_factor":2}})";records.push_back({track,{},{}});
    auto surface=record("surface");surface.surface=SurfaceDetail{};surface.surface->scale={2,3};surface.surface->area=12;
    records.push_back({surface,5,{}});records.push_back({record(),{}, {}});
    {auto t=normal.write();for(const auto& value:records){if(value.id)normal.insertObjectWithId(t,value.record,*value.id);else normal.upsertObject(t,value.record);}t.commit();}
    {auto t=initial.write();auto* db=ObjectStoreImplAccess::conn(t);Savepoint all(db);
        const auto ids=insertInitialObjects(initial,t,records);require(ids.size()==4,"identity count");
        require(ids[0].second==20&&ids[1].second==21&&ids[2].second==5&&ids[3].second==22,"mixed explicit/automatic IDs preserve input order");
        all.release();t.commit();}
    auto a=normal.read(),b=initial.read();
    for(const auto& value:records)require(comparable(*normal.objectByUuid(a,value.record.uuid))==comparable(*initial.objectByUuid(b,value.record.uuid)),"initial native record parity");
    if(spatial)require(scalar(ObjectStoreImplAccess::conn(a),"SELECT count(*) FROM object_bounds")==scalar(ObjectStoreImplAccess::conn(b),"SELECT count(*) FROM object_bounds"),"bounds parity");
}
void geometry(){
    TemporaryDirectory dir;OpenOptions o;o.author="test";o.history=HistoryMode::Off;o.spatialIndex=false;
    auto normal=ObjectStore::create(dir.path/"normal",o),initial=ObjectStore::create(dir.path/"initial",o);
    auto fiber=record("fiber");fiber.fiber=FiberDetail{};fiber.fiber->family="V";
    fiber.meta=R"({"source":{"line_points_attributes":[{"weight":1},{"weight":2}]}})";
    vc::geometry::Fiber data;data.linePoints={{1,2,3},{4,5,6}};const auto encoded=vc::geometry::encode(data);
    {auto t=normal.write();normal.upsertObject(t,fiber);normal.putGeometry(t,fiber.uuid,encoded);t.commit();}
    {auto t=initial.write();Savepoint all(ObjectStoreImplAccess::conn(t));InitialObjectRecord row{fiber,{},encoded};
        insertInitialObjects(initial,t,std::span(&row,1));all.release();t.commit();}
    auto a=normal.read(),b=initial.read();const auto ga=normal.currentGeometry(a,fiber.uuid),gb=initial.currentGeometry(b,fiber.uuid);
    require(ga&&gb,"both geometry heads present");
    for(const auto& array:encoded.arrays)require(normal.readGeometryArray(a,ga->uuid,array.name)==initial.readGeometryArray(b,gb->uuid,array.name),"exact geometry bytes");
    const auto ra=*normal.objectByUuid(a,fiber.uuid),rb=*initial.objectByUuid(b,fiber.uuid);
    require(ra.storageKind==rb.storageKind&&ra.dataChecksum==rb.dataChecksum&&ra.dataGeneration==rb.dataGeneration,"managed geometry identity and generation");
    require(nlohmann::json::parse(ra.meta)==nlohmann::json::parse(rb.meta),"source annotation ownership and coordinates preserved");
    require(initial.readGeometryArray(b,gb->uuid,"control_points").empty(),"no invented control points");
}
void rollbackAndLifecycle(){
    TemporaryDirectory dir;OpenOptions o;o.author="test";o.history=HistoryMode::Off;o.spatialIndex=false;o.objectIdRange={10,30};
    auto store=ObjectStore::create(dir.path/"objects",o);auto a=record(),b=record();std::string relation;
    {auto t=store.write();store.upsertObject(t,a);store.upsertObject(t,b);EdgeRecord e;e.kind="test";e.aUuid=a.uuid;e.bUuid=b.uuid;relation=store.addEdge(t,e);t.commit();}
    auto fails=[&](std::vector<InitialObjectRecord> rows){auto t=store.write();auto* db=ObjectStoreImplAccess::conn(t);
        const auto next=scalar(db,"SELECT next_id FROM key_allocators WHERE name='objects'");
        {Savepoint all(db);mustFail([&]{insertInitialObjects(store,t,rows);});}
        require(scalar(db,"SELECT count(*) FROM objects")==2,"failed stream inserts no records");
        require(scalar(db,"SELECT count(*) FROM geometry_chunks")==0,"failed initial geometry leaves no chunks");
        require(scalar(db,"SELECT next_id FROM key_allocators WHERE name='objects'")==next,"allocator rollback");};
    fails({{record(),{},{}},{a,{},{}}});fails({{record(),10,{}}});fails({{record(),31,{}}});
    auto bad=record();bad.uuid=relation;fails({{bad,{},{}}});bad=record();bad.meta="[]";fails({{bad,{},{}}});
    bad=record();bad.storageKind="sqlite";fails({{bad,{},{}}});
    auto duplicate=record();fails({{duplicate,{},{}},{duplicate,{},{}}});
    vc::geometry::Fiber data;data.linePoints={{1,2,3},{4,5,6}};
    auto fiber=record("fiber");fiber.meta=R"({"source":{"line_points_attributes":[{"label":"x"},{}]}})";
    fails({{fiber,{},vc::geometry::encode(data)},{a,{},{}}});
    {auto t=store.write();Savepoint all(ObjectStoreImplAccess::conn(t));auto retired=record();retired.retired=true;retired.lifecycleEpoch="epoch";
        InitialObjectRecord input{retired,30,{}};insertInitialObjects(store,t,std::span(&input,1));
        auto read=store.objectByUuid(t,retired.uuid);require(read&&read->retired&&read->lifecycleEpoch=="epoch","proxy lifecycle preserved with existing local edges");
        input={record(),20,{}};insertInitialObjects(store,t,std::span(&input,1)); // hole after exhausted allocator
        require(scalar(ObjectStoreImplAccess::conn(t),"SELECT next_id IS NULL FROM key_allocators WHERE name='objects'")==1,"explicit hole cannot reopen exhausted allocator");
        all.release();t.commit();}
    o.history=HistoryMode::On;auto history=ObjectStore::create(dir.path/"history",o);auto t=history.write();
    mustFail([&]{insertInitialObjects(history,t,{});});
}
}
int main(){try{parity(false);parity(true);geometry();rollbackAndLifecycle();std::cout<<"initial objects passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
