#include "TestSupport.hpp"
#include "internal/InitialObjects.hpp"
#include "internal/RecordSql.hpp"
#include "geometry/InitialGeometry.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/GeometryReader.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
using Json=nlohmann::json;

int main() try
{
    const std::string emptyDigest="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    require(formatDigest(geometryDigest({}))==emptyDigest,"binary SHA-256 preserves digest bytes");
    require(formatDigest(parseDigest(emptyDigest))==emptyDigest,"digest hex boundary round trips");
    mustFail([&]{parseDigest(std::string(64,'z'));});
    mustFail([&]{parseDigest("00");});
    TemporaryDirectory temp;
    OpenOptions options;options.author="initial-geometry-test";options.history=HistoryMode::Off;options.spatialIndex=false;
    auto scalar=ObjectStore::create(temp.path/"scalar.vcobj",options);
    auto initial=ObjectStore::create(temp.path/"initial.vcobj",options);
    std::vector<InitialObjectRecord> records;
    auto add=[&](const GeometryData& geometry,const Json& meta=Json::object()){
        InitialObjectRecord entry;entry.record.uuid=util::newUuid();entry.record.kind=geometry.kind;
        entry.record.meta=meta.dump();entry.geometry=geometry;
        if(geometry.kind=="fiber")entry.record.fiber.emplace();
        if(geometry.kind=="track")entry.record.track.emplace();
        if(geometry.kind=="surface")entry.record.surface.emplace();
        records.push_back(std::move(entry));
    };
    add(vc::geometry::encode(vc::geometry::Fiber{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6},{7,8,9}},{10,20}}),
        Json{{"label","annotated"},{"source",{{"control_points_attributes",Json::array({Json{{"note",std::string(1500,'x')}},Json{{"note","b"}}})},
            {"line_points_attributes",Json::array({Json::object(),Json{{"note","middle"}},Json::object()})},{"other",42}}}});
    add(vc::geometry::encode(vc::geometry::Surface{2,2,{1,1},{{1,1,2},{3,1,2},{1,4,2},{3,4,2}},{255,0,255,255}}));
    records.back().record.surface->gridHeight=99;records.back().record.surface->gridWidth=99;records.back().record.surface->area=99;
    add(vc::geometry::encode(vc::geometry::Fiber{}));
    add(vc::geometry::encode(vc::geometry::PointCollection{{7,9},{{2,3,4},{5,6,7}}}),
        Json{{"source",{{"point_attributes",Json::array({Json{{"key",1}},Json{{"key",2}}})}}}});
    add(vc::geometry::encode(vc::geometry::Track{{{11,12,13},{14,15,16}}}));
    add(vc::geometry::encode(vc::geometry::Umbilicus{{{31,32,33},{34,35,36}}}));
    {
        auto txn=scalar.write();
        for(const auto& entry:records){scalar.upsertObject(txn,entry.record);scalar.putGeometry(txn,entry.record.uuid,*entry.geometry);}
        txn.commit();
    }
    {
        auto txn=initial.write();auto* db=ObjectStoreImplAccess::conn(txn);Savepoint save(db);
        const auto identities=insertInitialObjects(initial,txn,records);
        require(identities.size()==records.size(),"initial geometry identity count");
        save.release();txn.commit();
    }
    {
        auto a=scalar.read(),b=initial.read();
        for(const auto* txn:{&a,&b}) {
            auto* db=ObjectStoreImplAccess::conn(*txn);
            Statement chunks(db,"SELECT count(*) FROM geometry_chunks WHERE typeof(hash)!='blob' OR length(hash)!=32");
            chunks.row();require(chunks.i64(0)==0,"geometry payload identities must be binary SHA-256");
            Statement descriptors(db,"SELECT count(*) FROM objects WHERE geometry_manifest IS NULL OR length(data_checksum)!=64");
            descriptors.row();require(descriptors.i64(0)==0,"current objects directly hold their geometry descriptors");
            Statement redundant(db,"SELECT count(*) FROM sqlite_schema WHERE name IN ('current_geometry','current_geometry_chunks','current_shared_payloads')");
            redundant.row();require(redundant.i64(0)==0,"no alternate history-off geometry schema");
        }
        for(const auto& entry:records){
            const auto expected=scalar.objectByUuid(a,entry.record.uuid),actual=initial.objectByUuid(b,entry.record.uuid);
            require(expected&&actual,"initial geometry object missing");
            require(Json::parse(expected->meta)==Json::parse(actual->meta),"initial geometry source annotations differ");
            require(expected->storageKind==actual->storageKind&&expected->storageRef==actual->storageRef&&
                expected->version==actual->version&&expected->dataChecksum==actual->dataChecksum,"initial managed fields differ");
            const auto ag=scalar.currentGeometry(a,entry.record.uuid),bg=initial.currentGeometry(b,entry.record.uuid);
            require(ag&&bg&&ag->contentHash==bg->contentHash,"initial manifest content differs");
            GeometryReader ar(a,ag->uuid),br(b,bg->uuid);
            require(ar.manifest()==br.manifest(),"initial shared sequence representation differs");
            for(const auto& array:entry.geometry->arrays)
                require(initial.readGeometryArray(b,bg->uuid,array.name)==array.bytes,"initial payload bytes differ");
            if(entry.record.kind=="surface")require(actual->surface&&expected->surface&&
                actual->surface->gridWidth==expected->surface->gridWidth&&actual->surface->gridHeight==expected->surface->gridHeight&&
                actual->surface->area==expected->surface->area&&actual->surface->area==3,"initial surface detail not derived exactly");
        }
    }
    auto fresh=ObjectStore::create(temp.path/"payload.vcobj",options);
    InitialObjectRecord fiber;fiber.record.uuid=util::newUuid();fiber.record.kind="fiber";fiber.record.fiber.emplace();
    fiber.geometry=vc::geometry::encode(vc::geometry::Fiber{{},{{103,104,105},{203,204,205}}, {}});
    // Identify by name rather than relying on encoder array order.
    const auto& line=*std::find_if(fiber.geometry->arrays.begin(),fiber.geometry->arrays.end(),[](const auto& x){return x.name=="line_points";});
    {
        auto txn=fresh.write();GeometryWork work;
        {GeometryWorkScope profile(work);Savepoint save(ObjectStoreImplAccess::conn(txn));
            insertInitialObjects(fresh,txn,std::span(&fiber,1));save.release();}
        require(work.writtenBytes["line_points"]==line.bytes.size()&&work.readBytes["line_points"]==0,"new payload was redundantly read back");
        auto duplicate=fiber;duplicate.record.uuid=util::newUuid();work={};
        {GeometryWorkScope profile(work);Savepoint save(ObjectStoreImplAccess::conn(txn));
            insertInitialObjects(fresh,txn,std::span(&duplicate,1));save.release();}
        require(work.writtenBytes["line_points"]==0&&work.readBytes["line_points"]==line.bytes.size(),"existing payload skipped collision validation");
        txn.commit();
    }
    {
        auto corrupt=ObjectStore::create(temp.path/"corrupt.vcobj",options);auto txn=corrupt.write();
        auto* db=ObjectStoreImplAccess::conn(txn);
        const auto hash=geometryHash(line.bytes);std::vector<uint8_t> wrong(line.bytes.size(),17);
        mustFail([&]{execute(db,"INSERT INTO geometry_chunks(hash,data) VALUES(?,?)",hash,wrong);});
        execute(db,"INSERT INTO geometry_chunks(hash,data) VALUES(?,?)",parseDigest(hash),wrong);
        mustFail([&]{Savepoint save(db);insertInitialObjects(corrupt,txn,std::span(&fiber,1));save.release();});
        require(!corrupt.objectByUuid(txn,fiber.record.uuid),"colliding payload left a partially imported object");
    }
    std::cout<<"initial geometry tests passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
