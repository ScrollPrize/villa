#include "TestSupport.hpp"
#include "internal/InitialEdges.hpp"
#include "internal/History.hpp"
#include "internal/EdgeSql.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <iostream>
#include <limits>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
int64_t scalar(sqlite3* db,const std::string& sql) {
    Statement row(db,sql);require(row.row(),"scalar row");return row.i64(0);
}
const std::string a="00000000-0000-4000-8000-000000000001";
const std::string b="00000000-0000-4000-8000-000000000002";
void objects(ObjectStore& store,WriteTxn& txn,bool geometry=false) {
    for(const auto& uuid:{a,b}) {
        ObjectRecord obj;obj.uuid=uuid;obj.kind=geometry?"fiber":"note";
        obj.meta=uuid==a?R"({"a":3,"null":null})":R"({"b":4})";
        store.upsertObject(txn,obj);
        if(geometry) {vc::geometry::Fiber fiber;fiber.linePoints={{0,0,0},{1,2,3}};
            store.putGeometry(txn,uuid,vc::geometry::encode(fiber));}
    }
}
EdgeRecord edge(bool points=false) {
    EdgeRecord e;e.uuid=util::newUuid();e.kind="perpendicular";e.aUuid=b;e.bUuid=a;
    e.aAnchor.metadataKeys={"b"};e.bAnchor.metadataKeys={"a","null","missing"};
    e.aAnchor.position=std::array<double,3>{1,2,3};e.bAnchor.direction=std::array<double,3>{4,5,6};
    e.meta=R"({"distance":1.5})";e.tags={"tag","","tag"};
    if(points) {e.aAnchor.points={{"line_points",{0}}};e.bAnchor.points={{"line_points",{1}}};}
    return e;
}
nlohmann::json snapshot(sqlite3* db) {
    nlohmann::json rows=nlohmann::json::array();
    Statement s(db,"SELECT e.uuid,a.uuid,b.uuid,e.kind,e.generation,e.approval_state,e.meta,e.a_dependencies,e.b_dependencies,"
        "e.relation_key "
        "FROM edges e JOIN objects a ON a.id=e.a_id JOIN objects b ON b.id=e.b_id "
        "ORDER BY e.id");
    while(s.row()) {nlohmann::json row=nlohmann::json::array();
        for(int i=0;i<10;++i) {
            if(i<3) row.push_back(formatUuid(s.blob16(i)));
            else if(i==4) row.push_back(s.i64(i));
            else row.push_back(s.text(i));
        } rows.push_back(row);}
    return rows;
}
void parity(bool geometry,bool defer,size_t sqlRows=1,size_t rowCount=2,int variables=0) {
    TemporaryDirectory dir;OpenOptions opts;opts.author="test";opts.history=HistoryMode::Off;
    auto ordinary=ObjectStore::create(dir.path/"ordinary.vcobj",opts),bulk=ObjectStore::create(dir.path/"bulk.vcobj",opts);
    std::vector<EdgeRecord> input;
    for(size_t i=0;i<rowCount;++i){auto e=edge(geometry);e.relationKey=std::to_string(i);input.push_back(std::move(e));}
    input[1].aAnchor.metadataKeys.clear();input[1].bAnchor.metadataKeys.clear();
    for(auto* store:{&ordinary,&bulk}) {
        auto txn=store->write();auto* db=ObjectStoreImplAccess::conn(txn);objects(*store,txn,geometry);
        if(store==&ordinary) for(const auto& e:input) store->addEdge(txn,e);
        else {
            exec(db,"CREATE INDEX custom_edge_index ON edges(relation_key)");
            if(variables)sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,variables);
            Savepoint whole(db);InitialEdgeInserter initial(*store,txn,defer,1,sqlRows);
            const auto first=rowCount>2?size_t(65):size_t(1);
            initial.append(std::span(input).first(first));initial.append(std::span(input).subspan(first));initial.finish();whole.release();
            require(scalar(db,"SELECT count(*) FROM sqlite_schema WHERE name='edges_a'")==1,"restored index");
            require(scalar(db,"SELECT count(*) FROM sqlite_schema WHERE name='custom_edge_index'")==1,"extension index retained");
            mustFail([&]{initial.append({});});
        }
        require(scalar(db,"SELECT count(*) FROM edge_tags")==int64_t(rowCount),"deduplicated tags");txn.commit();
    }
    auto x=ordinary.read(),y=bulk.read();
    require(snapshot(ObjectStoreImplAccess::conn(x))==snapshot(ObjectStoreImplAccess::conn(y)),"native/bulk current records identical");
    for(const auto& e:input) {
        const auto expected=ordinary.edgeByUuid(x,e.uuid),actual=bulk.edgeByUuid(y,e.uuid);
        require(expected&&actual&&expected->aUuid==actual->aUuid&&expected->bUuid==actual->bUuid&&
            expected->aAnchor==actual->aAnchor&&expected->bAnchor==actual->bAnchor&&expected->tags==actual->tags,
            "anchor values, endpoint order and tags preserved");
        require(actual->createdBy=="test"&&actual->changedBy=="test"&&
            actual->createdHost==actual->changedHost&&actual->createdAt==actual->changedAt&&
            actual->approvalState=="pending"&&!actual->approvedBy&&!actual->approvedAt&&
            !actual->approvedHost&&!actual->approvedGeneration,
            "shared batch provenance and SQL approval defaults preserved");
        require(currentRelationshipApplicable(ordinary,x,e.uuid)&&currentRelationshipApplicable(bulk,y,e.uuid),
            "ordinary and bulk relationships are valid against current endpoints");
        const auto os=ordinary.edgeGeometryState(x,e.uuid),bs=bulk.edgeGeometryState(y,e.uuid);
        require(!os.stale&&!bs.stale&&bool(os.aRevision)==bool(bs.aRevision)&&bool(os.bRevision)==bool(bs.bRevision),
            "ordinary and bulk geometry state resolves current endpoint descriptors");
    }
    auto tx=bulk.write();auto obj=*bulk.objectByUuid(tx,a);obj.meta=R"({"a":7,"null":null})";bulk.upsertObject(tx,obj);
    mustFail([&]{tx.commit();});
}
void failures(size_t sqlRows=1) {
    TemporaryDirectory dir;OpenOptions opts;opts.author="test";opts.history=HistoryMode::Off;
    auto store=ObjectStore::create(dir.path/"failures.vcobj",opts);
    {auto tx=store.write();objects(store,tx,true);tx.commit();}
    auto run=[&](const std::vector<EdgeRecord>& input) {
        auto tx=store.write();auto* db=ObjectStoreImplAccess::conn(tx);const auto before=scalar(db,"SELECT next_id FROM key_allocators WHERE name='edges'");
        {Savepoint whole(db);InitialEdgeInserter initial(store,tx,true,2,sqlRows);
            mustFail([&]{initial.append(input);});mustFail([&]{initial.finish();});}
        require(scalar(db,"SELECT count(*) FROM edges")==0,"whole stream rolled back");
        require(scalar(db,"SELECT next_id FROM key_allocators WHERE name='edges'")==before,"allocator rolled back");
        require(scalar(db,"SELECT count(*) FROM sqlite_schema WHERE name='edges_a'")==1,"deferred indexes rolled back");tx.commit();
    };
    auto good=edge(true),bad=good;run({good,bad});
    bad.uuid=util::newUuid();run({good,bad}); // native relation identity duplicate
    bad=edge(true);bad.uuid=a;run({good,bad});
    bad=edge(true);bad.aUuid=util::newUuid();run({good,bad});
    bad=edge(true);bad.meta="[]";run({good,bad});
    bad=edge(true);bad.uuid="invalid";run({good,bad});
    bad=edge(true);bad.aAnchor.position=std::array<double,3>{0,std::numeric_limits<double>::infinity(),0};run({good,bad});
    bad=edge(true);bad.aAnchor.points={{"line_points",{999}}};run({good,bad});
    bad=edge(true);bad.aUuid=bad.bUuid;bad.aAnchor.points=bad.bAnchor.points;run({good,bad});
    std::vector<EdgeRecord> across;
    for(int i=0;i<133;++i){auto e=edge(true);e.relationKey=std::to_string(i);across.push_back(std::move(e));}
    across.back()=across.front();run(across); // failure beyond two full SQL chunks
    across.back()=edge(true);across.back().uuid=a;run(across); // collision in a partial check chunk
    {auto tx=store.write();Savepoint whole(ObjectStoreImplAccess::conn(tx));InitialEdgeInserter initial(store,tx);
        initial.append(std::span(&good,1));initial.finish();whole.release();tx.commit();}
    {auto tx=store.write();mustFail([&]{InitialEdgeInserter initial(store,tx);});}
    opts.history=HistoryMode::On;auto historical=ObjectStore::create(dir.path/"history.vcobj",opts);
    auto tx=historical.write();mustFail([&]{InitialEdgeInserter initial(historical,tx);});
}
void invalidateAndExhaust() {
    TemporaryDirectory dir;OpenOptions opts;opts.author="test";opts.history=HistoryMode::Off;
    opts.edgeIdRange={std::numeric_limits<int64_t>::max()-1,std::numeric_limits<int64_t>::max()};
    auto store=ObjectStore::create(dir.path/"range.vcobj",opts);auto tx=store.write();auto* db=ObjectStoreImplAccess::conn(tx);
    objects(store,tx);Savepoint whole(db);InitialEdgeInserter initial(store,tx,false,2);
    auto first=edge();initial.append(std::span(&first,1));
    // Revalidate the first edge after changing its endpoint, then explicitly
    // invalidate the initial writer's immutable endpoint cache.
    auto obj=*store.objectByUuid(tx,a);obj.meta=R"({"a":9,"null":null})";store.upsertObject(tx,obj);
    store.retainRelationship(tx,first.uuid);initial.invalidateEndpoint(a);
    auto second=edge();second.relationKey="second";initial.append(std::span(&second,1));initial.finish();whole.release();
    require(scalar(db,"SELECT count(*) FROM edges WHERE id=9223372036854775807")==1,"maximum ID allocated without overflow");
    require(scalar(db,"SELECT next_id IS NULL FROM key_allocators WHERE name='edges'")==1,"range exhausted");tx.commit();
}
void rejectRetiredEndpoint() {
    TemporaryDirectory dir;OpenOptions opts;opts.author="test";opts.history=HistoryMode::Off;
    auto store=ObjectStore::create(dir.path/"retired.vcobj",opts);auto e=edge();
    e.aAnchor.metadataKeys.clear();e.bAnchor.metadataKeys.clear();
    {auto tx=store.write();objects(store,tx);store.retireObject(tx,a);tx.commit();}
    {auto tx=store.write();
        Savepoint whole(ObjectStoreImplAccess::conn(tx));InitialEdgeInserter initial(store,tx);
        mustFail([&]{initial.append(std::span(&e,1));});}
    {auto tx=store.read();require(!store.edgeByUuid(tx,e.uuid),"failed retired-endpoint import leaves no edge");}
    {auto tx=store.write();store.restoreObject(tx,a);tx.commit();}
    {auto tx=store.read();require(!store.edgeByUuid(tx,e.uuid),"restoration cannot revive rejected relationship");}
}
}
int main() {try {
    const std::string mixed="AbCdEFab-1234-ABcd-5678-90aBcDeF0123";
    require(util::canonicalUuid(mixed)==formatUuid(parseUuid(mixed)),"single-pass UUID normalization matches binary representation");
    for(size_t i=0;i<mixed.size();++i){auto invalid=mixed;invalid[i]='!';mustFail([&]{util::canonicalUuid(invalid);});}
    mustFail([]{util::canonicalUuid("");});
    parity(false,false);parity(false,true);parity(true,true);
    parity(false,true,64,133);parity(true,true,64,133,99);failures();failures(64);invalidateAndExhaust();rejectRetiredEndpoint();
    std::cout<<"initial edge tests passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
