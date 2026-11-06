#include "TestSupport.hpp"
#include "internal/InitialEdges.hpp"
#include "internal/History.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/SqlValues.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include <iostream>
#include <limits>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
int64_t scalar(sqlite3* db,const std::string& sql) {
    Statement row(db,sql);require(row.row(),"scalar row");return row.i64(0);
}
const std::string a="00000000-0000-4000-8000-00000000000a";
const std::string b="00000000-0000-4000-8000-00000000000b";
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
    Statement s(db,"SELECT e.uuid,e.a_uuid,e.b_uuid,e.kind,e.generation,e.approval_state,e.meta,e.a_dependencies,e.b_dependencies,"
        "e.uuid "
        "FROM edges e "
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
    for(size_t i=0;i<rowCount;++i){auto e=edge(geometry);
        e.kind=i%2?"parallel":"perpendicular";
        e.meta=i%5?nlohmann::json({{"row",i},{"text","different-"+std::to_string(i)}}).dump():"";
        e.tags={"tag-"+std::to_string(i),"","tag-"+std::to_string(i)};
        if(i%3==1){e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;
            e.aAnchor.points.clear();e.bAnchor.points.clear();e.aAnchor.metadataKeys.clear();e.bAnchor.metadataKeys.clear();}
        if(!geometry&&i%3==2){e.aAnchor={};e.bAnchor={};}
        if(i%4==2){e.aAnchor.position.reset();e.bAnchor.direction.reset();}
        input.push_back(std::move(e));}
    input[1].aAnchor.metadataKeys.clear();input[1].bAnchor.metadataKeys.clear();
    for(auto* store:{&ordinary,&bulk}) {
        auto txn=store->write();auto* db=ObjectStoreImplAccess::conn(txn);objects(*store,txn,geometry);
        if(store==&ordinary) for(const auto& e:input) store->addEdge(txn,e);
        else {
            exec(db,"CREATE INDEX custom_edge_index ON edges(generation)");
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
    bad=edge(true);bad.uuid=a;run({good,bad});
    bad=edge(true);bad.aUuid=util::newUuid();run({good,bad});
    bad=edge(true);bad.meta="[]";run({good,bad});
    bad=edge(true);bad.uuid="invalid";run({good,bad});
    bad=edge(true);bad.aAnchor.position=std::array<double,3>{0,std::numeric_limits<double>::infinity(),0};run({good,bad});
    bad=edge(true);bad.aAnchor.points={{"line_points",{999}}};run({good,bad});
    bad=edge(true);bad.aUuid=bad.bUuid;bad.aAnchor.points=bad.bAnchor.points;run({good,bad});
    std::vector<EdgeRecord> across;
    for(int i=0;i<513;++i){auto e=edge(true);across.push_back(std::move(e));}
    across.back()=across.front();run(across); // failure beyond two full UUID-check chunks
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
    auto second=edge();initial.append(std::span(&second,1));initial.finish();whole.release();
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

void sharedStatementBindings()
{
    TemporaryDirectory dir;OpenOptions options;options.author="shared-bindings";options.history=HistoryMode::Off;
    auto store=ObjectStore::create(dir.path/"shared.vcobj",options);auto txn=store.write();auto* db=ObjectStoreImplAccess::conn(txn);
    exec(db,"CREATE TEMP TABLE shared_values(label TEXT NOT NULL,epoch INTEGER NOT NULL,id INTEGER PRIMARY KEY,payload TEXT CHECK(payload!='reject'))");
    int64_t next=1;
    auto write=[&](size_t count,size_t requested,int variableLimit,const std::string& label,int64_t epoch,bool reject=false){
        const auto previousLimit=sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,variableLimit);
        struct RestoreLimit {sqlite3* db;int value;~RestoreLimit(){sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,value);}} restore{db,previousLimit};
        std::vector<std::optional<std::string>> values;values.reserve(count);
        for(size_t i=0;i<count;++i)values.push_back(i%3?std::optional("value-"+std::to_string(epoch)+"-"+std::to_string(i)):std::nullopt);
        if(reject)values.back()="reject";
        const auto first=next;size_t fixedCalls=0;
        writeSqlValuesShared(db,"INSERT INTO shared_values(label,epoch,id,payload)",2,2,count,requested,
            [](size_t offset){return "(?1,?2,?"+std::to_string(offset+1)+",?"+std::to_string(offset+2)+")";},
            [&](Statement& s){++fixedCalls;s.bindBorrowedText(1,label);s.bind(2,epoch);},
            [&](Statement& s,size_t index,int offset){s.bind(offset+1,first+int64_t(index));
                if(values[index])s.bindBorrowedText(offset+2,*values[index]);else s.bindNull(offset+2);});
        const auto capacity=std::min(requested,size_t(variableLimit-2)/2);
        const auto expectedFixed=count==0?0:(count>capacity&&count%capacity!=0?2:1);
        require(fixedCalls==expectedFixed,"shared bindings must be set once per SQL shape, not once per chunk");
        next+=int64_t(count);
        for(size_t i=0;i<count;++i){Statement row(db,"SELECT label,epoch,payload FROM shared_values WHERE id=?",first+int64_t(i));
            require(row.row()&&row.text(0)==label&&row.i64(1)==epoch&&row.optText(2)==values[i],"shared/nullable bindings leaked between chunks");}
    };
    write(31,32,1000,"first",1);write(32,32,1000,"second",2);write(65,32,1000,std::string("third\0label",11),3);
    write(7,32,8,"low-variable-limit",4);
    const auto before=scalar(db,"SELECT count(*) FROM shared_values");
    {Savepoint rollback(db);mustFail([&]{write(7,32,8,"failed",5,true);});}
    require(scalar(db,"SELECT count(*) FROM shared_values")==before,"failed shared batch rolls back all earlier chunks");
    write(7,32,8,"after-failure",6);
    std::string borrowed="temporary borrowed value";
    {Statement row(db,"SELECT ?1");row.bindBorrowedText(1,borrowed);require(row.row()&&row.text(0)==borrowed,"borrowed text read");}
    borrowed.assign(10000,'x');
    {Statement row(db,"SELECT ?1");require(row.row()&&row.isNull(0),"pooled statement retained borrowed binding beyond its scope");}
    txn.commit();
}

void resolvedMutableRows()
{
    TemporaryDirectory dir;OpenOptions options;options.author="resolved";options.history=HistoryMode::Off;
    auto db=ObjectStore::create(dir.path/"resolved.vcobj",options);
    std::vector<EdgeRecord> originals;
    {
        auto txn=db.write();auto* sql=ObjectStoreImplAccess::conn(txn);
        // Deliberately reverse numeric ID order relative to lexical UUID order.
        for(const auto& id:{b,a}){ObjectRecord object;object.uuid=id;object.kind="note";db.upsertObject(txn,object);}
        std::vector<EdgeRecord> records;
        for(int i=0;i<67;++i){auto record=edge();
            if(i%2)swapEndpoints(record);
            if(i%3==0){for(char& ch:record.aUuid)ch=char(std::toupper(static_cast<unsigned char>(ch)));
                for(char& ch:record.bUuid)ch=char(std::toupper(static_cast<unsigned char>(ch)));}
            record.kind=i%2?"parallel":"perpendicular";
            record.meta=nlohmann::json({{"index",i},{"payload",std::string(size_t(i+1),'x')}}).dump();
            originals.push_back(record);records.push_back(std::move(record));}
        Savepoint save(sql);InitialEdgeInserter writer(db,txn,true,2,32);
        writer.appendResolved(std::span(records).first(33));
        writer.appendResolved(std::span(records).subspan(33));
        writer.finish();save.release();
        // The insert must own bytes already; destroy/replace all borrowed input fields.
        for(auto& record:records){record.kind.assign(2000,'k');record.meta.assign(2000,'m');}
        records.clear();txn.commit();
    }
    auto read=db.read();
    for(auto expected:originals){const auto actual=db.edgeByUuid(read,expected.uuid);require(bool(actual),"resolved edge missing");
        expected.aUuid=util::canonicalUuid(expected.aUuid);expected.bUuid=util::canonicalUuid(expected.bUuid);
        if(expected.aUuid!=a)swapEndpoints(expected);
        require(actual->aUuid==a&&actual->bUuid==b&&actual->aAnchor==expected.aAnchor&&actual->bAnchor==expected.bAnchor&&
                actual->kind==expected.kind&&actual->meta==expected.meta&&actual->tags==std::vector<std::string>{"tag"},
                "mutable resolved batch lost UUID endpoint orientation or borrowed content");
        require(actual->createdBy=="resolved"&&actual->changedBy=="resolved"&&actual->createdAt==actual->changedAt,
                "resolved shared provenance mismatch");}
}
}
void immutableNamespaceCollisions() {
    TemporaryDirectory dir;OpenOptions options;options.author="test";options.history=HistoryMode::Off;
    const auto basePath=dir.path/"base.vcobj";ObjectKey external;
    {
        auto base=ObjectStore::create(basePath,options);auto tx=base.write();
        ObjectRecord object;object.uuid=util::newUuid();object.kind="note";
        external={base.uuid(tx),base.upsertObject(tx,object)};base.seal(tx);tx.commit();
    }
    for(bool packed:{false,true})for(bool foreign:{false,true}) {
        InitialImportOptions opt;opt.store.author="test";opt.dependencies={basePath};
        if(packed){opt.packed.emplace();opt.packed->memoryFirst=true;opt.packed->disposableBulk=true;
            opt.packed->backgroundFlush=false;opt.packed->residentBytes=64*1024*1024;}
        const auto output=dir.path/(std::string("failed-")+(packed?"packed-":"plain-")+(foreign?"external":"owned"));
        bool exhausted=false,caught=false;
        mustFail([&]{importInitialCollection(output,opt,[&](size_t,InitialShardImport& part){
            std::vector<InitialObjectRecord> rows(2);
            rows[0].record.uuid=a;rows[1].record.uuid=b;
            for(auto& row:rows)row.record.kind="note";
            part.appendObjects(rows);
            bool sent=false;
            try {part.importEdges([&](auto& batch){
                if(sent){exhausted=true;return false;}sent=true;
                EdgeRecord record;record.kind="perpendicular";
                record.uuid=a;
                batch.push_back({{part.storeUuid(),a},foreign?external:ObjectKey{part.storeUuid(),b},record});
                return true;
            });}catch(const std::exception& error){
                caught=true;require(std::string(error.what()).find("UUID already belongs")!=std::string::npos,
                    "namespace collision must report the actual validation failure");
            }
        });});
        require(exhausted&&caught,"private UUID collision checked at finish after source exhaustion");
        require(!std::filesystem::exists(output),"caught finish failure must still prevent publication");
        for(const auto& entry:std::filesystem::directory_iterator(dir.path))
            require(entry.path().filename().string().find(output.filename().string())!=0,
                "failed private import staging must be cleaned");
    }
}
int main() {try {
    const std::string mixed="AbCdEFab-1234-ABcd-5678-90aBcDeF0123";
    require(util::canonicalUuid(mixed)==formatUuid(parseUuid(mixed)),"single-pass UUID normalization matches binary representation");
    for(size_t i=0;i<mixed.size();++i){auto invalid=mixed;invalid[i]='!';mustFail([&]{util::canonicalUuid(invalid);});}
    mustFail([]{util::canonicalUuid("");});
    parity(false,false);parity(false,true);parity(true,true);
    parity(false,true,64,133);parity(true,true,64,133,99);failures();failures(64);invalidateAndExhaust();rejectRetiredEndpoint();sharedStatementBindings();resolvedMutableRows();immutableNamespaceCollisions();
    std::cout<<"initial edge tests passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
