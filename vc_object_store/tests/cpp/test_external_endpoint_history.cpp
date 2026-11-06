#include "TestSupport.hpp"
#include "internal/ExternalEndpoint.hpp"
#include "internal/History.hpp"
#include "internal/RecordSql.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;

int main() try {
    TemporaryDirectory directory;
    OpenOptions options;options.author="external-history";
    auto store=ObjectStore::create(directory.path/"store",options);
    const auto owner=util::newUuid(),a=util::newUuid(),b=util::newUuid(),originalA=util::newUuid(),originalB=util::newUuid();
    auto install=[&](const Txn& txn){
        auto& pool=*activeStatementPools.at(ObjectStoreImplAccess::conn(txn));
        pool.externalEndpointRecord=[&](const std::string& id){
            if(id!=a&&id!=b)throw std::runtime_error("unknown external key");
            ObjectRecord record;record.uuid=id;record.kind="fiber";record.generation=3;
            record.meta=R"({"quality":7})";return record;
        };
        pool.externalEndpointVersion=[&](const std::string& id){return externalEndpointToken(owner,id==a?originalA:originalB,7);};
    };
    std::string edge,oldVersion;
    {
        auto txn=store.write();auto* db=ObjectStoreImplAccess::conn(txn);install(txn);
        execute(db,"INSERT INTO external_object_keys(id,uuid,owner_uuid,object_uuid) VALUES(?,?,?,?)",int64_t(-1),parseUuid(a),parseUuid(owner),parseUuid(originalA));
        execute(db,"INSERT INTO external_object_keys(id,uuid,owner_uuid,object_uuid) VALUES(?,?,?,?)",int64_t(-2),parseUuid(b),parseUuid(owner),parseUuid(originalB));
        mustFail([&]{ObjectRecord collision;collision.uuid=a;collision.kind="fiber";store.upsertObject(txn,collision);});
        EdgeRecord record;record.kind="perpendicular";record.aUuid=a;record.bUuid=b;
        record.aAnchor.wholeGeometry=record.bAnchor.wholeGeometry=true;
        record.aAnchor.metadataKeys={"quality"};edge=store.addEdge(txn,record);txn.commit();
    }
    {
        auto txn=store.read();install(txn);auto* db=ObjectStoreImplAccess::conn(txn);
        Statement objects(db,"SELECT (SELECT count(*) FROM objects),(SELECT count(*) FROM object_revisions),(SELECT count(*) FROM object_head_records)");objects.row();
        require(objects.i64(0)==0&&objects.i64(1)==0&&objects.i64(2)==0,"external edges never fabricate objects or histories");
        const auto pinned=store.currentObjectRevision(txn,a);require(pinned&&pinned->uuid==externalEndpointToken(owner,originalA,7),"external endpoint has sealed version token");
        const auto relationships=store.recordedRelationships(txn,pinned->uuid);
        require(relationships.size()==1&&relationships[0].record.uuid==edge,"external-only relationship history remains discoverable");
        oldVersion=relationships[0].uuid;
        require(relationships[0].aRevision.starts_with("external:")&&relationships[0].bRevision.starts_with("external:"),"relationship pins external versions");
        const auto selected=store.relationshipsAt(txn,{{a,pinned->uuid},{b,externalEndpointToken(owner,originalB,7)}});
        require(selected.size()==1&&selected[0].applicable,"external sealed endpoints apply to historical graph");
    }
    {
        auto txn=store.read();mustFail([&]{store.currentObjectRevision(txn,a);});
    }
    {
        auto txn=store.write();install(txn);auto record=*store.edgeByUuid(txn,edge);record.meta=R"({"confidence":0.75})";
        store.updateEdge(txn,record);txn.commit();
    }
    {
        auto txn=store.read();install(txn);const auto history=store.recordedRelationships(txn,externalEndpointToken(owner,originalA,7));
        require(history.size()==1&&history[0].uuid!=oldVersion,"editing local relationship publishes a new relation, not an external object");
        auto& pool=*activeStatementPools.at(ObjectStoreImplAccess::conn(txn));
        pool.externalEndpointVersion=[&](const auto& id){return externalEndpointToken(owner,id==a?originalA:originalB,8);};
        mustFail([&]{store.objectRevision(txn,externalEndpointToken(owner,originalA,7));});
    }
    std::cout<<"external endpoint history tests passed\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
