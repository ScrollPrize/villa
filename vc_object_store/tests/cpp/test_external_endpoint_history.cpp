#include "TestSupport.hpp"
#include "UuidStorageTest.hpp"
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
    const auto owner=util::newUuid(),lineage=util::newUuid(),a=util::newUuid(),b=util::newUuid();
    auto install=[&](const Txn& txn){
        auto& pool=*activeStatementPools.at(ObjectStoreImplAccess::conn(txn));
        pool.externalEndpointOwner=[&](const std::string& id)->std::optional<std::string>{
            return id==a||id==b?std::optional(owner):std::nullopt;
        };
        pool.externalEndpointRecord=[&](const std::string& id){
            if(id!=a&&id!=b)throw std::runtime_error("unknown external key");
            ObjectRecord record;record.uuid=id;record.kind="fiber";record.version=3;record.versionGeneration=lineage;
            record.meta=R"({"quality":7})";return record;
        };
        pool.externalEndpointVersion=[&](const std::string& id){return externalEndpointToken(owner,id,lineage,7);};
    };
    std::string edge,oldVersion;
    {
        auto txn=store.write();auto* db=ObjectStoreImplAccess::conn(txn);install(txn);
        Statement aliases(db,"SELECT count(*) FROM sqlite_schema WHERE name IN ('external_object_keys','endpoint_keys')");
        require(aliases.row()&&aliases.i64(0)==0,"external history must not require endpoint registration tables");
        EdgeRecord record;record.kind="perpendicular";record.aUuid=a;record.bUuid=b;
        record.aAnchor.wholeGeometry=record.bAnchor.wholeGeometry=true;
        record.aAnchor.metadataKeys={"quality"};edge=store.addEdge(txn,record);txn.commit();
    }
    {
        auto txn=store.read();install(txn);auto* db=ObjectStoreImplAccess::conn(txn);requireBinaryUuidStorage(store,txn);
        Statement objects(db,"SELECT (SELECT count(*) FROM objects),(SELECT count(*) FROM object_revisions),(SELECT count(*) FROM object_head_records)");objects.row();
        require(objects.i64(0)==0&&objects.i64(1)==0&&objects.i64(2)==0,"external edges never fabricate objects or histories");
        const auto pinned=store.currentObjectRevision(txn,a);require(pinned&&pinned->uuid==externalEndpointToken(owner,a,lineage,7),"external endpoint has sealed version token");
        const auto selected=ObjectStoreImplAccess::relationshipsAt(store,txn,{{a,pinned->uuid},{b,externalEndpointToken(owner,b,lineage,7)}});
        require(selected.size()==1&&selected[0].relationship.record.uuid==edge,"external-only relationship history remains discoverable");
        oldVersion=selected[0].relationship.uuid;
        require(selected[0].relationship.aRevision.starts_with("external:")&&selected[0].relationship.bRevision.starts_with("external:"),"relationship pins external versions");
        require(selected[0].applicable,"external sealed endpoints apply to historical graph");
    }
    {
        auto txn=store.read();mustFail([&]{store.currentObjectRevision(txn,a);});
    }
    {
        auto txn=store.write();install(txn);auto record=*store.edgeByUuid(txn,edge);record.meta=R"({"confidence":0.75})";
        store.updateEdge(txn,record);txn.commit();
    }
    {
        auto txn=store.read();install(txn);const auto history=ObjectStoreImplAccess::relationshipsAt(store,txn,
            {{a,externalEndpointToken(owner,a,lineage,7)},{b,externalEndpointToken(owner,b,lineage,7)}});
        require(history.size()==1&&history[0].relationship.uuid!=oldVersion,"editing local relationship publishes a new relation, not an external object");
        auto& pool=*activeStatementPools.at(ObjectStoreImplAccess::conn(txn));
        pool.externalEndpointVersion=[&](const auto& id){return externalEndpointToken(owner,id,lineage,8);};
        mustFail([&]{ObjectStoreImplAccess::objectRevision(store,txn,externalEndpointToken(owner,a,lineage,7));});
        pool.externalEndpointVersion=[&](const auto& id){return externalEndpointToken(owner,id,util::newUuid(),7);};
        mustFail([&]{ObjectStoreImplAccess::objectRevision(store,txn,externalEndpointToken(owner,a,lineage,7));});
    }
    std::cout<<"external endpoint history tests passed\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
