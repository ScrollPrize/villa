#include "TestSupport.hpp"
#include "UuidStorageTest.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/CollectionReads.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <iostream>
using namespace vc::objects;
using namespace vc::objects::detail;

void check(HistoryMode history) {
    TemporaryDirectory directory;
    OpenOptions options;options.author="reference-history";options.history=history;
    const auto basePath=directory.path/"base",overlayPath=directory.path/"overlay";
    const auto a=util::newUuid(),b=util::newUuid();
    ObjectVersion aVersion,bVersion;
    {
        auto base=ObjectStore::create(basePath,options);auto w=base.write();
        for(const auto& id:{a,b}){ObjectRecord r;r.uuid=id;r.kind="fiber";r.meta=R"({"quality":7})";base.upsertObject(w,r);}
        base.seal(w);w.commit();auto r=base.read();
        aVersion=base.versionReference(r,*base.currentObjectRevision(r,a));
        bVersion=base.versionReference(r,*base.currentObjectRevision(r,b));
    }
    options.history=HistoryMode::On;
    std::string edge;
    {
        auto overlay=ObjectCollection::create(overlayPath,{basePath},options);auto w=overlay.write();
        EdgeRecord record;record.kind="perpendicular";record.aAnchor.metadataKeys={"quality"};
        edge=overlay.addEdge(w,{{},a},{{},b},record).objectUuid;w.commit();
    }
    options.physicalPartitionOnly=true;
    auto store=ObjectStore::open(overlayPath,options);
    for(int iteration=0;iteration<2;++iteration){
        auto txn=store.read();auto* db=ObjectStoreImplAccess::conn(txn);
        auto scope=attachCollectionReferenceResolver(db,overlayPath);
        require(store.currentVersion(txn,a)==aVersion.version&&store.currentVersion(txn,b)==bVersion.version,
            "current version reads use the same visible-object resolver");
        requireBinaryUuidStorage(store,txn);
        Statement count(db,"SELECT (SELECT count(*) FROM objects),(SELECT count(*) FROM object_revisions)");count.row();
        require(count.i64(0)==0&&count.i64(1)==0,"relations do not fabricate endpoint objects or history");
        const auto selected=store.relationshipsAt(txn,std::vector<ObjectVersion>{aVersion,bVersion});
        require(selected.size()==1&&selected[0].relationship.record.uuid==edge&&selected[0].applicable,"normal exact-version selection resolves pinned endpoints");
        require(bool(resolveObjectRevision(db,selected[0].relationship.aRevision)),"endpoint references resolve through the normal exact-version reader");
        auto invalid=aVersion;++invalid.version;
        require(!store.objectAtVersion(txn,invalid),"unavailable versions cannot become current versions");
        invalid=aVersion;invalid.generation=util::newUuid();
        require(!store.objectAtVersion(txn,invalid),"lineage is checked");
    }
}
int main()try{check(HistoryMode::Off);check(HistoryMode::On);std::cout<<"referenced endpoint history tests passed\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
