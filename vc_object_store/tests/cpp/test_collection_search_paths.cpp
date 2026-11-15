#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "internal/Database.hpp"
#include <fstream>
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace fs=std::filesystem;
namespace {
OpenOptions writable(){OpenOptions o;o.author="search-path-test";o.history=HistoryMode::Off;o.spatialIndex=false;return o;}
std::string base(const fs::path& path){
    auto store=ObjectStore::create(path,writable());auto w=store.write();ObjectRecord o;o.kind="note";o.name="found";
    auto uuid=store.upsertObject(w,o);store.seal(w);w.commit();store.checkpoint();return uuid;
}
void found(const fs::path& path,const OpenOptions& options,const std::string& uuid,const fs::path& dependency){
    auto collection=ObjectCollection::open(path,options);auto r=collection.read();
    const auto object=collection.object(r,{{},uuid});
    require(object&&object->record.name=="found","relocated store_reference object not readable");
    require(collection.members().at(1).locator==dependency,"dependency candidate precedence changed");
}
void plainAndNested(){
    TemporaryDirectory temp;fs::create_directories(temp.path/"source");fs::create_directories(temp.path/"moved");
    const auto original=temp.path/"source"/"base.vcobj",root=temp.path/"overlay.vcobj";
    const auto uuid=base(original);
    {auto collection=ObjectCollection::create(root,{original},writable());}
    const auto duplicate=temp.path/"moved"/"duplicate";fs::copy_file(original,duplicate);
    OpenOptions options;options.readOnly=true;options.storeReferenceFiles={duplicate};
    found(root,options,uuid,original); // Recorded location always wins.
    // Absolute serialized locators get the same relocation behavior as relative ones.
    {auto o=writable();o.physicalPartitionOnly=true;auto raw=ObjectStore::open(root,o);auto w=raw.write();
        execute(ObjectStoreImplAccess::conn(w),"UPDATE collection_members SET locator=?",original.string());w.commit();raw.checkpoint();}
    const auto moved=temp.path/"moved"/"renamed-database.bin";fs::rename(original,moved);
    mustFail([&]{ObjectCollection::open(root,OpenOptions{.readOnly=true});});
    const auto unrelated=temp.path/"moved"/"unrelated";base(unrelated);
    const auto stale=temp.path/"moved"/"stale";fs::copy_file(moved,stale);
    {std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)> sql(openConnection(stale,SQLITE_OPEN_READWRITE,{},false),sqlite3_close_v2);
        exec(sql.get(),"UPDATE store_manifest SET commit_sequence=commit_sequence+1;");}
    options.storeReferenceFiles={unrelated,stale,moved};found(root,options,uuid,moved);
    options.storeReferenceFiles={unrelated,stale};mustFail([&]{ObjectCollection::open(root,options);});
    // Directory discovery is not filename-based and ignores non-store files.
    {std::ofstream noise(temp.path/"moved"/"00-readme.txt");noise<<"not a database";}
    options.storeReferenceFiles.clear();options.storeReferenceDirectories={temp.path/"missing-directory",temp.path/"moved"};
    // The lexically first identity-matching copy is selected deterministically.
    found(root,options,uuid,duplicate);
    auto editing=options;editing.readOnly=false;editing.author="search-path-test";
    {auto collection=ObjectCollection::open(root,editing);auto w=collection.write();collection.seal(w);w.commit();collection.checkpoint();}
    const auto child=temp.path/"child.vcobj";
    {auto collection=ObjectCollection::create(child,{root},editing);}
    const auto movedRoot=temp.path/"moved"/"parent-renamed.vcobj";fs::rename(root,movedRoot);
    options.storeReferenceFiles={movedRoot,moved};options.storeReferenceDirectories.clear();
    auto collection=ObjectCollection::open(child,options);auto r=collection.read();
    require(collection.object(r,{{},uuid}).has_value(),"search options were not propagated down store_reference chains");
}
void shardedAlias(){
    TemporaryDirectory temp;fs::create_directory(temp.path/"aliases");
    InitialImportOptions options;options.store=writable();options.shards.resize(4);
    for(size_t i=0;i<options.shards.size();++i)options.shards[i].objectIds={int64_t(i*100+1),int64_t((i+1)*100)};
    std::vector<std::string> ids(4);
    for(size_t i=0;i<ids.size();++i)do{ids[i]=util::newUuid();}while(uuidShardIndex(ids[i],ids.size())!=i);
    const auto result=importInitialCollection(temp.path/"fibers.vcobj",options,[&](size_t index,InitialShardImport& shard){
        InitialObjectRecord object;object.record.kind="note";object.record.uuid=ids[index];object.record.name="via alias";
        shard.appendObjects(std::span(&object,1));
    });
    const auto alias=temp.path/"aliases"/"source.vcobj";
    fs::create_symlink(fs::relative(result.path,alias.parent_path()),alias);
    {auto collection=ObjectCollection::open(alias,OpenOptions{.readOnly=true});auto txn=collection.read();
        for(const auto& uuid:ids)require(collection.object(txn,{{},uuid}).has_value(),"sharded root alias resolves its relative partitions");}
    {auto store=ObjectStore::open(alias,OpenOptions{.readOnly=true});auto txn=store.read();
        require(store.objectsByKind(txn,"note").size()==ids.size(),"ordinary store alias reads complete logical sharded store");}
    const auto overlay=temp.path/"overlay.vcobj";
    {auto collection=ObjectCollection::create(overlay,{alias},writable());}
    {auto collection=ObjectCollection::open(overlay,OpenOptions{.readOnly=true});auto txn=collection.read();
        for(const auto& uuid:ids)require(collection.object(txn,{{},uuid}).has_value(),"aliased dependency resolves its own relative partitions");}
}
void packedCandidate(){
    TemporaryDirectory temp;const auto original=temp.path/"plain.vcobj",packed=temp.path/"original.pack";
    const auto uuid=base(original);
    const auto wrapper=temp.path/"wrapper.vcobj";
    {auto collection=ObjectCollection::create(wrapper,{original},writable());auto w=collection.write();collection.seal(w);w.commit();collection.checkpoint();}
    packObjectCollection(wrapper,packed);
    const auto root=temp.path/"overlay.vcobj";
    {auto collection=ObjectCollection::create(root,{packed},writable());}
    fs::create_directory(temp.path/"search");const auto moved=temp.path/"search"/"renamed-container.bin";fs::rename(packed,moved);
    mustFail([&]{ObjectCollection::open(root,OpenOptions{.readOnly=true});});
    OpenOptions options;options.readOnly=true;options.storeReferenceDirectories={temp.path/"search"};
    auto collection=ObjectCollection::open(root,options);auto r=collection.read();
    require(collection.object(r,{{},uuid}).has_value(),"directory search did not resolve packed dependency");
    require(packedContainerPath(collection.members().at(1).locator)==moved,"packed relocation used wrong container");
    // A plain store's pinned physical UUID may also be supplied by a packed copy.
    const auto plainRoot=temp.path/"plain-overlay.vcobj";
    {auto overlay=ObjectCollection::create(plainRoot,{original},writable());}
    fs::rename(original,temp.path/"not-a-candidate");
    options.storeReferenceDirectories.clear();options.storeReferenceFiles={moved};
    auto plainOverlay=ObjectCollection::open(plainRoot,options);auto read=plainOverlay.read();
    require(plainOverlay.object(read,{{},uuid}).has_value(),"plain dependency was not located inside supplied packed file");
}

ExtensionSource position(ObjectCollection& collection) {
    auto txn=collection.read();const auto p=collection.provisioning(txn);
    return {collection.uuid(txn),p.generation,p.commitSequence};
}
void sourcePositions() {
    TemporaryDirectory temp;
    const auto plain=temp.path/"plain.vcobj";base(plain);
    auto source=ObjectCollection::open(plain,OpenOptions{.readOnly=true});
    const auto pin=position(source);
    {auto txn=source.read();require(resolveStorePosition(source,txn,pin)==fs::canonical(plain),"plain source root resolves exactly");
        for(int field:{0,1,2}){auto wrong=pin;
            if(field==0)wrong.storeUuid=util::newUuid();
            if(field==1)wrong.generation=util::newUuid();
            if(field==2)++wrong.commitSequence;
            mustFail([&]{resolveStorePosition(source,txn,wrong);});}}
    const auto parent=temp.path/"parent.vcobj",child=temp.path/"child.vcobj";
    {auto collection=ObjectCollection::create(parent,{plain},writable());auto txn=collection.write();collection.seal(txn);txn.commit();}
    {auto collection=ObjectCollection::create(child,{parent},writable());}
    auto overlay=ObjectCollection::open(child,OpenOptions{.readOnly=true});
    {auto txn=overlay.read();require(resolveStorePosition(overlay,txn,pin)==fs::canonical(plain),"nested source references resolve from provisioned metadata");
        auto foreign=source.read();mustFail([&]{resolveStorePosition(overlay,foreign,pin);});}
    for(bool packed:{false,true}) {
        InitialImportOptions options;options.store=writable();options.uuidHashRouting=false;
        if(packed)options.packed=PackedStoreOptions{};
        options.shards={InitialShardSpec{{1,100}},InitialShardSpec{{101,200}}};
        const auto result=importInitialCollection(temp.path/(packed?"packed":"sharded"),options,
            [](size_t i,auto& shard){InitialObjectRecord row;row.record.uuid=util::nameUuid("source-position-"+std::to_string(i));
                row.record.kind="note";shard.appendObjects(std::span(&row,1));});
        auto collection=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});const auto expected=position(collection);
        auto txn=collection.read();
        require(resolveStorePosition(collection,txn,expected)==fs::canonical(result.path),
                "shards sharing one source position do not become ambiguous public roots");
    }
    // Ambiguous physical roots with one logical position must be rejected,
    // either when their collection is admitted or when the source is selected.
    {
        const auto owner=util::newUuid(),generation=util::newUuid();std::vector<fs::path> copies;ExtensionSource ambiguousPin;
        for(int i:{0,1}){auto options=writable();options.creationStoreUuid=owner;options.creationGeneration=generation;
            options.objectIdRange={i*100+1,(i+1)*100};const auto path=temp.path/("ambiguous-"+std::to_string(i));
            {auto store=ObjectStore::create(path,options);auto txn=store.write();store.seal(txn);txn.commit();}
            auto copy=ObjectCollection::open(path,OpenOptions{.readOnly=true});const auto copyPin=position(copy);
            if(i==0)ambiguousPin=copyPin;else require(copyPin==ambiguousPin,"ambiguity fixture has matching positions");
            copies.push_back(path);}
        mustFail([&]{auto collection=ObjectCollection::create(temp.path/"ambiguous.vcobj",copies,writable());
            auto txn=collection.read();resolveStorePosition(collection,txn,ambiguousPin);});
    }
}
}
int main()try{plainAndNested();shardedAlias();packedCandidate();sourcePositions();std::cout<<"collection search paths passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
