#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
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
    require(object&&object->record.name=="found","relocated linked object not readable");
    require(collection.members().at(1).locator==dependency,"dependency candidate precedence changed");
}
void plainAndNested(){
    TemporaryDirectory temp;fs::create_directories(temp.path/"source");fs::create_directories(temp.path/"moved");
    const auto original=temp.path/"source"/"base.vcobj",root=temp.path/"overlay.vcobj";
    const auto uuid=base(original);
    {auto collection=ObjectCollection::create(root,{original},writable());}
    const auto duplicate=temp.path/"moved"/"duplicate";fs::copy_file(original,duplicate);
    OpenOptions options;options.readOnly=true;options.linkedStoreFiles={duplicate};
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
    options.linkedStoreFiles={unrelated,stale,moved};found(root,options,uuid,moved);
    options.linkedStoreFiles={unrelated,stale};mustFail([&]{ObjectCollection::open(root,options);});
    // Directory discovery is not filename-based and ignores non-store files.
    {std::ofstream noise(temp.path/"moved"/"00-readme.txt");noise<<"not a database";}
    options.linkedStoreFiles.clear();options.linkedStoreDirectories={temp.path/"missing-directory",temp.path/"moved"};
    // The lexically first identity-matching copy is selected deterministically.
    found(root,options,uuid,duplicate);
    auto editing=options;editing.readOnly=false;editing.author="search-path-test";
    {auto collection=ObjectCollection::open(root,editing);auto w=collection.write();collection.seal(w);w.commit();collection.checkpoint();}
    const auto child=temp.path/"child.vcobj";
    {auto collection=ObjectCollection::create(child,{root},editing);}
    const auto movedRoot=temp.path/"moved"/"parent-renamed.vcobj";fs::rename(root,movedRoot);
    options.linkedStoreFiles={movedRoot,moved};options.linkedStoreDirectories.clear();
    auto collection=ObjectCollection::open(child,options);auto r=collection.read();
    require(collection.object(r,{{},uuid}).has_value(),"search options were not propagated down linked chains");
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
    OpenOptions options;options.readOnly=true;options.linkedStoreDirectories={temp.path/"search"};
    auto collection=ObjectCollection::open(root,options);auto r=collection.read();
    require(collection.object(r,{{},uuid}).has_value(),"directory search did not resolve packed dependency");
    require(packedContainerPath(collection.members().at(1).locator)==moved,"packed relocation used wrong container");
    // A plain store's pinned physical UUID may also be supplied by a packed copy.
    const auto plainRoot=temp.path/"plain-overlay.vcobj";
    {auto overlay=ObjectCollection::create(plainRoot,{original},writable());}
    fs::rename(original,temp.path/"not-a-candidate");
    options.linkedStoreDirectories.clear();options.linkedStoreFiles={moved};
    auto plainOverlay=ObjectCollection::open(plainRoot,options);auto read=plainOverlay.read();
    require(plainOverlay.object(read,{{},uuid}).has_value(),"plain dependency was not located inside supplied packed file");
}
}
int main()try{plainAndNested();packedCandidate();std::cout<<"collection search paths passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
