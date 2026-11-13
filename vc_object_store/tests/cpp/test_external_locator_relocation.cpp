#include "TestSupport.hpp"
#include "internal/RecordJson.hpp"
#include "vc_object_store/migration/ConvertStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include <sqlite3.h>
#include <fstream>
#include <iostream>
#include <map>

using namespace vc::objects;
using Json=nlohmann::json;
namespace fs=std::filesystem;

using Records=std::map<std::string,Json>;
Records records(const fs::path& path,const std::vector<std::string>& ids)
{
    auto store=ObjectStore::open(path,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto read=store.read();
    Records result;
    for(const auto& id:ids) {
        if(const auto current=store.objectByUuid(read,id))result["current:"+id]=detail::objectToJson(*current);
        for(const auto& revision:store.objectHistory(read,id,true))
            result[revision.uuid]=detail::objectToJson(revision.record);
    }
    return result;
}

void compare(const Records& before,Records after,const fs::path& source,const fs::path& destination)
{
    require(before.size()==after.size(),"relocation retains all current and historical records");
    for(const auto& [key,original]:before) {
        auto expected=original;
        const auto locator=[&](Json& old,Json& current,const char* name) {
            if(!old.contains(name))return;
            const auto a=old.at(name).get<std::string>(), b=current.at(name).get<std::string>();
            if(!a.empty()&&a.find("://")==std::string::npos&&fs::path(a).is_relative()) {
                require(fs::absolute(source.parent_path()/a).lexically_normal()==
                        fs::absolute(destination.parent_path()/b).lexically_normal(),"external locator retains its exact target");
                old[name]=b;
            }
        };
        auto& actual=after.at(key);
        if(expected.value("storage_kind",std::string())=="file")locator(expected,actual,"storage_ref");
        if(expected.contains("track"))locator(expected["track"],actual["track"],"dataset_ref");
        require(expected==actual,"relocation preserves record contents and identities apart from equivalent path spelling");
    }
}

int main() try {
    TemporaryDirectory tmp;
    for(const auto policy:{HistoryMode::On,HistoryMode::Off}) {
        OpenOptions options;options.history=policy;options.author="relocation-test";
        const auto root=tmp.path/std::to_string(int(policy));
        for(const auto* name:{"source","copy","pack","repack"})fs::create_directories(root/name);
        const auto source=root/"source/original.vcobj", copied=root/"copy/copy.vcobj",
            packed=root/"pack/packed.vcobj", repacked=root/"repack/packed.vcobj";
        fs::create_directories(root/"source/data");
        for(const auto* label:{"first","second","third"}) {
            std::ofstream(root/"source/data"/(std::string(label)+".json"))<<"[]";
            fs::create_directory(root/"source/data"/(std::string(label)+".tracks"));
        }
        std::ofstream(root/"absolute.json")<<"[]";
        std::vector<std::string> ids;
        {
            auto store=ObjectStore::create(source,options);
            {auto write=store.write();
                for(const auto& path:{std::string("data/first.json"),std::string("https://example.org/file"),(root/"absolute.json").string()}) {
                    ObjectRecord object;object.kind="fiber";object.fiber.emplace();object.storageKind="file";object.storageRef=path;
                    ids.push_back(store.upsertObject(write,object));
                }
                ObjectRecord track;track.kind="track";track.track.emplace();track.track->datasetRef="data/first.tracks";
                track.track->externalKey="one";ids.push_back(store.upsertObject(write,track));write.commit();
            }
            for(int step=0;step<2;++step) {
                TransactionRequest request;
                {auto read=store.read();const auto status=store.status(read);
                    request.requestUuid=util::newUuid();request.storeUuid=status.storeUuid;
                    request.generation=status.generation;request.author="relocation-test";
                    for(const auto& id:ids){auto record=*store.objectByUuid(read,id);record.name="version"+std::to_string(step);
                        expectObject(request,store.currentObjectRevision(read,id)->reference());
                        if(id==ids.front())record.storageRef=step==0?"data/second.json":"data/third.json";
                        if(record.track)record.track->datasetRef=step==0?"data/second.tracks":"data/third.tracks";
                        request.operations.push_back(UpdateObjectOp{detail::objectToJson(record).dump()});}
                }
                store.submit(request);
            }
            auto write=store.write();store.seal(write);write.commit();
        }
        const auto before=records(source,ids);
        copyObjectCollection(source,copied,options);compare(before,records(copied,ids),source,copied);
        packObjectCollection(source,packed);compare(before,records(packed,ids),source,packed);
        packObjectCollection(packed,repacked);compare(before,records(repacked,ids),source,repacked);
        const auto compacted=root/"copy/compacted.vcobj";
        compactPackedStore(packed,compacted);
        compare(before,records(compacted,ids),source,compacted);
        copyObjectCollection(packed,root/"copy/from-packed.vcobj",options);
        compare(before,records(root/"copy/from-packed.vcobj",ids),source,root/"copy/from-packed.vcobj");
        // Collection closure: the dependency moves into a sidecar for copying,
        // or into another logical database of the same packed container.
        const auto overlay=root/"source/overlay.vcobj",overlayCopy=root/"copy/overlay.vcobj",
            overlayPack=root/"pack/overlay.vcobj";
        {auto collection=ObjectCollection::create(overlay,{source},options);auto write=collection.write();
            collection.seal(write);write.commit();}
        copyObjectCollection(overlay,overlayCopy,options);packObjectCollection(overlay,overlayPack);
        const auto overlayCompact=root/"repack/overlay.vcobj";
        compactPackedStore(overlayPack,overlayCompact);
        for(const auto& output:{overlayCopy,overlayPack,overlayCompact}) {
            auto collection=ObjectCollection::open(output,OpenOptions{.readOnly=true});
            const auto members=collection.members();require(members.size()==2,"collection copy retains its dependency");
            const auto physical=members.back().locator;
            compare(before,records(physical,ids),source,isPackedDatabasePath(physical)?packedContainerPath(physical):physical);
        }
        require(records(source,ids)==before,"exports do not modify the source");
        require(!fs::exists(root/"copy/data")&&!fs::exists(root/"pack/data"),"external payloads are not copied");
    }
    // Authentic upstream histories exercise conversion across directories,
    // including their complete bases, metadata patches and cached head records.
    for(int schema:{17,18,19,20}) {
        const auto root=tmp.path/("convert"+std::to_string(schema));fs::create_directories(root/"elsewhere");
        const auto source=root/"source.vcobj",baseline=root/"baseline.vcobj",moved=root/"elsewhere/converted.vcobj";
        std::ifstream input(fs::path(VC_OBJECT_STORE_TEST_DATA)/("convert_schema"+std::to_string(std::min(schema,19))+".sql"));
        const std::string sql{std::istreambuf_iterator<char>(input),{}};
        sqlite3* db=nullptr;require(sqlite3_open(source.c_str(),&db)==SQLITE_OK,"open upstream fixture");
        const auto rc=sqlite3_exec(db,sql.c_str(),nullptr,nullptr,nullptr);
        if(schema==20)require(sqlite3_exec(db,"DROP TABLE sync_replica; PRAGMA user_version=20",nullptr,nullptr,nullptr)==SQLITE_OK,"schema20 fixture");
        sqlite3_close(db);require(rc==SQLITE_OK,"load authentic upstream fixture");
        migration::convertStore(source,baseline);migration::convertStore(source,moved);
        std::ifstream expected(fs::path(VC_OBJECT_STORE_TEST_DATA)/"convert_expected.json");
        const auto ids=Json::parse(expected).at("objects").get<std::vector<std::string>>();
        compare(records(baseline,ids),records(moved,ids),baseline,moved);
    }
    std::cout<<"external locator relocation passed\n";
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
