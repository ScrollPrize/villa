#include "TestSupport.hpp"
#include "vc_object_store/objects/imports/ShardQueue.hpp"
#include "vc_object_store/objects/imports/ParallelImportStream.hpp"
#include <future>
#include <iostream>

using namespace vc::objects::imports;
using namespace vc::objects;
using namespace std::chrono_literals;
namespace {
void backpressureAndClose(){
    StreamFailure failure;ShardQueues<int> queue(1,1,failure);
    queue.push(0,{1},20); // One oversize batch still makes progress.
    std::promise<void> started;
    auto writer=std::async(std::launch::async,[&]{started.set_value();queue.push(0,{2},1);});
    started.get_future().wait();
    require(writer.wait_for(60ms)==std::future_status::timeout,"queue ignored backpressure");
    std::vector<int> rows;
    require(queue.pop(0,rows,[]{})&&rows==std::vector<int>{1},"queue did not preserve first batch");
    writer.get();queue.close();
    require(queue.pop(0,rows,[]{})&&rows==std::vector<int>{2},"close dropped a queued batch");
    require(!queue.pop(0,rows,[]{}),"closed drained queue did not finish");
}
void cancellation(){
    for(const bool producer:{false,true}){
        StreamFailure failure;ShardQueues<int> queue(1,1,failure);
        if(producer)queue.push(0,{1},1);
        std::promise<void> started;
        auto blocked=std::async(std::launch::async,[&]{
            started.set_value();
            if(producer)queue.push(0,{2},1);
            else {std::vector<int> rows;queue.pop(0,rows,[]{});}
        });
        started.get_future().wait();
        require(blocked.wait_for(60ms)==std::future_status::timeout,"queue did not wait before cancellation");
        failure.fail(std::make_exception_ptr(std::runtime_error("peer failed")));
        require(blocked.wait_for(2s)==std::future_status::ready,"peer failure left queue blocked");
        mustFail([&]{blocked.get();});
    }
    StreamFailure failure;ShardQueues<int> queue(1,1,failure);std::vector<int> rows;
    mustFail([&]{queue.pop(0,rows,[]{throw std::runtime_error("builder cancelled");});});
}
void parallelStream(){
    TemporaryDirectory dir;constexpr size_t count=80,workers=4;
    std::vector<InitialObjectRecord> records(count);std::vector<std::atomic<int>> visits(count);
    for(size_t i=0;i<count;++i){records[i].record.uuid=util::nameUuid("stream-test-"+std::to_string(i));records[i].record.kind="note";}
    InitialImportOptions options;options.store.author="stream-test";options.shards.clear();
    for(int64_t i=0;i<int64_t(workers);++i)options.shards.push_back({{i*100+1,(i+1)*100},{i*100+1,(i+1)*100}});
    ParallelImportStream stream({workers,workers,3,512,512});
    stream.start(count,count,[&](size_t,size_t task,ObjectEmitter& out){
        ++visits[task];
        // Exercise owned and borrowed payloads in the same bounded stream.
        if(task%2)out.emit(ImportObject{&records[task]},sizeof(ImportObject),records[task].record.uuid);
        else out.emit(ImportObject{records[task]},sizeof(ImportObject),records[task].record.uuid);
    },[&](size_t,size_t task,EdgeEmitter& out){
        EdgeRecord e;e.uuid=util::nameUuid("stream-edge-"+std::to_string(task));e.kind="test";
        e.aUuid=records[task].record.uuid;e.bUuid=records[(task+1)%count].record.uuid;
        out.emit({{"",e.aUuid},{"",e.bUuid},e},sizeof(e),e.uuid);
    });
    const auto result=importInitialCollection(dir.path/"stream",options,[&](size_t shard,InitialShardImport& writer){try{
        std::vector<ImportObject> rows;std::vector<const InitialObjectRecord*> refs;
        while(stream.popObjects(shard,rows,[&]{writer.checkCancelled();})){
            refs.clear();for(const auto& row:rows)refs.push_back(&row.get());writer.appendObjectRefs(refs);
        }
    }catch(...){stream.fail(std::current_exception());throw;}},[&](size_t shard,InitialShardImport& writer){try{
        writer.importEdges([&](auto& rows){return stream.popEdges(shard,rows,[&]{writer.checkCancelled();});});
    }catch(...){stream.fail(std::current_exception());throw;}});
    stream.finish();auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto txn=store.read();
    for(size_t i=0;i<count;++i){require(visits[i]==1,"work stealing lost or duplicated a task");
        require(bool(store.object(txn,{"",records[i].record.uuid})),"stream lost an object");
        require(bool(store.edge(txn,{"",util::nameUuid("stream-edge-"+std::to_string(i))})),"stream lost an edge");}
    // A source failure cancels blocked consumers, and cannot publish a partial store.
    ParallelImportStream failed({2,1,1,1,1});failed.start(1,0,[](size_t,size_t,ObjectEmitter&){throw std::runtime_error("bad source");},{});
    InitialImportOptions failedOptions;failedOptions.store.author="stream-test";
    mustFail([&]{importInitialCollection(dir.path/"failed",failedOptions,[&](size_t shard,InitialShardImport& writer){
        std::vector<ImportObject> rows;while(failed.popObjects(shard,rows,[&]{writer.checkCancelled();})){}
    });});
    mustFail([&]{failed.finish();});require(!std::filesystem::exists(dir.path/"failed"),"failed stream was published");
}
}
int main()try{backpressureAndClose();cancellation();parallelStream();std::cout<<"shard queue passed\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
