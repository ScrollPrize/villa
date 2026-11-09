#pragma once
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/imports/ShardQueue.hpp"
#include <algorithm>
#include <future>
#include <stdexcept>
#include <variant>

namespace vc::objects::imports {
// Shared transport for decoded files, prepared memory and future native sources.
// A task is a coarse source chunk, NOT a destination shard. Readers steal tasks;
// each record is then routed by its UUID to a single independent SQLite writer.
struct ParallelImportOptions {
    size_t readers=8,shards=1,batchRows=256,readerBytes=8*1024*1024,queueBytes=256*1024*1024;
};
struct ImportObject {
    std::variant<InitialObjectRecord,const InitialObjectRecord*> value;
    const InitialObjectRecord& get() const {
        if(const auto* owned=std::get_if<InitialObjectRecord>(&value))return *owned;
        const auto* pointer=std::get<const InitialObjectRecord*>(value);
        if(!pointer)throw std::invalid_argument("null borrowed import object");
        return *pointer;
    }
};
template<class T> class ShardEmitter {
    ShardQueues<T>& queues_;
    StreamFailure& failure_;
    const ParallelImportOptions& options_;
    std::vector<std::vector<T>> batches_;
    std::vector<size_t> bytes_;
    size_t pending_=0;
    void flush(size_t shard) {
        if(batches_[shard].empty())return;
        pending_-=bytes_[shard];
        const auto start=std::chrono::steady_clock::now();
        queues_.push(shard,std::move(batches_[shard]),bytes_[shard]);
        waitSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        batches_[shard].clear();bytes_[shard]=0;
    }
public:
    double waitSeconds=0;
    ShardEmitter(ShardQueues<T>& queues,StreamFailure& failure,const ParallelImportOptions& options)
        :queues_(queues),failure_(failure),options_(options),batches_(options.shards),bytes_(options.shards){}
    void emit(T row,size_t bytes,const std::string& uuid) {
        failure_.check();const auto shard=uuidShardIndex(uuid,options_.shards);
        batches_[shard].push_back(std::move(row));bytes_[shard]+=bytes;pending_+=bytes;
        if(batches_[shard].size()>=options_.batchRows)flush(shard);
        // Flush the largest buffer, not every shard's mostly empty batch.
        while(pending_>=options_.readerBytes) {
            const auto largest=size_t(std::max_element(bytes_.begin(),bytes_.end())-bytes_.begin());
            flush(largest);
        }
    }
    void finish(){for(size_t i=0;i<options_.shards;++i)flush(i);}
};
using ObjectEmitter=ShardEmitter<ImportObject>;
using EdgeEmitter=ShardEmitter<InitialCollectionEdge>;
struct StreamReaderStats {size_t objectTasks=0,edgeTasks=0;double objectSeconds=0,edgeSeconds=0,queueWaitSeconds=0;};

class ParallelImportStream {
    ParallelImportOptions options_;
    StreamFailure failure_;
    ShardQueues<ImportObject> objects_;
    ShardQueues<InitialCollectionEdge> edges_;
    std::atomic<size_t> nextObject_{0},nextEdge_{0},objectsRemaining_,edgesRemaining_;
    std::vector<std::future<void>> jobs_;
    bool started_=false;
public:
    std::vector<StreamReaderStats> readers;
    explicit ParallelImportStream(ParallelImportOptions options)
        :options_(options),objects_(options.shards,std::max(size_t(1),options.queueBytes/std::max(size_t(1),options.shards)),failure_),
         edges_(options.shards,std::max(size_t(1),options.queueBytes/std::max(size_t(1),options.shards)),failure_),
         objectsRemaining_(options.readers),edgesRemaining_(options.readers),readers(options.readers) {
        if(!options.readers||!options.shards||!options.batchRows||!options.readerBytes||!options.queueBytes)
            throw std::invalid_argument("parallel import budgets and worker counts must be positive");
    }
    ~ParallelImportStream(){
        if(!jobs_.empty()){fail(std::make_exception_ptr(std::runtime_error("parallel import abandoned")));
            for(auto& job:jobs_)if(job.valid())job.wait();}
    }
    void fail(std::exception_ptr error){failure_.fail(error);}
    // Each callback runs only on its indexed reader. Source cursors may be kept
    // there and reused for successive tasks. All callbacks are joined on finish.
    void start(size_t objectTasks,size_t edgeTasks,
        std::function<void(size_t,size_t,ObjectEmitter&)> objectTask,
        std::function<void(size_t,size_t,EdgeEmitter&)> edgeTask,
        std::function<void(size_t)> readerFinished={}) {
        if(started_)throw std::logic_error("parallel import stream already started");
        started_=true;
        if((objectTasks&&!objectTask)||(edgeTasks&&!edgeTask))throw std::invalid_argument("missing source callback");
        // Reserve before launching: a vector allocation failure must not destroy
        // a live async future while it waits for consumers not yet started.
        jobs_.reserve(options_.readers);
        try {for(size_t worker=0;worker<options_.readers;++worker)
            jobs_.push_back(std::async(std::launch::async,[=,this]{
                try {
                    auto& stats=readers[worker];const auto begin=std::chrono::steady_clock::now();
                    ObjectEmitter objects(objects_,failure_,options_);
                    for(size_t task;(task=nextObject_.fetch_add(1))<objectTasks;){failure_.check();objectTask(worker,task,objects);++stats.objectTasks;}
                    objects.finish();stats.objectSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()-objects.waitSeconds;
                    if(objectsRemaining_.fetch_sub(1)==1)objects_.close();
                    // No reader join/barrier: early finishers can decode edges
                    // while other readers finish their last geometry chunks.
                    const auto edgeStart=std::chrono::steady_clock::now();EdgeEmitter edges(edges_,failure_,options_);
                    for(size_t task;(task=nextEdge_.fetch_add(1))<edgeTasks;){failure_.check();edgeTask(worker,task,edges);++stats.edgeTasks;}
                    edges.finish();stats.edgeSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-edgeStart).count()-edges.waitSeconds;
                    stats.queueWaitSeconds=objects.waitSeconds+edges.waitSeconds;
                    if(readerFinished)readerFinished(worker);
                    if(edgesRemaining_.fetch_sub(1)==1)edges_.close();
                }catch(...){fail(std::current_exception());throw;}
            }));
        }catch(...){auto error=std::current_exception();fail(error);
            for(auto& job:jobs_)if(job.valid())job.wait();jobs_.clear();std::rethrow_exception(error);}
    }
    template<class Check> bool popObjects(size_t shard,std::vector<ImportObject>& batch,Check&& check){return objects_.pop(shard,batch,std::forward<Check>(check));}
    template<class Check> bool popEdges(size_t shard,std::vector<InitialCollectionEdge>& batch,Check&& check){return edges_.pop(shard,batch,std::forward<Check>(check));}
    void finish(){for(auto& job:jobs_)try{job.get();}catch(...){fail(std::current_exception());}jobs_.clear();failure_.check();}
};
} // namespace vc::objects::imports
