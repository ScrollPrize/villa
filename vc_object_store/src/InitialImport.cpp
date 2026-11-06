#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/Publication.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/InitialEndpoints.hpp"
#include "vc_reader/publication.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <thread>

namespace vc::objects {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start)
{return std::chrono::duration<double>(Clock::now()-start).count();}

struct Failure {
    std::atomic<bool> failed=false;
    std::mutex mutex;
    std::condition_variable changed;
    std::exception_ptr error;
    void record(std::exception_ptr current) {
        {std::lock_guard lock(mutex);
            if(!error)error=std::move(current);
            failed.store(true,std::memory_order_release);}
        changed.notify_all();
    }
    void check() const {
        if(failed.load(std::memory_order_acquire))
            throw std::runtime_error("initial import cancelled after another producer failed");
    }
};
struct PrivateDirectory {
    std::filesystem::path path;
    explicit PrivateDirectory(const std::filesystem::path& destination)
        :path(destination.parent_path()/("."+destination.filename().string()+"-"+util::newUuid())) {
        if(!std::filesystem::create_directory(path))
            throw std::runtime_error("initial import cannot create private directory");
    }
    ~PrivateDirectory(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}
};
void validateRanges(const std::vector<InitialShardSpec>& shards,bool objects)
{
    std::vector<std::array<int64_t,2>> ranges;
    for(const auto& shard:shards){
        const auto range=objects?shard.objectIds:shard.edgeIds;
        if(range[0]<1 || range[1]<range[0])throw std::invalid_argument("initial import invalid shard ID range");
        ranges.push_back(range);
    }
    std::sort(ranges.begin(),ranges.end());
    for(size_t i=1;i<ranges.size();++i)
        if(ranges[i][0]<=ranges[i-1][1])throw std::invalid_argument("initial import overlapping shard ID ranges");
}
}

struct InitialShardImport::Impl {
    ObjectCollection& store;
    CollectionTxn& transaction;
    const InitialEdgeImportOptions& edgeOptions;
    const std::string& owner;
    Failure& failure;
    InitialShardResult result;
    std::thread::id thread=std::this_thread::get_id();
    bool edgesStarted=false,insideProducer=false;
    bool phasedObjects=false,objectsClosed=false;
    std::unique_ptr<detail::InitialEndpointDirectory> endpoints;
    void check() const {
        if(std::this_thread::get_id()!=thread){
            auto error=std::make_exception_ptr(std::runtime_error("initial shard used outside its producer thread"));
            failure.record(error);std::rethrow_exception(error);
        }
        failure.check();
    }
    [[noreturn]] void reject(const char* message) {
        auto error=std::make_exception_ptr(std::logic_error(message));failure.record(error);std::rethrow_exception(error);
    }
};

InitialShardImport::InitialShardImport(std::unique_ptr<Impl> impl):impl_(std::move(impl)){}
InitialShardImport::~InitialShardImport()=default;
const std::string& InitialShardImport::storeUuid() const {impl_->check();return impl_->owner;}

std::vector<InitialObjectIdentity> InitialShardImport::appendObjects(std::span<const InitialObjectRecord> objects)
{
    auto& p=*impl_;p.check();
    if(p.edgesStarted || p.insideProducer || p.objectsClosed)p.reject("initial objects must precede edge import");
    try {
        if(objects.size()>uint64_t(INT64_MAX-p.result.objects))throw std::overflow_error("initial object count overflow");
        const auto start=Clock::now();
        auto identities=p.store.importInitialObjects(p.transaction,objects);
        if(p.endpoints) {
            if(identities.size()!=objects.size())throw std::logic_error("initial object identity count differs from input");
            for(size_t i=0;i<identities.size();++i) {
                detail::InitialEndpoint endpoint;endpoint.id=identities[i].id;endpoint.retired=objects[i].record.retired;
                if(objects[i].geometry)endpoint.geometry=detail::prepareDependencies(*objects[i].geometry);
                if(!p.endpoints->emplace(identities[i].key,std::move(endpoint)).second)
                    throw std::runtime_error("initial object has a duplicate qualified identity");
            }
        }
        p.result.objectSeconds+=elapsed(start);
        p.result.objects+=static_cast<int64_t>(identities.size());
        return identities;
    } catch(...){p.failure.record(std::current_exception());throw;}
}

int64_t InitialShardImport::importEdges(const std::function<bool(std::vector<InitialCollectionEdge>&)>& producer)
{
    auto& p=*impl_;p.check();
    if(p.edgesStarted || !producer || p.phasedObjects)p.reject("initial edges require one nonempty producer callback after the object phase");
    p.edgesStarted=true;
    try {
        const auto start=Clock::now();double sourceSeconds=0;
        const auto count=p.store.importInitialEdges(p.transaction,[&](auto& batch){
            p.check();const auto sourceStart=Clock::now();p.insideProducer=true;
            bool more;
            try {more=producer(batch);}catch(...){p.insideProducer=false;throw;}
            p.insideProducer=false;sourceSeconds+=elapsed(sourceStart);p.check();return more;
        },p.edgeOptions);
        p.result.producerSeconds+=sourceSeconds;
        p.result.edgeSeconds+=std::max(0.0,elapsed(start)-sourceSeconds);
        p.result.edges=count;
        return count;
    } catch(...){p.failure.record(std::current_exception());throw;}
}

InitialImportResult importInitialCollection(const std::filesystem::path& target,
    const InitialImportOptions& options,const std::function<void(size_t,InitialShardImport&)>& producer)
{
    return importInitialCollection(target,options,producer,{});
}

InitialImportResult importInitialCollection(const std::filesystem::path& target,
    const InitialImportOptions& options,const std::function<void(size_t,InitialShardImport&)>& producer,
    const std::function<void(size_t,InitialShardImport&)>& edgeProducer)
{
    const auto start=Clock::now();
    if(!producer || options.shards.empty())throw std::invalid_argument("initial import needs shards and a producer");
    if(options.store.history!=HistoryMode::Off || options.store.spatialIndex || options.store.readOnly)
        throw std::invalid_argument("initial import requires writable history-off, spatial-off provisioning");
    if(options.store.author.empty())throw std::invalid_argument("initial import requires an author");
    if(options.store.creationStoreUuid)throw std::invalid_argument("initial import owns its fresh store UUID");
    if(!options.edges.maxCachedEndpoints || !options.edges.sqlBatchRows)
        throw std::invalid_argument("initial import edge batch/cache limits must be positive");
    validateRanges(options.shards,true);validateRanges(options.shards,false);
    const auto destination=std::filesystem::absolute(target).lexically_normal();
    if(std::filesystem::exists(destination) || std::filesystem::is_symlink(destination))
        throw std::runtime_error("initial import destination already exists");
    if(!std::filesystem::is_directory(destination.parent_path()))
        throw std::runtime_error("initial import destination parent is absent");

    InitialImportResult result;
    result.path=!options.packed && options.shards.size()>1?destination/"collection.vcobj":destination;
    result.storeUuid=util::newUuid();
    auto common=options.store;common.creationStoreUuid=result.storeUuid;
    std::unique_ptr<PrivateDirectory> directory;
    std::unique_ptr<detail::StagedDatabase> single;
    std::unique_ptr<PackedStoreBuilder> packed;
    const auto manifestId=util::newUuid();
    std::vector<std::filesystem::path> paths;
    std::filesystem::path manifest;
    if(options.packed){
        // Relative external locators must be based on the eventual container,
        // not on a temporary subdirectory that disappears after publication.
        packed=std::make_unique<PackedStoreBuilder>(destination,*options.packed);
        for(size_t i=0;i<options.shards.size();++i)paths.push_back(packed->addDatabase(util::newUuid()));
        manifest=paths.size()==1?paths.front():packed->addDatabase(manifestId);
    }else if(options.shards.size()==1){
        // A sibling filename retains the same external-relative-link base.
        single=std::make_unique<detail::StagedDatabase>(destination,false);
        paths.push_back(single->path);manifest=paths.front();
    }else{
        directory=std::make_unique<PrivateDirectory>(destination);
        for(size_t i=0;i<options.shards.size();++i)paths.push_back(directory->path/("part-"+std::to_string(i)+".vcobj"));
        manifest=directory->path/"collection.vcobj";
    }

    Failure failure;
    std::vector<std::unique_ptr<detail::InitialEndpointDirectory>> workerEndpoints(paths.size());
    std::shared_ptr<const detail::InitialEndpointDirectory> endpoints;
    size_t objectsRemaining=paths.size();
    std::vector<std::future<InitialShardResult>> jobs;
    try {
        for(size_t i=0;i<paths.size();++i)jobs.push_back(std::async(std::launch::async,[&,i]{
            try {
                failure.check();const auto creation=Clock::now();
                auto opts=common;opts.objectIdRange=options.shards[i].objectIds;opts.edgeIdRange=options.shards[i].edgeIds;
                auto collection=detail::InitialCollectionAccess::create(paths[i],options.dependencies,opts);
                auto transaction=collection.write();
                auto state=std::make_unique<InitialShardImport::Impl>(InitialShardImport::Impl{
                    collection,transaction,options.edges,result.storeUuid,failure});
                state->result.creationSeconds=elapsed(creation);
                state->phasedObjects=bool(edgeProducer);
                if(edgeProducer)state->endpoints=std::make_unique<detail::InitialEndpointDirectory>();
                InitialShardImport shard(std::move(state));
                const auto producing=Clock::now();
                producer(i,shard);
                shard.impl_->check();
                double totalProducer=elapsed(producing);
                if(edgeProducer) {
                    shard.impl_->objectsClosed=true;shard.impl_->phasedObjects=false;
                    // Each worker accumulates its own compact validation state.
                    // A single merge publishes it only after every object phase
                    // has finished; no other thread touches a worker's SQLite DB.
                    {
                        std::unique_lock lock(failure.mutex);
                        workerEndpoints[i]=std::move(shard.impl_->endpoints);
                        if(--objectsRemaining==0 && !failure.failed.load(std::memory_order_acquire)) {
                            auto combined=std::make_shared<detail::InitialEndpointDirectory>();
                            for(auto& local:workerEndpoints) {
                                combined->merge(*local);
                                if(!local->empty())throw std::runtime_error("initial shards contain duplicate qualified object identities");
                                local.reset();
                            }
                            endpoints=std::move(combined);failure.changed.notify_all();
                        } else failure.changed.wait(lock,[&]{return endpoints || failure.failed.load(std::memory_order_acquire);});
                    }
                    failure.check();
                    detail::InitialCollectionAccess::setEndpoints(collection,endpoints);
                    const auto edgesStart=Clock::now();edgeProducer(i,shard);shard.impl_->check();
                    totalProducer+=elapsed(edgesStart);
                }
                // Includes producer preparation between API calls, but excludes
                // native object/edge construction and index finalization.
                shard.impl_->result.producerSeconds=std::max(0.0,totalProducer-
                    shard.impl_->result.objectSeconds-shard.impl_->result.edgeSeconds);
                const auto seal=Clock::now();collection.seal(transaction);transaction.commit();collection.checkpoint();
                shard.impl_->result.sealCommitSeconds=elapsed(seal);
                return shard.impl_->result;
            } catch(...){failure.record(std::current_exception());throw;}
        }));
    } catch(...){failure.record(std::current_exception());}
    for(auto& job:jobs){try {result.shards.push_back(job.get());}catch(...){failure.record(std::current_exception());}}
    if(failure.error)std::rethrow_exception(failure.error);
    result.ingestionSeconds=elapsed(start);
    const auto manifestStart=Clock::now();
    if(paths.size()>1){auto joined=ObjectCollection::publishShards(manifest,paths,common);}
    result.manifestSeconds=elapsed(manifestStart);
    const auto persistence=Clock::now();
    if(packed)packed->finalize(paths.size()==1?packedDatabaseUuid(paths.front()):manifestId);
    else if(single)single->publish();
    else {
        vcr::platform::sync_directory(directory->path,"sync initial import members");
        vcr::files::publishDirectory(directory->path,destination,false);
        vcr::platform::sync_directory(destination.parent_path(),"publish initial import collection");
    }
    result.persistenceSeconds=elapsed(persistence);result.totalSeconds=elapsed(start);
    return result;
}
}
