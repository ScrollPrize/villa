// Benchmark-only access to the actual pre-reader decoded cache implementation.
#pragma once
#include "vc/core/render/ChunkCache.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <functional>
#include <limits>
#include <stdexcept>

namespace vc::render {
class CacheBenchmarkAccess {
    using Cache=ChunkCache;
    std::shared_ptr<DecodedChunkCacheBudget> budget_;
    std::shared_ptr<Cache::State> state_;
public:
    explicit CacheBenchmarkAccess(std::size_t budget)
        : budget_(std::make_shared<DecodedChunkCacheBudget>(budget)) {
        Cache::Options options;
        options.metadataEntryCapacity=std::numeric_limits<std::size_t>::max();
        options.detectAllFillChunks=false;
        state_=std::make_shared<Cache::State>(std::vector<Cache::LevelInfo>{{{16,16,16},{16,16,16},{}}},
            std::vector<std::shared_ptr<IChunkFetcher>>{nullptr},0,ChunkDtype::Opaque,options,budget_,VolumeSourceId{1},"synthetic");
        Cache::registerStateBudget(state_);
    }
    ~CacheBenchmarkAccess() { Cache::unregisterStateBudget(*state_); }
    static ChunkKey key(const std::string& text) {
        constexpr std::string_view prefix="volume:synthetic/chunk/";
        std::uint64_t id=0;
        const auto parsed=std::from_chars(text.data()+prefix.size(),text.data()+text.size(),id);
        if(parsed.ec!=std::errc{})throw std::invalid_argument("invalid fixture key");
        return {0,int(id>>48),int((id>>24)&0xffffff),int(id&0xffffff),VolumeSourceId{1}};
    }
    std::pair<bool,std::shared_ptr<const DecodedChunkPayload>> get(const std::string& text) {
        const auto k=key(text);
        std::lock_guard lock(state_->mutex_);
        const auto it=state_->entries_.find(k);
        if(it==state_->entries_.end())return {false,{}};
        const auto result=Cache::resultFromEntryLocked(*state_,k,it->second,true);
        return {result.status==ChunkStatus::Data,result.payload};
    }
    void put(const std::string& text,std::shared_ptr<const DecodedChunkPayload> payload) {
        const auto k=key(text);
        if(payload->residentBytes()>budget_->maximumBytes())return;
        {
            std::lock_guard lock(state_->mutex_);
            state_->entries_.try_emplace(k);
            ChunkFetchResult fetch;
            fetch.status=ChunkFetchStatus::Found;fetch.payload=std::move(payload);
            Cache::storeFetchResultLocked(state_,k,std::move(fetch),false);
        }
        Cache::enforceSharedBudget(state_);
    }
    std::size_t nbytes()const{return budget_->stats().decodedBytes;}
    void resize(std::size_t bytes){budget_->setMaximumBytes(bytes);}
    std::size_t max_bytes()const{return budget_->maximumBytes();}
};
}

// Surface matching the common fixture, not an alternative cache implementation.
namespace vcr {
struct Chunk final : std::vector<std::uint8_t>, vc::render::DecodedChunkPayload {
    using std::vector<std::uint8_t>::vector;
    std::size_t residentBytes()const noexcept override{return size();}
};
using ChunkPtr=std::shared_ptr<const Chunk>;
class MemoryCache {
    vc::render::CacheBenchmarkAccess cache_;
public:
    struct AutoOptions {
        double smoothing_seconds=0;
        std::chrono::milliseconds refresh_interval{1000};
        std::function<std::uint64_t()> available;
    };
    struct Timing {std::uint64_t wait_ns=0,hold_ns=0;};
    struct Profile {Timing get,put;};
    struct MaintenanceStats {
        std::uint64_t evicted_entries=0,batches=0,selection_scans=0,reclaim_claims=0,assist_puts=0;
        std::size_t headroom_bytes=0,reclaiming_bytes=0;
    };
    explicit MemoryCache(std::size_t bytes,std::size_t shards):cache_(bytes){
        if(shards!=1)throw std::invalid_argument("historical control uses one source, not sharded storage");
    }
    explicit MemoryCache(AutoOptions options,std::size_t shards):cache_(options.available()/2),options_(std::move(options)){
        if(shards!=1)throw std::invalid_argument("historical control uses one source, not sharded storage");
    }
    auto get(const std::string& k){return cache_.get(k);}
    void put(const std::string& k,ChunkPtr value){cache_.put(k,std::move(value));}
    std::size_t nbytes()const{return cache_.nbytes();}
    std::size_t max_bytes()const{return cache_.max_bytes();}
    void refresh_budget(){if(options_.available)cache_.resize((options_.available()+nbytes())/2);}
    void enable_profiling(bool on){if(on)throw std::invalid_argument("historical profiling unsupported");}
    void enable_maintenance(std::size_t,std::size_t){
        throw std::invalid_argument("historical cache has no background maintenance");
    }
    MaintenanceStats maintenance_stats()const{return {};}
    Profile profile()const{return {};}
private:
    AutoOptions options_;
};
}
