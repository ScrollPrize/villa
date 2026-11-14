#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unordered_map>
#include <utility>
#include "coordination_trace.hpp"

namespace vcr {
// One producer per exact key. Shard locks protect registration/completion only;
// neither the producer nor future waiters hold them during source work.
template<class Value> class SingleFlight {
    struct Entry {
        std::promise<Value> promise;
        std::shared_future<Value> future=promise.get_future().share();
    };
    struct Shard {
        std::mutex mutex;
        std::unordered_map<std::string,std::shared_ptr<Entry>> active;
    };
public:
    explicit SingleFlight(std::size_t shards=std::max(1u,std::thread::hardware_concurrency())) {
        if(!shards)throw std::invalid_argument("single-flight shard count must be positive");
        shards_.reserve(shards);for(std::size_t i=0;i<shards;++i)shards_.push_back(std::make_unique<Shard>());
    }
    class Claim {
        Shard* shard_;
        std::string key_;
        std::shared_ptr<Entry> entry_;
        bool owner_;
        friend class SingleFlight;
        Claim(Shard& shard,const std::string& key,std::shared_ptr<Entry> entry,bool owner)
            :shard_(&shard),key_(key),entry_(std::move(entry)),owner_(owner){}
        void fail(std::exception_ptr error) {
            diagnostics::SchedulerLock lock(shard_->mutex,diagnostics::SchedulerMetric::DependencyCompleteLock);
            entry_->promise.set_exception(error);shard_->active.erase(key_);owner_=false;
        }
    public:
        Claim(const Claim&)=delete;
        Claim& operator=(const Claim&)=delete;
        Claim(Claim&& other) noexcept
            :shard_(other.shard_),key_(std::move(other.key_)),entry_(std::move(other.entry_)),
             owner_(std::exchange(other.owner_,false)){}
        ~Claim() {
            if(owner_)try{fail(std::make_exception_ptr(std::runtime_error("abandoned chunk load")));}catch(...){}
        }
        bool owner() const {return owner_;}
        bool ready() const {return entry_->future.wait_for(std::chrono::seconds(0))==std::future_status::ready;}
        template<class Producer> Value resolve(Producer&& producer) {
            if(!owner_){diagnostics::Scope waiting(diagnostics::ChunkWait);
                diagnostics::SchedulerElapsed elapsed(diagnostics::SchedulerMetric::DependencyWait);return entry_->future.get();}
            try {
                diagnostics::schedulerRecord(diagnostics::SchedulerMetric::Producer,1);
                diagnostics::Scope producing(diagnostics::Producer);
                Value value=producer();
                {
                    diagnostics::SchedulerLock lock(shard_->mutex,diagnostics::SchedulerMetric::DependencyCompleteLock);
                    entry_->promise.set_value(value);shard_->active.erase(key_);owner_=false;
                }
                return value;
            } catch(...) {fail(std::current_exception());throw;}
        }
    };
    Claim acquire(const std::string& key) {
        auto& shard=*shards_[std::hash<std::string>{}(key)%shards_.size()];
        diagnostics::Scope registering(diagnostics::Registration);
        diagnostics::SchedulerLock lock(shard.mutex,diagnostics::SchedulerMetric::DependencyRegisterLock);
        const auto found=shard.active.find(key);
        if(found!=shard.active.end())return Claim(shard,key,found->second,false);
        auto entry=std::make_shared<Entry>();shard.active.emplace(key,entry);
        return Claim(shard,key,std::move(entry),true);
    }
private:
    std::vector<std::unique_ptr<Shard>> shards_;
};
} // namespace vcr
