#pragma once
#include "platform_io.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>
#include <typeinfo>

namespace vcr {
struct ScratchStats {
    std::atomic<size_t> bytes{0},peak{0},allocations{0},shrinks{0},idle_releases{0};
    void resize(size_t old,size_t next) {
        const auto current=next>=old?bytes.fetch_add(next-old,std::memory_order_relaxed)+next-old:
            bytes.fetch_sub(old-next,std::memory_order_relaxed)-(old-next);
        auto high=peak.load(std::memory_order_relaxed);
        while(current>high&&!peak.compare_exchange_weak(high,current,std::memory_order_relaxed)){}
    }
};

// A pool belongs to one executing thread. Only an idle scavenger may claim its
// inactive frames; no worker takes buffers from another worker's free-list.
class ScratchPool {
public:
    using Clock=std::chrono::steady_clock;
    using Now=Clock::time_point(*)();
    struct Policy {Clock::duration idle=std::chrono::seconds(2),small=std::chrono::seconds(10);};
private:
    struct Buffer {
        std::unique_ptr<uint8_t[]> data;
        size_t capacity=0,small_peak=0;
        Clock::time_point large_use{},last_use{};
        bool used=false;
    };
    struct Frame {
        std::atomic<bool> busy{false};
        std::vector<Buffer> buffers;
        struct Resource {
            const std::type_info* type;
            std::shared_ptr<void> value;
            Clock::time_point last_use{};
            bool used=false;
        };
        std::vector<Resource> resources;
    };
    std::shared_ptr<ScratchStats> stats_;
    Policy policy_;
    Now now_;
    std::mutex frames_mutex_; // growth/snapshot only, not ordinary reuse
    std::vector<std::unique_ptr<Frame>> frames_;
    size_t depth_=0;
    static Clock::time_point now(){return Clock::now();}
    void release(std::vector<Buffer>& buffers) {
        for(auto& b:buffers)if(b.capacity){stats_->resize(b.capacity,0);b.data.reset();b.capacity=0;}
    }
    void detachExpired(Frame& frame,Clock::time_point time,std::vector<Buffer>& drop,
                       std::vector<std::shared_ptr<void>>& resources) {
        for(auto& b:frame.buffers)if(b.capacity&&time-b.last_use>=policy_.idle) {
            drop.push_back(std::move(b));b=Buffer{};
        }
        for(auto& r:frame.resources)if(r.value&&time-r.last_use>=policy_.idle) {
            resources.push_back(std::move(r.value));r.used=false;
        }
        if(!drop.empty())++stats_->idle_releases;
    }
public:
    explicit ScratchPool(std::shared_ptr<ScratchStats> stats,Policy policy,Now clock=&now)
        :stats_(std::move(stats)),policy_(policy),now_(clock){}
    ~ScratchPool(){for(auto& frame:frames_)release(frame->buffers);}
    class Lease {
        ScratchPool& pool_;
        Frame* frame_;
        Clock::time_point time_;
    public:
        explicit Lease(ScratchPool& pool):pool_(pool),time_(pool.now_()) {
            const auto index=pool.depth_;
            if(index==pool.frames_.size()) {
                auto frame=std::make_unique<Frame>();
                std::lock_guard lock(pool.frames_mutex_);pool.frames_.push_back(std::move(frame));
            }
            frame_=pool.frames_[index].get();
            bool expected=false;
            while(!frame_->busy.compare_exchange_weak(expected,true,std::memory_order_acquire)) {
                expected=false;std::this_thread::yield();
            }
            ++pool.depth_;
            for(auto& buffer:frame_->buffers)buffer.used=false;
            for(auto& resource:frame_->resources)resource.used=false;
        }
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
        ~Lease(){
            const auto time=pool_.now_();
            for(auto& b:frame_->buffers)if(b.used)b.last_use=time;
            for(auto& r:frame_->resources)if(r.used)r.last_use=time;
            std::vector<Buffer> drop;
            std::vector<std::shared_ptr<void>> resources;
            // At a task boundary no returned scratch span is still live.
            // Opportunistic cleanup must not make lease destruction throw.
            try{pool_.detachExpired(*frame_,time,drop,resources);}catch(...){}
            --pool_.depth_;frame_->busy.store(false,std::memory_order_release);pool_.release(drop);
        }
        // A leased frame owns codec state just as it owns byte scratch: nested
        // operations get another frame; idle cleanup cannot touch active state.
        template<class T> T& resource() {
            for(auto& r:frame_->resources)if(*r.type==typeid(T)) {
                if(!r.value)r.value=std::make_shared<T>();
                r.used=true;return *static_cast<T*>(r.value.get());
            }
            auto value=std::make_shared<T>();
            frame_->resources.push_back({&typeid(T),value,time_,true});return *value;
        }
        // Scratch is uninitialized. Every byte read by the caller must first be
        // written; requesting a slot again may invalidate that slot's old span.
        std::span<uint8_t> bytes(size_t slot,size_t count) {
            if(slot>=frame_->buffers.size())frame_->buffers.resize(slot+1);
            auto& b=frame_->buffers[slot];
            if(!count)return {};
            b.used=true;
            size_t target=b.capacity;
            if(count>b.capacity) {
                target=std::max(count,b.capacity+b.capacity/2);
                b.large_use=time_;b.small_peak=0;
            } else if(count>=b.capacity/2) {b.large_use=time_;b.small_peak=0;}
            else {
                b.small_peak=std::max(b.small_peak,count);
                if(time_-b.large_use>=pool_.policy_.small)target=b.small_peak;
            }
            if(target!=b.capacity) {
                auto next=std::make_unique_for_overwrite<uint8_t[]>(target);
                if(target<b.capacity)++pool_.stats_->shrinks;
                ++pool_.stats_->allocations;pool_.stats_->resize(b.capacity,target);
                b.data=std::move(next);b.capacity=target;b.large_use=time_;b.small_peak=0;
            }
            return {b.data.get(),count};
        }
    };
    Lease lease(){return Lease(*this);}
    void trim(Clock::time_point time) {
        std::vector<Frame*> frames;
        {std::lock_guard lock(frames_mutex_);for(auto& frame:frames_)frames.push_back(frame.get());}
        for(auto* frame:frames) {
            bool expected=false;
            if(!frame->busy.compare_exchange_strong(expected,true,std::memory_order_acquire))continue;
            std::vector<Buffer> drop;
            std::vector<std::shared_ptr<void>> resources;
            try{detachExpired(*frame,time,drop,resources);}catch(const std::bad_alloc&){}
            frame->busy.store(false,std::memory_order_release);
            release(drop); // frees never hold the frame claim
        }
    }
};

class ScratchMaintenance {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::weak_ptr<ScratchPool>> pools_;
    bool stop_=false;
    std::thread worker_;
    std::once_flag start_;
    void loop() {
        std::unique_lock lock(mutex_);
        while(!changed_.wait_for(lock,std::chrono::milliseconds(250),[&]{return stop_;})) {
            std::vector<std::shared_ptr<ScratchPool>> pools;
            std::erase_if(pools_,[&](const auto& weak){if(auto p=weak.lock()){pools.push_back(std::move(p));return false;}return true;});
            lock.unlock();const auto time=ScratchPool::Clock::now();
            for(auto& p:pools)p->trim(time);
            pools.clear();lock.lock();
        }
    }
public:
    const long pid=platform::process_id();
    std::shared_ptr<ScratchStats> stats=std::make_shared<ScratchStats>();
    ScratchMaintenance()=default;
    void start(){std::call_once(start_,[this]{worker_=std::thread([this]{loop();});});}
    ~ScratchMaintenance(){{std::lock_guard lock(mutex_);stop_=true;}changed_.notify_one();if(worker_.joinable())worker_.join();}
    std::shared_ptr<ScratchPool> add() {
        auto pool=std::make_shared<ScratchPool>(stats,ScratchPool::Policy{});
        std::lock_guard lock(mutex_);pools_.push_back(pool);return pool;
    }
};
inline ScratchMaintenance& scratchMaintenance() {
    struct Holder {
        std::atomic<ScratchMaintenance*> current{nullptr};
        ~Holder(){auto* p=current.load();if(p&&p->pid==platform::process_id())delete p;}
    };
    static Holder holder;
    auto* current=holder.current.load(std::memory_order_acquire);
    if(!current||current->pid!=platform::process_id()) {
        auto fresh=std::make_unique<ScratchMaintenance>();
        if(holder.current.compare_exchange_strong(current,fresh.get(),std::memory_order_acq_rel))current=fresh.release();
    }
    current->start();return *current;
}
inline ScratchPool& workerScratch() {
    struct Local {long pid;std::shared_ptr<ScratchPool> pool;};
    thread_local Local local{platform::process_id(),scratchMaintenance().add()};
    if(local.pid!=platform::process_id())local={platform::process_id(),scratchMaintenance().add()};
    return *local.pool;
}
} // namespace vcr
