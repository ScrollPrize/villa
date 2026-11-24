#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <cmath>
#include <string>

// Diagnostic only. A short monitor supplies snapshots; no reader/cache locks
// or scientific work are changed. Times and counters are supplied explicitly
// so cutoff/warmup behavior can be tested without sleeps or fake geometry.
class ChunkSaturationWindow {
public:
    struct Sample {
        double seconds=0.,cpuSeconds=0.;
        size_t active=0,remaining=0;
        std::array<uint64_t,3> counts{}; // measured constraints, normal samples, emitted constraints
        size_t completedWorkers=0;
    };
    ChunkSaturationWindow(size_t workers,double warmup,double cap,bool requireWorkerCompletion=false)
        :workers_(workers),warmup_(warmup),cap_(cap),requireWorkerCompletion_(requireWorkerCompletion) {
        if(!workers || !std::isfinite(warmup)||warmup<0. || !std::isfinite(cap)||cap<=0.)
            throw std::invalid_argument("Saturated chunk benchmark requires positive workers and duration");
    }
    bool done()const{return done_.load(std::memory_order_acquire);}
    void abortUnstarted(const Sample& sample) {
        std::lock_guard lock(mutex_);if(done()||start_)return;
        end_=sample;reason_="pool_not_saturated";done_.store(true,std::memory_order_release);
    }
    void observe(const Sample& sample) {
        std::lock_guard lock(mutex_);if(done())return;
        completedWorkers_=sample.completedWorkers;
        const bool ready=!requireWorkerCompletion_||completedWorkers_>=workers_;
        // Freeze saturated statistics at the tail boundary, but do not kill a
        // worker before its first completion. Waiting is outside that window.
        if(end_) {
            if(ready){completionWait_=sample.seconds-end_->seconds;done_.store(true,std::memory_order_release);}
            return;
        }
        if(!full_&&sample.active>=workers_)full_=sample;
        if(!start_&&full_&&sample.seconds-full_->seconds>=warmup_&&sample.remaining>=workers_) {
            start_=last_=sample;minimum_=maximum_=sample.active;
        }
        if(start_) {
            activeIntegral_+=(sample.seconds-last_->seconds)*double(last_->active);
            last_=sample;minimum_=std::min(minimum_,sample.active);maximum_=std::max(maximum_,sample.active);
        }
        const bool drained=sample.remaining<workers_;
        const bool capped=start_&&sample.seconds-start_->seconds>=cap_&&
            ready;
        if(drained||capped) {
            end_=sample;reason_=drained?"remaining_jobs_below_worker_count":"measurement_cap";
            if(ready)done_.store(true,std::memory_order_release);
        }
    }
    nlohmann::json summary()const {
        std::lock_guard lock(mutex_);
        nlohmann::json result={{"valid",bool(start_&&end_)},{"reason",reason_},
            {"worker_count",workers_},{"warmup_seconds",warmup_},{"cap_seconds",cap_}};
        if(!start_||!end_)return result;
        const auto elapsed=end_->seconds-start_->seconds;
        result["window_seconds"]=elapsed;result["window_start_pool_seconds"]=start_->seconds;
        result["window_end_pool_seconds"]=end_->seconds;
        result["remaining_at_start"]=start_->remaining;result["remaining_at_end"]=end_->remaining;
        result["completed_workers"]=completedWorkers_;
        result["worker_completion_wait_seconds"]=completionWait_;
        result["worker_completion_requirement_met"]=!requireWorkerCompletion_||completedWorkers_>=workers_;
        result["process_cpu_seconds"]=end_->cpuSeconds-start_->cpuSeconds;
        result["effective_parallelism"]=(end_->cpuSeconds-start_->cpuSeconds)/std::max(elapsed,1e-9);
        result["mean_active_jobs"]=activeIntegral_/std::max(elapsed,1e-9);
        result["minimum_active_jobs"]=minimum_;result["peak_active_jobs"]=maximum_;
        const std::array<const char*,3> names{"measured_constraints","normal_samples","emitted_constraints"};
        for(size_t i=0;i<names.size();++i) {
            const auto count=end_->counts[i]-start_->counts[i];result[names[i]]=count;
            result[std::string(names[i])+"_per_second"]=double(count)/std::max(elapsed,1e-9);
        }
        return result;
    }
private:
    size_t workers_;double warmup_,cap_;bool requireWorkerCompletion_;
    size_t completedWorkers_=0;double completionWait_=0.;
    mutable std::mutex mutex_;std::atomic<bool> done_{false};
    std::optional<Sample> full_,start_,last_,end_;
    size_t minimum_=0,maximum_=0;double activeIntegral_=0.;std::string reason_="not_saturated";
};
