#pragma once
// Opt-in sampled diagnostics. Per-task updates touch thread-local atomics, not
// shared counters; registry locks are only used on thread/job registration.
#include <atomic>
#include <array>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <ctime>
#include <vector>
#include <nlohmann/json.hpp>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace vcr::diagnostics {
enum Flag : unsigned { Planning=1,Task=2,Producer=4,ChunkWait=8,BatchWait=16,Registration=32,Helper=64,ChunkBatch=128,ChunkTask=256,Io=512,Decode=1024,Allocate=2048,Copy=4096 };
enum class SchedulerStage : unsigned {Other,Planning,Resolve,Interpolation,Count};
enum class SchedulerMetric : unsigned {Calls,Submitted,CallerItems,HelperItems,SubmitLock,RetireLock,SelectLock,CompletionLock,BatchLock,BatchWait,HelperIdle,Producer,DependencyWait,DependencyRegisterLock,DependencyCompleteLock,SourceRead,DecodeBlock,Size0,Size1,Size2to4,Size5to16,Size17to64,Size65to256,Size257plus,QueuePush,QueuePop,QueueFull,QueueStale,FileOpen,FileStat,FileRead,FileClose,Count};
inline constexpr const char* schedulerStageNames[]={"other","planning","resolve","interpolation"};
inline constexpr const char* schedulerMetricNames[]={"calls","submitted","caller_items","helper_items","submit_lock","retire_lock","select_lock","completion_lock","batch_lock","batch_wait","helper_idle","producer","dependency_wait","dependency_register_lock","dependency_complete_lock","source_read","decode_block","size_0","size_1","size_2_4","size_5_16","size_17_64","size_65_256","size_257_plus","queue_push","queue_pop","queue_full","queue_stale","file_open","file_stat","file_read","file_close"};
struct SchedulerCounter {
    std::atomic<uint64_t> count{0},units{0},samples{0},wait_ns{0},hold_ns{0};
};
struct SchedulerCounters {
    std::array<std::array<SchedulerCounter,unsigned(SchedulerMetric::Count)>,unsigned(SchedulerStage::Count)> values;
};
struct alignas(64) ThreadState {
    std::atomic<unsigned> flags{0};long tid=0;
    std::unique_ptr<SchedulerCounters> scheduler;
};
inline bool schedulerProfiling(){static const bool on=std::getenv("VC_READER_SCHEDULER_PROFILE")!=nullptr;return on;}
inline thread_local SchedulerStage schedulerStage=SchedulerStage::Other;
struct JobState {std::atomic<size_t> ready{0};bool chunks=false;};
class Trace {
    std::mutex mutex_;
    std::vector<std::shared_ptr<ThreadState>> threads_;
    std::vector<std::weak_ptr<JobState>> jobs_;
    std::ofstream output_;
    std::atomic<bool> stop_{false};
    std::thread monitor_;
    void loop() {
        const bool wchan=std::getenv("VC_READER_COORDINATION_WCHAN")!=nullptr;
        const auto start=std::chrono::steady_clock::now();
        unsigned iteration=0;
        while(!stop_.load(std::memory_order_relaxed)) {
            size_t planning=0,tasks=0,producers=0,waiters=0,joins=0,registration=0,helpers=0,helperWaiters=0,ready=0,jobs=0,chunkTasks=0,chunkReady=0,io=0,decode=0,allocate=0,copy=0;
            std::vector<long> ownerTids;
            const bool snapshot=schedulerProfiling()&&iteration++%10==0;
            std::array<std::array<std::array<uint64_t,5>,unsigned(SchedulerMetric::Count)>,unsigned(SchedulerStage::Count)> scheduler{};
            {
                std::lock_guard lock(mutex_);
                for(const auto& t:threads_) {
                    if(snapshot&&t->scheduler)for(unsigned s=0;s<unsigned(SchedulerStage::Count);++s)
                        for(unsigned m=0;m<unsigned(SchedulerMetric::Count);++m){
                            const auto& v=t->scheduler->values[s][m];auto& sum=scheduler[s][m];
                            sum[0]+=v.count.load(std::memory_order_relaxed);sum[1]+=v.units.load(std::memory_order_relaxed);
                            sum[2]+=v.samples.load(std::memory_order_relaxed);sum[3]+=v.wait_ns.load(std::memory_order_relaxed);sum[4]+=v.hold_ns.load(std::memory_order_relaxed);
                        }
                    const auto f=t->flags.load(std::memory_order_relaxed);
                    planning+=bool(f&Planning);tasks+=bool(f&Task);producers+=bool(f&Producer);
                    chunkTasks+=bool((f&ChunkTask)||((f&Task)&&(f&ChunkBatch)));io+=bool(f&Io);decode+=bool(f&Decode);
                    allocate+=bool(f&Allocate);copy+=bool(f&Copy);
                    if(wchan&&(f&Producer)&&t->tid)ownerTids.push_back(t->tid);
                    waiters+=bool(f&ChunkWait);joins+=bool(f&BatchWait);registration+=bool(f&Registration);
                    helpers+=bool((f&Helper)&&(f&Task));helperWaiters+=bool((f&Helper)&&(f&ChunkWait));
                }
                std::erase_if(jobs_,[&](const auto& weak) {
                    if(auto job=weak.lock()){const auto n=job->ready.load(std::memory_order_relaxed);ready+=n;if(job->chunks)chunkReady+=n;++jobs;return false;}
                    return true;
                });
            }
            nlohmann::json waits=nlohmann::json::object();
#if defined(__linux__)
            for(const auto tid:ownerTids){std::string wait;std::ifstream("/proc/self/task/"+std::to_string(tid)+"/wchan")>>wait;
                if(wait.empty())wait="unknown";waits[wait]=waits.value(wait,0)+1;}
#endif
            nlohmann::json row={{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
                {"monotonic_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()},
                {"planning",planning},{"tasks",tasks},{"producers",producers},{"chunk_waiters",waiters},
                {"batch_waiters",joins},{"registering",registration},{"helper_tasks",helpers},
                {"helper_chunk_waiters",helperWaiters},{"ready",ready},{"chunk_ready",chunkReady},{"chunk_tasks",chunkTasks},{"io",io},{"decode",decode},{"allocate",allocate},{"copy",copy},{"producer_wchan",waits},{"jobs",jobs}};
            if(snapshot)for(unsigned s=0;s<unsigned(SchedulerStage::Count);++s)
                for(unsigned m=0;m<unsigned(SchedulerMetric::Count);++m){const auto& v=scheduler[s][m];
                    row["scheduler"][schedulerStageNames[s]][schedulerMetricNames[m]]={{"count",v[0]},{"units",v[1]},{"samples",v[2]},{"wait_ns",v[3]},{"hold_ns",v[4]}};}
            output_<<row.dump()<<'\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(wchan?30:10));
        }
    }
public:
    Trace() {if(const auto* path=std::getenv("VC_READER_COORDINATION_TRACE")) {
        output_.open(path);if(!output_)throw std::runtime_error("cannot open reader coordination trace");
        monitor_=std::thread([this]{loop();});}}
    ~Trace(){stop_=true;if(monitor_.joinable())monitor_.join();}
    static Trace& instance(){static Trace trace;return trace;}
    static bool enabled(){static const bool on=std::getenv("VC_READER_COORDINATION_TRACE")!=nullptr;return on;}
    std::shared_ptr<ThreadState> thread() {
        auto value=std::make_shared<ThreadState>();
        if(schedulerProfiling())value->scheduler=std::make_unique<SchedulerCounters>();
#if defined(__linux__)
        value->tid=::syscall(SYS_gettid);
#endif
        std::lock_guard lock(mutex_);threads_.push_back(value);return value;
    }
    std::shared_ptr<JobState> job(size_t n,bool chunks) {
        auto value=std::make_shared<JobState>();value->ready=n;value->chunks=chunks;
        std::lock_guard lock(mutex_);jobs_.push_back(value);return value;
    }
};
inline ThreadState* threadState() {
    struct Local {std::shared_ptr<ThreadState> value=Trace::instance().thread();~Local(){value->flags=0;}};
    thread_local Local local;return local.value.get();
}
inline SchedulerCounter* schedulerCounter(SchedulerMetric metric,uint64_t units=0) {
    if(!schedulerProfiling())return nullptr;
    auto* counter=&threadState()->scheduler->values[unsigned(schedulerStage)][unsigned(metric)];
    counter->units.fetch_add(units,std::memory_order_relaxed);return counter;
}
inline void schedulerRecord(SchedulerMetric metric,uint64_t units=0) {
    if(auto* value=schedulerCounter(metric,units))value->count.fetch_add(1,std::memory_order_relaxed);
}
struct SchedulerStageScope {
    SchedulerStage previous=schedulerStage;bool on=schedulerProfiling();
    explicit SchedulerStageScope(SchedulerStage stage){if(on)schedulerStage=stage;}
    ~SchedulerStageScope(){if(on)schedulerStage=previous;}
};
// Timing one in 64 acquisitions avoids clock calls on every item. Counters are
// per thread; the sampled monitor alone aggregates them. CV sleeps are excluded
// from lock hold time and reported separately, including CV reacquisition.
class SchedulerLock {
    using Clock=std::chrono::steady_clock;
    SchedulerCounter* counter_=nullptr;bool sampled_=false;Clock::time_point acquired_{};
    void held(){if(sampled_)counter_->hold_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-acquired_).count(),std::memory_order_relaxed);}
public:
    std::unique_lock<std::mutex> lock;
    SchedulerLock(std::mutex& mutex,SchedulerMetric metric):lock(mutex,std::defer_lock) {
        counter_=schedulerCounter(metric);
        if(counter_)sampled_=counter_->count.fetch_add(1,std::memory_order_relaxed)%64==0;
        const auto before=sampled_?Clock::now():Clock::time_point{};
        lock.lock();
        if(sampled_){acquired_=Clock::now();counter_->samples.fetch_add(1,std::memory_order_relaxed);
            counter_->wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(acquired_-before).count(),std::memory_order_relaxed);}
    }
    ~SchedulerLock(){held();}
    template<class Predicate>void wait(std::condition_variable& cv,Predicate predicate,SchedulerMetric metric) {
        if(predicate())return;
        held();auto* counter=schedulerCounter(metric);const auto before=counter?Clock::now():Clock::time_point{};
        cv.wait(lock,predicate);
        if(counter){counter->count.fetch_add(1,std::memory_order_relaxed);counter->samples.fetch_add(1,std::memory_order_relaxed);
            counter->wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-before).count(),std::memory_order_relaxed);}
        if(sampled_)acquired_=Clock::now();
    }
};
class SchedulerElapsed {
    using Clock=std::chrono::steady_clock;SchedulerCounter* counter_;Clock::time_point start_{};bool sampled_=false;
public:
    explicit SchedulerElapsed(SchedulerMetric metric):counter_(schedulerCounter(metric)) {
        if(counter_)sampled_=counter_->count.fetch_add(1,std::memory_order_relaxed)%64==0;
        if(sampled_)start_=Clock::now();
    }
    ~SchedulerElapsed(){if(sampled_){counter_->samples.fetch_add(1,std::memory_order_relaxed);counter_->wait_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start_).count(),std::memory_order_relaxed);}}
};
// For source syscalls, wait_ns is sampled elapsed time and hold_ns is sampled
// executing-thread CPU time. Their difference includes blocking and descheduling,
// not just storage-device latency. Disabled profiling incurs no clock calls.
class SourceElapsed {
    using Clock=std::chrono::steady_clock;
    SchedulerCounter* counter_;bool sampled_=false;
    Clock::time_point start_{};uint64_t cpu_=0;
    static uint64_t cpuTime() noexcept {
#if defined(CLOCK_THREAD_CPUTIME_ID)
        timespec t{};
        if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0)
            return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_nsec);
#endif
        return 0;
    }
public:
    explicit SourceElapsed(SchedulerMetric metric,uint64_t bytes=0):counter_(schedulerCounter(metric,bytes)) {
        if(counter_)sampled_=counter_->count.fetch_add(1,std::memory_order_relaxed)%64==0;
        if(sampled_){cpu_=cpuTime();start_=Clock::now();}
    }
    ~SourceElapsed(){if(sampled_){const auto wall=Clock::now()-start_;const auto cpu=cpuTime();
        counter_->samples.fetch_add(1,std::memory_order_relaxed);
        counter_->wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(wall).count(),std::memory_order_relaxed);
        counter_->hold_ns.fetch_add(cpu>=cpu_?cpu-cpu_:0,std::memory_order_relaxed);}}
};
struct Scope {
    ThreadState* state=nullptr;unsigned previous=0;
    explicit Scope(unsigned flag) {
        if(Trace::enabled()){state=threadState();previous=state->flags.load(std::memory_order_relaxed);
            state->flags.store(previous|flag,std::memory_order_relaxed);}
    }
    ~Scope(){if(state)state->flags.store(previous,std::memory_order_relaxed);}
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
// Disposable replay input, native-endian and same-host only. Capturing is
// independent of sampled tracing and is never enabled by ordinary reads.
inline void recordCoordinates(const float* xyz,size_t count,float spacing) {
    static const char* path=std::getenv("VC_READER_COORDINATE_REPLAY");
    if(!path)return;
    static std::mutex mutex;std::lock_guard lock(mutex);
    static std::ofstream out(path,std::ios::binary);
    if(!out)throw std::runtime_error("cannot write coordinate replay");
    const uint64_t n=count;
    out.write(reinterpret_cast<const char*>(&n),sizeof(n));
    out.write(reinterpret_cast<const char*>(&spacing),sizeof(spacing));
    out.write(reinterpret_cast<const char*>(xyz),count*3*sizeof(float));
    if(!out)throw std::runtime_error("coordinate replay write failed");
}
} // namespace vcr::diagnostics
