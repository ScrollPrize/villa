#pragma once

#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#if (defined(__linux__) || defined(__APPLE__)) && __has_include(<execinfo.h>) && __has_include(<dlfcn.h>)
#define VC_SQLITE_PROFILE_STACKS 1
#include <execinfo.h>
#include <dlfcn.h>
#else
#define VC_SQLITE_PROFILE_STACKS 0
#endif

namespace vc::objects::benchmark::sqlite_profile {
// Opt-in diagnostic forwarding overlay, shared by standalone import tools.
// Adapted from the existing embedded-import STATIC_MEM profiler. No lock is
// removed or replaced: allocation and entry always call SQLite's own methods.
// Installation must precede any live SQLite connection or worker thread.
struct Counters {
    uint64_t calls=0, samples=0, waitNanoseconds=0, maxWaitNanoseconds=0;
};
inline sqlite3_mutex_methods original{};
inline std::array<std::atomic<sqlite3_mutex*>,14> staticMutexes{};
inline thread_local std::array<Counters,14> counters{};
inline bool installed=false;
inline bool captureStacks=false;
struct StackSample {
    std::array<void*,8> frames{};
    int depth=0;
    uint64_t count=0;
};
inline thread_local std::array<StackSample,32> stacks{};
inline thread_local uint64_t droppedStacks=0;
inline constexpr std::array<const char*,14> names={
    "dynamic_or_unknown", "unused", "static_main", "static_mem",
    "static_open", "static_prng", "static_lru", "static_pmem",
    "static_app1", "static_app2", "static_app3", "static_vfs1",
    "static_vfs2", "static_vfs3"};

inline sqlite3_mutex* allocate(int type)
{
    auto* result=original.xMutexAlloc(type);
    if(type>=2 && type<int(staticMutexes.size()))
        staticMutexes[type].store(result,std::memory_order_relaxed);
    return result;
}

inline void capturePmemStack()
{
#if VC_SQLITE_PROFILE_STACKS
    StackSample sample;sample.depth=backtrace(sample.frames.data(),int(sample.frames.size()));
    for(auto& entry:stacks){
        if(!entry.count){entry=sample;entry.count=1;return;}
        if(entry.depth==sample.depth && entry.frames==sample.frames){++entry.count;return;}
    }
    ++droppedStacks;
#endif
}

inline void enter(sqlite3_mutex* mutex)
{
    // SQLite's public sqlite3_mutex_enter(NULL) is a no-op; do not classify it
    // as one of the not-yet-allocated static mutexes.
    if(!mutex){original.xMutexEnter(mutex);return;}
    size_t category=0;
    for(size_t i=2;i<staticMutexes.size();++i)
        if(staticMutexes[i].load(std::memory_order_relaxed)==mutex){category=i;break;}
    auto& value=counters[category];
    ++value.calls;
    // Capture before entering the lock, never while holding the shared PMEM
    // lock. Stack storage itself is fixed-size and thread-local. Symbol lookup
    // happens only in the much less frequent progress snapshot.
    if(captureStacks && category==SQLITE_MUTEX_STATIC_PMEM && value.calls%65536==0)
        capturePmemStack();
    if((value.calls%1024)==0){
        const auto start=std::chrono::steady_clock::now();
        original.xMutexEnter(mutex);
        const auto nanoseconds=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
        ++value.samples;
        value.waitNanoseconds+=nanoseconds;
        value.maxWaitNanoseconds=std::max(value.maxWaitNanoseconds,uint64_t(nanoseconds));
        return;
    }
    original.xMutexEnter(mutex);
}

inline void install(bool withStacks=false)
{
    if(installed)throw std::logic_error("SQLite mutex profiler already installed");
#if !VC_SQLITE_PROFILE_STACKS
    if(withStacks)throw std::runtime_error("SQLite stack profiling is unsupported on this platform");
#endif
    if(sqlite3_initialize()!=SQLITE_OK || sqlite3_shutdown()!=SQLITE_OK)
        throw std::runtime_error("SQLite profile initialization failed");
    if(sqlite3_config(SQLITE_CONFIG_GETMUTEX,&original)!=SQLITE_OK ||
       !original.xMutexEnter || !original.xMutexAlloc)
        throw std::runtime_error("SQLite native mutex methods unavailable");
    auto methods=original;methods.xMutexAlloc=allocate;methods.xMutexEnter=enter;
    if(sqlite3_config(SQLITE_CONFIG_MUTEX,&methods)!=SQLITE_OK)
        throw std::runtime_error("Cannot install SQLite mutex profiler");
    installed=true;
    captureStacks=withStacks;
}

// Read only the calling thread's counters. Safe while other workers run;
// no process-global registry lock, allocations or atomic counters per entry.
inline nlohmann::json snapshot()
{
    nlohmann::json result=nlohmann::json::object();
    for(size_t i=0;i<counters.size();++i){
        const auto& value=counters[i];
        if(!value.calls && !(captureStacks && i==SQLITE_MUTEX_STATIC_PMEM))continue;
        result[names[i]]={{"entry_calls",value.calls},{"samples",value.samples},
            {"sampled_entry_seconds",double(value.waitNanoseconds)*1e-9},
            {"estimated_entry_seconds",value.samples ? double(value.waitNanoseconds)*1e-9*
                double(value.calls)/double(value.samples):0.0},
            {"max_sampled_entry_seconds",double(value.maxWaitNanoseconds)*1e-9}};
    }
#if VC_SQLITE_PROFILE_STACKS
    if(captureStacks){
        auto samples=nlohmann::json::array();
        for(const auto& sample:stacks){
            if(!sample.count)continue;
            auto frames=nlohmann::json::array();
            for(int i=0;i<sample.depth;++i){
                Dl_info info{};const bool resolved=dladdr(sample.frames[i],&info)!=0;
                frames.push_back({{"address",reinterpret_cast<uintptr_t>(sample.frames[i])},
                    {"image_offset",resolved ? nlohmann::json(reinterpret_cast<uintptr_t>(sample.frames[i])-
                        reinterpret_cast<uintptr_t>(info.dli_fbase)):nlohmann::json(nullptr)},
                    {"symbol",resolved && info.dli_sname ? info.dli_sname:""},
                    {"image",resolved && info.dli_fname ? info.dli_fname:""}});
            }
            samples.push_back({{"count",sample.count},{"frames",std::move(frames)}});
        }
        result[names[SQLITE_MUTEX_STATIC_PMEM]]["stacks"]=std::move(samples);
        result[names[SQLITE_MUTEX_STATIC_PMEM]]["dropped_stacks"]=droppedStacks;
    }
#endif
    return result;
}
}
