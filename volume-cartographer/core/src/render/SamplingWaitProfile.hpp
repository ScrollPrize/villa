#pragma once

// Opt-in attribution for batch CLI diagnostics. No shared counters or changes
// to locking/scheduling. Flush only outside cache locks, from the sampler.
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <syncstream>
#include <thread>

namespace vc::render::detail {
inline bool samplingWaitProfileEnabled() {
    static const bool enabled = std::getenv("VC_SAMPLING_WAIT_PROFILE") != nullptr;
    return enabled;
}
struct SamplingWaitStats {
    std::array<double,12> seconds{};
    std::array<size_t,12> calls{};
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
inline thread_local SamplingWaitStats samplingWaitStats;
inline void recordSamplingWait(size_t phase,double seconds) {
    if (!samplingWaitProfileEnabled()) return;
    samplingWaitStats.seconds[phase] += seconds;
    ++samplingWaitStats.calls[phase];
}
class SamplingWaitTimer {
    size_t phase_;
    std::chrono::steady_clock::time_point start_;
public:
    explicit SamplingWaitTimer(size_t phase):phase_(phase) {
        if(samplingWaitProfileEnabled()) start_=std::chrono::steady_clock::now();
    }
    ~SamplingWaitTimer() {
        if(samplingWaitProfileEnabled()) recordSamplingWait(phase_,
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count());
    }
};
inline void flushSamplingWaits() {
    if(!samplingWaitProfileEnabled()) return;
    auto& s=samplingWaitStats;
    const auto now=std::chrono::steady_clock::now();
    if(now-s.last<std::chrono::seconds(10)) return;
    constexpr const char* names[]={"blocking_total","blocking_lock","prefetch_total",
        "prefetch_lock","cached_total","cached_lock","corner_prepare_layout","corner_pin","corner_gather",
        "payload_wait","entry_budget","shared_budget"};
    std::osyncstream out(std::cerr);
    out<<"sampling-waits thread="<<std::this_thread::get_id();
    for(size_t i=0;i<s.seconds.size();++i) out<<' '<<names[i]<<'='<<s.seconds[i]<<'/'<<s.calls[i];
    out<<'\n';
    s.seconds={}; s.calls={}; s.last=now;
}
}
