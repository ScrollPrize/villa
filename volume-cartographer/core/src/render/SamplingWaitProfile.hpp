#pragma once

// Opt-in attribution for batch CLI diagnostics. No shared counters or changes
// to locking/scheduling. Flush only outside cache locks, from the sampler.
#include <array>
#include <chrono>
#include <cstdlib>

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
// Defined in one translation unit to avoid duplicate dynamic TLS
// initializers on MinGW. No state is constructed while profiling is disabled.
SamplingWaitStats& samplingWaitStats();
inline void recordSamplingWait(size_t phase,double seconds) {
    if (!samplingWaitProfileEnabled()) return;
    auto& stats = samplingWaitStats();
    stats.seconds[phase] += seconds;
    ++stats.calls[phase];
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
void flushSamplingWaits();
}
