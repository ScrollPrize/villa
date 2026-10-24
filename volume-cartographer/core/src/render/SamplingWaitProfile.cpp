#include "SamplingWaitProfile.hpp"

#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace vc::render::detail {

SamplingWaitStats& samplingWaitStats()
{
    static thread_local SamplingWaitStats stats;
    return stats;
}

void flushSamplingWaits()
{
    if (!samplingWaitProfileEnabled())
        return;
    auto& stats = samplingWaitStats();
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.last < std::chrono::seconds(10))
        return;
    constexpr const char* names[] = {
        "blocking_total", "blocking_lock", "prefetch_total", "prefetch_lock",
        "cached_total", "cached_lock", "corner_prepare_layout", "corner_pin",
        "corner_gather", "payload_wait", "entry_budget", "shared_budget"};
    std::ostringstream out;
    out << "sampling-waits thread=" << std::this_thread::get_id();
    for (size_t i = 0; i < stats.seconds.size(); ++i)
        out << ' ' << names[i] << '=' << stats.seconds[i] << '/' << stats.calls[i];
    out << '\n';
    // Serialize complete diagnostic lines without requiring libc++ syncstream.
    // This is called outside cache locks and only when logging is enabled.
    static std::mutex outputMutex;
    {
        std::lock_guard lock(outputMutex);
        std::cerr << out.str();
    }
    stats.seconds = {};
    stats.calls = {};
    stats.last = now;
}

} // namespace vc::render::detail
