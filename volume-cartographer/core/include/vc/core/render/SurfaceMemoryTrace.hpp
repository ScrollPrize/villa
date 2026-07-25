#pragma once

#include "vc/core/util/Logging.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

namespace vc::render {

// Opt-in diagnostics for the complete SurfaceCache render pipeline. This is
// intentionally independent of the normal profile flag: memory diagnosis is
// noisy and reads /proc plus allocator counters, so it must have zero work in
// ordinary builds unless explicitly requested.
//
//   VC_SURFACE_CACHE_MEMORY_TRACE=1
//   VC_SURFACE_CACHE_MEMORY_TRACE_FILL_EVERY=64  (optional; 1 logs every fill)
//
// Each record includes process RSS plus glibc's live arena, free arena and
// direct-mmap accounting. Comparing those numbers with the event's explicit
// owner sizes tells us whether bytes are still owned by the pipeline or merely
// retained by the allocator.
inline bool surfaceMemoryTraceEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("VC_SURFACE_CACHE_MEMORY_TRACE");
        if (!value || !*value)
            return false;
        const std::string_view text(value);
        return text != "0" && text != "false" && text != "FALSE" &&
               text != "off" && text != "OFF";
    }();
    return enabled;
}

inline std::size_t surfaceMemoryTraceFillEvery()
{
    static const std::size_t every = [] {
        const char* value =
            std::getenv("VC_SURFACE_CACHE_MEMORY_TRACE_FILL_EVERY");
        if (!value || !*value)
            return std::size_t{64};
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end != value && parsed > 0 ? std::size_t(parsed)
                                         : std::size_t{64};
    }();
    return every;
}

inline bool surfaceMemoryTraceSampleFill(std::uint64_t ordinal)
{
    return surfaceMemoryTraceEnabled() &&
           ordinal % surfaceMemoryTraceFillEvery() == 0;
}

struct SurfaceMemorySnapshot {
    std::size_t rssBytes = 0;
    std::size_t arenaLiveBytes = 0;
    std::size_t arenaFreeBytes = 0;
    std::size_t mmapBytes = 0;
};

inline SurfaceMemorySnapshot surfaceMemorySnapshot()
{
    SurfaceMemorySnapshot snapshot;
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long long totalPages = 0;
    long long residentPages = 0;
    if (statm >> totalPages >> residentPages && residentPages > 0) {
        const long pageSize = ::sysconf(_SC_PAGESIZE);
        if (pageSize > 0) {
            snapshot.rssBytes =
                std::size_t(residentPages) * std::size_t(pageSize);
        }
    }
#endif
#if defined(__GLIBC__)
    const struct mallinfo2 info = ::mallinfo2();
    snapshot.arenaLiveBytes = static_cast<std::size_t>(info.uordblks);
    snapshot.arenaFreeBytes = static_cast<std::size_t>(info.fordblks);
    snapshot.mmapBytes = static_cast<std::size_t>(info.hblkhd);
#endif
    return snapshot;
}

inline void surfaceMemoryTrace(std::string_view event,
                               std::string_view details = {})
{
    if (!surfaceMemoryTraceEnabled())
        return;

    static const auto started = std::chrono::steady_clock::now();
    static std::atomic_uint64_t sequence{0};
    const auto snapshot = surfaceMemorySnapshot();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    const auto thread =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    Logger()->info(
        "[surface-mem] seq={} ms={} tid={} event={} rss={} arena_live={} "
        "arena_free={} mmap={} {}",
        sequence.fetch_add(1, std::memory_order_relaxed), elapsedMs, thread,
        event, snapshot.rssBytes, snapshot.arenaLiveBytes,
        snapshot.arenaFreeBytes, snapshot.mmapBytes, details);
}

} // namespace vc::render
