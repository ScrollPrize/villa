#pragma once

#include <cstdio>
#include <cstdlib>

namespace vc::cache {

// RAII wrapper that closes the FILE* on destruction (i.e., program exit).
struct CacheDebugLogHandle {
    FILE* fp = nullptr;
    explicit CacheDebugLogHandle(FILE* f) : fp(f) {}
    ~CacheDebugLogHandle() { if (fp) std::fclose(fp); }
    CacheDebugLogHandle(const CacheDebugLogHandle&) = delete;
    CacheDebugLogHandle& operator=(const CacheDebugLogHandle&) = delete;
};

inline FILE* cacheDebugLog()
{
    static CacheDebugLogHandle handle = [] {
        const char* path = std::getenv("VC_CACHE_DEBUG_LOG");
        if (!path || path[0] == '\0') return CacheDebugLogHandle{nullptr};
        FILE* fp = std::fopen(path, "a");
        if (fp) std::setvbuf(fp, nullptr, _IOLBF, 0);  // line-buffered
        return CacheDebugLogHandle{fp};
    }();
    return handle.fp;
}

}  // namespace vc::cache
