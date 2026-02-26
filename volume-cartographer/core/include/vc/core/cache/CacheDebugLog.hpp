#pragma once

#include <cstdio>
#include <cstdlib>

namespace vc::cache {

inline FILE* cacheDebugLog()
{
    static FILE* f = [] () -> FILE* {
        const char* path = std::getenv("VC_CACHE_DEBUG_LOG");
        if (!path || path[0] == '\0') return nullptr;
        FILE* fp = std::fopen(path, "a");
        if (fp) std::setvbuf(fp, nullptr, _IOLBF, 0);  // line-buffered
        return fp;
    }();
    return f;
}

}  // namespace vc::cache
