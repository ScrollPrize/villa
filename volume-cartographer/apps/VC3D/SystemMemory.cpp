#include "SystemMemory.hpp"

#include "VCSettings.hpp"

#include <algorithm>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#else
#  include <unistd.h>
#endif

namespace vc3d {

std::uint64_t physicalMemoryBytes()
{
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<std::uint64_t>(status.ullTotalPhys);
    }
    return 0;
#elif defined(__APPLE__)
    std::uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0) {
        return bytes;
    }
    return 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && pageSize > 0) {
        return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(pageSize);
    }
    return 0;
#endif
}

int defaultRamCacheSizeGB()
{
    constexpr int maxDefault = settings::perf::RAM_CACHE_SIZE_GB_DEFAULT;
    const std::uint64_t gib = physicalMemoryBytes() >> 30;
    if (gib == 0) {
        return maxDefault;
    }
    return std::clamp(static_cast<int>(gib / 3), 2, maxDefault);
}

}  // namespace vc3d
