#include "RamStats.hpp"

#include "CState.hpp"
#include "ViewerManager.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/cache/BlockPipeline.hpp"

#include <cstdio>
#include <fstream>
#include <string>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(VC_HAVE_MIMALLOC)
#include <mimalloc.h>
#endif

namespace {

struct ProcStatus {
    long vmRssKB = 0;
    long vmSizeKB = 0;
    long vmHwmKB = 0;
    long vmSwapKB = 0;
};

ProcStatus readProcStatus()
{
    ProcStatus s;
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        auto parseKB = [&](const char* prefix, long& out) {
            const auto n = std::char_traits<char>::length(prefix);
            if (line.rfind(prefix, 0) == 0) {
                out = std::strtol(line.c_str() + n, nullptr, 10);
            }
        };
        parseKB("VmRSS:", s.vmRssKB);
        parseKB("VmSize:", s.vmSizeKB);
        parseKB("VmHWM:", s.vmHwmKB);
        parseKB("VmSwap:", s.vmSwapKB);
    }
#endif
    return s;
}

std::size_t estimateLoadedSurfaceBytes(CState* state)
{
    if (!state) return 0;
    std::size_t total = 0;
    for (const auto& surf : state->surfaces()) {
        auto quad = std::dynamic_pointer_cast<QuadSurface>(surf);
        if (!quad) continue;
        if (!quad->isLoaded()) continue;
        const cv::Mat_<cv::Vec3f>* p = quad->rawPointsPtr();
        if (!p) continue;
        total += static_cast<std::size_t>(p->rows) * p->cols * sizeof(cv::Vec3f);
    }
    return total;
}

}  // namespace

namespace vc3d::ramstats {

void dumpOnce(ViewerManager* viewerManager, CState* state)
{
    const ProcStatus ps = readProcStatus();
#if defined(__GLIBC__) && !defined(VC_HAVE_MIMALLOC)
    // mallinfo2 is glibc-only (>=2.33). Skipped on macOS / musl / with
    // mimalloc (which overrides malloc and makes the legacy struct noise).
    const struct mallinfo2 mi = mallinfo2();
#else
    struct { int uordblks=0, fordblks=0, hblkhd=0; } mi;
    (void)mi;
#endif

    std::size_t blockBytes = 0;
    std::size_t patchCount = 0;
    std::size_t surfaceCount = 0;
    if (viewerManager) {
        if (auto* vol = state ? state->currentVolume().get() : nullptr) {
            if (auto* bp = vol->tieredCache()) {
                const auto stats = bp->stats();
                blockBytes = stats.blocks * 4096;
            }
        }
        if (auto* spi = viewerManager->surfacePatchIndex()) {
            patchCount = spi->patchCount();
            surfaceCount = spi->surfaceCount();
        }
    }

    const std::size_t surfaceBytes = estimateLoadedSurfaceBytes(state);

#if defined(VC_HAVE_MIMALLOC)
    // std::fprintf(stderr,
    //     "[RAM] rss=%ldMB hwm=%ldMB swap=%ldMB"
    //     " | blocks=%zuMB | spi=%zu_patches/%zu_surfs | surface_points=%zuMB\n",
    //     ps.vmRssKB / 1024,
    //     ps.vmHwmKB / 1024,
    //     ps.vmSwapKB / 1024,
    //     blockBytes / (1024*1024),
    //     patchCount,
    //     surfaceCount,
    //     surfaceBytes / (1024*1024));
#else
    // std::fprintf(stderr,
    //     "[RAM] rss=%ldMB hwm=%ldMB swap=%ldMB | malloc:in_use=%zuMB free=%zuMB mmap=%zuMB"
    //     " | blocks=%zuMB | spi=%zu_patches/%zu_surfs | surface_points=%zuMB\n",
    //     ps.vmRssKB / 1024,
    //     ps.vmHwmKB / 1024,
    //     ps.vmSwapKB / 1024,
    //     std::size_t(mi.uordblks) / (1024*1024),
    //     std::size_t(mi.fordblks) / (1024*1024),
    //     std::size_t(mi.hblkhd) / (1024*1024),
    //     blockBytes / (1024*1024),
    //     patchCount,
    //     surfaceCount,
    //     surfaceBytes / (1024*1024));
#endif
    (void)ps; (void)mi; (void)blockBytes; (void)patchCount;
    (void)surfaceCount; (void)surfaceBytes;
}

}  // namespace vc3d::ramstats
