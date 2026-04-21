// vc_render_bench: Benchmark the live VC3D plane-render path.
//
// Exercises the fused ARGB32 sampler that VC3D's viewer actually uses
// (samplePlaneCompositeBestEffortARGB32 → dispatchCompositeAdaptive →
//  sampleSingleLayerAdaptiveImpl for nL=1), so before/after timings on this
// bench match the hotspots the interactive profiler sees.
//
// Usage:
//   vc_render_bench <volume_path_or_url> [options]
//
// Options:
//   --tile-size N       Tile size in pixels (default: 256)
//   --io-threads N      I/O threads for chunk fetching (default: 8)
//   --hot-gb N          Hot cache budget in GB (default: 8)
//   --iters N           Iterations per test (default: 100)
//   --warm-timeout N    Seconds to wait for cache warm-up (default: 60)
//   --composite N       Number of composite layers (default: 1 = single slice)
//
// Determinism
//   • Tile positions are computed from a fixed lattice (no RNG).
//   • Every test phase starts from a drained IOPool (all prior-frame fetches
//     resolved) so a given tile's timing reflects only steady-state work.
//   • The hot-render phase pre-fetches the full tile set and waits for
//     pipeline drain before timing — what it measures is pure sampler cost.
//
// What the phases mean
//   • "Cold stream"  : every tile is fresh to the cache; times include
//                      fetch+decode+render. Panning-from-zero behaviour.
//   • "Hot render"   : all tiles fully cached; times are sampler cost only.
//                      Steady-state in-viewport frame cost.
//   • "Pan stream"   : tiles stream in as you slide; tile N's fetches start
//                      while tile N-1 renders. Realistic panning.
//   • "Z-scroll"     : same (x,y) at N successive z offsets; exercises
//                      z-axis chunk boundaries and nearby-block hit rate.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

using Clock = std::chrono::steady_clock;

namespace {

// Fused ARGB32 plane render — same path CAdaptiveVolumeViewer hits on
// every paint. Route through samplePlaneCompositeBestEffortARGB32 with
// numLayers=1 so we reach dispatchCompositeAdaptive → the Nearest-mode
// sampleSingleLayerAdaptiveImpl specialization that VC3D uses.
// Returns elapsed seconds.
double renderTileARGB32(
    Volume* vol,
    uint32_t* outBuf,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    int tileW, int tileH,
    int level,
    const uint32_t lut[256])
{
    (void)level;  // level selection is handled by the adaptive dispatcher
    const cv::Vec3f normal(0, 0, 1);  // single-layer: normal direction irrelevant
    const auto t0 = Clock::now();
    vol->samplePlaneCompositeBestEffortARGB32(
        outBuf, tileW, origin, vxStep, vyStep, normal,
        /*zStep=*/1.0f, /*zStart=*/0, /*numLayers=*/1,
        tileW, tileH, "mean", lut);
    const auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// Fused ARGB32 composite render (numLayers>1) — routes through
// sampleAdaptiveARGB32 so we hit the same code VC3D runs (vs. the legacy
// samplePlaneCompositeARGB32 path). Matches CAdaptiveVolumeViewer.cpp:520.
double renderTileCompositeARGB32(
    Volume* vol,
    uint32_t* outBuf,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    const cv::Vec3f& normal,
    int numLayers, int zStart, float zStep,
    int tileW, int tileH,
    const uint32_t lut[256])
{
    const auto t0 = Clock::now();
    // Mirror VC3D's path: Nearest for composite (averaging is the low-pass).
    sampleAdaptiveARGB32(
        outBuf, tileW, vol->tieredCache(),
        /*desiredLevel=*/0, /*numLevels=*/int(vol->numScales()),
        /*coords=*/nullptr, &origin, &vxStep, &vyStep,
        /*normals=*/nullptr, &normal,
        numLayers, zStart, zStep,
        tileW, tileH, "mean", lut,
        vc::Sampling::Nearest,
        /*lightParams=*/nullptr,
        /*levelOut=*/nullptr, /*levelStride=*/0);
    const auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

struct BenchResult {
    std::string name;
    std::vector<double> times;  // per-iteration seconds

    int iterations() const { return static_cast<int>(times.size()); }
    double totalSec() const {
        return std::accumulate(times.begin(), times.end(), 0.0);
    }
    double mean() const {
        return iterations() ? totalSec() / iterations() : 0.0;
    }
    double pct(double p) const {
        if (times.empty()) return 0.0;
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min(sorted.size() - 1,
                              static_cast<size_t>(p * sorted.size()));
        return sorted[idx];
    }
    double stdev() const {
        if (iterations() < 2) return 0.0;
        double m = mean();
        double s = 0.0;
        for (double t : times) s += (t - m) * (t - m);
        return std::sqrt(s / (iterations() - 1));
    }
};

void printResult(const BenchResult& r) {
    printf("  %-22s %4d iters  avg %6.2f ms  p50 %6.2f  p99 %6.2f  stdev %6.2f  throughput %7.1f fps\n",
           r.name.c_str(), r.iterations(),
           r.mean() * 1000.0,
           r.pct(0.50) * 1000.0,
           r.pct(0.99) * 1000.0,
           r.stdev() * 1000.0,
           1.0 / std::max(r.mean(), 1e-9));
}

// Block until the pipeline has no pending I/O or a timeout elapses. Returns
// true if drained cleanly. Used to ensure bench phases start from a
// reproducible cache state.
bool waitForDrain(vc::cache::BlockPipeline* cache, double timeoutSec) {
    if (!cache) return true;
    const auto deadline = Clock::now() + std::chrono::duration<double>(timeoutSec);
    while (Clock::now() < deadline) {
        if (cache->stats().ioPending == 0) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return cache->stats().ioPending == 0;
}

// Identity gray LUT so the LUT step is a straight passthrough and doesn't
// inject colourmap cost into the sampler timing.
void buildIdentityLut(uint32_t lut[256]) {
    for (int i = 0; i < 256; ++i) {
        const uint32_t v = static_cast<uint32_t>(i);
        lut[i] = 0xFF000000u | (v << 16) | (v << 8) | v;
    }
}

bool isRemoteUrl(const std::string& path) {
    return path.starts_with("s3://") ||
           path.starts_with("s3+") ||
           path.starts_with("http://") ||
           path.starts_with("https://");
}

// Deterministic lattice of tile origins around (cx, cy, cz). Returns
// iterations positions as (panX, panY, zOff) in plane-local coords.
struct TileSpec { float panX, panY, zOff; };
std::vector<TileSpec> latticeTiles(int iterations, int tileW, int tileH) {
    std::vector<TileSpec> out;
    out.reserve(iterations);
    // Fixed 10x10 grid around centre, walked in scan order. Falls back to
    // wrapping for iterations>100 so the test length is tunable.
    for (int i = 0; i < iterations; ++i) {
        const int row = (i / 10) % 10 - 5;
        const int col = (i % 10) - 5;
        out.push_back({ float(col) * tileW, float(row) * tileH, 0.0f });
    }
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: vc_render_bench <volume_path_or_url> [--tile-size N] "
            "[--io-threads N] [--hot-gb N] [--iters N] [--warm-timeout N] "
            "[--composite N]\n");
        return 1;
    }

    std::string volumePath = argv[1];
    int tileSize = 256;
    int ioThreads = 8;
    int hotGb = 8;
    int iters = 100;
    double warmTimeout = 60.0;
    int compositeLayers = 1;
    std::array<float, 3> centerVoxel = {-1.f, -1.f, -1.f};  // -1 = use geometric centre

    for (int i = 2; i < argc; i++) {
        std::string_view a = argv[i];
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value\n", what);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--tile-size")    tileSize = std::atoi(need("--tile-size"));
        else if (a == "--io-threads")   ioThreads = std::atoi(need("--io-threads"));
        else if (a == "--hot-gb")       hotGb = std::atoi(need("--hot-gb"));
        else if (a == "--iters")        iters = std::atoi(need("--iters"));
        else if (a == "--warm-timeout") warmTimeout = std::atof(need("--warm-timeout"));
        else if (a == "--composite")    compositeLayers = std::atoi(need("--composite"));
        else if (a == "--center") {
            const char* v = need("--center");
            float z=-1, y=-1, x=-1;
            if (std::sscanf(v, "%f,%f,%f", &z, &y, &x) != 3) {
                fprintf(stderr, "--center expects Z,Y,X (got '%s')\n", v);
                return 1;
            }
            centerVoxel = {z, y, x};
        }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    printf("vc_render_bench\n");
    printf("  Volume:       %s\n", volumePath.c_str());
    printf("  Tile size:    %d x %d\n", tileSize, tileSize);
    printf("  IO threads:   %d\n", ioThreads);
    printf("  Hot cache:    %d GB\n", hotGb);
    printf("  Iterations:   %d\n", iters);
    printf("  Warm timeout: %.1f s\n", warmTimeout);
    printf("  Composite:    %d layer(s)\n", compositeLayers);
    printf("\n");

    // Open volume
    printf("Opening volume...\n");
    const auto t0 = Clock::now();
    std::shared_ptr<Volume> vol;
    if (isRemoteUrl(volumePath)) vol = Volume::NewFromUrl(volumePath);
    else                          vol = Volume::New(volumePath);
    vol->setCacheBudget(static_cast<size_t>(hotGb) << 30);
    vol->setIOThreads(ioThreads);
    auto shape = vol->shape();
    const int numLevels = static_cast<int>(vol->numScales());
    const auto t1 = Clock::now();
    printf("  Shape: %d x %d x %d  (%d pyramid levels)\n",
           shape[0], shape[1], shape[2], numLevels);
    printf("  Open time: %.2f s\n\n",
           std::chrono::duration<double>(t1 - t0).count());

    auto* cache = vol->tieredCache();

    // Plane oriented along +Z through the chosen centre (defaults to
    // volume centre). For sparse volumes, pass --center to point at a
    // region with actual data.
    const float cx = (centerVoxel[0] >= 0) ? centerVoxel[0] : shape[0] / 2.0f;
    const float cy = (centerVoxel[1] >= 0) ? centerVoxel[1] : shape[1] / 2.0f;
    const float cz = (centerVoxel[2] >= 0) ? centerVoxel[2] : shape[2] / 2.0f;
    printf("  Plane centre: %.1f, %.1f, %.1f\n\n", cx, cy, cz);
    PlaneSurface plane(cv::Vec3f(cx, cy, cz), cv::Vec3f(0, 0, 1));

    const cv::Vec3f vx = plane.basisX();
    const cv::Vec3f vy = plane.basisY();
    const cv::Vec3f origin0 = plane.origin();
    const cv::Vec3f normal = plane.normal(cv::Vec3f(0, 0, 0));

    auto makeTileOrigin = [&](const TileSpec& t) -> cv::Vec3f {
        return origin0 + vx * t.panX + vy * t.panY + normal * t.zOff;
    };

    const int tileW = tileSize, tileH = tileSize;
    std::vector<uint32_t> tileBuf(size_t(tileW) * tileH);
    uint32_t lut[256]; buildIdentityLut(lut);

    const auto tiles = latticeTiles(iters, tileW, tileH);
    std::vector<BenchResult> results;

    auto renderOneTile = [&](const TileSpec& t, int level) -> double {
        cv::Vec3f tileOrigin = makeTileOrigin(t);
        if (compositeLayers <= 1) {
            return renderTileARGB32(vol.get(), tileBuf.data(),
                tileOrigin, vx, vy, tileW, tileH, level, lut);
        } else {
            return renderTileCompositeARGB32(vol.get(), tileBuf.data(),
                tileOrigin, vx, vy, normal,
                compositeLayers, -compositeLayers / 2, 1.0f,
                tileW, tileH, lut);
        }
    };

    // ==== Phase 1: warmup + populate cache ====
    printf("Warmup: streaming %d tiles into cache...\n", iters);
    for (const auto& t : tiles) renderOneTile(t, 0);
    if (!waitForDrain(cache, warmTimeout)) {
        printf("  WARN: pipeline didn't fully drain in %.1fs "
               "(pending=%zu); hot-render numbers may include I/O latency.\n",
               warmTimeout, cache ? cache->stats().ioPending : 0);
    } else {
        printf("  Drained in %.2f s.\n",
               std::chrono::duration<double>(Clock::now() - t1).count());
    }
    printf("\n");

    // ==== Phase 2: Hot render — all tiles resident, pure sampler cost ====
    {
        BenchResult r; r.name = "Hot render";
        for (const auto& t : tiles) r.times.push_back(renderOneTile(t, 0));
        results.push_back(std::move(r));
    }

    // ==== Phase 3: Hot render, coarse level (pyramid) ====
    if (numLevels >= 3) {
        const int level = 2;
        BenchResult r; r.name = "Hot render level 2";
        for (const auto& t : tiles) r.times.push_back(renderOneTile(t, level));
        results.push_back(std::move(r));
    }

    // ==== Phase 4: Pan stream — slide +1 tileW per iter, fresh data each ====
    // Pre-clear so tiles are cold. clearAll wipes caches but keeps disk.
    if (cache) cache->clearAll();
    {
        BenchResult r; r.name = "Pan stream (cold)";
        for (int i = 0; i < iters; ++i) {
            TileSpec t{ float(i) * tileW, 0.0f, 0.0f };
            r.times.push_back(renderOneTile(t, 0));
        }
        results.push_back(std::move(r));
    }

    // ==== Phase 5: Z-scroll — 20 z steps through centre ====
    if (cache) cache->clearAll();
    {
        BenchResult r; r.name = "Z-scroll (cold)";
        for (int i = 0; i < std::min(iters, 50); ++i) {
            TileSpec t{ 0.0f, 0.0f, float(i - 25) };
            r.times.push_back(renderOneTile(t, 0));
        }
        results.push_back(std::move(r));
    }

    // ==== Report ====
    printf("Results:\n");
    for (const auto& r : results) printResult(r);

    if (cache) {
        auto s = cache->stats();
        const uint64_t total = std::max<uint64_t>(1,
            s.blockHits + s.coldHits + s.iceFetches + s.misses);
        auto pct = [&](uint64_t n) { return 100.0 * n / total; };
        printf("\nCache stats:\n");
        printf("  Block hits:  %8llu  (%5.1f%%)\n", (unsigned long long)s.blockHits, pct(s.blockHits));
        printf("  Cold hits:   %8llu  (%5.1f%%)\n", (unsigned long long)s.coldHits,  pct(s.coldHits));
        printf("  Ice fetches: %8llu  (%5.1f%%)\n", (unsigned long long)s.iceFetches, pct(s.iceFetches));
        printf("  Misses:      %8llu  (%5.1f%%)\n", (unsigned long long)s.misses,    pct(s.misses));
        printf("  Shard hits:  %8llu    Shard misses: %llu\n",
               (unsigned long long)s.shardHits, (unsigned long long)s.shardMisses);
        printf("  Disk writes: %8llu    Disk bytes:   %.1f MB\n",
               (unsigned long long)s.diskWrites, s.diskBytes / (1024.0 * 1024.0));
    }

    printf("\nDone.\n");
    return 0;
}
