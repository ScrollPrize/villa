// vc_render_bench: Benchmark the volume rendering pipeline.
//
// Exercises the same sampling paths used by TileRenderer (fused plane,
// coordinate-based, best-effort pyramid fallback) without Qt.
//
// Usage:
//   vc_render_bench <volume_path_or_url> [options]
//
// Input can be:
//   /path/to/volume                (local filesystem)
//   s3://bucket/path/volume.zarr   (S3, uses AWS env credentials)
//   s3+us-east-1://bucket/...      (S3 with explicit region)
//   https://...                    (HTTP remote zarr)
//
// Options:
//   --tile-size N    Tile size in pixels (default: 256)
//   --io-threads N   I/O threads for chunk fetching (default: 8)
//   --hot-gb N       Hot cache budget in GB (default: 8)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/util/PlaneSurface.hpp"

using Clock = std::chrono::high_resolution_clock;

// Render one tile using the fused plane path (same as TileRenderer for PlaneSurface).
// Returns the elapsed time in seconds.
static double renderPlaneTile(
    Volume* vol,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    int tileW, int tileH,
    int level)
{
    cv::Mat_<uint8_t> gray;
    vc::SampleParams sp;
    sp.level = level;
    sp.method = (level >= 3) ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

    auto t0 = Clock::now();
    vol->samplePlaneBestEffort(gray, origin, vxStep, vyStep, tileW, tileH, sp);
    auto t1 = Clock::now();

    return std::chrono::duration<double>(t1 - t0).count();
}

// Render one tile using the coordinate-based path (for QuadSurface compatibility).
static double renderCoordTile(
    Volume* vol,
    PlaneSurface& plane,
    float surfX, float surfY, float zOff,
    float scale,
    int tileW, int tileH,
    int level)
{
    cv::Mat_<cv::Vec3f> coords;
    plane.gen(&coords, nullptr, cv::Size(tileW, tileH),
              cv::Vec3f(0, 0, 0), scale,
              {surfX * scale, surfY * scale, zOff});

    cv::Mat_<uint8_t> gray;
    vc::SampleParams sp;
    sp.level = level;
    sp.method = (level >= 3) ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

    auto t0 = Clock::now();
    vol->sampleBestEffort(gray, coords, sp);
    auto t1 = Clock::now();

    return std::chrono::duration<double>(t1 - t0).count();
}

struct BenchResult {
    std::string name;
    int tileCount;
    std::vector<double> times;  // per-tile seconds

    double totalSec() const {
        return std::accumulate(times.begin(), times.end(), 0.0);
    }
    double tilesPerSec() const {
        return tileCount / totalSec();
    }
    double avgMs() const {
        return totalSec() / tileCount * 1000.0;
    }
    double p99Ms() const {
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        int idx = std::max(0, (int)(sorted.size() * 0.99) - 1);
        return sorted[idx] * 1000.0;
    }
};

static void printResult(const BenchResult& r, int tileW, int tileH) {
    double voxelsPerTile = (double)tileW * tileH;
    double totalVoxels = voxelsPerTile * r.tileCount;
    double totalSec = r.totalSec();
    double voxelsPerSec = totalVoxels / totalSec;
    double mbPerSec = voxelsPerSec / (1024.0 * 1024.0);  // 1 byte/voxel (uint8)

    printf("  %-30s %6d tiles  %8.1f tiles/s  %6.2f ms/tile avg  %6.2f ms/tile p99  %8.1f Mvox/s  %7.1f MB/s\n",
           r.name.c_str(), r.tileCount,
           r.tilesPerSec(), r.avgMs(), r.p99Ms(),
           voxelsPerSec / 1e6, mbPerSec);
}

static void printCacheStats(vc::cache::TieredChunkCache* cache) {
    auto s = cache->stats();
    uint64_t total = s.hotHits + s.coldHits + s.iceFetches + s.misses;
    if (total == 0) total = 1;

    auto pct = [&](uint64_t n) { return 100.0 * n / total; };

    printf("\n  Cache stats:\n");
    printf("    Hot  hits:  %8llu  (%5.1f%%)\n", (unsigned long long)s.hotHits,  pct(s.hotHits));
    printf("    Cold hits:  %8llu  (%5.1f%%)\n", (unsigned long long)s.coldHits, pct(s.coldHits));
    printf("    Ice fetch:  %8llu  (%5.1f%%)\n", (unsigned long long)s.iceFetches, pct(s.iceFetches));
    printf("    Misses:     %8llu  (%5.1f%%)\n", (unsigned long long)s.misses,  pct(s.misses));
    printf("    Hot bytes:  %8.1f MB\n", s.hotBytes / (1024.0 * 1024.0));
    printf("    IO pending: %8zu\n", s.ioPending);
    printf("    Disk writes:%8llu\n", (unsigned long long)s.diskWrites);
}

static bool isRemoteUrl(const std::string& path) {
    return path.starts_with("s3://") ||
           path.starts_with("s3+") ||
           path.starts_with("http://") ||
           path.starts_with("https://");
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: vc_render_bench <volume_path_or_url> [--tile-size N] [--io-threads N] [--hot-gb N]\n");
        return 1;
    }

    std::string volumePath = argv[1];
    int tileSize = 256;
    int ioThreads = 8;
    int hotGb = 8;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc)
            tileSize = atoi(argv[++i]);
        else if (strcmp(argv[i], "--io-threads") == 0 && i + 1 < argc)
            ioThreads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hot-gb") == 0 && i + 1 < argc)
            hotGb = atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    printf("vc_render_bench\n");
    printf("  Volume:     %s\n", volumePath.c_str());
    printf("  Tile size:  %d x %d\n", tileSize, tileSize);
    printf("  IO threads: %d\n", ioThreads);
    printf("  Hot cache:  %d GB\n", hotGb);
    printf("\n");

    // Open volume
    printf("Opening volume...\n");
    auto t0 = Clock::now();

    std::shared_ptr<Volume> vol;
    if (isRemoteUrl(volumePath)) {
        vol = Volume::NewFromUrl(volumePath);
    } else {
        vol = Volume::New(volumePath);
    }

    vol->setCacheBudget((size_t)hotGb << 30);
    vol->setIOThreads(ioThreads);

    auto shape = vol->shape();
    int numLevels = (int)vol->numScales();
    auto t1 = Clock::now();
    printf("  Shape: %d x %d x %d  (%d pyramid levels)\n",
           shape[0], shape[1], shape[2], numLevels);
    printf("  Open time: %.2f s\n\n", std::chrono::duration<double>(t1 - t0).count());

    // Set up a PlaneSurface centered in the volume
    float cx = shape[0] / 2.0f;
    float cy = shape[1] / 2.0f;
    float cz = shape[2] / 2.0f;

    PlaneSurface plane(cv::Vec3f(cx, cy, cz), cv::Vec3f(0, 0, 1));

    // Helper: compute fused plane parameters for a given tile position and zoom
    auto makePlaneParams = [&](float panX, float panY, float zOff, float scale)
        -> std::tuple<cv::Vec3f, cv::Vec3f, cv::Vec3f>
    {
        float m = 1.0f / scale;
        cv::Vec3f vx = plane.basisX();
        cv::Vec3f vy = plane.basisY();
        cv::Vec3f origin = plane.origin() + plane.normal(cv::Vec3f(0,0,0)) * zOff;
        cv::Vec3f vxStep = vx * m;
        cv::Vec3f vyStep = vy * m;
        cv::Vec3f planeOrigin = vx * panX + vy * panY + origin;
        return {planeOrigin, vxStep, vyStep};
    };

    int tileW = tileSize;
    int tileH = tileSize;

    std::vector<BenchResult> results;

    // ==== Warmup: 100 tiles at default view ====
    {
        printf("Warmup: 100 tiles at center...\n");
        auto [origin, vxStep, vyStep] = makePlaneParams(0, 0, 0, 1.0f);
        for (int i = 0; i < 100; i++) {
            float offX = (float)((i % 10) - 5) * tileW;
            float offY = (float)((i / 10) - 5) * tileH;
            cv::Vec3f tileOrigin = origin + vxStep * offX + vyStep * offY;
            renderPlaneTile(vol.get(), tileOrigin, vxStep, vyStep, tileW, tileH, 0);
        }
        printf("  Done.\n\n");
    }

    // ==== Test 1: 100 tiles at zoom level 0 (full res) ====
    {
        BenchResult r;
        r.name = "Zoom 0 (full res)";
        r.tileCount = 100;
        auto [origin, vxStep, vyStep] = makePlaneParams(0, 0, 0, 1.0f);
        for (int i = 0; i < 100; i++) {
            float offX = (float)((i % 10) - 5) * tileW;
            float offY = (float)((i / 10) - 5) * tileH;
            cv::Vec3f tileOrigin = origin + vxStep * offX + vyStep * offY;
            r.times.push_back(renderPlaneTile(vol.get(), tileOrigin, vxStep, vyStep, tileW, tileH, 0));
        }
        results.push_back(r);
    }

    // ==== Test 2: 100 tiles at zoom level 2 (4x downsampled) ====
    {
        int level = std::min(2, numLevels - 1);
        BenchResult r;
        r.name = "Zoom 2 (4x downsample)";
        r.tileCount = 100;
        float scale = 1.0f / (1 << level);  // 0.25 at level 2
        auto [origin, vxStep, vyStep] = makePlaneParams(0, 0, 0, scale);
        for (int i = 0; i < 100; i++) {
            float offX = (float)((i % 10) - 5) * tileW;
            float offY = (float)((i / 10) - 5) * tileH;
            cv::Vec3f tileOrigin = origin + vxStep * offX + vyStep * offY;
            r.times.push_back(renderPlaneTile(vol.get(), tileOrigin, vxStep, vyStep, tileW, tileH, level));
        }
        results.push_back(r);
    }

    // ==== Test 3: Pan sequence (50 adjacent tiles) ====
    {
        BenchResult r;
        r.name = "Pan (50 adjacent)";
        r.tileCount = 50;
        auto [origin, vxStep, vyStep] = makePlaneParams(0, 0, 0, 1.0f);
        for (int i = 0; i < 50; i++) {
            float offX = (float)i * tileW;
            cv::Vec3f tileOrigin = origin + vxStep * offX;
            r.times.push_back(renderPlaneTile(vol.get(), tileOrigin, vxStep, vyStep, tileW, tileH, 0));
        }
        results.push_back(r);
    }

    // ==== Test 4: Z-scroll sequence (20 slices at same XY position) ====
    {
        BenchResult r;
        r.name = "Z-scroll (20 slices)";
        r.tileCount = 20;
        for (int i = 0; i < 20; i++) {
            float zOff = (float)(i - 10);
            auto [origin, vxStep, vyStep] = makePlaneParams(0, 0, zOff, 1.0f);
            r.times.push_back(renderPlaneTile(vol.get(), origin, vxStep, vyStep, tileW, tileH, 0));
        }
        results.push_back(r);
    }

    // ==== Test 5: Zoom sequence (10 tiles at 10 different zoom levels) ====
    {
        BenchResult r;
        r.name = "Zoom sequence (10 levels)";
        r.tileCount = 10;
        for (int i = 0; i < 10; i++) {
            // Scale from 0.1 to 4.0 in 10 steps
            float scale = 0.1f + (4.0f - 0.1f) * i / 9.0f;
            int level = 0;
            // Pick appropriate pyramid level for this scale
            float s = scale;
            while (s < 0.5f && level + 1 < numLevels) {
                s *= 2.0f;
                level++;
            }
            auto [origin, vxStep, vyStep] = makePlaneParams(0, 0, 0, scale);
            r.times.push_back(renderPlaneTile(vol.get(), origin, vxStep, vyStep, tileW, tileH, level));
        }
        results.push_back(r);
    }

    // ==== Report ====
    printf("Results:\n");
    printf("  %-30s %6s  %13s  %17s  %17s  %12s  %9s\n",
           "Test", "Tiles", "Tiles/s", "Avg ms/tile", "P99 ms/tile", "Mvox/s", "MB/s");
    printf("  %s\n", std::string(130, '-').c_str());
    for (auto& r : results)
        printResult(r, tileW, tileH);

    // Cache stats
    auto* cache = vol->tieredCache();
    if (cache)
        printCacheStats(cache);

    printf("\nDone.\n");
    return 0;
}
