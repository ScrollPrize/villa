// vc_io_bench: Benchmark the chunk I/O + decode pipeline.
//
// Measures end-to-end "submit N chunk keys → all arrive in BlockCache"
// throughput across the downloader → encoder → loader → decoder chain that
// BlockPipeline runs on every cold viewport. The bench deliberately
// enumerates a contiguous 3D region of chunks (not a single tile's worth)
// to stress parallel fetch and shard-cache hit rates the way an active
// streaming session does.
//
// What the phases mean
//   • "Cold pipeline": start from a fully cleared pipeline (blocks + shard
//                      cache + negative cache all wiped). Measures the
//                      worst case: every shard must be fetched, every chunk
//                      decoded, every block inserted. This is what the user
//                      sees the first time they open a new volume.
//   • "Warm shards" :  start with shard cache populated but block cache
//                      empty. Measures decode + insert without network
//                      transfer — how fast the CPU side of the pipeline is.
//   • "Fully warm"  :  start with block cache populated. Measures the
//                      no-op/dedup path — most calls should short-circuit
//                      via fetchInteractive's hash dedup and the block
//                      cache's containsBatch.
//
// Usage:
//   vc_io_bench <volume_path_or_url> [options]
//
// Options:
//   --region-size N     Side length of chunk region (default: 8 → 512 chunks)
//   --level N           Pyramid level to fetch (default: 0)
//   --io-threads N      IO threads for downloader pool (default: 16)
//   --hot-gb N          Block cache budget in GB (default: 8)
//   --timeout N         Seconds to wait for a phase to drain (default: 300)
//   --poll-hz N         Stats polling rate in Hz (default: 10)
//   --center Z,Y,X      Centre the fetch region at level-0 voxel (Z,Y,X)
//                       instead of the volume centre. Use this to point the
//                       bench at a known-populated region — sparse volumes
//                       can have all-zero sentinel chunks at the centre
//                       that negative-cache and bypass real I/O.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "vc/core/cache/BlockPipeline.hpp"
#include "vc/core/cache/ChunkKey.hpp"
#include "vc/core/types/Volume.hpp"

using Clock = std::chrono::steady_clock;
namespace {

bool isRemoteUrl(const std::string& p) {
    return p.starts_with("s3://") || p.starts_with("s3+") ||
           p.starts_with("http://") || p.starts_with("https://");
}

double elapsedSec(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

// Snapshot of cache stats + wall time so phase deltas are trivially
// computable by subtracting two snapshots.
struct StatSnapshot {
    Clock::time_point when;
    vc::cache::BlockPipeline::Stats stats;
};

StatSnapshot snap(vc::cache::BlockPipeline* cache) {
    StatSnapshot s;
    s.when = Clock::now();
    if (cache) s.stats = cache->stats();
    return s;
}

// Enumerate a contiguous 3D lattice of chunk keys around centerVoxel (in
// level-0 voxel coords, -1 to use volume centre). Using a lattice (not
// random) keeps runs deterministic and exercises shard-cache locality.
std::vector<vc::cache::ChunkKey> enumerateRegion(
    Volume& vol, vc::cache::BlockPipeline& cache, int level, int regionSize,
    std::array<int, 3> centerVoxel)
{
    auto shape = vol.shape();
    const int sz = std::max(1, shape[0] >> level);
    const int sy = std::max(1, shape[1] >> level);
    const int sx = std::max(1, shape[2] >> level);
    auto cs = cache.chunkShape(level);
    if (cs[0] <= 0 || cs[1] <= 0 || cs[2] <= 0) return {};
    const int chunksZ = (sz + cs[0] - 1) / cs[0];
    const int chunksY = (sy + cs[1] - 1) / cs[1];
    const int chunksX = (sx + cs[2] - 1) / cs[2];

    // Centre chunk: either explicit (scale level-0 voxel coord down) or
    // geometric volume centre.
    const int chunkCenterZ = (centerVoxel[0] >= 0)
        ? std::clamp((centerVoxel[0] >> level) / cs[0], 0, chunksZ - 1)
        : chunksZ / 2;
    const int chunkCenterY = (centerVoxel[1] >= 0)
        ? std::clamp((centerVoxel[1] >> level) / cs[1], 0, chunksY - 1)
        : chunksY / 2;
    const int chunkCenterX = (centerVoxel[2] >= 0)
        ? std::clamp((centerVoxel[2] >> level) / cs[2], 0, chunksX - 1)
        : chunksX / 2;

    const int half = regionSize / 2;
    const int cz0 = std::max(0, chunkCenterZ - half);
    const int cy0 = std::max(0, chunkCenterY - half);
    const int cx0 = std::max(0, chunkCenterX - half);
    const int cz1 = std::min(chunksZ, cz0 + regionSize);
    const int cy1 = std::min(chunksY, cy0 + regionSize);
    const int cx1 = std::min(chunksX, cx0 + regionSize);

    std::vector<vc::cache::ChunkKey> out;
    out.reserve(size_t(cz1 - cz0) * (cy1 - cy0) * (cx1 - cx0));
    for (int iz = cz0; iz < cz1; ++iz)
      for (int iy = cy0; iy < cy1; ++iy)
        for (int ix = cx0; ix < cx1; ++ix)
            out.push_back({level, iz, iy, ix});
    return out;
}

// Wait for the pipeline's ioPending to hit zero (or timeout). Polls at
// pollHz. Returns elapsed seconds — always returns, even on timeout, so
// the caller can decide what to do.
double waitDrain(vc::cache::BlockPipeline* cache, double timeoutSec, int pollHz,
                 Clock::time_point phaseStart)
{
    if (!cache) return 0.0;
    const auto deadline = phaseStart + std::chrono::duration<double>(timeoutSec);
    const auto pollDt = std::chrono::milliseconds(1000 / std::max(1, pollHz));
    while (Clock::now() < deadline) {
        if (cache->stats().ioPending == 0) break;
        std::this_thread::sleep_for(pollDt);
    }
    return elapsedSec(phaseStart, Clock::now());
}

struct PhaseResult {
    std::string name;
    int chunks;
    double seconds;
    StatSnapshot before;
    StatSnapshot after;
};

void printPhase(const PhaseResult& r) {
    const auto& b = r.before.stats;
    const auto& a = r.after.stats;
    const uint64_t dDisk   = a.diskWrites - b.diskWrites;
    const uint64_t dDiskB  = a.diskBytes - b.diskBytes;
    const uint64_t dColdH  = a.coldHits - b.coldHits;
    const uint64_t dIce    = a.iceFetches - b.iceFetches;
    const uint64_t dShardH = a.shardHits - b.shardHits;
    const uint64_t dShardM = a.shardMisses - b.shardMisses;
    const uint64_t dMiss   = a.misses - b.misses;
    const double   chunksPerSec = r.chunks / std::max(r.seconds, 1e-9);
    const double   mbPerSec     = (dDiskB / (1024.0 * 1024.0)) / std::max(r.seconds, 1e-9);

    printf("  %-20s %6d chunks  %7.2fs  %7.1f chunks/s  %8.1f MB/s written\n",
           r.name.c_str(), r.chunks, r.seconds, chunksPerSec, mbPerSec);
    printf("      diskWrites +%6llu   coldHits +%6llu   iceFetches +%6llu   misses +%6llu\n",
           (unsigned long long)dDisk, (unsigned long long)dColdH,
           (unsigned long long)dIce,  (unsigned long long)dMiss);
    printf("      shardHits  +%6llu   shardMiss +%5llu   shardHitRate %5.1f%%\n",
           (unsigned long long)dShardH, (unsigned long long)dShardM,
           (dShardH + dShardM) ? 100.0 * dShardH / (dShardH + dShardM) : 0.0);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: vc_io_bench <volume_path_or_url> [--region-size N] "
            "[--level N] [--io-threads N] [--hot-gb N] [--timeout N] "
            "[--poll-hz N]\n");
        return 1;
    }

    std::string volumePath = argv[1];
    int regionSize = 8;
    int level = 0;
    int ioThreads = 16;
    int hotGb = 8;
    double timeoutSec = 300.0;
    int pollHz = 10;
    std::array<int, 3> centerVoxel = {-1, -1, -1};  // -1 = use geometric centre

    for (int i = 2; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires a value\n", what); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--region-size") regionSize = std::atoi(need("--region-size"));
        else if (a == "--level")       level      = std::atoi(need("--level"));
        else if (a == "--io-threads")  ioThreads  = std::atoi(need("--io-threads"));
        else if (a == "--hot-gb")      hotGb      = std::atoi(need("--hot-gb"));
        else if (a == "--timeout")     timeoutSec = std::atof(need("--timeout"));
        else if (a == "--poll-hz")     pollHz     = std::atoi(need("--poll-hz"));
        else if (a == "--center") {
            const char* v = need("--center");
            int z=-1, y=-1, x=-1;
            if (std::sscanf(v, "%d,%d,%d", &z, &y, &x) != 3) {
                fprintf(stderr, "--center expects Z,Y,X (got '%s')\n", v);
                return 1;
            }
            centerVoxel = {z, y, x};
        }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    printf("vc_io_bench\n");
    printf("  Volume:       %s\n", volumePath.c_str());
    printf("  Region:       %d^3 chunks (= %d chunks)\n",
           regionSize, regionSize * regionSize * regionSize);
    printf("  Level:        %d\n", level);
    printf("  IO threads:   %d\n", ioThreads);
    printf("  Hot cache:    %d GB\n", hotGb);
    printf("  Timeout:      %.1f s/phase\n", timeoutSec);
    printf("\n");

    // Open volume
    printf("Opening volume...\n");
    const auto tOpen = Clock::now();
    std::shared_ptr<Volume> vol;
    if (isRemoteUrl(volumePath)) vol = Volume::NewFromUrl(volumePath);
    else                          vol = Volume::New(volumePath);
    vol->setCacheBudget(static_cast<size_t>(hotGb) << 30);
    vol->setIOThreads(ioThreads);
    const int numLevels = static_cast<int>(vol->numScales());
    printf("  Shape: %d x %d x %d  (%d pyramid levels)\n",
           vol->shape()[0], vol->shape()[1], vol->shape()[2], numLevels);
    printf("  Open time: %.2f s\n\n", elapsedSec(tOpen, Clock::now()));

    auto* cache = vol->tieredCache();
    if (!cache) { fprintf(stderr, "No tieredCache on volume\n"); return 1; }

    if (centerVoxel[0] >= 0) {
        printf("  Center:       %d, %d, %d (level-0 voxel)\n",
               centerVoxel[0], centerVoxel[1], centerVoxel[2]);
    }
    auto keys = enumerateRegion(*vol, *cache, level, regionSize, centerVoxel);
    if (keys.empty()) {
        fprintf(stderr, "Failed to enumerate chunks (bad level? empty volume?)\n");
        return 1;
    }
    printf("Chunks enumerated: %zu\n\n", keys.size());

    std::vector<PhaseResult> results;

    // ==== Phase 1: Cold pipeline (clean slate) ====
    cache->clearAll();
    {
        PhaseResult r; r.name = "Cold pipeline";
        r.chunks = static_cast<int>(keys.size());
        r.before = snap(cache);
        const auto t0 = Clock::now();
        cache->fetchInteractive(keys, level);
        r.seconds = waitDrain(cache, timeoutSec, pollHz, t0);
        r.after = snap(cache);
        if (cache->stats().ioPending != 0)
            printf("  (cold phase: timed out with %zu chunks still pending)\n",
                   cache->stats().ioPending);
        results.push_back(std::move(r));
    }

    // ==== Phase 2: Warm shards (clear block cache, keep shard cache) ====
    cache->clearMemory();  // keeps diskLevels + on-disk shards + shard RAM cache
    {
        PhaseResult r; r.name = "Warm shards";
        r.chunks = static_cast<int>(keys.size());
        r.before = snap(cache);
        const auto t0 = Clock::now();
        cache->fetchInteractive(keys, level);
        r.seconds = waitDrain(cache, timeoutSec, pollHz, t0);
        r.after = snap(cache);
        results.push_back(std::move(r));
    }

    // ==== Phase 3: Fully warm (everything resident) ====
    // Don't clear anything. fetchInteractive should hit the dedup fast-path
    // on the second call, and the containsBatch inside should filter out
    // every already-resident chunk.
    {
        PhaseResult r; r.name = "Fully warm (dedup)";
        r.chunks = static_cast<int>(keys.size());
        r.before = snap(cache);
        const auto t0 = Clock::now();
        cache->fetchInteractive(keys, level);
        // No drain — dedup should make this immediate.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        r.seconds = elapsedSec(t0, Clock::now());
        r.after = snap(cache);
        results.push_back(std::move(r));
    }

    // ==== Report ====
    printf("Results:\n");
    for (const auto& r : results) printPhase(r);
    printf("\nDone.\n");
    return 0;
}
