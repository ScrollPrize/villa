// vc_blockcache_bench: Multi-threaded BlockCache put/get throughput.
//
// The BlockCache is the render hot path's deepest sync point. Every sampler
// call hits blockAt → cache.get under a shared lock, and every decoded
// chunk inserts 512 blocks via BatchPut under an exclusive lock. This
// bench measures both paths isolated from any volume loading logic so
// regressions show up against a known baseline.
//
// Workloads
//   get-hot      : N threads read the same small working set (always hits).
//                  Isolates get() path + slot-cache traversal.
//   get-scattered: N threads read random keys across a large arena
//                  (~80% hit / ~20% miss after warmup). Measures eviction
//                  + clock-sweep behaviour under pressure.
//   put-batch    : N threads call BatchPut::acquire + fill 512 blocks each.
//                  Measures the chunk→block insert path that
//                  insertChunkAsBlocks takes after every decode.
//   contains-batch: N threads call containsBatch with K keys each. Models
//                   fetchInteractive's probe step.
//
// Options:
//   --threads N       Worker threads (default: 8)
//   --arena-gb N      BlockCache budget in GB (default: 1)
//   --workset-mb N    Working-set size per phase in MB (default: 64)
//   --iters N         Iterations per thread (default: 1_000_000 for get,
//                     10_000 for put)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

#include "vc/core/cache/BlockCache.hpp"

using Clock = std::chrono::steady_clock;
using namespace vc::cache;

namespace {

double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct Phase {
    std::string name;
    uint64_t totalOps = 0;
    double seconds = 0.0;
    double opsPerSec() const { return seconds ? totalOps / seconds : 0.0; }
};

void printPhase(const Phase& p, int threads) {
    printf("  %-18s %2d thr  %12llu ops  %7.2fs  %12.0f ops/s  (%7.0f ns/op avg)\n",
           p.name.c_str(), threads,
           (unsigned long long)p.totalOps, p.seconds, p.opsPerSec(),
           p.seconds * 1e9 / std::max<double>(p.totalOps, 1));
}

// Populate `cache` with a working set of known keys. Returns the keys so
// tests can issue hits against them.
std::vector<BlockKey> populate(BlockCache& cache, size_t nKeys) {
    std::vector<BlockKey> keys;
    keys.reserve(nKeys);
    std::vector<uint8_t> pattern(kBlockBytes);
    for (size_t i = 0; i < kBlockBytes; ++i) pattern[i] = static_cast<uint8_t>(i);
    for (size_t i = 0; i < nKeys; ++i) {
        BlockKey k{0, int(i / 1024), int((i / 32) & 31), int(i & 31)};
        keys.push_back(k);
        cache.put(k, pattern.data(), cache.generation());
    }
    return keys;
}

Phase benchGetHot(BlockCache& cache, int threads, uint64_t itersPerThread,
                  const std::vector<BlockKey>& keys)
{
    Phase p; p.name = "get-hot";
    std::atomic<uint64_t> totalHits{0};
    const auto t0 = Clock::now();
    std::vector<std::jthread> ws;
    for (int t = 0; t < threads; ++t) {
        ws.emplace_back([&, tid = t](std::stop_token) {
            uint64_t hits = 0;
            std::mt19937 rng(uint32_t(tid) * 0x9E37u + 1);
            for (uint64_t i = 0; i < itersPerThread; ++i) {
                const auto& k = keys[rng() % keys.size()];
                if (cache.get(k)) ++hits;
            }
            totalHits.fetch_add(hits, std::memory_order_relaxed);
        });
    }
    ws.clear();
    p.seconds = elapsed(t0, Clock::now());
    p.totalOps = uint64_t(threads) * itersPerThread;
    return p;
}

Phase benchGetScattered(BlockCache& cache, int threads, uint64_t itersPerThread,
                        const std::vector<BlockKey>& keys, uint64_t spaceSize)
{
    // Mix hits (indexes into keys) with misses (random keys outside keys),
    // ratio ~80/20.
    Phase p; p.name = "get-scattered";
    const auto t0 = Clock::now();
    std::vector<std::jthread> ws;
    for (int t = 0; t < threads; ++t) {
        ws.emplace_back([&, tid = t](std::stop_token) {
            std::mt19937 rng(uint32_t(tid) * 0x9E37u + 3);
            for (uint64_t i = 0; i < itersPerThread; ++i) {
                if ((rng() & 7) < 6) {
                    (void)cache.get(keys[rng() % keys.size()]);
                } else {
                    BlockKey k{0, int(rng() % spaceSize),
                                   int(rng() % spaceSize),
                                   int(rng() % spaceSize)};
                    (void)cache.get(k);
                }
            }
        });
    }
    ws.clear();
    p.seconds = elapsed(t0, Clock::now());
    p.totalOps = uint64_t(threads) * itersPerThread;
    return p;
}

Phase benchPutBatch(BlockCache& cache, int threads, uint64_t chunkIterPerThread)
{
    // Each "chunk" == 512 blocks written under one BatchPut. Models a
    // 128³ chunk arriving from decode. Uses put() so the bench works on
    // both baseline and optimized builds.
    Phase p; p.name = "put-batch";
    std::vector<uint8_t> pattern(kBlockBytes);
    for (size_t i = 0; i < kBlockBytes; ++i) pattern[i] = static_cast<uint8_t>(i ^ 0xA5);
    const auto t0 = Clock::now();
    std::vector<std::jthread> ws;
    for (int t = 0; t < threads; ++t) {
        ws.emplace_back([&, tid = t, patPtr = pattern.data()](std::stop_token) {
            for (uint64_t c = 0; c < chunkIterPerThread; ++c) {
                // Disjoint bz ranges per thread per chunk so threads don't
                // thrash on identical keys — models concurrent chunk
                // insertion from different levels/regions.
                const int baseBz = tid * 64 + int((c & 0xff) * threads);
                BlockCache::BatchPut batch(cache, cache.generation());
                for (int bi = 0; bi < 8; ++bi)
                  for (int bj = 0; bj < 8; ++bj)
                    for (int bk = 0; bk < 8; ++bk) {
                        BlockKey k{0, baseBz + bi, bj, bk};
                        batch.put(k, patPtr);
                    }
            }
        });
    }
    ws.clear();
    p.seconds = elapsed(t0, Clock::now());
    p.totalOps = uint64_t(threads) * chunkIterPerThread * 512;  // blocks inserted
    return p;
}

Phase benchContainsBatch(BlockCache& cache, int threads, uint64_t iterPerThread,
                          const std::vector<BlockKey>& keys)
{
    // K=512 keys per call — matches the two-blocks-per-chunk x 256-chunks
    // probe profile fetchInteractive does in the worst case.
    Phase p; p.name = "contains-batch";
    const size_t K = 512;
    const auto t0 = Clock::now();
    std::vector<std::jthread> ws;
    for (int t = 0; t < threads; ++t) {
        ws.emplace_back([&, tid = t](std::stop_token) {
            std::mt19937 rng(uint32_t(tid) * 0x9E37u + 7);
            std::vector<BlockKey> probe(K);
            std::vector<uint8_t> out;
            for (uint64_t i = 0; i < iterPerThread; ++i) {
                for (size_t j = 0; j < K; ++j)
                    probe[j] = keys[rng() % keys.size()];
                cache.containsBatch(probe, out);
            }
        });
    }
    ws.clear();
    p.seconds = elapsed(t0, Clock::now());
    p.totalOps = uint64_t(threads) * iterPerThread * K;
    return p;
}

}  // namespace

int main(int argc, char** argv)
{
    int threads = 8;
    int arenaGb = 1;
    int worksetMb = 64;
    uint64_t getIters = 1'000'000;
    uint64_t putIters = 10'000;
    uint64_t containsIters = 100'000;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need = [&](const char* w) { if (i+1>=argc){fprintf(stderr,"%s needs value\n",w);std::exit(1);} return argv[++i]; };
        if      (a == "--threads")     threads = std::atoi(need("--threads"));
        else if (a == "--arena-gb")    arenaGb = std::atoi(need("--arena-gb"));
        else if (a == "--workset-mb")  worksetMb = std::atoi(need("--workset-mb"));
        else if (a == "--iters")       getIters = std::atoll(need("--iters"));
        else if (a == "--put-iters")   putIters = std::atoll(need("--put-iters"));
        else if (a == "--contains-iters") containsIters = std::atoll(need("--contains-iters"));
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    BlockCache::Config cfg;
    cfg.bytes = size_t(arenaGb) << 30;
    printf("vc_blockcache_bench\n");
    printf("  Threads:    %d\n", threads);
    printf("  Arena:      %d GB (%llu slots)\n", arenaGb,
           (unsigned long long)(cfg.bytes / kBlockBytes));
    printf("  Workset:    %d MB (%llu blocks)\n", worksetMb,
           (unsigned long long)((size_t(worksetMb) << 20) / kBlockBytes));
    printf("\n");

    BlockCache cache(cfg);
    const size_t worksetBlocks = (size_t(worksetMb) << 20) / kBlockBytes;
    auto keys = populate(cache, worksetBlocks);
    printf("Populated %zu keys. Arena size=%zu.\n\n", keys.size(), cache.size());

    std::vector<Phase> results;
    results.push_back(benchGetHot(cache, threads, getIters, keys));
    results.push_back(benchGetScattered(cache, threads, getIters, keys, 10'000));
    results.push_back(benchContainsBatch(cache, threads, containsIters, keys));
    // Put test populates beyond arena → exercises eviction too.
    results.push_back(benchPutBatch(cache, threads, putIters));

    printf("Results:\n");
    for (const auto& r : results) printPhase(r, threads);
    printf("\nDone.\n");
    return 0;
}
