// vc_transpose_bench: Chunk→block transpose throughput.
//
// Measures the inner loop of BlockPipeline::insertChunkAsBlocks: given a
// fully decoded 128³ (or other multiple-of-16) chunk in row-major layout,
// slice it into 16³ blocks and write each into the BlockCache arena via
// BatchPut::acquire. This path runs on every chunk the pipeline decodes,
// so even small regressions here compound with streaming volume.
//
// The bench isolates the transpose from the rest of the pipeline by:
//   1) Pre-allocating a synthetic "decoded chunk" buffer.
//   2) Replaying the exact insert loop N times under T worker threads
//      using disjoint (bz,by,bx) ranges so threads don't alias slots.
//
// What the numbers mean
//   blocks/s   : raw per-block insert rate (16³ voxel copies).
//   chunks/s   : 128³ chunks inserted per second = blocks/s / 512.
//   MB/s       : bytes moved through the transpose (useful for comparing
//                against memory bandwidth).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// The transpose loop lifted verbatim from BlockPipeline::insertChunkAsBlocks
// so the bench measures exactly that code path. Kept local so it doesn't
// depend on internal pipeline state.
void insertChunkAsBlocks(BlockCache& cache, int level, int baseBz, int baseBy, int baseBx,
                         const uint8_t* src, int cz, int cy, int cx)
{
    const int bzN = cz / kBlockSize;
    const int byN = cy / kBlockSize;
    const int bxN = cx / kBlockSize;
    const int strideY = cx;
    const int strideZ = cx * cy;
    // Use put() via BatchPut so this bench works on both baseline and
    // optimized builds (the zero-copy acquire() API is only in the
    // optimized branch). The 4 KiB tmp + memcpy-to-slot is the same
    // pattern the baseline insertChunkAsBlocks uses, so this bench
    // measures the baseline variant apples-to-apples.
    uint8_t tmp[kBlockBytes];
    BlockCache::BatchPut batch(cache);
    for (int bi = 0; bi < bzN; ++bi) {
      for (int bj = 0; bj < byN; ++bj) {
        for (int bk = 0; bk < bxN; ++bk) {
            uint8_t* dst = tmp;
            for (int lz = 0; lz < kBlockSize; ++lz) {
                const uint8_t* zRow = src + (bi * kBlockSize + lz) * strideZ;
                for (int ly = 0; ly < kBlockSize; ++ly) {
                    const uint8_t* p = zRow + (bj * kBlockSize + ly) * strideY + bk * kBlockSize;
                    std::memcpy(dst, p, kBlockSize);
                    dst += kBlockSize;
                }
            }
            BlockKey bkKey{level, baseBz + bi, baseBy + bj, baseBx + bk};
            batch.put(bkKey, tmp);
        }
      }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    int threads = 8;
    int arenaGb = 4;
    int chunkDim = 128;
    uint64_t chunksPerThread = 2000;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need = [&](const char* w) { if(i+1>=argc){fprintf(stderr,"%s needs value\n",w);std::exit(1);} return argv[++i]; };
        if      (a == "--threads")   threads = std::atoi(need("--threads"));
        else if (a == "--arena-gb")  arenaGb = std::atoi(need("--arena-gb"));
        else if (a == "--chunk-dim") chunkDim = std::atoi(need("--chunk-dim"));
        else if (a == "--chunks")    chunksPerThread = std::atoll(need("--chunks"));
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }
    if (chunkDim % kBlockSize != 0) {
        fprintf(stderr, "chunk-dim must be multiple of %d\n", kBlockSize);
        return 1;
    }

    const size_t chunkBytes = size_t(chunkDim) * chunkDim * chunkDim;
    const int bpcPerAxis = chunkDim / kBlockSize;
    const uint64_t blocksPerChunk = uint64_t(bpcPerAxis) * bpcPerAxis * bpcPerAxis;

    BlockCache::Config cfg;
    cfg.bytes = size_t(arenaGb) << 30;
    printf("vc_transpose_bench\n");
    printf("  Threads:       %d\n", threads);
    printf("  Arena:         %d GB\n", arenaGb);
    printf("  Chunk dim:     %d (= %llu blocks/chunk)\n", chunkDim,
           (unsigned long long)blocksPerChunk);
    printf("  Chunks/thread: %llu\n", (unsigned long long)chunksPerThread);
    printf("  Chunk bytes:   %zu (%.1f MB)\n", chunkBytes, chunkBytes / (1024.0 * 1024.0));
    printf("\n");

    BlockCache cache(cfg);

    // One shared source buffer per thread, filled with a non-zero pattern so
    // we don't accidentally trigger an isEmpty shortcut in any downstream
    // consumer (this bench doesn't, but keep it realistic).
    std::vector<std::vector<uint8_t>> srcs(threads);
    for (int t = 0; t < threads; ++t) {
        srcs[t].resize(chunkBytes);
        for (size_t i = 0; i < chunkBytes; ++i)
            srcs[t][i] = static_cast<uint8_t>(i ^ (t * 17));
    }

    const auto t0 = Clock::now();
    std::vector<std::jthread> ws;
    for (int t = 0; t < threads; ++t) {
        ws.emplace_back([&, tid = t](std::stop_token) {
            for (uint64_t c = 0; c < chunksPerThread; ++c) {
                // Disjoint (bz, by, bx) per thread per chunk so threads
                // insert into distinct arena slots concurrently.
                const int baseBz = tid * 1024 + int(c % 64) * bpcPerAxis;
                const int baseBy = int((c / 64) % 64) * bpcPerAxis;
                const int baseBx = int((c / 4096) % 64) * bpcPerAxis;
                insertChunkAsBlocks(cache, 0, baseBz, baseBy, baseBx,
                                    srcs[tid].data(),
                                    chunkDim, chunkDim, chunkDim);
            }
        });
    }
    ws.clear();
    const double sec = elapsed(t0, Clock::now());

    const uint64_t totalChunks = uint64_t(threads) * chunksPerThread;
    const uint64_t totalBlocks = totalChunks * blocksPerChunk;
    const double bytesPerSec = (totalChunks * double(chunkBytes)) / sec;

    printf("Results:\n");
    printf("  Elapsed:    %7.2f s\n", sec);
    printf("  Chunks/s:   %12.0f\n", totalChunks / sec);
    printf("  Blocks/s:   %12.0f\n", totalBlocks / sec);
    printf("  Throughput: %8.1f MB/s\n", bytesPerSec / (1024.0 * 1024.0));
    printf("  Per-chunk:  %7.2f us avg\n",
           sec * 1e6 / std::max<double>(totalChunks, 1));
    printf("\nDone.\n");
    return 0;
}
