#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BlockCache.hpp"
#include "ChunkData.hpp"
#include "ChunkKey.hpp"
#include "VolumeSource.hpp"
#include "IOPool.hpp"
#include <utils/zarr.hpp>
#include <utils/c3d_codec.hpp>

namespace vc { class VcDataset; }

namespace vc::cache {

// Block-granular cache pipeline. Data flows ice (remote) → cold (disk) →
// decoded bytes → 16^3 blocks in the BlockCache.
//
// Callers see only blocks. Chunks are an on-disk/IO artifact used internally
// to amortize S3 and codec overhead.
class BlockPipeline {
public:
    struct Config {
        size_t bytes = 10ULL << 30;          // 10 GiB block cache
        // RAM cache of compressed canonical shard files. The loader pool
        // checks this before hitting disk; on a miss it mmaps the whole
        // shard file once and caches the handle, so subsequent inner-chunk
        // reads from the same shard are zero-syscall memcpy from page
        // cache. Since shards are mmap'd (not heap-allocated), a generous
        // budget costs nothing beyond virtual-address space — the kernel
        // drops unused pages under memory pressure automatically. Set
        // to 0 to disable shard caching (loader goes straight to disk
        // every time).
        size_t shardCacheBytes = 4ULL << 30; // 4 GiB default
        std::string volumeId;
        // Defaults to hardware_concurrency(); see constructor.
        int ioThreads = 0;

        // c3d encode parameters used when re-encoding non-canonical
        // source chunks into the canonical 256³ disk cache.
        // target_ratio is the only knob; 50 ≈ 40 dB PSNR on scroll CT.
        utils::C3dCodecParams c3dEncodeParams = {};

        // Backpressure ceiling on the downloader → encoder hand-off.
        // Each staged chunk is one decoded canonical ChunkData (~16 MiB at
        // 256³ u8). Without a cap the downloader pool floods RAM on a
        // warm network / slow disk. 64 chunks ≈ 1 GiB max staged.
        size_t maxEncodeStagingChunks = 64;

        // Max concurrent dz.read_whole_shard() calls. Each returns a freshly-
        // allocated std::vector<std::byte> sized to the shard file
        // (~256 MiB on sharded c3d volumes). With many loader workers all
        // reading freshly-written shards at once, in-flight shard bytes
        // balloon past the shardCacheBytes LRU budget — that budget only
        // caps *cached* shards, not reads in progress. 8 ≈ 2 GiB worst
        // case in flight. Shutdown-aware.
        size_t maxConcurrentShardReads = 8;

        // Backpressure ceiling on the loader → decoder hand-off.
        // decodeStaging_ holds compressed inner-chunk bytes awaiting
        // decode (~200 KiB–2 MiB each depending on target_ratio). On
        // heavy pans the loader can outrun the decoder and queue
        // hundreds of MiB here. 256 MiB ≈ a few thousand chunks max.
        size_t maxDecodeStagingBytes = 256ULL << 20;

        // When non-zero, declares the source is byte-identical to our local
        // canonical c3d disk format: zarr v3, 4096³ shards with 256³ inner
        // C3DC chunks. The downloader then bypasses the encoder entirely —
        // fetchWholeShard from source, write the bytes verbatim to disk,
        // forward chunk keys directly to the loader. Local shard shape MUST
        // match this for byte-passthrough to be valid; every inner chunk is
        // magic-checked for "C3DC".
        std::array<int, 3> canonicalSourceShard = {0, 0, 0};
    };

    BlockPipeline(
        Config config,
        BlockCache& blockCache,
        std::unique_ptr<VolumeSource> source,
        DecompressFn decompress,
        std::vector<std::unique_ptr<utils::ZarrArray>> diskLevels = {});

    ~BlockPipeline();

    BlockPipeline(const BlockPipeline&) = delete;
    BlockPipeline& operator=(const BlockPipeline&) = delete;

    // Transfer ownership of a fallback BlockCache into this pipeline.
    // Must be called immediately after construction, before any work.
    void ownBlockCache(std::unique_ptr<BlockCache> cache) { ownedBlockCache_ = std::move(cache); }

    // --- Block-level access ---
    // Returns a shared_ptr to the 16^3 block, or null if not in RAM.
    // Evicted blocks stay alive while any caller still holds a shared_ptr.
    [[nodiscard]] BlockPtr blockAt(const BlockKey& key) noexcept;

    // --- Interactive fetch (for viewport chunks) ---
    // Chunk keys are still the IO unit — after decode, each chunk is split
    // into 16^3 blocks and inserted into the block cache. targetLevel is
    // the pyramid level the viewer is currently displaying at; shards at
    // that level get the highest IO priority.
    void fetchInteractive(const std::vector<ChunkKey>& keys, int targetLevel = 0);

    // --- Cache management ---
    void clearMemory();
    void clearAll();

    // Explicitly stop all worker pools and release caches. After this call
    // the pipeline is inert — the destructor becomes a no-op. Use in
    // volume-switch paths to ensure the old pipeline's abortAll() doesn't
    // poison a new pipeline that's already running.
    void shutdown();

    [[nodiscard]] int numLevels() const noexcept;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept;

    // OME-Zarr scale factor for a vector index (e.g. 4.0 for directory "2").
    // Falls back to 2^vectorIndex if not set.
    [[nodiscard]] float levelScaleFactor(int vectorIndex) const noexcept;

    // --- Logical data bounds ---
    struct DataBoundsL0 {
        int minX = 0, maxX = 0;
        int minY = 0, maxY = 0;
        int minZ = 0, maxZ = 0;
        bool valid = false;
        constexpr bool operator==(const DataBoundsL0&) const noexcept = default;
    };

    void setDataBounds(int minX, int maxX, int minY, int maxY, int minZ, int maxZ);
    [[nodiscard]] DataBoundsL0 dataBounds() const;

    [[nodiscard]] bool isNegativeCached(const ChunkKey& key) const;

    // Counts how many of the given chunks are either already decoded
    // (first block in block cache) or known-empty.
    [[nodiscard]] size_t countAvailable(const std::vector<ChunkKey>& keys) const;

    // Return chunks that are not resident, not known-empty, and not already
    // present in the canonical disk cache. Used by explicit cache prefetch so
    // already-downloaded chunks are not fetched again.
    [[nodiscard]] std::vector<ChunkKey> chunksMissingFromCache(
        const std::vector<ChunkKey>& keys) const;

    // --- Notifications ---
    using ChunkReadyCallback = std::function<void(const ChunkKey&)>;
    using ChunkReadyCallbackId = uint64_t;

    [[nodiscard]] ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb);
    void removeChunkReadyListener(ChunkReadyCallbackId id);
    void clearChunkArrivedFlag() noexcept;

    // --- Stats ---
    struct Stats {
        uint64_t blockHits = 0;
        uint64_t coldHits = 0;
        uint64_t iceFetches = 0;
        uint64_t misses = 0;
        size_t blocks = 0;
        size_t ioPending = 0;             // download + encode + load
        size_t downloadPending = 0;        // s3 → staged ChunkData queue
        size_t encodePending = 0;          // staged ChunkData → h265 disk queue
        size_t loadPending = 0;            // disk → staged bytes queue
        size_t decodePending = 0;          // staged bytes → decoded + block cache
        size_t encodeStagingChunks = 0;    // chunks sitting in encodeStaging_ (16 MiB ea)
        size_t decodeStagingBytes = 0;     // bytes sitting in decodeStaging_ (compressed)
        size_t inflightShardReads = 0;     // read_whole_shard calls currently in progress
        size_t inflightShardBytes = 0;     // bytes held by in-progress shard reads
        uint64_t shardHits = 0;            // loader found shard in RAM cache
        uint64_t shardMisses = 0;          // loader had to read shard from disk
        size_t shardCacheBytes = 0;        // current shard cache occupancy
        size_t shardCacheEntries = 0;
        uint64_t diskWrites = 0;
        size_t negativeCount = 0;
        size_t diskBytes = 0;
        size_t diskShards = 0;
        uint64_t totalSubmitted = 0;
        bool sharded = false;
    };

    [[nodiscard]] Stats stats() const;

private:
    Config config_;
    std::vector<std::unique_ptr<utils::ZarrArray>> diskLevels_;
    std::unique_ptr<VolumeSource> source_;
    DecompressFn decompress_;
    // Four fully independent pools — each specialised for one stage so no
    // stage can starve another.
    //   downloaderPool_ : s3 fetch + source decode + re-chunk → staged
    //     ChunkData. Network-bound. Never touches disk or block cache.
    //   encodePool_     : take staged ChunkData → h265 encode → disk.
    //     CPU-bound. Never touches the network or block cache.
    //   loaderPool_     : disk read (or shard-cache memcpy) → staged
    //     compressed bytes. I/O-bound. Never decodes.
    //   decodePool_     : take staged compressed bytes → h265 decode →
    //     insert blocks, fire chunk-ready callbacks. Pure CPU.
    // Submission in fetchInteractive triages on the disk shard index:
    // present → loaderPool_, whose completion forwards to decodePool_;
    // absent → downloaderPool_, whose completion forwards to encodePool_,
    // whose completion forwards to loaderPool_.
    IOPool downloaderPool_;
    IOPool encodePool_;
    IOPool loaderPool_;
    IOPool decodePool_;
    // Hand-off buffer between downloader and encoder. Download inserts
    // (key → decoded ChunkData) after assembling, encoder takes it out.
    // CV gates the downloader when encodeStaging_ is at capacity so a
    // fast network can't run RAM to swap while encode drains to disk.
    mutable std::mutex encodeStagingMutex_;
    std::unordered_map<ChunkKey, ChunkDataPtr, ChunkKeyHash> encodeStaging_;
    std::condition_variable encodeStagingCv_;
    std::atomic<bool> shuttingDown_{false};

    // Backpressure for shardBytesFor() → dz.read_whole_shard(). Gates
    // concurrent shard reads by count; tracks bytes separately for stats.
    mutable std::mutex inflightShardMutex_;
    std::condition_variable inflightShardCv_;
    size_t inflightShardReads_ = 0;                 // guarded by inflightShardMutex_
    std::atomic<size_t> inflightShardBytes_{0};
    // Hand-off buffer between loader and decoder. Loader inserts
    // (key → compressed inner-chunk bytes); decoder pops and decodes.
    // CV gates the loader when decodeStagingBytesAtomic_ would exceed
    // maxDecodeStagingBytes. Atomic counter avoids walking the map in
    // the CV predicate (hot path).
    mutable std::mutex decodeStagingMutex_;
    std::unordered_map<ChunkKey, std::vector<uint8_t>, ChunkKeyHash> decodeStaging_;
    std::condition_variable decodeStagingCv_;
    std::atomic<size_t> decodeStagingBytesAtomic_{0};

    // Shard-level LRU cache of compressed canonical h265 shard files.
    // Populated on loader misses. Bytes-budgeted; when exceeded the
    // least-recently-used shard is evicted. shared_ptr on the buffer so
    // concurrent loaders can serve from the same shard without the cache
    // mutex blocking them.
    //
    // Sharded by hash(ShardKey) to reduce mutex contention: with N loader
    // threads all hitting the same mutex, the LRU splice on every hit
    // serializes the hot fetch path. kShardCacheBuckets=16 drops contention
    // to ~1/16 for uniformly distributed shard accesses.
    //
    // Budget is SHARED across buckets via the atomic shardCacheGlobalBytes_
    // counter: any bucket can hold as much as it wants so long as the total
    // stays under shardCacheBytes. An insert that pushes the global over
    // budget evicts LRU entries from its OWN bucket until under — preserves
    // per-bucket LRU semantics while making capacity fluid across uneven
    // hash distributions (avoids the per-bucket cap starving a hot bucket).
    struct ShardCacheEntry {
        ShardKey key;
        std::shared_ptr<utils::ShardBytes> bytes;
    };
    static constexpr size_t kShardCacheBuckets = 16;
    struct ShardCacheBucket {
        mutable std::mutex mutex;
        std::list<ShardCacheEntry> lru;  // front = most recent
        std::unordered_map<ShardKey,
                           std::list<ShardCacheEntry>::iterator,
                           ShardKeyHash> map;
        size_t bytes = 0;
    };
    mutable std::array<ShardCacheBucket, kShardCacheBuckets> shardCacheBuckets_;
    std::atomic<size_t> shardCacheGlobalBytes_{0};
    // Hits/misses so the status bar can surface them.
    std::atomic<uint64_t> statShardHits_{0};
    std::atomic<uint64_t> statShardMisses_{0};

    // Translate a ChunkKey into the shard it lives in (zarr v3 sharded
    // grid coordinates). Empty/non-sharded arrays return the zero shard.
    [[nodiscard]] ShardKey canonicalShardKey(const ChunkKey& key) const noexcept;

    // Pull the whole shard file for `key` through the LRU cache. First
    // hit from any thread reads the file once; subsequent hits just bump
    // the LRU head and return the shared buffer.
    //
    // Returns a raw pointer valid until the calling thread next invokes
    // shardBytesFor() — a per-thread shared_ptr keeps the bytes alive
    // across the caller's brief synchronous use. Returning a raw pointer
    // instead of shared_ptr eliminates ~2 refcount atomics per call that
    // were cache-line ping-ponging under 12-thread decode (perf showed
    // the shared_ptr dtor's LDADDAL at ~20% of total CPU).
    const utils::ShardBytes* shardBytesFor(
        const ChunkKey& key, utils::ZarrArray& dz);

    // Map ShardKey → bucket index in shardCacheBuckets_.
    static size_t shardCacheBucketIndex(const ShardKey& sk) noexcept;
    // Insert into a specific bucket, evicting per-bucket LRU until under its
    // share of the total budget.
    void shardCacheInsertLocked(ShardCacheBucket& b,
                                const ShardKey& sk,
                                std::shared_ptr<utils::ShardBytes> bytes);

    std::unique_ptr<BlockCache> ownedBlockCache_;  // per-pipeline fallback when no shared cache
    BlockCache& blockCache_;

    // Assemble a canonical 128^3 chunk from one or more source chunks at
    // `canonKey.level`, rechunking as needed. Null if the canonical region
    // is entirely absent from the source.
    [[nodiscard]] ChunkDataPtr assembleCanonicalChunk(const ChunkKey& canonKey);

    // Split a decoded chunk into 16^3 blocks and insert into blockCache_.
    void insertChunkAsBlocks(const ChunkKey& key, const ChunkData& chunk);

    // Wake viewer/listeners when a chunk's visible state changes, including
    // decoded data, all-zero chunks, and confirmed-absent chunks. The
    // chunk-arrival flag coalesces bursts to one UI wake per render tick.
    void notifyChunkReady(const ChunkKey& key);

    // All-zero canonical chunks: record their key instead of materialising
    // 512 identical zero blocks in the arena. blockAt() returns a pointer
    // to a single static zero-block when the block's canonical chunk is
    // in this set.
    //
    // Lock-free hash set. Readers probe with acquire-atomic-loads; writers
    // (insertChunkAsBlocks / clear) serialize via emptyChunksWriteMutex_
    // and publish with release-atomic-stores. Every blockAt miss used to
    // take a shared_mutex → ~2% of CPU in pthread_rwlock at the 12-thread
    // render path; now just one atomic load per probe step. Entries: 64
    // bits = [state:2 | chunkHash:62]. False positives on chunkHash are
    // OK — an "empty" hit is cheap (returns the static zero block) and
    // correctness is unaffected because the arena still holds the
    // authoritative data if it's there. In practice false positives are
    // vanishingly rare (62-bit hash).
    static constexpr size_t kEmptyChunksBits = 14;  // 16K slots × 8 B = 128 KB
    static constexpr size_t kEmptyChunksSize = size_t(1) << kEmptyChunksBits;
    static constexpr size_t kEmptyChunksMask = kEmptyChunksSize - 1;
    static constexpr uint64_t kEmptyChunksStateMask = 0xC000000000000000ull;
    static constexpr uint64_t kEmptyChunksOccupied  = 0x8000000000000000ull;
    static constexpr uint64_t kEmptyChunksHashMask  = 0x3FFFFFFFFFFFFFFFull;
    std::array<std::atomic<uint64_t>, kEmptyChunksSize> emptyChunksTable_{};
    mutable std::mutex emptyChunksWriteMutex_;
    // Lock-free probe for "is this chunk known to be all-zero?".
    [[nodiscard]] bool isEmptyChunk(const ChunkKey& k) const noexcept {
        const uint64_t fh = emptyChunkFullHash(k);
        size_t idx = fh & kEmptyChunksMask;
        for (size_t probe = 0; probe < kEmptyChunksSize; ++probe) {
            const uint64_t e = emptyChunksTable_[(idx + probe) & kEmptyChunksMask]
                                  .load(std::memory_order_acquire);
            if (e == 0) return false;  // empty → end of probe
            if ((e & kEmptyChunksStateMask) == kEmptyChunksOccupied
                && (e & kEmptyChunksHashMask) == fh) {
                return true;
            }
            // mismatched hash: keep probing
        }
        return false;
    }
    void addEmptyChunk(const ChunkKey& k) noexcept {
        std::lock_guard lk(emptyChunksWriteMutex_);
        const uint64_t fh = emptyChunkFullHash(k);
        size_t idx = fh & kEmptyChunksMask;
        const uint64_t entry = kEmptyChunksOccupied | fh;
        for (size_t probe = 0; probe < kEmptyChunksSize; ++probe) {
            const size_t pos = (idx + probe) & kEmptyChunksMask;
            const uint64_t e = emptyChunksTable_[pos].load(std::memory_order_relaxed);
            if (e == 0) {
                emptyChunksTable_[pos].store(entry, std::memory_order_release);
                return;
            }
            if ((e & kEmptyChunksHashMask) == fh) return;  // already present
        }
        // Table full — in practice won't happen, we have 16K slots.
    }
    void clearEmptyChunks() noexcept {
        std::lock_guard lk(emptyChunksWriteMutex_);
        for (auto& e : emptyChunksTable_) e.store(0, std::memory_order_relaxed);
    }
    static uint64_t emptyChunkFullHash(const ChunkKey& k) noexcept {
        // 62-bit hash of (level, iz, iy, ix). Collision rate 1/2^62 —
        // negligible even over the program's lifetime.
        uint64_t h = ChunkKeyHash{}(k);
        h = (h ^ (h >> 31)) * 0x9E3779B97F4A7C15ull;
        h = (h ^ (h >> 27)) * 0xBF58476D1CE4E5B9ull;
        h ^= h >> 32;
        h &= kEmptyChunksHashMask;
        return h == 0 ? 1 : h;  // reserve 0 for "empty slot" sentinel
    }

    // Negative cache (same design as before).
    static constexpr size_t kBloomBits = 65536;
    std::array<std::atomic<uint64_t>, kBloomBits / 64> negativeBloom_{};
    void bloomAdd(const ChunkKey& key) noexcept;
    [[nodiscard]] bool bloomMayContain(const ChunkKey& key) const noexcept;
    void bloomClear() noexcept;
    mutable std::mutex negativeMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> negativeCache_;

    mutable std::mutex callbackMutex_;
    std::vector<std::pair<ChunkReadyCallbackId, ChunkReadyCallback>> chunkReadyListeners_;
    std::atomic<ChunkReadyCallbackId> nextListenerId_{1};
    std::atomic<bool> chunkArrivedFlag_{false};

    // fetchInteractive dedup: the renderer calls fetchInteractive every
    // frame, but when the viewport is idle the keys + targetLevel are
    // identical back-to-back, and identical across multiple viewers. A
    // commutative XOR hash of (key, targetLevel) + the BlockCache eviction
    // version is enough to detect "nothing has changed since we last did
    // the expensive probe/classify/updateInteractive work" and skip.
    // Guarded by fetchInteractiveDedupMutex_ so cross-viewer calls
    // serialize their compare-and-update of the stored state — without it
    // two identical calls could both see a stale match and each do the
    // work, defeating the dedup.
    mutable std::mutex fetchInteractiveDedupMutex_;
    uint64_t lastFetchInteractiveHash_ = 0;
    uint64_t lastFetchInteractiveEviction_ = 0;
    std::array<uint64_t, 4> lastFetchInteractiveIoVersions_{};
    int lastFetchInteractiveTargetLevel_ = -1;
    bool haveLastFetchInteractive_ = false;

    mutable std::mutex dataBoundsMutex_;
    DataBoundsL0 dataBoundsL0_;

    // Hits from the empty-chunk canonical zero block (cold path in blockAt).
    // The hot-path block hit counter lives in BlockCache::shards_ — a single
    // atomic here saturated a cacheline under 12-thread render.
    mutable std::atomic<uint64_t> statEmptyHits_{0};
    std::atomic<uint64_t> statColdHits_{0};
    std::atomic<uint64_t> statIceFetches_{0};
    std::atomic<uint64_t> statDiskWrites_{0};
    std::atomic<uint64_t> statTotalSubmitted_{0};
    // Cumulative bytes written to the canonical disk cache this session.
    std::atomic<uint64_t> statDiskBytes_{0};
    // Distinct shard files touched this session.
    mutable std::mutex writtenShardsMutex_;
    std::unordered_set<ShardKey, ShardKeyHash> writtenShards_;
    mutable std::atomic<uint64_t> statMisses_{0};

    // Seeded from a startup scan of the on-disk cache so stats show real
    // usage immediately; session-scoped writes accumulate on top.
    size_t initialDiskBytes_ = 0;
    size_t initialDiskShards_ = 0;
};

// Convenience: open a single-level BlockPipeline against a local zarr dataset
// (no disk tier; filesystem serves as ice). Used by CLI tools, tracer, etc.
std::unique_ptr<BlockPipeline> openFilesystemPipeline(
    VcDataset* ds, size_t maxBytes, const std::filesystem::path& datasetPath,
    BlockCache* sharedCache = nullptr);

}  // namespace vc::cache
