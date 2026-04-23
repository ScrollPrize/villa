#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
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
#include <utils/video_codec.hpp>

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
        // RAM cache of compressed canonical h265 shard files. The loader
        // pool checks this before hitting disk; on a miss it reads the
        // whole shard file once and caches it, so subsequent inner-chunk
        // reads from the same shard are zero-syscall memcpy. Set to 0 to
        // disable shard caching (loader goes straight to disk every time).
        size_t shardCacheBytes = 1ULL << 30; // 1 GiB default
        std::string volumeId;
        // Defaults to hardware_concurrency(); see constructor.
        int ioThreads = 0;
        // H.265 encode parameters used when re-encoding non-canonical source
        // chunks into the canonical disk cache. depth/height/width are filled
        // in per-chunk; qp/air_clamp/shift_n carry the configured values.
        // Default qp=36 matches the historical hard-coded value.
        utils::VideoCodecParams encodeParams = {.qp = 36};

        // When non-zero, declares the source is byte-identical to our local
        // canonical disk format: zarr v3, sharded with these dims, 128^3
        // inner H.265 chunks. The downloader then bypasses the encoder
        // entirely — fetchWholeShard from source, write the bytes verbatim
        // to disk, forward chunk keys directly to the loader. Skipping the
        // decode→re-encode round trip is the whole point.
        // Local shard shape (currently 1024^3) MUST match this for the
        // byte-passthrough to be valid.
        std::array<int, 3> canonicalSourceShard = {0, 0, 0};
    };

    BlockPipeline(
        Config config,
        BlockCache& blockCache,
        std::unique_ptr<VolumeSource> source,
        DecompressFn decompress,
        std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels = {});

    ~BlockPipeline();

    BlockPipeline(const BlockPipeline&) = delete;
    BlockPipeline& operator=(const BlockPipeline&) = delete;

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

    [[nodiscard]] int numLevels() const noexcept;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept;

    // OME-Zarr scale factor for a vector index (e.g. 4.0 for directory "2").
    // Falls back to 2^vectorIndex if not set.
    void setLevelScaleFactors(std::vector<float> factors);
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
    std::vector<float> levelScaleFactors_;  // OME-Zarr scale per vector index
    std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels_;
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
    mutable std::mutex encodeStagingMutex_;
    std::unordered_map<ChunkKey, ChunkDataPtr, ChunkKeyHash> encodeStaging_;
    // Hand-off buffer between loader and decoder. Loader inserts
    // (key → compressed inner-chunk bytes); decoder pops and decodes.
    mutable std::mutex decodeStagingMutex_;
    std::unordered_map<ChunkKey, std::vector<uint8_t>, ChunkKeyHash> decodeStaging_;

    // Shard-level LRU cache of compressed canonical h265 shard files.
    // Populated on loader misses. Bytes-budgeted; when exceeded the
    // least-recently-used shard is evicted. shared_ptr on the buffer so
    // concurrent loaders can serve from the same shard without the cache
    // mutex blocking them.
    mutable std::mutex shardCacheMutex_;
    struct ShardCacheEntry {
        ShardKey key;
        std::shared_ptr<std::vector<std::byte>> bytes;
    };
    std::list<ShardCacheEntry> shardCacheLru_;  // front = most recent
    std::unordered_map<ShardKey,
                       std::list<ShardCacheEntry>::iterator,
                       ShardKeyHash> shardCacheMap_;
    size_t shardCacheTotalBytes_ = 0;
    // Hits/misses so the status bar can surface them.
    std::atomic<uint64_t> statShardHits_{0};
    std::atomic<uint64_t> statShardMisses_{0};

    // Translate a ChunkKey into the shard it lives in (zarr v3 sharded
    // grid coordinates). Empty/non-sharded arrays return the zero shard.
    [[nodiscard]] ShardKey canonicalShardKey(const ChunkKey& key) const noexcept;

    // Pull the whole shard file for `key` through the LRU cache. First
    // hit from any thread reads the file once; subsequent hits just bump
    // the LRU head and return the shared buffer.
    std::shared_ptr<std::vector<std::byte>> shardBytesFor(
        const ChunkKey& key, utils::ZarrArray& dz);

    // Insert into the shard cache, evicting LRU until under budget.
    void shardCacheInsertLocked(const ShardKey& sk,
                                std::shared_ptr<std::vector<std::byte>> bytes);

    BlockCache& blockCache_;
    uint64_t cacheGen_;  // captured at construction from blockCache_.generation()

    // Assemble a canonical 128^3 chunk from one or more source chunks at
    // `canonKey.level`, rechunking as needed. Null if the canonical region
    // is entirely absent from the source.
    [[nodiscard]] ChunkDataPtr assembleCanonicalChunk(const ChunkKey& canonKey);

    // Split a decoded chunk into 16^3 blocks and insert into blockCache_.
    void insertChunkAsBlocks(const ChunkKey& key, const ChunkData& chunk);

    // All-zero canonical chunks: record their key instead of materialising
    // 512 identical zero blocks in the arena. blockAt() returns a pointer
    // to a single static zero-block when the block's canonical chunk is
    // in this set.
    mutable std::mutex emptyChunksMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> emptyChunks_;

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

    mutable std::mutex dataBoundsMutex_;
    DataBoundsL0 dataBoundsL0_;

    mutable std::atomic<uint64_t> statBlockHits_{0};
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
