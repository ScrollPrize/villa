#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "BlockCache.hpp"
#include "ChunkData.hpp"
#include "ChunkKey.hpp"
#include "VolumeSource.hpp"
#include "IOPool.hpp"
#include <utils/zarr.hpp>

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
        size_t bytes = 10ULL << 30;  // 10 GiB
        std::string volumeId;
        // Defaults to hardware_concurrency(); see constructor.
        int ioThreads = 0;
    };

    BlockPipeline(
        Config config,
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
    // into 16^3 blocks and inserted into the block cache.
    void fetchInteractive(const std::vector<ChunkKey>& keys);

    // --- Cache management ---
    void clearMemory();
    void clearAll();

    [[nodiscard]] int numLevels() const noexcept;
    [[nodiscard]] std::array<int, 3> chunkShape(int level) const noexcept;
    [[nodiscard]] std::array<int, 3> levelShape(int level) const noexcept;

    void flushPersistentState();

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
        size_t ioPending = 0;
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
    std::vector<std::shared_ptr<utils::ZarrArray>> diskLevels_;
    std::unique_ptr<VolumeSource> source_;
    DecompressFn decompress_;
    IOPool ioPool_;

    BlockCache blockCache_;

    // Assemble a canonical 128^3 chunk from one or more source chunks at
    // `canonKey.level`, rechunking as needed. Null if the canonical region
    // is entirely absent from the source.
    [[nodiscard]] ChunkDataPtr assembleCanonicalChunk(const ChunkKey& canonKey);

    // Split a decoded chunk into 16^3 blocks and insert into blockCache_.
    void insertChunkAsBlocks(const ChunkKey& key, const ChunkData& chunk);

    // Negative cache (same design as before).
    static constexpr size_t kBloomBits = 65536;
    std::array<std::atomic<uint64_t>, kBloomBits / 64> negativeBloom_{};
    void bloomAdd(const ChunkKey& key) noexcept;
    [[nodiscard]] bool bloomMayContain(const ChunkKey& key) const noexcept;
    void bloomClear() noexcept;
    mutable std::mutex negativeMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> negativeCache_;
    void loadNegativeCache();
    void saveNegativeCache() const;

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
    VcDataset* ds, size_t maxBytes, const std::filesystem::path& datasetPath);

}  // namespace vc::cache
