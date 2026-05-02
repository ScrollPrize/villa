#pragma once

#include "vc/core/render/IChunkedArray.hpp"

#include <utils/thread_pool.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vc::render {

class ChunkCache final : public IChunkedArray {
public:
    struct LevelInfo {
        std::array<int, 3> shape{};
        std::array<int, 3> chunkShape{};
        LevelTransform transform{};
    };

    struct Options {
        std::size_t decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        // Number of process-wide chunk I/O workers used by this cache. The
        // pool is shared by caches with the same worker count and is not
        // destroyed when a viewer is closed.
        std::size_t maxConcurrentReads = 16;
        std::optional<std::filesystem::path> persistentCachePath;
    };

    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype);
    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype,
               Options options);
    ~ChunkCache() override;

    int numLevels() const override;
    std::array<int, 3> shape(int level) const override;
    std::array<int, 3> chunkShape(int level) const override;
    ChunkDtype dtype() const override;
    double fillValue() const override;
    LevelTransform levelTransform(int level) const override;

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override;
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait) override;

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) override;
    void removeChunkReadyListener(ChunkReadyCallbackId id) override;

    void invalidate();

private:
    enum class EntryStatus {
        InFlight,
        AllFill,
        Data,
        Error
    };

    struct Entry {
        EntryStatus status = EntryStatus::InFlight;
        std::shared_ptr<const std::vector<std::byte>> bytes;
        std::string error;
        std::size_t decodedBytes = 0;
        bool persisted = false;
        bool inLru = false;
        std::list<ChunkKey>::iterator lruIt;
    };

    struct State {
        State(std::vector<LevelInfo> levels,
              std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
              double fillValue,
              ChunkDtype dtype,
              Options options)
            : levels_(std::move(levels))
            , fetchers_(std::move(fetchers))
            , fillValue_(fillValue)
            , dtype_(dtype)
            , options_(std::move(options))
        {}

        std::vector<LevelInfo> levels_;
        std::vector<std::shared_ptr<IChunkFetcher>> fetchers_;
        double fillValue_ = 0.0;
        ChunkDtype dtype_ = ChunkDtype::UInt8;
        Options options_;

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
        std::list<ChunkKey> lru_;
        std::size_t decodedBytes_ = 0;
        std::uint64_t generation_ = 0;
        ChunkReadyCallbackId nextCallbackId_ = 1;
        std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> callbacks_;
    };

    static ChunkResult resultFromEntryLocked(State& state, const ChunkKey& key, Entry& entry);
    static void queueFetchLocked(const std::shared_ptr<State>& state, const ChunkKey& key, std::uint64_t generation);
    static void fetchAndStore(const std::shared_ptr<State>& state, ChunkKey key, std::uint64_t generation);
    static void storeFetchResultLocked(const std::shared_ptr<State>& state,
                                       const ChunkKey& key,
                                       ChunkFetchResult fetch,
                                       bool loadedFromPersistentCache);
    static std::optional<std::vector<std::byte>> readPersistent(const State& state, const ChunkKey& key);
    static bool queuePersistentWrite(const std::shared_ptr<State>& state,
                                     const ChunkKey& key,
                                     std::shared_ptr<const std::vector<std::byte>> bytes);
    static void writePersistent(const State& state, const ChunkKey& key, const std::vector<std::byte>& bytes);
    static std::filesystem::path persistentPath(const State& state, const ChunkKey& key);
    static void touchLocked(State& state, const ChunkKey& key, Entry& entry);
    static void enforceCapacityLocked(const std::shared_ptr<State>& state);
    static bool isValidKey(const State& state, const ChunkKey& key);
    static bool isAllFill(const State& state, const std::vector<std::byte>& bytes);
    static std::size_t dtypeSize(ChunkDtype dtype);
    static std::size_t expectedChunkBytes(const State& state, const ChunkKey& key);
    static void notifyListeners(const std::shared_ptr<State>& state);
    static void waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key);

    std::shared_ptr<State> state_;
};

} // namespace vc::render
