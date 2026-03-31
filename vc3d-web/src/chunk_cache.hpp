#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// Simplified chunk cache for WASM: in-memory LRU, no threads, no disk.
// Chunks arrive pre-decompressed from JavaScript (which handles fetch + blosc).

struct ChunkKey {
    int level;
    int iz, iy, ix;

    bool operator==(const ChunkKey& o) const {
        return level == o.level && iz == o.iz && iy == o.iy && ix == o.ix;
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const {
        uint64_t h = uint64_t(k.level) * 73856093ULL ^
                     uint64_t(k.iz) * 19349669ULL ^
                     uint64_t(k.iy) * 83492791ULL ^
                     uint64_t(k.ix) * 50331653ULL;
        return static_cast<size_t>(h);
    }
};

struct ChunkData {
    std::vector<uint8_t> bytes;
    std::array<int, 3> shape{0, 0, 0}; // {z, y, x}

    const uint8_t* data() const { return bytes.data(); }
    size_t totalBytes() const { return bytes.size(); }
    int strideZ() const { return shape[1] * shape[2]; }
    int strideY() const { return shape[2]; }
};

using ChunkDataPtr = std::shared_ptr<ChunkData>;

struct LevelMeta {
    std::array<int, 3> shape;      // volume dimensions {z, y, x}
    std::array<int, 3> chunkShape; // chunk dimensions {z, y, x}
};

class ChunkCache {
public:
    void setLevels(std::vector<LevelMeta> levels) { levels_ = std::move(levels); }
    int numLevels() const { return static_cast<int>(levels_.size()); }

    std::array<int, 3> chunkShape(int level) const {
        if (level < 0 || level >= numLevels()) return {0, 0, 0};
        return levels_[level].chunkShape;
    }

    std::array<int, 3> levelShape(int level) const {
        if (level < 0 || level >= numLevels()) return {0, 0, 0};
        return levels_[level].shape;
    }

    // Load a chunk from JS. Data must be raw decompressed uint8 voxels
    // in ZYX row-major order.
    void loadChunk(int level, int iz, int iy, int ix,
                   const uint8_t* data, size_t size) {
        ChunkKey key{level, iz, iy, ix};
        auto chunk = std::make_shared<ChunkData>();
        chunk->bytes.assign(data, data + size);
        chunk->shape = chunkShape(level);
        chunks_[key] = chunk;
        totalBytes_ += size;
        evictIfNeeded();
    }

    ChunkDataPtr get(const ChunkKey& key) const {
        auto it = chunks_.find(key);
        return (it != chunks_.end()) ? it->second : nullptr;
    }

    void setMaxBytes(size_t maxBytes) { maxBytes_ = maxBytes; }
    size_t totalBytes() const { return totalBytes_; }
    size_t count() const { return chunks_.size(); }

private:
    void evictIfNeeded() {
        // Simple eviction: if over budget, remove oldest entries
        // (unordered_map iteration order is arbitrary but good enough)
        while (totalBytes_ > maxBytes_ && !chunks_.empty()) {
            auto it = chunks_.begin();
            totalBytes_ -= it->second->totalBytes();
            chunks_.erase(it);
        }
    }

    std::vector<LevelMeta> levels_;
    std::unordered_map<ChunkKey, ChunkDataPtr, ChunkKeyHash> chunks_;
    size_t totalBytes_ = 0;
    size_t maxBytes_ = 512 * 1024 * 1024; // 512 MB default
};
