#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace vc::core::util {

using Shape3 = std::array<size_t, 3>;

struct SparseChunkIndex {
    uint32_t z;
    uint32_t y;
    uint32_t x;

    bool operator==(const SparseChunkIndex& other) const noexcept {
        return z == other.z && y == other.y && x == other.x;
    }
};

struct SparseChunkIndexHash {
    size_t operator()(const SparseChunkIndex& c) const noexcept;
};

bool sparseChunkIndexLess(const SparseChunkIndex& a, const SparseChunkIndex& b);

struct SparseChunkRecordU8x7 {
    uint8_t z;
    uint8_t y;
    uint8_t x;
    std::array<uint8_t, 7> values;
};

struct SparseChunkSpoolStats {
    size_t touchedChunks = 0;
    size_t spillFiles = 0;
    uint64_t appendedRecords = 0;
    size_t inMemoryBytes = 0;
    size_t inMemoryBudgetBytes = 0;
};

class SparseChunkSpool {
public:
    SparseChunkSpool(std::filesystem::path spoolDir,
                     const Shape3& chunkShape,
                     const Shape3& volumeShape,
                     size_t inMemoryMaxBytes);
    ~SparseChunkSpool();

    const Shape3& chunkShape() const { return chunkShape_; }
    const Shape3& volumeShape() const { return volumeShape_; }
    const std::filesystem::path& spoolDir() const { return spoolDir_; }

    void appendChunkRecords(const SparseChunkIndex& chunk,
                            const std::vector<SparseChunkRecordU8x7>& records);

    bool readChunkRecords(const SparseChunkIndex& chunk,
                          std::vector<SparseChunkRecordU8x7>& out) const;

    std::vector<SparseChunkIndex> touchedChunks() const;
    SparseChunkSpoolStats stats() const;

private:
    std::filesystem::path spoolPathFor(const SparseChunkIndex& chunk) const;

    struct Impl;
    std::filesystem::path spoolDir_;
    Shape3 chunkShape_;
    Shape3 volumeShape_;
    size_t inMemoryMaxBytes_;
    std::unique_ptr<Impl> impl_;
};

class SparseChunkSpoolBuffer {
public:
    explicit SparseChunkSpoolBuffer(SparseChunkSpool& spool);

    void emit(size_t z, size_t y, size_t x, const std::array<uint8_t, 7>& values);
    void flushAll();

private:
    void flushChunk(const SparseChunkIndex& chunk);

    SparseChunkSpool* spool_;
    std::unordered_map<SparseChunkIndex,
                       std::vector<SparseChunkRecordU8x7>,
                       SparseChunkIndexHash> buffers_;
};

} // namespace vc::core::util
