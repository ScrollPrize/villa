#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "ChunkKey.hpp"
#include "HugePageAllocator.hpp"

namespace vc::cache {

// Decompressed chunk data, ready for sampling.
// Stores raw bytes with shape metadata. Callers cast to the appropriate type
// (uint8_t, uint16_t, float, etc.) via the data<T>() accessors.
//
// Storage: uses a 2MB huge-page-aligned buffer from HugePageAllocator when
// the data fits (<=2MB). Falls back to std::vector for larger buffers.
struct ChunkData {
    std::vector<uint8_t> bytes;          // fallback for >2MB or legacy path
    HugePageBuffer hugeBuf;              // 2MB-aligned buffer from pool
    std::array<int, 3> shape{0, 0, 0};   // {z, y, x}
    int elementSize = 1;                  // bytes per element (1=u8, 2=u16, 4=f32)

    [[nodiscard]] size_t numElements() const noexcept
    {
        return static_cast<size_t>(shape[0]) * shape[1] * shape[2];
    }

    [[nodiscard]] size_t totalBytes() const noexcept
    {
        return hugeBuf.ptr ? hugeBuf.size : bytes.size();
    }

    // Resize storage: uses huge page pool for sizes <= 2MB, vector otherwise.
    void resizeBytes(size_t n)
    {
        if (n <= HugePageAllocator::k2MB) {
            hugeBuf.resize(n);
            bytes.clear();
        } else {
            hugeBuf = HugePageBuffer{};
            bytes.resize(n);
        }
    }

    // Raw byte pointer (whichever storage is active).
    [[nodiscard]] uint8_t* rawData() noexcept
    {
        return hugeBuf.ptr ? hugeBuf.ptr : bytes.data();
    }
    [[nodiscard]] const uint8_t* rawData() const noexcept
    {
        return hugeBuf.ptr ? hugeBuf.ptr : bytes.data();
    }

    template <typename T>
    [[nodiscard]] T* data() noexcept
    {
        return reinterpret_cast<T*>(rawData());
    }

    template <typename T>
    [[nodiscard]] const T* data() const noexcept
    {
        return reinterpret_cast<const T*>(rawData());
    }

    // Stride helpers for (z, y, x) indexing into the flat buffer.
    // Physical layout is row-major: z varies slowest, x varies fastest.
    [[nodiscard]] int strideZ() const noexcept { return shape[1] * shape[2]; }
    [[nodiscard]] int strideY() const noexcept { return shape[2]; }
    [[nodiscard]] int strideX() const noexcept { return 1; }
};

using ChunkDataPtr = std::shared_ptr<ChunkData>;

// Compressed chunk bytes (warm tier / on-disk storage).
struct CompressedChunk {
    std::vector<uint8_t> data;  // compressed bytes
};

// Callback signature for decompressing raw bytes into ChunkData.
// The cache itself is compression-agnostic; the caller provides this.
using DecompressFn = std::function<ChunkDataPtr(
    const std::vector<uint8_t>& compressed,
    const ChunkKey& key)>;

// Optional callback for recompressing chunks before writing to disk cache.
// Called with the original compressed bytes from the remote source.
// Returns recompressed bytes (e.g., video codec compressed).
// If null, chunks are stored to disk in their original format.
using RecompressFn = std::function<std::vector<uint8_t>(
    const std::vector<uint8_t>& original,
    const ChunkKey& key)>;

}  // namespace vc::cache
