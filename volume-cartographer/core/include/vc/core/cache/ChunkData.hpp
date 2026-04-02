#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "ChunkKey.hpp"

namespace vc::cache {

// Decompressed chunk data, ready for sampling.
// Stores raw bytes with shape metadata. Callers cast to the appropriate type
// (uint8_t, uint16_t, float, etc.) via the data<T>() accessors.
struct ChunkData {
    std::vector<uint8_t> bytes;
    std::array<int, 3> shape{0, 0, 0};   // {z, y, x}
    int elementSize = 1;                  // bytes per element (1=u8, 2=u16, 4=f32)
    bool isEmpty = false;                 // true = all voxels are zero (skip sampling)

    [[nodiscard]] constexpr size_t numElements() const noexcept
    {
        return static_cast<size_t>(shape[0]) * shape[1] * shape[2];
    }

    [[nodiscard]] constexpr size_t totalBytes() const noexcept
    {
        return bytes.size();
    }

    void resizeBytes(size_t n)
    {
        bytes.resize(n);
    }

    [[nodiscard]] uint8_t* rawData() noexcept
    {
        return bytes.data();
    }
    [[nodiscard]] const uint8_t* rawData() const noexcept
    {
        return bytes.data();
    }

    template <typename T>
    [[nodiscard]] T* data() noexcept
    {
        return reinterpret_cast<T*>(bytes.data());
    }

    template <typename T>
    [[nodiscard]] const T* data() const noexcept
    {
        return reinterpret_cast<const T*>(bytes.data());
    }

    // Stride helpers for (z, y, x) indexing into the flat buffer.
    // Physical layout is row-major: z varies slowest, x varies fastest.
    [[nodiscard]] constexpr int strideZ() const noexcept { return shape[1] * shape[2]; }
    [[nodiscard]] constexpr int strideY() const noexcept { return shape[2]; }
    [[nodiscard]] constexpr int strideX() const noexcept { return 1; }
};

using ChunkDataPtr = std::shared_ptr<ChunkData>;

// Callback signature for decompressing raw bytes into ChunkData.
// The cache itself is compression-agnostic; the caller provides this.
using DecompressFn = std::function<ChunkDataPtr(
    const std::vector<uint8_t>& compressed,
    const ChunkKey& key)>;

}  // namespace vc::cache
