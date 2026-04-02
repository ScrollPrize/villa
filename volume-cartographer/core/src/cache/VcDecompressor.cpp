#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#if __has_include("utils/video_codec.hpp")
#include "utils/video_codec.hpp"
#endif

namespace vc::cache {

// Check if all bytes in a chunk are zero.
static bool isChunkEmpty(const uint8_t* data, size_t n)
{
    // Check 8 bytes at a time
    const uint64_t* p = reinterpret_cast<const uint64_t*>(data);
    size_t n8 = n / 8;
    for (size_t i = 0; i < n8; i++)
        if (p[i]) return false;
    for (size_t i = n8 * 8; i < n; i++)
        if (data[i]) return false;
    return true;
}


// Thread-local pool of pre-allocated ChunkData objects to avoid
// malloc/free churn on every decompress (0.46% of CPU in profiles).
// Each thread keeps up to 4 recycled buffers. When a ChunkDataPtr's
// refcount drops to 1 (only the pool holds it), it's available for reuse.
//
// Buffers are backed by the HugePageAllocator (2MB-aligned pages) when
// the decoded chunk fits within 2MB (128^3 uint8 = 2MB exactly).
static ChunkDataPtr acquireChunkData(size_t bytesNeeded)
{
    // Small thread-local free list
    constexpr size_t kPoolSize = 4;
    struct Pool {
        ChunkDataPtr bufs[kPoolSize];
        int count = 0;
    };
    thread_local Pool pool;

    // Try to reuse a pooled buffer with sufficient capacity
    for (int i = 0; i < pool.count; i++) {
        if (pool.bufs[i].use_count() == 1 &&
            pool.bufs[i]->totalBytes() >= bytesNeeded) {
            auto result = pool.bufs[i];
            result->resizeBytes(bytesNeeded);
            return result;
        }
    }

    // Allocate fresh — uses huge page pool for sizes <= 2MB
    auto result = std::make_shared<ChunkData>();
    result->resizeBytes(bytesNeeded);

    // Try to add to pool for future reuse
    if (pool.count < static_cast<int>(kPoolSize)) {
        pool.bufs[pool.count++] = result;
    } else {
        // Replace the first entry with refcount > 1 (in active use, won't
        // be recycled anyway), or the first entry if all are reclaimable.
        for (int i = 0; i < static_cast<int>(kPoolSize); i++) {
            if (pool.bufs[i].use_count() > 1) {
                pool.bufs[i] = result;
                break;
            }
        }
    }

    return result;
}

DecompressFn makeVcDecompressor(const std::vector<vc::VcDataset*>& datasets)
{
    return [datasets](const std::vector<uint8_t>& compressed,
                      const ChunkKey& key) -> ChunkDataPtr {
        if (key.level < 0 ||
            key.level >= static_cast<int>(datasets.size()) ||
            !datasets[key.level]) {
            return nullptr;
        }

        vc::VcDataset& ds = *datasets[key.level];
        const auto& chunkShape = ds.defaultChunkShape();
        const size_t chunkSize = ds.defaultChunkSize();

        // Determine required buffer size upfront to reuse pooled buffers
        const auto dtype = ds.getDtype();
        size_t bufferBytes = (dtype == vc::VcDtype::uint16) ? chunkSize * 2 : chunkSize;

#ifdef UTILS_HAS_VIDEO_CODEC
        // Check for VC3D video codec magic header
        if (utils::is_video_compressed(
                std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(compressed.data()),
                    compressed.size()))) {
            auto dims = utils::video_header_dims(
                std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(compressed.data()),
                    compressed.size()));

            utils::VideoCodecParams vp;
            vp.depth = dims[0];
            vp.height = dims[1];
            vp.width = dims[2];

            auto decoded = utils::video_decode(
                std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(compressed.data()),
                    compressed.size()),
                size_t(dims[0]) * dims[1] * dims[2], vp);

            int cz = static_cast<int>(chunkShape[0]);
            int cy = static_cast<int>(chunkShape[1]);
            int cx = static_cast<int>(chunkShape[2]);
            size_t copySize = std::min(decoded.size(), chunkSize);

            auto result = acquireChunkData(chunkSize);
            result->shape = {cz, cy, cx};
            result->elementSize = 1;

            std::memcpy(result->rawData(), decoded.data(), copySize);
            result->blockLayout = false;
            result->isEmpty = isChunkEmpty(result->rawData(), chunkSize);
            return result;
        }
#endif

        int cz = static_cast<int>(chunkShape[0]);
        int cy = static_cast<int>(chunkShape[1]);
        int cx = static_cast<int>(chunkShape[2]);

        auto result = acquireChunkData(bufferBytes);
        result->shape = {cz, cy, cx};
        result->elementSize = 1;

        // Normal zarr decompression path
        if (dtype == vc::VcDtype::uint8) {
            ds.decompress(compressed, result->rawData(), chunkSize);
        } else if (dtype == vc::VcDtype::uint16) {
            ds.decompress(compressed, result->rawData(), chunkSize);

            auto* dst = result->rawData();
            auto* src = reinterpret_cast<const uint16_t*>(dst);
            for (size_t i = 0; i < chunkSize; i++) {
                dst[i] = static_cast<uint8_t>(src[i] / 257);
            }
            result->resizeBytes(chunkSize);
        } else {
            return nullptr;
        }

        result->isEmpty = isChunkEmpty(result->rawData(), chunkSize);
        return result;
    };
}

DecompressFn makeVcDecompressor(vc::VcDataset* ds)
{
    return makeVcDecompressor(std::vector<vc::VcDataset*>{ds});
}

}  // namespace vc::cache
