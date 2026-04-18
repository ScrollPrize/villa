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


static ChunkDataPtr acquireChunkData(size_t bytesNeeded)
{
    auto result = std::make_unique<ChunkData>();
    result->resizeBytes(bytesNeeded);
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

            const size_t decodedSize = size_t(dims[0]) * dims[1] * dims[2];
            int cz = static_cast<int>(chunkShape[0]);
            int cy = static_cast<int>(chunkShape[1]);
            int cx = static_cast<int>(chunkShape[2]);

            auto result = acquireChunkData(chunkSize);
            result->shape = {cz, cy, cx};
            result->elementSize = 1;

            // Decode straight into the result buffer when it matches; saves
            // one 2 MiB allocation + memcpy per chunk on the hot path.
            // Fall back to a temp buffer when the shapes mismatch (rare).
            if (decodedSize == chunkSize) {
                utils::video_decode_into(
                    std::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(compressed.data()),
                        compressed.size()),
                    std::span<std::byte>(
                        reinterpret_cast<std::byte*>(result->rawData()),
                        chunkSize),
                    vp);
            } else {
                auto decoded = utils::video_decode(
                    std::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(compressed.data()),
                        compressed.size()),
                    decodedSize, vp);
                std::memcpy(result->rawData(), decoded.data(),
                            std::min(decoded.size(), chunkSize));
            }
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

            // Forward scan is technically safe (dst[i] only writes byte i
            // while src[i] reads bytes 2i..2i+1, which are ahead of the
            // write cursor), but aliasing a uint16_t* over a buffer that's
            // concurrently being written via uint8_t* violates strict
            // aliasing and is a minefield under TBAA + -ffast-math. Use
            // memcpy to read each uint16 before writing the uint8.
            auto* dst = result->rawData();
            for (size_t i = 0; i < chunkSize; ++i) {
                uint16_t v;
                std::memcpy(&v, dst + i * 2, sizeof(v));
                dst[i] = static_cast<uint8_t>(v / 257);
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
