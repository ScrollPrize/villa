#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <cstring>
#include <stdexcept>

#if __has_include("utils/video_codec.hpp")
#include "utils/video_codec.hpp"
#endif

namespace vc::cache {

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

        auto result = std::make_shared<ChunkData>();
        result->shape = {
            static_cast<int>(chunkShape[0]),
            static_cast<int>(chunkShape[1]),
            static_cast<int>(chunkShape[2])};
        result->elementSize = 1;

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

            // The video codec always produces uint8 data
            result->bytes.resize(chunkSize);
            size_t copySize = std::min(decoded.size(), chunkSize);
            std::memcpy(result->bytes.data(), decoded.data(), copySize);
            return result;
        }
#endif

        // Normal zarr decompression path
        const auto dtype = ds.getDtype();

        if (dtype == vc::VcDtype::uint8) {
            result->bytes.resize(chunkSize);
            ds.decompress(compressed, result->bytes.data(), chunkSize);
        } else if (dtype == vc::VcDtype::uint16) {
            result->bytes.resize(chunkSize * 2);
            ds.decompress(compressed, result->bytes.data(), chunkSize);

            auto* src = reinterpret_cast<const uint16_t*>(result->bytes.data());
            for (size_t i = 0; i < chunkSize; i++) {
                result->bytes[i] = static_cast<uint8_t>(src[i] / 257);
            }
            result->bytes.resize(chunkSize);
        } else {
            return nullptr;
        }

        return result;
    };
}

DecompressFn makeVcDecompressor(vc::VcDataset* ds)
{
    return makeVcDecompressor(std::vector<vc::VcDataset*>{ds});
}

RecompressFn makeVideoRecompressor(
    const std::vector<vc::VcDataset*>& datasets,
    int codecType, int qp)
{
#ifdef UTILS_HAS_VIDEO_CODEC
    return [datasets, codecType, qp](const std::vector<uint8_t>& original,
                                     const ChunkKey& key) -> std::vector<uint8_t> {
        if (key.level < 0 ||
            key.level >= static_cast<int>(datasets.size()) ||
            !datasets[key.level]) {
            return {};  // Can't recompress without dataset info
        }

        // Don't re-recompress already video-compressed data
        if (utils::is_video_compressed(
                std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(original.data()),
                    original.size()))) {
            return {};  // Already video compressed
        }

        vc::VcDataset& ds = *datasets[key.level];
        const auto& chunkShape = ds.defaultChunkShape();
        const size_t chunkSize = ds.defaultChunkSize();
        const auto dtype = ds.getDtype();

        // Decompress original data
        std::vector<uint8_t> raw(dtype == vc::VcDtype::uint16 ? chunkSize * 2 : chunkSize);
        try {
            ds.decompress(original, raw.data(), chunkSize);
        } catch (...) {
            return {};  // Can't decompress, skip recompression
        }

        // Convert uint16 -> uint8 if needed
        if (dtype == vc::VcDtype::uint16) {
            auto* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < chunkSize; i++) {
                raw[i] = static_cast<uint8_t>(src[i] / 257);
            }
            raw.resize(chunkSize);
        }

        // Encode with video codec
        utils::VideoCodecParams vp;
        vp.type = static_cast<utils::VideoCodecType>(codecType);
        vp.qp = qp;
        vp.depth = static_cast<int>(chunkShape[0]);
        vp.height = static_cast<int>(chunkShape[1]);
        vp.width = static_cast<int>(chunkShape[2]);

        auto encoded = utils::video_encode(
            std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(raw.data()), raw.size()),
            vp);

        // Convert std::byte -> uint8_t
        std::vector<uint8_t> result(encoded.size());
        std::memcpy(result.data(), encoded.data(), encoded.size());
        return result;
    };
#else
    (void)datasets;
    (void)codecType;
    (void)qp;
    return nullptr;
#endif
}

}  // namespace vc::cache
