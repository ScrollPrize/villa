#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/types/VcDataset.hpp"

#include <stdexcept>

namespace vc::cache {

DecompressFn makeVcDecompressor(const std::vector<vc::VcDataset*>& datasets)
{
    // Capture the dataset pointers (caller guarantees lifetime)
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
        const auto dtype = ds.getDtype();

        auto result = std::make_shared<ChunkData>();
        result->shape = {
            static_cast<int>(chunkShape[0]),
            static_cast<int>(chunkShape[1]),
            static_cast<int>(chunkShape[2])};

        if (dtype == vc::VcDtype::uint8) {
            result->elementSize = 1;
            result->bytes.resize(chunkSize);
            ds.decompress(compressed, result->bytes.data(), chunkSize);
        } else if (dtype == vc::VcDtype::uint16) {
            // Decompress uint16 directly into the result buffer (which is
            // large enough for uint16 data), then convert in-place to uint8.
            result->elementSize = 1;
            result->bytes.resize(chunkSize * 2);
            ds.decompress(compressed, result->bytes.data(), chunkSize);

            // Convert uint16 -> uint8 in-place (read from front, write
            // from front; source stride is 2x dest so no overlap issues).
            auto* src = reinterpret_cast<const uint16_t*>(result->bytes.data());
            for (size_t i = 0; i < chunkSize; i++) {
                result->bytes[i] = static_cast<uint8_t>(src[i] / 257);
            }
            result->bytes.resize(chunkSize);
        } else {
            return nullptr;  // unsupported dtype
        }

        return result;
    };
}

DecompressFn makeVcDecompressor(vc::VcDataset* ds)
{
    return makeVcDecompressor(std::vector<vc::VcDataset*>{ds});
}

}  // namespace vc::cache
