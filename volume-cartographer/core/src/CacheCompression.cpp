#include "vc/core/util/CacheCompression.hpp"

#include <stdexcept>
#include <string>

#include <zstd.h>

namespace vc {

std::vector<std::byte> cacheCompress(std::span<const std::byte> input, int level)
{
    const std::size_t bound = ZSTD_compressBound(input.size());
    std::vector<std::byte> output(bound);
    const std::size_t rc = ZSTD_compress(
        output.data(), bound, input.data(), input.size(), level);
    if (ZSTD_isError(rc)) {
        throw std::runtime_error(
            std::string("cacheCompress: ZSTD_compress failed: ") +
            ZSTD_getErrorName(rc));
    }
    output.resize(rc);
    return output;
}

std::optional<std::vector<std::byte>> cacheDecompress(
    std::span<const std::byte> input,
    std::size_t expectedSize)
{
    std::vector<std::byte> output(expectedSize);
    const std::size_t rc = ZSTD_decompress(
        output.data(), output.size(), input.data(), input.size());
    if (ZSTD_isError(rc) || rc != expectedSize)
        return std::nullopt;
    return output;
}

} // namespace vc
