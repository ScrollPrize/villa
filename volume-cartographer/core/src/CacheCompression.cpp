#include "vc/core/util/CacheCompression.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <zstd.h>

namespace vc {

namespace {

constexpr std::size_t kHeaderSize = 20;
constexpr unsigned char kMagic[4] = {'V', 'C', 'Z', '1'};
constexpr unsigned char kFormatVersion = 1;

// In-place backward difference along z, then y, then x. Iterating each pass
// from the end keeps every subtraction reading not-yet-filtered values; the
// element-wise arithmetic wraps mod 2^8/2^16, so the inverse is exact.
template <typename T>
void deltaZyxFilter(T* data, std::size_t z, std::size_t y, std::size_t x)
{
    const std::size_t slice = y * x;
    const std::size_t n = z * slice;
    for (std::size_t i = n; i-- > slice;)
        data[i] = static_cast<T>(data[i] - data[i - slice]);
    for (std::size_t s = 0; s < z; ++s) {
        T* p = data + s * slice;
        for (std::size_t i = slice; i-- > x;)
            p[i] = static_cast<T>(p[i] - p[i - x]);
    }
    for (std::size_t r = 0; r < z * y; ++r) {
        T* p = data + r * x;
        for (std::size_t i = x; i-- > 1;)
            p[i] = static_cast<T>(p[i] - p[i - 1]);
    }
}

// Inverse: forward prefix sums along x, then y, then z.
template <typename T>
void deltaZyxUnfilter(T* data, std::size_t z, std::size_t y, std::size_t x)
{
    const std::size_t slice = y * x;
    const std::size_t n = z * slice;
    for (std::size_t r = 0; r < z * y; ++r) {
        T* p = data + r * x;
        for (std::size_t i = 1; i < x; ++i)
            p[i] = static_cast<T>(p[i] + p[i - 1]);
    }
    for (std::size_t s = 0; s < z; ++s) {
        T* p = data + s * slice;
        for (std::size_t i = x; i < slice; ++i)
            p[i] = static_cast<T>(p[i] + p[i - x]);
    }
    for (std::size_t i = slice; i < n; ++i)
        data[i] = static_cast<T>(data[i] + data[i - slice]);
}

void writeU32(std::byte* out, std::uint32_t value)
{
    out[0] = static_cast<std::byte>(value & 0xFF);
    out[1] = static_cast<std::byte>((value >> 8) & 0xFF);
    out[2] = static_cast<std::byte>((value >> 16) & 0xFF);
    out[3] = static_cast<std::byte>((value >> 24) & 0xFF);
}

std::uint32_t readU32(const std::byte* in)
{
    return static_cast<std::uint32_t>(in[0]) |
           (static_cast<std::uint32_t>(in[1]) << 8) |
           (static_cast<std::uint32_t>(in[2]) << 16) |
           (static_cast<std::uint32_t>(in[3]) << 24);
}

bool hasVcz1Header(std::span<const std::byte> input)
{
    return input.size() > kHeaderSize &&
           std::memcmp(input.data(), kMagic, sizeof(kMagic)) == 0 &&
           static_cast<unsigned char>(input[4]) == kFormatVersion;
}

std::vector<std::byte> zstdCompressFrame(std::span<const std::byte> input,
                                         int level,
                                         std::size_t headerReserve)
{
    const std::size_t bound = ZSTD_compressBound(input.size());
    std::vector<std::byte> output(headerReserve + bound);
    const std::size_t rc = ZSTD_compress(
        output.data() + headerReserve, bound, input.data(), input.size(), level);
    if (ZSTD_isError(rc)) {
        throw std::runtime_error(
            std::string("cacheCompress: ZSTD_compress failed: ") +
            ZSTD_getErrorName(rc));
    }
    output.resize(headerReserve + rc);
    return output;
}

} // namespace

std::vector<std::byte> cacheCompress(std::span<const std::byte> input,
                                     std::array<int, 3> shapeZYX,
                                     std::size_t elemSize,
                                     int level)
{
    const bool shapeValid =
        (elemSize == 1 || elemSize == 2) &&
        shapeZYX[0] > 0 && shapeZYX[1] > 0 && shapeZYX[2] > 0 &&
        static_cast<std::size_t>(shapeZYX[0]) *
                static_cast<std::size_t>(shapeZYX[1]) *
                static_cast<std::size_t>(shapeZYX[2]) * elemSize ==
            input.size();
    if (!shapeValid) {
        throw std::invalid_argument(
            "cacheCompress: chunk shape/element size does not match payload");
    }

    const auto z = static_cast<std::size_t>(shapeZYX[0]);
    const auto y = static_cast<std::size_t>(shapeZYX[1]);
    const auto x = static_cast<std::size_t>(shapeZYX[2]);

    std::vector<std::byte> filtered(input.begin(), input.end());
    if (elemSize == 1)
        deltaZyxFilter(reinterpret_cast<std::uint8_t*>(filtered.data()), z, y, x);
    else
        deltaZyxFilter(reinterpret_cast<std::uint16_t*>(filtered.data()), z, y, x);

    auto output = zstdCompressFrame(
        std::span<const std::byte>(filtered.data(), filtered.size()),
        level,
        kHeaderSize);
    std::memcpy(output.data(), kMagic, sizeof(kMagic));
    output[4] = static_cast<std::byte>(kFormatVersion);
    output[5] = static_cast<std::byte>(elemSize);
    output[6] = std::byte{0};
    output[7] = std::byte{0};
    writeU32(output.data() + 8, static_cast<std::uint32_t>(z));
    writeU32(output.data() + 12, static_cast<std::uint32_t>(y));
    writeU32(output.data() + 16, static_cast<std::uint32_t>(x));
    return output;
}

std::optional<std::vector<std::byte>> cacheDecompress(
    std::span<const std::byte> input,
    std::size_t expectedSize)
{
    if (!hasVcz1Header(input))
        return std::nullopt;

    const auto elemSize = static_cast<std::size_t>(input[5]);
    const std::size_t z = readU32(input.data() + 8);
    const std::size_t y = readU32(input.data() + 12);
    const std::size_t x = readU32(input.data() + 16);
    if ((elemSize != 1 && elemSize != 2) || z == 0 || y == 0 || x == 0 ||
        z * y * x * elemSize != expectedSize)
        return std::nullopt;

    std::vector<std::byte> output(expectedSize);
    const std::size_t rc = ZSTD_decompress(
        output.data(), output.size(),
        input.data() + kHeaderSize, input.size() - kHeaderSize);
    if (ZSTD_isError(rc) || rc != expectedSize)
        return std::nullopt;

    if (elemSize == 1)
        deltaZyxUnfilter(reinterpret_cast<std::uint8_t*>(output.data()), z, y, x);
    else
        deltaZyxUnfilter(reinterpret_cast<std::uint16_t*>(output.data()), z, y, x);
    return output;
}

} // namespace vc
