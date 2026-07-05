#include "vc/core/util/CacheCompression.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
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

// Snap to the nearest multiple of `width` (bins centered on multiples, so 0
// stays exactly 0 and masked background survives untouched). Idempotent:
// bin centers map to themselves.
template <typename T>
void quantizeValues(T* data, std::size_t n, int width)
{
    constexpr int maxVal = std::numeric_limits<T>::max();
    const int half = width / 2;
    if constexpr (sizeof(T) == 1) {
        T lut[256];
        for (int v = 0; v < 256; ++v)
            lut[v] = static_cast<T>(std::min((v + half) / width * width, maxVal));
        for (std::size_t i = 0; i < n; ++i)
            data[i] = lut[data[i]];
    } else {
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<T>(
                std::min((data[i] + half) / width * width, maxVal));
    }
}

// ---- order-0 rANS (ryg's rans64 construction): 12-bit frequency tables,
// eight interleaved 64-bit states, 32-bit renormalization. Encoding walks
// the symbols back to front and emits renorm words back to front, so the
// decoder streams strictly forward. Division-free encode via 64-bit
// reciprocals (Alverson); exact for all 64-bit states. ----

constexpr int kRansScaleBits = 12;
constexpr std::uint32_t kRansM = 1u << kRansScaleBits;   // total frequency
constexpr std::uint64_t kRansL = 1ull << 31;             // renorm lower bound
constexpr std::size_t kRansLanes = 8;
constexpr std::size_t kRansTableBytes = 256 * 2;         // uint16 per symbol
constexpr std::size_t kRansStateBytes = kRansLanes * 8;

struct RansEncSym {
    std::uint64_t rcp;   // reciprocal so q = mulhi(x, rcp) >> shift = x/freq
    std::uint64_t bias;  // cum, or cum + M - 1 for freq 1 (rcp = 2^64 - 1)
    std::uint32_t cmpl;  // M - freq
    std::uint32_t shift;
};

inline std::uint64_t ransMulHi(std::uint64_t a, std::uint64_t b)
{
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(a) * b) >> 64);
}

// Histogram -> frequencies summing to exactly kRansM, every present symbol
// nonzero. Rounding drift is settled against the most frequent symbol.
void ransNormalize(const std::uint64_t hist[256],
                   std::uint64_t total,
                   std::uint32_t freq[256])
{
    std::uint64_t assigned = 0;
    for (int s = 0; s < 256; ++s) {
        freq[s] = hist[s]
            ? static_cast<std::uint32_t>(
                  std::max<std::uint64_t>(1, (hist[s] * kRansM) / total))
            : 0;
        assigned += freq[s];
    }
    while (assigned != kRansM) {
        int s = 0;
        for (int i = 1; i < 256; ++i)
            if (freq[i] > freq[s]) s = i;
        if (assigned > kRansM) {
            const auto cut = std::min<std::uint64_t>(freq[s] - 1, assigned - kRansM);
            freq[s] -= static_cast<std::uint32_t>(cut);
            assigned -= cut;
        } else {
            freq[s] += static_cast<std::uint32_t>(kRansM - assigned);
            assigned = kRansM;
        }
    }
}

void ransBuildEncTable(const std::uint32_t freq[256], RansEncSym enc[256])
{
    std::uint32_t cum = 0;
    for (int s = 0; s < 256; ++s) {
        const std::uint32_t f = freq[s];
        RansEncSym& e = enc[s];
        e.cmpl = kRansM - f;
        if (f == 0) {
            e = {0, 0, 0, 0};
        } else if (f == 1) {
            // mulhi(x, 2^64 - 1) = x - 1; the off-by-(M-1) folds into bias
            e.rcp = ~0ull;
            e.shift = 0;
            e.bias = static_cast<std::uint64_t>(cum) + kRansM - 1;
        } else {
            std::uint32_t k = 1;
            while ((1u << k) < f) ++k;  // k = ceil(log2 f)
            e.shift = k - 1;
            const auto num =
                ((static_cast<unsigned __int128>(1) << (k + 63)) + f - 1);
            e.rcp = static_cast<std::uint64_t>(num / f);
            e.bias = cum;
        }
        cum += f;
    }
}

// Compresses `input` into [headerReserve][freq table][states+stream].
std::vector<std::byte> ransCompressFrame(std::span<const std::byte> input,
                                         std::size_t headerReserve)
{
    const std::size_t n = input.size();
    const auto* in = reinterpret_cast<const std::uint8_t*>(input.data());

    std::uint64_t h4[4][256] = {};
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        h4[0][in[i]]++;
        h4[1][in[i + 1]]++;
        h4[2][in[i + 2]]++;
        h4[3][in[i + 3]]++;
    }
    for (; i < n; ++i) h4[0][in[i]]++;
    std::uint64_t hist[256];
    for (int s = 0; s < 256; ++s)
        hist[s] = h4[0][s] + h4[1][s] + h4[2][s] + h4[3][s];

    std::uint32_t freq[256];
    ransNormalize(hist, n, freq);
    RansEncSym enc[256];
    ransBuildEncTable(freq, enc);

    // Worst case is ~1.5 bytes/symbol (12-bit code ceiling); 2n is safe.
    const std::size_t streamCap = 2 * n + kRansStateBytes + 64;
    std::vector<std::byte> output(headerReserve + kRansTableBytes + streamCap);
    auto* base = reinterpret_cast<std::uint8_t*>(output.data());

    std::uint8_t* table = base + headerReserve;
    for (int s = 0; s < 256; ++s) {
        table[2 * s] = static_cast<std::uint8_t>(freq[s] & 0xFF);
        table[2 * s + 1] = static_cast<std::uint8_t>(freq[s] >> 8);
    }

    std::uint8_t* end = base + output.size();
    std::uint8_t* p = end;
    std::uint64_t x[kRansLanes];
    std::fill(std::begin(x), std::end(x), kRansL);
    for (std::size_t j = n; j-- > 0;) {
        const RansEncSym& e = enc[in[j]];
        std::uint64_t& s = x[j & (kRansLanes - 1)];
        const std::uint64_t xmax = ((kRansL >> kRansScaleBits) << 32) * (kRansM - e.cmpl);
        if (s >= xmax) {
            p -= 4;
            const auto w = static_cast<std::uint32_t>(s);
            std::memcpy(p, &w, 4);
            s >>= 32;
        }
        const std::uint64_t q = ransMulHi(s, e.rcp) >> e.shift;
        s = s + e.bias + q * e.cmpl;
    }
    for (std::size_t k = kRansLanes; k-- > 0;) {
        p -= 8;
        std::memcpy(p, &x[k], 8);
    }

    const std::size_t streamSize = static_cast<std::size_t>(end - p);
    std::memmove(base + headerReserve + kRansTableBytes, p, streamSize);
    output.resize(headerReserve + kRansTableBytes + streamSize);
    return output;
}

// Decodes a rANS frame produced by ransCompressFrame. Returns false on any
// inconsistency (bad table, short stream, or states/stream not ending
// exactly where encoding started them).
bool ransDecompressFrame(std::span<const std::byte> frame,
                         std::byte* outBytes,
                         std::size_t n)
{
    if (frame.size() < kRansTableBytes + kRansStateBytes)
        return false;
    const auto* table = reinterpret_cast<const std::uint8_t*>(frame.data());

    std::uint32_t freq[256];
    std::uint32_t cum[256];
    std::uint32_t total = 0;
    for (int s = 0; s < 256; ++s) {
        freq[s] = static_cast<std::uint32_t>(table[2 * s]) |
                  (static_cast<std::uint32_t>(table[2 * s + 1]) << 8);
        cum[s] = total;
        total += freq[s];
    }
    if (total != kRansM)
        return false;
    static thread_local std::uint8_t slot2sym[kRansM];
    for (int s = 0; s < 256; ++s)
        if (freq[s])
            std::memset(slot2sym + cum[s], s, freq[s]);

    const auto* p = reinterpret_cast<const std::uint8_t*>(frame.data()) +
                    kRansTableBytes;
    const auto* end = reinterpret_cast<const std::uint8_t*>(frame.data()) +
                      frame.size();
    std::uint64_t x[kRansLanes];
    for (std::size_t k = 0; k < kRansLanes; ++k) {
        std::memcpy(&x[k], p, 8);
        p += 8;
    }

    auto* out = reinterpret_cast<std::uint8_t*>(outBytes);
    std::size_t i = 0;
    for (; i + kRansLanes <= n; i += kRansLanes) {
        for (std::size_t k = 0; k < kRansLanes; ++k) {
            std::uint64_t& s = x[k];
            const auto slot = static_cast<std::uint32_t>(s) & (kRansM - 1);
            const std::uint8_t sym = slot2sym[slot];
            out[i + k] = sym;
            s = static_cast<std::uint64_t>(freq[sym]) * (s >> kRansScaleBits) +
                slot - cum[sym];
            if (s < kRansL) {
                std::uint32_t w = 0;
                if (p + 4 > end)
                    return false;
                std::memcpy(&w, p, 4);
                p += 4;
                s = (s << 32) | w;
            }
        }
    }
    for (; i < n; ++i) {
        std::uint64_t& s = x[i & (kRansLanes - 1)];
        const auto slot = static_cast<std::uint32_t>(s) & (kRansM - 1);
        const std::uint8_t sym = slot2sym[slot];
        out[i] = sym;
        s = static_cast<std::uint64_t>(freq[sym]) * (s >> kRansScaleBits) +
            slot - cum[sym];
        if (s < kRansL) {
            std::uint32_t w = 0;
            if (p + 4 > end)
                return false;
            std::memcpy(&w, p, 4);
            p += 4;
            s = (s << 32) | w;
        }
    }

    // Encoding started every lane at kRansL and the decoder must consume the
    // stream exactly; anything else means a corrupt payload.
    if (p != end)
        return false;
    for (std::size_t k = 0; k < kRansLanes; ++k)
        if (x[k] != kRansL)
            return false;
    return true;
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
                                     int level,
                                     int quantBinWidth,
                                     CacheCodec codec)
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
    if (quantBinWidth < 1 || quantBinWidth > 255) {
        throw std::invalid_argument(
            "cacheCompress: quantization bin width must be in [1, 255]");
    }

    const auto z = static_cast<std::size_t>(shapeZYX[0]);
    const auto y = static_cast<std::size_t>(shapeZYX[1]);
    const auto x = static_cast<std::size_t>(shapeZYX[2]);

    std::vector<std::byte> filtered(input.begin(), input.end());
    cacheQuantize({filtered.data(), filtered.size()}, elemSize, quantBinWidth);
    if (elemSize == 1)
        deltaZyxFilter(reinterpret_cast<std::uint8_t*>(filtered.data()), z, y, x);
    else
        deltaZyxFilter(reinterpret_cast<std::uint16_t*>(filtered.data()), z, y, x);

    const std::span<const std::byte> filteredSpan(filtered.data(),
                                                  filtered.size());
    auto output = codec == CacheCodec::Rans
        ? ransCompressFrame(filteredSpan, kHeaderSize)
        : zstdCompressFrame(filteredSpan, level, kHeaderSize);
    std::memcpy(output.data(), kMagic, sizeof(kMagic));
    output[4] = static_cast<std::byte>(kFormatVersion);
    output[5] = static_cast<std::byte>(elemSize);
    output[6] = static_cast<std::byte>(quantBinWidth);
    output[7] = static_cast<std::byte>(codec);
    writeU32(output.data() + 8, static_cast<std::uint32_t>(z));
    writeU32(output.data() + 12, static_cast<std::uint32_t>(y));
    writeU32(output.data() + 16, static_cast<std::uint32_t>(x));
    return output;
}

void cacheQuantize(std::span<std::byte> data,
                   std::size_t elemSize,
                   int quantBinWidth)
{
    if (quantBinWidth <= 1)
        return;
    if (elemSize == 1)
        quantizeValues(reinterpret_cast<std::uint8_t*>(data.data()),
                       data.size(), quantBinWidth);
    else
        quantizeValues(reinterpret_cast<std::uint16_t*>(data.data()),
                       data.size() / 2, quantBinWidth);
}

std::optional<int> cacheQuantBinWidth(std::span<const std::byte> input)
{
    if (!hasVcz1Header(input))
        return std::nullopt;
    return std::max(1, static_cast<int>(input[6]));
}

std::optional<CacheCodec> cacheCodec(std::span<const std::byte> input)
{
    if (!hasVcz1Header(input))
        return std::nullopt;
    switch (static_cast<unsigned char>(input[7])) {
    case static_cast<unsigned char>(CacheCodec::Zstd):
        return CacheCodec::Zstd;
    case static_cast<unsigned char>(CacheCodec::Rans):
        return CacheCodec::Rans;
    default:
        return std::nullopt;
    }
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
    const auto codec = cacheCodec(input);
    if (!codec)
        return std::nullopt;

    std::vector<std::byte> output(expectedSize);
    if (*codec == CacheCodec::Rans) {
        if (!ransDecompressFrame(input.subspan(kHeaderSize), output.data(),
                                 expectedSize))
            return std::nullopt;
    } else {
        const std::size_t rc = ZSTD_decompress(
            output.data(), output.size(),
            input.data() + kHeaderSize, input.size() - kHeaderSize);
        if (ZSTD_isError(rc) || rc != expectedSize)
            return std::nullopt;
    }

    if (elemSize == 1)
        deltaZyxUnfilter(reinterpret_cast<std::uint8_t*>(output.data()), z, y, x);
    else
        deltaZyxUnfilter(reinterpret_cast<std::uint16_t*>(output.data()), z, y, x);
    return output;
}

} // namespace vc
