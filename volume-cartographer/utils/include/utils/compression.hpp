#pragma once
#include <vector>
#include <span>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <functional>
#include <optional>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <cstring>

// Backend detection
#if __has_include(<zstd.h>)
#   define UTILS_HAS_ZSTD 1
#   include <zstd.h>
#else
#   define UTILS_HAS_ZSTD 0
#endif

#if __has_include(<lz4.h>)
#   define UTILS_HAS_LZ4 1
#   include <lz4.h>
#else
#   define UTILS_HAS_LZ4 0
#endif

#if __has_include(<lz4hc.h>)
#   define UTILS_HAS_LZ4HC 1
#   include <lz4hc.h>
#else
#   define UTILS_HAS_LZ4HC 0
#endif

#if __has_include(<zlib.h>)
#   define UTILS_HAS_ZLIB 1
#   include <zlib.h>
#else
#   define UTILS_HAS_ZLIB 0
#endif

#if __has_include(<blosc.h>)
#   define UTILS_HAS_BLOSC 1
#   include <blosc.h>
#else
#   define UTILS_HAS_BLOSC 0
#endif

namespace utils {

// ---------------------------------------------------------------------------
// Codec enum
// ---------------------------------------------------------------------------

enum class Codec : std::uint8_t {
    none = 0,
    zstd,
    lz4,
    lz4hc,
    zlib,
    gzip,
    blosc_lz4,
    blosc_lz4hc,
    blosc_zstd,
    blosc_zlib,
    blosc_snappy,
    blosc_blosclz,
};

[[nodiscard]] constexpr std::string_view codec_name(Codec c) noexcept {
    switch (c) {
        case Codec::none:          return "none";
        case Codec::zstd:          return "zstd";
        case Codec::lz4:           return "lz4";
        case Codec::lz4hc:         return "lz4hc";
        case Codec::zlib:          return "zlib";
        case Codec::gzip:          return "gzip";
        case Codec::blosc_lz4:     return "blosc_lz4";
        case Codec::blosc_lz4hc:   return "blosc_lz4hc";
        case Codec::blosc_zstd:    return "blosc_zstd";
        case Codec::blosc_zlib:    return "blosc_zlib";
        case Codec::blosc_snappy:  return "blosc_snappy";
        case Codec::blosc_blosclz: return "blosc_blosclz";
    }
    return "unknown";
}

namespace detail {
    struct CodecEntry { std::string_view name; Codec codec; };
    inline constexpr CodecEntry codec_table[] = {
        {"none",          Codec::none},
        {"zstd",          Codec::zstd},
        {"lz4",           Codec::lz4},
        {"lz4hc",         Codec::lz4hc},
        {"zlib",          Codec::zlib},
        {"gzip",          Codec::gzip},
        {"blosc_lz4",     Codec::blosc_lz4},
        {"blosc_lz4hc",   Codec::blosc_lz4hc},
        {"blosc_zstd",    Codec::blosc_zstd},
        {"blosc_zlib",    Codec::blosc_zlib},
        {"blosc_snappy",  Codec::blosc_snappy},
        {"blosc_blosclz", Codec::blosc_blosclz},
    };
} // namespace detail

[[nodiscard]] constexpr std::optional<Codec> parse_codec(std::string_view name) noexcept {
    for (auto& e : detail::codec_table)
        if (e.name == name) return e.codec;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Compression parameters
// ---------------------------------------------------------------------------

struct CompressParams {
    Codec codec          = Codec::zstd;
    int level            = 5;
    int shuffle          = 1;       // 0=none, 1=byte, 2=bit (blosc)
    std::size_t typesize  = 1;      // Element size for shuffle (blosc)
    std::size_t blocksize = 0;      // 0 = auto (blosc)
    std::size_t num_threads = 1;
};

// ---------------------------------------------------------------------------
// Codec registry
// ---------------------------------------------------------------------------

using CompressFn = std::function<std::vector<std::byte>(
    std::span<const std::byte>, const CompressParams&)>;
using DecompressFn = std::function<std::vector<std::byte>(
    std::span<const std::byte>, std::size_t expected_size)>;

struct CodecImpl {
    CompressFn compress;
    DecompressFn decompress;
};

namespace detail {

inline auto& codec_registry() noexcept {
    static std::unordered_map<std::uint8_t, CodecImpl> reg;
    return reg;
}

inline auto& named_codec_registry() noexcept {
    static std::unordered_map<std::string, CodecImpl> reg;
    return reg;
}

// Helper: prepend a 4-byte little-endian size prefix
inline std::vector<std::byte> prepend_size(std::span<const std::byte> compressed,
                                           std::uint32_t original_size) {
    std::vector<std::byte> out(4 + compressed.size());
    std::memcpy(out.data(), &original_size, 4);
    std::memcpy(out.data() + 4, compressed.data(), compressed.size());
    return out;
}

// Helper: read 4-byte LE size prefix
inline std::uint32_t read_size_prefix(std::span<const std::byte> data) {
    if (data.size() < 4)
        throw std::runtime_error("compression: data too small for size prefix");
    std::uint32_t sz{};
    std::memcpy(&sz, data.data(), 4);
    return sz;
}

} // namespace detail

inline void register_codec(Codec id, CodecImpl impl) {
    detail::codec_registry()[static_cast<std::uint8_t>(id)] = std::move(impl);
}

inline void register_codec(std::string_view name, CodecImpl impl) {
    detail::named_codec_registry()[std::string(name)] = std::move(impl);
}

[[nodiscard]] inline bool codec_available(Codec c) noexcept {
    if (c == Codec::none) return true;
    auto& reg = detail::codec_registry();
    return reg.contains(static_cast<std::uint8_t>(c));
}

// ---------------------------------------------------------------------------
// Backend implementations
// ---------------------------------------------------------------------------

#if UTILS_HAS_ZSTD
namespace detail::zstd_backend {

inline std::vector<std::byte> do_compress(std::span<const std::byte> in,
                                          const CompressParams& p) {
    auto bound = ZSTD_compressBound(in.size());
    std::vector<std::byte> out(bound);
    auto rc = ZSTD_compress(out.data(), out.size(), in.data(), in.size(), p.level);
    if (ZSTD_isError(rc))
        throw std::runtime_error(std::string("zstd compress: ") + ZSTD_getErrorName(rc));
    out.resize(rc);
    return out;
}

inline std::vector<std::byte> do_decompress(std::span<const std::byte> in,
                                            std::size_t expected) {
    if (expected == 0) {
        auto content_size = ZSTD_getFrameContentSize(in.data(), in.size());
        if (content_size == ZSTD_CONTENTSIZE_ERROR)
            throw std::runtime_error("zstd decompress: not a valid zstd frame");
        if (content_size == ZSTD_CONTENTSIZE_UNKNOWN)
            throw std::runtime_error("zstd decompress: unknown content size, provide expected_size");
        expected = static_cast<std::size_t>(content_size);
    }
    std::vector<std::byte> out(expected);
    auto rc = ZSTD_decompress(out.data(), out.size(), in.data(), in.size());
    if (ZSTD_isError(rc))
        throw std::runtime_error(std::string("zstd decompress: ") + ZSTD_getErrorName(rc));
    out.resize(rc);
    return out;
}

} // namespace detail::zstd_backend
#endif

#if UTILS_HAS_LZ4
namespace detail::lz4_backend {

inline std::vector<std::byte> do_compress(std::span<const std::byte> in,
                                          const CompressParams& /*p*/) {
    auto bound = LZ4_compressBound(static_cast<int>(in.size()));
    std::vector<std::byte> tmp(bound);
    auto rc = LZ4_compress_default(
        reinterpret_cast<const char*>(in.data()),
        reinterpret_cast<char*>(tmp.data()),
        static_cast<int>(in.size()), bound);
    if (rc <= 0)
        throw std::runtime_error("lz4 compress: failed with code " + std::to_string(rc));
    tmp.resize(static_cast<std::size_t>(rc));
    return prepend_size(tmp, static_cast<std::uint32_t>(in.size()));
}

inline std::vector<std::byte> do_decompress(std::span<const std::byte> in,
                                            std::size_t expected) {
    if (expected == 0) {
        expected = read_size_prefix(in);
        in = in.subspan(4);
    }
    std::vector<std::byte> out(expected);
    auto rc = LZ4_decompress_safe(
        reinterpret_cast<const char*>(in.data()),
        reinterpret_cast<char*>(out.data()),
        static_cast<int>(in.size()), static_cast<int>(expected));
    if (rc < 0)
        throw std::runtime_error("lz4 decompress: failed with code " + std::to_string(rc));
    out.resize(static_cast<std::size_t>(rc));
    return out;
}

} // namespace detail::lz4_backend
#endif

#if UTILS_HAS_LZ4HC
namespace detail::lz4hc_backend {

inline std::vector<std::byte> do_compress(std::span<const std::byte> in,
                                          const CompressParams& p) {
    auto bound = LZ4_compressBound(static_cast<int>(in.size()));
    std::vector<std::byte> tmp(bound);
    auto rc = LZ4_compress_HC(
        reinterpret_cast<const char*>(in.data()),
        reinterpret_cast<char*>(tmp.data()),
        static_cast<int>(in.size()), bound, p.level);
    if (rc <= 0)
        throw std::runtime_error("lz4hc compress: failed with code " + std::to_string(rc));
    tmp.resize(static_cast<std::size_t>(rc));
    return detail::prepend_size(tmp, static_cast<std::uint32_t>(in.size()));
}

} // namespace detail::lz4hc_backend
#endif

#if UTILS_HAS_ZLIB
namespace detail::zlib_backend {

inline std::vector<std::byte> do_compress(std::span<const std::byte> in,
                                          const CompressParams& p, bool gzip_mode) {
    z_stream strm{};
    int window_bits = gzip_mode ? (15 + 16) : 15;
    if (deflateInit2(&strm, p.level, Z_DEFLATED, window_bits, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("zlib deflateInit2 failed");
    auto bound = deflateBound(&strm, static_cast<uLong>(in.size()));
    std::vector<std::byte> out(bound);
    strm.next_in  = reinterpret_cast<Bytef*>(const_cast<std::byte*>(in.data()));
    strm.avail_in = static_cast<uInt>(in.size());
    strm.next_out = reinterpret_cast<Bytef*>(out.data());
    strm.avail_out = static_cast<uInt>(out.size());
    auto rc = deflate(&strm, Z_FINISH);
    deflateEnd(&strm);
    if (rc != Z_STREAM_END)
        throw std::runtime_error("zlib deflate failed with code " + std::to_string(rc));
    out.resize(strm.total_out);
    return out;
}

inline std::vector<std::byte> do_decompress(std::span<const std::byte> in,
                                            std::size_t expected, bool gzip_mode) {
    z_stream strm{};
    int window_bits = gzip_mode ? (15 + 16) : 15;
    if (inflateInit2(&strm, window_bits) != Z_OK)
        throw std::runtime_error("zlib inflateInit2 failed");

    if (expected == 0) expected = in.size() * 4;  // heuristic
    std::vector<std::byte> out(expected);
    strm.next_in  = reinterpret_cast<Bytef*>(const_cast<std::byte*>(in.data()));
    strm.avail_in = static_cast<uInt>(in.size());

    std::size_t total = 0;
    int rc = Z_OK;
    while (rc != Z_STREAM_END) {
        if (total == out.size()) out.resize(out.size() * 2);
        strm.next_out  = reinterpret_cast<Bytef*>(out.data() + total);
        strm.avail_out = static_cast<uInt>(out.size() - total);
        rc = inflate(&strm, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) {
            inflateEnd(&strm);
            throw std::runtime_error("zlib inflate failed with code " + std::to_string(rc));
        }
        total = strm.total_out;
    }
    inflateEnd(&strm);
    out.resize(total);
    return out;
}

} // namespace detail::zlib_backend
#endif

#if UTILS_HAS_BLOSC
namespace detail::blosc_backend {

inline const char* blosc_compressor_name(Codec c) {
    switch (c) {
        case Codec::blosc_lz4:     return "lz4";
        case Codec::blosc_lz4hc:   return "lz4hc";
        case Codec::blosc_zstd:    return "zstd";
        case Codec::blosc_zlib:    return "zlib";
        case Codec::blosc_snappy:  return "snappy";
        case Codec::blosc_blosclz: return "blosclz";
        default: return "blosclz";
    }
}

inline std::vector<std::byte> do_compress(std::span<const std::byte> in,
                                          const CompressParams& p) {
    auto bound = in.size() + BLOSC_MAX_OVERHEAD;
    std::vector<std::byte> out(bound);
    auto rc = blosc_compress_ctx(
        p.level, p.shuffle,
        p.typesize, in.size(),
        in.data(), out.data(), out.size(),
        blosc_compressor_name(p.codec),
        p.blocksize, 1);
    if (rc <= 0)
        throw std::runtime_error("blosc compress: failed with code " + std::to_string(rc));
    out.resize(static_cast<std::size_t>(rc));
    return out;
}

inline std::vector<std::byte> do_decompress(std::span<const std::byte> in,
                                            std::size_t expected) {
    if (expected == 0) {
        std::size_t nbytes{}, cbytes{}, blocksize{};
        blosc_cbuffer_sizes(in.data(), &nbytes, &cbytes, &blocksize);
        expected = nbytes;
    }
    std::vector<std::byte> out(expected);
    auto rc = blosc_decompress_ctx(in.data(), out.data(), out.size(), 1);
    if (rc < 0)
        throw std::runtime_error("blosc decompress: failed with code " + std::to_string(rc));
    out.resize(static_cast<std::size_t>(rc));
    return out;
}

} // namespace detail::blosc_backend
#endif

// ---------------------------------------------------------------------------
// Auto-registration
// ---------------------------------------------------------------------------

namespace detail {

inline const bool backends_registered = [] {
    // none: passthrough
    register_codec(Codec::none, CodecImpl{
        [](std::span<const std::byte> in, const CompressParams&) {
            return std::vector<std::byte>(in.begin(), in.end());
        },
        [](std::span<const std::byte> in, std::size_t) {
            return std::vector<std::byte>(in.begin(), in.end());
        }
    });

#if UTILS_HAS_ZSTD
    register_codec(Codec::zstd, CodecImpl{
        zstd_backend::do_compress, zstd_backend::do_decompress});
#endif

#if UTILS_HAS_LZ4
    register_codec(Codec::lz4, CodecImpl{
        lz4_backend::do_compress, lz4_backend::do_decompress});
#endif

#if UTILS_HAS_LZ4HC && UTILS_HAS_LZ4
    register_codec(Codec::lz4hc, CodecImpl{
        lz4hc_backend::do_compress, lz4_backend::do_decompress});
#endif

#if UTILS_HAS_ZLIB
    register_codec(Codec::zlib, CodecImpl{
        [](std::span<const std::byte> in, const CompressParams& p) {
            return zlib_backend::do_compress(in, p, false);
        },
        [](std::span<const std::byte> in, std::size_t expected) {
            return zlib_backend::do_decompress(in, expected, false);
        }
    });
    register_codec(Codec::gzip, CodecImpl{
        [](std::span<const std::byte> in, const CompressParams& p) {
            return zlib_backend::do_compress(in, p, true);
        },
        [](std::span<const std::byte> in, std::size_t expected) {
            return zlib_backend::do_decompress(in, expected, true);
        }
    });
#endif

#if UTILS_HAS_BLOSC
    for (auto c : {Codec::blosc_lz4, Codec::blosc_lz4hc, Codec::blosc_zstd,
                   Codec::blosc_zlib, Codec::blosc_snappy, Codec::blosc_blosclz}) {
        register_codec(c, CodecImpl{
            blosc_backend::do_compress, blosc_backend::do_decompress});
    }
#endif

    return true;
}();

} // namespace detail

// ---------------------------------------------------------------------------
// Core compress / decompress
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::vector<std::byte> compress(
    std::span<const std::byte> input,
    const CompressParams& params)
{
    (void)detail::backends_registered; // ensure init
    auto& reg = detail::codec_registry();
    auto it = reg.find(static_cast<std::uint8_t>(params.codec));
    if (it == reg.end())
        throw std::runtime_error(
            std::string("compress: codec '") + std::string(codec_name(params.codec)) + "' not available");
    return it->second.compress(input, params);
}

[[nodiscard]] inline std::vector<std::byte> decompress(
    std::span<const std::byte> input,
    Codec codec,
    std::size_t expected_size)
{
    (void)detail::backends_registered;
    auto& reg = detail::codec_registry();
    auto it = reg.find(static_cast<std::uint8_t>(codec));
    if (it == reg.end())
        throw std::runtime_error(
            std::string("decompress: codec '") + std::string(codec_name(codec)) + "' not available");
    return it->second.decompress(input, expected_size);
}

// Convenience: uint8_t input
[[nodiscard]] inline std::vector<std::byte> compress(
    std::span<const std::uint8_t> input,
    const CompressParams& params)
{
    return compress(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(input.data()), input.size()),
        params);
}

[[nodiscard]] inline std::vector<std::uint8_t> decompress_u8(
    std::span<const std::byte> input,
    Codec codec,
    std::size_t expected_size)
{
    auto raw = decompress(input, codec, expected_size);
    std::vector<std::uint8_t> out(raw.size());
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// ---------------------------------------------------------------------------
// CodecPipeline
// ---------------------------------------------------------------------------

class CodecPipeline final {
public:
    CodecPipeline& add(CompressParams params) {
        stages_.push_back(std::move(params));
        return *this;
    }

    [[nodiscard]] std::vector<std::byte> encode(std::span<const std::byte> input) const {
        std::vector<std::byte> buf(input.begin(), input.end());
        for (auto const& stage : stages_)
            buf = compress(buf, stage);
        return buf;
    }

    [[nodiscard]] std::vector<std::byte> decode(
        std::span<const std::byte> input,
        std::size_t expected_size) const
    {
        std::vector<std::byte> buf(input.begin(), input.end());
        for (auto it = stages_.rbegin(); it != stages_.rend(); ++it) {
            // Only pass expected_size for the last stage applied (first in reverse)
            std::size_t sz = (it == stages_.rbegin()) ? expected_size : 0;
            buf = decompress(buf, it->codec, sz);
        }
        return buf;
    }

    [[nodiscard]] std::size_t size() const noexcept { return stages_.size(); }
    [[nodiscard]] bool empty() const noexcept { return stages_.empty(); }

private:
    std::vector<CompressParams> stages_;
};

} // namespace utils
