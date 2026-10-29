// Chunk codecs implemented in C++: raw, blosc, zstd, zlib/gzip, LZ4, LZW, vc-delta3d and volcomp.
// Every codec but LZW also encodes (Zarr writes and disk caches).
#pragma once

#include "lzw.hpp"

#include <blosc.h>
#include <lz4.h>
#include <volcomp_lib.h>
#include <zlib.h>
#include <zstd.h>
#include <vc_delta3d/codec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vcr {

enum class Codec { Raw, Blosc, Zstd, Zlib, Gzip, Lz4, Lzw, Delta3d, Volcomp };

inline Codec parse_codec(const std::string& id)
{
    if (id.empty() || id == "raw")
        return Codec::Raw;
    if (id == "blosc")
        return Codec::Blosc;
    if (id == "zstd")
        return Codec::Zstd;
    if (id == "zlib")
        return Codec::Zlib;
    if (id == "gzip")
        return Codec::Gzip;
    if (id == "lz4")
        return Codec::Lz4;
    if (id == "lzw" || id == "imagecodecs_lzw")
        return Codec::Lzw;
    if (id == "vc-delta3d")
        return Codec::Delta3d;
    if (id == "volcomp")
        return Codec::Volcomp;
    throw std::invalid_argument("C++ reader does not support codec '" + id + "'");
}

inline void check_raw_size(std::size_t n, std::size_t expected)
{
    if (n != expected)
        throw std::runtime_error("raw chunk has " + std::to_string(n) + " bytes, expected " +
                                 std::to_string(expected));
}

// Inflates a zlib or gzip stream (the header is detected) into exactly `dst_n` bytes.
inline void inflate_exact(const std::uint8_t* src, std::size_t n, std::uint8_t* dst, std::size_t dst_n)
{
    z_stream z{};
    if (inflateInit2(&z, 15 + 32) != Z_OK)
        throw std::runtime_error("zlib init failed");
    z.next_in = const_cast<Bytef*>(src);
    z.next_out = dst;
    int status = Z_OK;
    // avail_in/avail_out are 32-bit: feed large buffers in pieces.
    std::size_t in_left = n, out_left = dst_n;
    do {
        const auto in_step = static_cast<uInt>(std::min<std::size_t>(in_left, UINT32_MAX));
        const auto out_step = static_cast<uInt>(std::min<std::size_t>(out_left, UINT32_MAX));
        z.avail_in = in_step;
        z.avail_out = out_step;
        status = inflate(&z, Z_NO_FLUSH);
        in_left -= in_step - z.avail_in;
        out_left -= out_step - z.avail_out;
    } while (status == Z_OK && (in_left || out_left));
    inflateEnd(&z);
    if (status != Z_STREAM_END || out_left)
        throw std::runtime_error("zlib chunk decodes to " + std::to_string(dst_n - out_left) +
                                 " bytes, expected " + std::to_string(dst_n) +
                                 (status == Z_STREAM_END ? "" : " (corrupt or truncated stream)"));
}

// Decodes `n` stored bytes into exactly `dst_n` bytes at `dst`; throws on corrupt input.
inline void decode(Codec codec, const std::uint8_t* src, std::size_t n, std::uint8_t* dst, std::size_t dst_n)
{
    switch (codec) {
    case Codec::Raw:
        check_raw_size(n, dst_n);
        std::memcpy(dst, src, n);
        return;
    case Codec::Blosc: {
        std::size_t nbytes = 0, cbytes = 0, blocksize = 0;
        if (n < BLOSC_MIN_HEADER_LENGTH)
            throw std::runtime_error("blosc chunk too short");
        blosc_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize);
        if (nbytes != dst_n || cbytes > n)
            throw std::runtime_error("blosc chunk decodes to " + std::to_string(nbytes) + " bytes, expected " +
                                     std::to_string(dst_n));
        if (blosc_decompress_ctx(src, dst, dst_n, 1) != static_cast<int>(dst_n))
            throw std::runtime_error("blosc decode failed");
        return;
    }
    case Codec::Zstd: {
        const auto size = ZSTD_decompress(dst, dst_n, src, n);
        if (ZSTD_isError(size))
            throw std::runtime_error(std::string("zstd decode failed: ") + ZSTD_getErrorName(size));
        if (size != dst_n)
            throw std::runtime_error("zstd chunk decodes to " + std::to_string(size) + " bytes, expected " +
                                     std::to_string(dst_n));
        return;
    }
    case Codec::Zlib:
    case Codec::Gzip:
        inflate_exact(src, n, dst, dst_n);
        return;
    case Codec::Lz4: {  // numcodecs LZ4: little-endian uint32 decoded size, then one LZ4 block
        if (n < 4)
            throw std::runtime_error("lz4 chunk too short");
        std::uint32_t size = 0;
        for (int i = 0; i < 4; ++i)
            size |= std::uint32_t(src[i]) << (8 * i);
        if (size != dst_n || n - 4 > std::size_t(INT32_MAX) || dst_n > std::size_t(INT32_MAX))
            throw std::runtime_error("lz4 chunk decodes to " + std::to_string(size) + " bytes, expected " +
                                     std::to_string(dst_n));
        const int got = LZ4_decompress_safe(reinterpret_cast<const char*>(src + 4), reinterpret_cast<char*>(dst),
                                            int(n - 4), int(dst_n));
        if (got != int(dst_n))
            throw std::runtime_error("lz4 decode failed");
        return;
    }
    case Codec::Lzw:
        decode_lzw(src, n, dst, dst_n);
        return;
    case Codec::Volcomp: {
        if (dst_n != VOLCOMP_LIB_CHUNK_BYTES)
            throw std::runtime_error("volcomp chunks are 128^3 uint8, not " + std::to_string(dst_n) + " bytes");
        const int status = volcomp_lib_decode(src, n, dst, dst_n);
        if (status != VOLCOMP_LIB_OK)
            throw std::runtime_error(std::string("volcomp decode failed: ") + volcomp_lib_status_string(status));
        return;
    }
    case Codec::Delta3d:
        if (!vc_delta3d::decompressInto(std::span(reinterpret_cast<const std::byte*>(src), n),
                                        std::span(reinterpret_cast<std::byte*>(dst), dst_n)))
            throw std::runtime_error("vc-delta3d decode failed");
        return;
    }
}

// A Delta3D payload written losslessly (quantization bin width 0 or 1).
inline bool delta3d_lossless(const std::uint8_t* p, std::size_t n)
{
    return n >= 7 && std::memcmp(p, "D3D1", 4) == 0 && p[6] <= 1;
}

inline std::vector<std::uint8_t> encode_delta3d(const std::uint8_t* src, std::size_t n,
                                                const std::array<int, 3>& shape, std::size_t itemsize)
{
    auto out = vc_delta3d::compress(std::span(reinterpret_cast<const std::byte*>(src), n), shape, itemsize,
                                    vc_delta3d::kQuantizationLossless);
    std::vector<std::uint8_t> bytes(out.size());
    std::memcpy(bytes.data(), out.data(), out.size());
    return bytes;
}

// Settings of an encoder, from a numcodecs compressor config.
struct EncodeOptions {
    std::string cname = "lz4";  // blosc
    int level = -1;             // blosc clevel; zstd/zlib/gzip level; lz4 acceleration (-1: codec default)
    int shuffle = 1;            // blosc
    int blocksize = 0;          // blosc
    float q = 8.0f;             // volcomp quantiser step (0: lossless)
    bool checksum = false;     // zstd frame checksum
};

// Encodes one decoded chunk of `itemsize`-byte elements and C-order `shape`.
inline std::vector<std::uint8_t> encode(Codec codec, const EncodeOptions& o, const std::uint8_t* src, std::size_t n,
                                        std::size_t itemsize, const std::vector<std::int64_t>& shape)
{
    std::vector<std::uint8_t> out;
    switch (codec) {
    case Codec::Raw:
        out.assign(src, src + n);
        return out;
    case Codec::Blosc: {
        out.resize(n + BLOSC_MAX_OVERHEAD);
        const int written = blosc_compress_ctx(o.level < 0 ? 5 : o.level, o.shuffle, itemsize, n, src, out.data(),
                                               out.size(), o.cname.c_str(), std::size_t(o.blocksize), 1);
        if (written <= 0)
            throw std::runtime_error("blosc encode failed (" + std::to_string(written) + ")");
        out.resize(std::size_t(written));
        return out;
    }
    case Codec::Zstd: {
        out.resize(ZSTD_compressBound(n));
        std::size_t written;
        if (o.checksum) {
            std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> context(ZSTD_createCCtx(), ZSTD_freeCCtx);
            if (!context)
                throw std::runtime_error("zstd context allocation failed");
            const auto level = ZSTD_CCtx_setParameter(context.get(), ZSTD_c_compressionLevel, o.level < 0 ? 3 : o.level);
            const auto checksum = ZSTD_CCtx_setParameter(context.get(), ZSTD_c_checksumFlag, 1);
            if (ZSTD_isError(level) || ZSTD_isError(checksum))
                throw std::invalid_argument("invalid zstd encoding parameters");
            written = ZSTD_compress2(context.get(), out.data(), out.size(), src, n);
        } else {
            written = ZSTD_compress(out.data(), out.size(), src, n, o.level < 0 ? 3 : o.level);
        }
        if (ZSTD_isError(written))
            throw std::runtime_error(std::string("zstd encode failed: ") + ZSTD_getErrorName(written));
        out.resize(written);
        return out;
    }
    case Codec::Zlib:
    case Codec::Gzip: {
        z_stream z{};
        if (deflateInit2(&z, o.level < 0 ? Z_DEFAULT_COMPRESSION : o.level, Z_DEFLATED,
                         codec == Codec::Gzip ? 15 + 16 : 15, 8, Z_DEFAULT_STRATEGY) != Z_OK)
            throw std::runtime_error("zlib init failed");
        if (n > UINT32_MAX) {
            deflateEnd(&z);
            throw std::runtime_error("zlib chunks are limited to 4 GiB");
        }
        out.resize(deflateBound(&z, uLong(n)));
        z.next_in = const_cast<Bytef*>(src);
        z.avail_in = uInt(n);
        z.next_out = out.data();
        z.avail_out = uInt(out.size());
        const int status = deflate(&z, Z_FINISH);
        out.resize(z.total_out);
        deflateEnd(&z);
        if (status != Z_STREAM_END)
            throw std::runtime_error("zlib encode failed");
        return out;
    }
    case Codec::Lz4: {
        if (n > std::size_t(LZ4_MAX_INPUT_SIZE))
            throw std::runtime_error("lz4 chunks are limited to " + std::to_string(LZ4_MAX_INPUT_SIZE) + " bytes");
        const int bound = LZ4_compressBound(int(n));
        out.resize(4 + std::size_t(bound));
        for (int i = 0; i < 4; ++i)
            out[std::size_t(i)] = std::uint8_t(std::uint32_t(n) >> (8 * i));
        const int written = LZ4_compress_fast(reinterpret_cast<const char*>(src), reinterpret_cast<char*>(out.data() + 4),
                                              int(n), bound, std::max(o.level, 1));
        if (written <= 0)
            throw std::runtime_error("lz4 encode failed");
        out.resize(4 + std::size_t(written));
        return out;
    }
    case Codec::Lzw:
        throw std::invalid_argument("LZW chunks can be read, not written");
    case Codec::Delta3d:
        if (shape.size() != 3 || (itemsize != 1 && itemsize != 2))
            throw std::invalid_argument("vc-delta3d stores 3-D uint8 or uint16 chunks");
        return encode_delta3d(src, n, {int(shape[0]), int(shape[1]), int(shape[2])}, itemsize);
    case Codec::Volcomp: {
        if (n != VOLCOMP_LIB_CHUNK_BYTES || itemsize != 1)
            throw std::invalid_argument("volcomp stores 128^3 uint8 chunks");
        out.resize(volcomp_lib_encode_bound());
        std::size_t written = 0;
        const int status = volcomp_lib_encode(src, o.q, out.data(), out.size(), &written);
        if (status != VOLCOMP_LIB_OK)
            throw std::runtime_error(std::string("volcomp encode failed: ") + volcomp_lib_status_string(status));
        out.resize(written);
        out.shrink_to_fit();  // the bound is ~14 MiB; chunks are usually 20-200 KiB
        return out;
    }
    }
    throw std::invalid_argument("unknown codec");
}

}  // namespace vcr
