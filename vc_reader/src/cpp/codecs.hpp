// Chunk codecs implemented in C++: raw, blosc, zstd, zlib/gzip, LZW, vc-delta3d and volcomp.
#pragma once

#include "lzw.hpp"

#include <blosc.h>
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
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vcr {

enum class Codec { Raw, Blosc, Zstd, Zlib, Gzip, Lzw, Delta3d, Volcomp };

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

}  // namespace vcr
