#include "utils/volcodec_ng_codec.hpp"

#include <volcodec_ng_lib.h>

#include <stdexcept>
#include <string>

namespace utils {

namespace {
[[noreturn]] void fail(const char* what, int st)
{
    throw std::runtime_error(std::string(what) + ": " + volcodec_ng_lib_status_string(st));
}
}  // namespace

const char* volcodec_ng_kernels() noexcept { return volcodec_ng_lib_kernels(); }

unsigned volcodec_ng_format_revision() noexcept { return volcodec_ng_lib_format_revision(); }

bool is_volcodec_ng_compressed(std::span<const std::byte> data) noexcept
{
    return volcodec_ng_lib_is_chunk(data.data(), data.size()) != 0;
}

int volcodec_ng_chunk_mode(std::span<const std::byte> data) noexcept
{
    return volcodec_ng_lib_chunk_mode(data.data(), data.size());
}

std::vector<std::byte> volcodec_ng_encode(std::span<const std::byte> raw)
{
    if (raw.size() != kVolcodecNgChunkBytes) {
        throw std::runtime_error(
            "volcodec_ng_encode: input must be exactly 128^3 bytes, got " +
            std::to_string(raw.size()));
    }
    std::vector<std::byte> out(volcodec_ng_lib_encode_bound());
    std::size_t n = 0;
    const int st = volcodec_ng_lib_encode(reinterpret_cast<const uint8_t*>(raw.data()),
                                          out.data(), out.size(), &n);
    if (st != VOLCODEC_NG_LIB_OK) fail("volcodec_ng_encode", st);
    out.resize(n);
    // The bound is a raw chunk plus the header; most chunks are far smaller.
    out.shrink_to_fit();
    return out;
}

void volcodec_ng_decode_into(std::span<const std::byte> compressed, std::span<std::byte> out)
{
    if (out.size() != kVolcodecNgChunkBytes) {
        throw std::runtime_error(
            "volcodec_ng_decode: output must be exactly 128^3 bytes, got " +
            std::to_string(out.size()));
    }
    if (!is_volcodec_ng_compressed(compressed)) {
        throw std::runtime_error("volcodec_ng_decode: input missing VCNG magic");
    }
    const int st = volcodec_ng_lib_decode(compressed.data(), compressed.size(),
                                          reinterpret_cast<uint8_t*>(out.data()), out.size());
    if (st != VOLCODEC_NG_LIB_OK) fail("volcodec_ng_decode", st);
}

std::vector<std::byte> volcodec_ng_decode(std::span<const std::byte> compressed,
                                          std::size_t out_size)
{
    std::vector<std::byte> out(out_size);
    volcodec_ng_decode_into(compressed, out);
    return out;
}

}  // namespace utils
