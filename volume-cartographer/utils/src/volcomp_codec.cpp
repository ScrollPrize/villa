#include "utils/volcomp_codec.hpp"

#include <volcomp_lib.h>

#include <stdexcept>
#include <string>

namespace utils {

namespace {
[[noreturn]] void fail(const char* what, int st)
{
    throw std::runtime_error(std::string(what) + ": " + volcomp_lib_status_string(st));
}
}  // namespace

bool volcomp_available() noexcept { return volcomp_lib_available() != 0; }

bool is_volcomp_compressed(std::span<const std::byte> data) noexcept
{
    return volcomp_lib_is_chunk(data.data(), data.size()) != 0;
}

float volcomp_chunk_q(std::span<const std::byte> data) noexcept
{
    return volcomp_lib_chunk_q(data.data(), data.size());
}

std::vector<std::byte> volcomp_encode(std::span<const std::byte> raw,
                                      const VolcompCodecParams& params)
{
    if (raw.size() != kVolcompChunkBytes) {
        throw std::runtime_error(
            "volcomp_encode: input must be exactly 128^3 bytes, got " +
            std::to_string(raw.size()));
    }
    if (!(params.q >= VOLCOMP_LIB_Q_MIN && params.q <= VOLCOMP_LIB_Q_MAX)) {
        throw std::runtime_error("volcomp_encode: q must be in [1, 255]");
    }
    if (!volcomp_available()) fail("volcomp_encode", VOLCOMP_LIB_UNSUPPORTED);

    std::vector<std::byte> out(volcomp_lib_encode_bound());
    std::size_t n = 0;
    const int st = volcomp_lib_encode(
        reinterpret_cast<const uint8_t*>(raw.data()), params.q,
        out.data(), out.size(), &n);
    if (st != VOLCOMP_LIB_OK) fail("volcomp_encode", st);
    out.resize(n);
    // The bound is ~14 MiB while real chunks are 20-200 KiB; release the slack
    // so callers that hold many encoded chunks don't pay for it.
    out.shrink_to_fit();
    return out;
}

void volcomp_decode_into(std::span<const std::byte> compressed, std::span<std::byte> out)
{
    if (out.size() != kVolcompChunkBytes) {
        throw std::runtime_error(
            "volcomp_decode: output must be exactly 128^3 bytes, got " +
            std::to_string(out.size()));
    }
    if (!is_volcomp_compressed(compressed)) {
        throw std::runtime_error("volcomp_decode: input missing VOLC magic");
    }
    if (!volcomp_available()) fail("volcomp_decode", VOLCOMP_LIB_UNSUPPORTED);
    const int st = volcomp_lib_decode(compressed.data(), compressed.size(),
                                      reinterpret_cast<uint8_t*>(out.data()), out.size());
    if (st != VOLCOMP_LIB_OK) fail("volcomp_decode", st);
}

std::vector<std::byte> volcomp_decode(std::span<const std::byte> compressed,
                                      std::size_t out_size)
{
    std::vector<std::byte> out(out_size);
    volcomp_decode_into(compressed, out);
    return out;
}

void volcomp_decode_block_into(std::span<const std::byte> compressed,
                               unsigned bz, unsigned by, unsigned bx,
                               std::span<std::byte> out)
{
    if (out.size() < 4096) {
        throw std::runtime_error("volcomp_decode_block: output must hold 16^3 bytes");
    }
    if (!is_volcomp_compressed(compressed)) {
        throw std::runtime_error("volcomp_decode_block: input missing VOLC magic");
    }
    if (!volcomp_available()) fail("volcomp_decode_block", VOLCOMP_LIB_UNSUPPORTED);
    const int st = volcomp_lib_decode_block(compressed.data(), compressed.size(), bz, by, bx,
                                            reinterpret_cast<uint8_t*>(out.data()), out.size());
    if (st != VOLCOMP_LIB_OK) fail("volcomp_decode_block", st);
}

}  // namespace utils
