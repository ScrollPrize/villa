#include "utils/c3d_codec.hpp"

#include <c3d.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace utils {

static constexpr std::size_t kC3dChunkBytes =
    static_cast<std::size_t>(C3D_CHUNK_SIDE) * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE;

bool is_c3d_compressed(std::span<const std::byte> data) noexcept
{
    return c3d_is_chunk(
        reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

namespace {
// Move-only RAII wrapper around aligned_alloc.  std::vector doesn't
// guarantee the 32-byte alignment c3d's kernels assert on, so callers
// with unaligned buffers stage through one of these.
// Move-only (copy would double-free).
struct AlignedBuf {
    uint8_t* p = nullptr;
    AlignedBuf() = default;
    explicit AlignedBuf(std::size_t n)
        // aligned_alloc requires size to be a multiple of alignment —
        // c3d's C3D_ALIGN is 32 and n is always 256^3, so divisible.
        : p(static_cast<uint8_t*>(std::aligned_alloc(C3D_ALIGN, n)))
    {
        if (!p) throw std::bad_alloc();
    }
    ~AlignedBuf() { std::free(p); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
    AlignedBuf& operator=(AlignedBuf&& o) noexcept {
        if (this != &o) { std::free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
};
static bool is_aligned(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & (C3D_ALIGN - 1)) == 0;
}
}  // namespace

std::vector<std::byte> c3d_encode(std::span<const std::byte> raw,
                                  const C3dCodecParams& params)
{
    if (raw.size() != kC3dChunkBytes) {
        throw std::runtime_error(
            "c3d_encode: input must be exactly 256^3 bytes, got " +
            std::to_string(raw.size()));
    }
    if (params.depth != 256 || params.height != 256 || params.width != 256) {
        throw std::runtime_error("c3d_encode: codec atom is fixed at 256^3");
    }
    if (!(params.target_ratio > 1.0f)) {
        throw std::runtime_error("c3d_encode: target_ratio must be > 1.0");
    }

    const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(raw.data());
    AlignedBuf staging;  // lifetime extends through the c3d_chunk_encode call
    if (!is_aligned(in_ptr)) {
        staging = AlignedBuf(kC3dChunkBytes);
        std::memcpy(staging.p, in_ptr, kC3dChunkBytes);
        in_ptr = staging.p;
    }

    std::vector<std::byte> out(c3d_chunk_encode_max_size());
    const std::size_t n = c3d_chunk_encode(
        in_ptr,
        params.target_ratio,
        /*ctx=*/nullptr,
        reinterpret_cast<uint8_t*>(out.data()),
        out.size());
    out.resize(n);
    return out;
}

std::vector<std::byte> c3d_decode(std::span<const std::byte> compressed,
                                  std::size_t out_size,
                                  const C3dCodecParams& /*params*/)
{
    if (out_size != kC3dChunkBytes) {
        throw std::runtime_error(
            "c3d_decode: output must be exactly 256^3 bytes, got " +
            std::to_string(out_size));
    }
    const auto* in  = reinterpret_cast<const uint8_t*>(compressed.data());
    const std::size_t in_len = compressed.size();
    if (!c3d_is_chunk(in, in_len)) {
        throw std::runtime_error("c3d_decode: input missing C3DC magic");
    }
    // Validate-then-decode: libc3d is fatal-on-error.  Validation rejects
    // malformed bytes cleanly instead of panicking inside decode.
    if (!c3d_chunk_validate(in, in_len)) {
        throw std::runtime_error("c3d_decode: structural validation failed");
    }

    // Decode needs a 32-byte-aligned output; std::vector doesn't guarantee
    // that so decode into an aligned staging buffer then copy out.
    // Zero the staging buffer first — c3d's byte-progressive decode can
    // leave regions of the output untouched if the input is truncated or
    // only carries a coarser LOD prefix.  Zeroing guarantees any such
    // regions come out as black pixels rather than uninitialized memory
    // (visible as noise in the rendered tiles).
    AlignedBuf staging(out_size);
    std::memset(staging.p, 0, out_size);
    c3d_chunk_decode(in, in_len, /*ctx=*/nullptr, staging.p);
    std::vector<std::byte> out(out_size);
    std::memcpy(out.data(), staging.p, out_size);
    return out;
}

}  // namespace utils
