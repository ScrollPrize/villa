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

// Thread-local c3d_decoder reuse.  c3d_decoder_new() allocates ~80 MiB of
// scratch arenas; reusing across calls in the same worker thread saves
// 50-100 ms/chunk per the c3d header note.  A c3d_decoder is not
// thread-safe, so we bind one per worker thread.
struct C3dDecoderDeleter {
    void operator()(c3d_decoder* d) const noexcept {
        if (d) c3d_decoder_free(d);
    }
};
using DecoderPtr = std::unique_ptr<c3d_decoder, C3dDecoderDeleter>;

static c3d_decoder* thread_decoder() {
    thread_local DecoderPtr dec(c3d_decoder_new());
    return dec.get();
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
        reinterpret_cast<uint8_t*>(out.data()),
        out.size());
    out.resize(n);
    // c3d_chunk_encode_max_size is ~16 MiB but typical compressed payloads
    // are 200 KiB-2 MiB.  resize() alone keeps the full capacity around,
    // so callers that retain many encoded chunks (e.g. vc_zarr_recompress
    // accumulating 4096 chunks per shard) were paying 16 MiB × N of dead
    // heap per shard.  Release the slack.
    out.shrink_to_fit();
    return out;
}

std::vector<std::byte> c3d_decode(std::span<const std::byte> compressed,
                                  std::size_t out_size,
                                  const C3dCodecParams& params)
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

    // Decoder needs 32-byte-aligned output. glibc's malloc returns page-
    // aligned memory for 16 MiB allocations so the vector's storage is
    // already aligned in practice; stage only on the fallback. No upfront
    // memset — the decoder writes every output voxel (c3d.c:3659 for the
    // lod_end==0 fast path, c3d.c:3778 otherwise, including the §T9
    // truncated-entropy case where zero-fill happens in the coefficient
    // buffer before the final output loop converts it to u8).
    c3d_decoder* d = thread_decoder();
    c3d_decoder_set_denoise(d, !params.skip_denoise);
    std::vector<std::byte> out(out_size);
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    if (is_aligned(out_ptr)) {
        c3d_decoder_chunk_decode(d, in, in_len, out_ptr);
    } else {
        AlignedBuf staging(out_size);
        c3d_decoder_chunk_decode(d, in, in_len, staging.p);
        std::memcpy(out.data(), staging.p, out_size);
    }
    return out;
}

std::vector<std::byte> c3d_decode_lod(std::span<const std::byte> compressed,
                                      std::uint8_t lod)
{
    if (lod > 5) {
        throw std::runtime_error("c3d_decode_lod: lod must be in [0, 5]");
    }
    const auto* in  = reinterpret_cast<const uint8_t*>(compressed.data());
    const std::size_t in_len = compressed.size();
    if (!c3d_is_chunk(in, in_len)) {
        throw std::runtime_error("c3d_decode_lod: input missing C3DC magic");
    }
    if (!c3d_chunk_validate(in, in_len)) {
        throw std::runtime_error("c3d_decode_lod: structural validation failed");
    }
    const std::size_t side    = static_cast<std::size_t>(C3D_CHUNK_SIDE) >> lod;
    const std::size_t out_size = side * side * side;

    std::vector<std::byte> out(out_size);
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data());
    if (is_aligned(out_ptr)) {
        c3d_decoder_chunk_decode_lod(thread_decoder(), in, in_len, lod, out_ptr);
    } else {
        AlignedBuf staging(out_size);
        c3d_decoder_chunk_decode_lod(thread_decoder(), in, in_len, lod, staging.p);
        std::memcpy(out.data(), staging.p, out_size);
    }
    return out;
}

}  // namespace utils
