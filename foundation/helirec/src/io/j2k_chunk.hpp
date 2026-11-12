#pragma once
// Frames stored compressed: one JPEG 2000 codestream per HDF5 chunk, wrapped in a blosc2 frame (HDF5 filter
// 32026 with the JPEG 2000 codec, as ESRF BM18 writes projections: chunks of (1, r, cols), HTJ2K, irreversible).
//
// The container is parsed here (no blosc2 library): a blosc2 frame that holds one chunk of one block whose only
// stream is the codestream. The codestream is decoded with OpenJPEG (>= 2.5 for HTJ2K) when helirec is built
// with it (HELIREC_WITH_OPENJPEG); without it, available() is false and decode() throws.
//
// An irreversible codestream does not define its decoded integers to the last count: they are OpenJPEG's
// (rounded to nearest, held to 0..65535). Another decoder may differ by a count.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace helirec::j2k {

inline constexpr unsigned kBlosc2FilterId = 32026;

bool available();

struct Codestream {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
};

// The codestream inside the bytes of one HDF5 chunk as the blosc2 filter stored them. `nbytes`: the chunk's
// size in values times two, which the container must state. Throws on any other container layout.
Codestream codestream_of_chunk(const std::uint8_t* chunk, std::size_t size, std::size_t nbytes);

// What the codestream's main header states (SIZ and COD markers); no decoder needed.
struct Info {
    int width = 0, height = 0, components = 0, bit_depth = 0;
    bool is_signed = false;
    bool reversible = false;       // 5/3 integer wavelet; false: 9/7, values are not reproduced exactly
    bool high_throughput = false;  // HTJ2K block coder (ISO/IEC 15444-15)
    int levels = 0, layers = 0;
};
Info info(const Codestream& cs);

// One unsigned component of width x height values -> out. Throws if the codestream is something else.
void decode(const Codestream& cs, int width, int height, std::uint16_t* out);

// For tests and synthetic records: the chunk bytes for (height, width) values, as codestream_of_chunk reads
// them, with a reversible (exact) codestream.
std::vector<std::uint8_t> encode_chunk(const std::uint16_t* values, int width, int height);

}  // namespace helirec::j2k
