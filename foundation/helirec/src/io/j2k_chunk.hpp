#pragma once
// Frames stored compressed: one JPEG 2000 codestream per HDF5 chunk, wrapped in a blosc2 frame (HDF5 filter
// 32026 with the JPEG 2000 codec, as ESRF BM18 writes projections: chunks of (1, r, cols), HTJ2K, irreversible).
//
// The container is parsed here (no blosc2 library): a blosc2 frame that holds one chunk of one block whose only
// stream is the codestream or a B2JPF32 float wrapper around it. OpenJPEG (>= 2.5) handles uint16 detector
// frames; OpenJPH handles seekable HTJ2K reads and the 32-bit samples of reconstructed float32 volumes.
// available() reports OpenJPEG support; openjph_available() reports OpenJPH support.
//
// An irreversible codestream does not define its decoded integers to the last count: they are OpenJPEG's
// (rounded to nearest, held to 0..65535). Another decoder may differ by a count.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace helirec::j2k {

inline constexpr unsigned kBlosc2FilterId = 32026;

bool available();
bool openjph_available();

// A bounded view of an HDF5 chunk. Offsets are relative to that chunk; the caller
// owns the file and accounts for I/O. No filter plugin or whole-chunk read is needed.
using ReadAt = std::function<void(void*, std::size_t, std::size_t)>;
struct ChunkHeader {
    std::size_t offset = 0, size = 0;
    bool floating = false, constant = false;
    int quant_bits = 0;
    double scale_min = 0, scale_max = 0;
};
ChunkHeader chunk_header(const ReadAt& read, std::size_t size, std::size_t nbytes, int sample_bytes);

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
Info info(const ReadAt& read, const ChunkHeader& chunk);

// One unsigned component of width x height values -> out. Throws if the codestream is something else.
// Always the whole chunk: OpenJPEG 2.5.4 returns only the level shift (32768) for a decode area of an HTJ2K
// codestream (opj_set_decode_area), so an area cannot be decoded alone.
//
// reduce > 0: the codestream's own lower resolution, `reduce` levels of its wavelet left out. width and height
// are then the reduced ones (the stored ones over 2^reduce, rounded up) and value k of a line is the low band's
// sample at the stored position k 2^reduce: the stored values through the wavelet's low-pass (of unit gain),
// not the mean of 2^reduce values, and not centred between them. Decoding costs about 2^-2reduce of the whole
// (Paris4, one chunk of 128 x 4416: 8.5, 2.5, 0.7, 0.2 ms for reduce 0 to 3). Throws if the codestream has fewer
// levels.
void decode(const Codestream& cs, int width, int height, std::uint16_t* out, int reduce = 0);

// OpenJPH: one HTJ2K component, uint16 or B2JPF32 float32, read through seekable
// callbacks. `selective=false` reads the full codestream before the same decode,
// for comparison and for storage where a sequential read is preferable.
void decode_htj2k(const ReadAt& read, std::size_t size, std::size_t nbytes, int sample_bytes,
                  int stored_width, int stored_height, int reduce, void* out, bool selective = true);

// For tests and synthetic records: the chunk bytes for (height, width) values, as codestream_of_chunk reads
// them, with a reversible (exact) codestream.
std::vector<std::uint8_t> encode_chunk(const std::uint16_t* values, int width, int height);

}  // namespace helirec::j2k
