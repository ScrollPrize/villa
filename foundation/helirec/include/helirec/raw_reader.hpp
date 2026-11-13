#pragma once
// Fast reader for raw detector frames stored one HDF5 chunk per frame
// (Lima files: dataset (n, rows, cols) uint16, chunks (1, rows, cols), no
// filters) or as a contiguous dataset. HDF5 is only used to locate each frame's
// bytes; the data is then read with pread(), in parallel over files.
//
// Also read: frames stored as JPEG 2000, one codestream per chunk of (1, r, cols)
// through the blosc2 filter (id 32026), as ESRF BM18 writes projections. Only the
// chunks a request touches are decoded, in parallel over chunks. This needs helirec
// built with OpenJPEG; the values of an irreversible codestream are that decoder's
// (src/io/j2k_chunk.hpp). Any other compression is refused.
//
// At a reduced resolution (reduce = 1, 2 or 3: a half, a quarter, an eighth) the reader gives frames of
// (rows, cols) / 2^reduce values, from the lower resolutions a JPEG 2000 codestream holds: everything below
// (rows(), cols(), every row index and count, the cache) is then in reduced values. Value (j, k) of a reduced
// frame is the wavelet's low band at the stored pixel (j, k) 2^reduce (src/io/j2k_chunk.hpp): a geometry in
// stored pixels holds for the reduced frame with every position divided by 2^reduce, nothing added. Only
// compressed frames have such resolutions (plain ones are refused), and the stored rows, columns and rows per
// chunk must be multiples of 2^reduce for uint16.
//
// dtype="float32": also reads reconstructed B2JPF32 HTJ2K volumes using OpenJPH
// (fetched by CMake). Odd frame dimensions round up at lower resolutions. Whole
// frame chunks may be odd; internal chunk boundaries must align with 2^reduce.
// selective=true seeks over unused resolution packets; the bytes saved depend on
// packet order. selective=false preloads the codestream before the same decoding.
// This reduces X/Y only: frames along Z remain independently addressable.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

class RawReader {
  public:
    // rows, cols: of a frame as stored
    RawReader(std::vector<std::string> files, std::string dataset, int rows, int cols, int reduce = 0,
              std::string dtype = "uint16", bool selective = true);
    ~RawReader();

    int rows() const { return rows_; }  // of a frame as it is read (reduced)
    int cols() const { return cols_; }
    int reduce() const { return reduce_; }
    const std::string& dtype() const { return dtype_; }
    // Payload bytes requested through pread, including compressed headers. Excludes
    // HDF5 metadata and any read-ahead/caching performed by the filesystem.
    std::uint64_t bytes_read() const;
    std::size_t n_files() const { return files_.size(); }

    // How a file stores its frames (only headers of the first compressed chunk are read).
    struct Storage {
        bool compressed = false;
        int chunk_rows = 0;         // rows per chunk of a compressed file
        std::string codec;          // "" for plain values
        bool reversible = true;     // false: the stored values are not the detector's
        int bit_depth = 16;
        std::int64_t frames = 0;
        double bytes_per_frame = 0;  // as stored
    };
    Storage storage(int file);

    // For i in [0, n): rows [row0[i], row0[i] + nrows) of frame (file_index[i], frame_index[i])
    // -> out[i] (nrows, cols) uint16. Rows outside [0, rows) are filled with 0.
    // out_stride: elements between consecutive outputs (0: nrows * cols)
    void read_rows(const int32_t* file_index, const int32_t* frame_index, const int32_t* row0, std::size_t n,
                   int nrows, uint16_t* out, int n_threads = 0, std::size_t out_stride = 0);

    void read_rows(const int32_t* file_index, const int32_t* frame_index, const int32_t* row0, std::size_t n,
                   int nrows, float* out, int n_threads = 0, std::size_t out_stride = 0);

    // Decoded chunks of compressed frames kept between calls, up to `bytes` (0, the default: none), the least
    // recently used dropped first. For readers that come back to the same frames (blocks of a calibration that
    // overlap); a reader that takes every frame once should leave it off. Plain frames are never kept.
    void set_cache_bytes(std::size_t bytes);
    // chunks taken from the cache and chunks decoded, since the reader was made
    void cache_counts(std::uint64_t& hits, std::uint64_t& decoded) const;

    // Decoding ahead: `threads` workers of the reader's own (0, the default: none) that decode the chunks of
    // rows announced with prefetch_rows into the cache while the caller does something else; a later read_rows
    // takes them from there, waiting for one that is being decoded. Nothing is announced twice. Needs the cache.
    void set_prefetch_threads(int threads);
    // the rows of a coming read_rows (same arguments); returns at once. Errors are left to the read itself.
    void prefetch_rows(const int32_t* file_index, const int32_t* frame_index, const int32_t* row0, std::size_t n, int nrows);

    // Whole frames: out (n, rows, cols)
    void read_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, uint16_t* out,
                     int n_threads = 0);
    void read_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, float* out,
                     int n_threads = 0);
    // Whole frames in pieces of `nrows` rows: visit(i, r0, nr, data) with data = rows [r0, r0 + nr) of frame i, in
    // row order for each frame, from the reading threads (frames in parallel, any order). `data` is that thread's
    // buffer, valid during the call (small pieces stay in its cache: the frame is traversed once).
    void visit_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, int nrows,
                      const std::function<void(std::size_t, int, int, const uint16_t*)>& visit, int n_threads = 0);

  private:
    struct FileInfo;
    const FileInfo& info(int file);  // lazily inspects the file (thread safe)

    void decode_chunk(int fd, int file, const FileInfo& fi, std::size_t at,
                      std::vector<std::uint8_t>& raw, void* tile);
    void read_rows_impl(const int32_t*, const int32_t*, const int32_t*, std::size_t, int, void*, int, std::size_t);
    std::string dtype_;
    int sample_bytes_;
    bool selective_;
    std::vector<std::string> files_;
    std::string dataset_;
    int rows_, cols_;  // as read
    int stored_rows_, stored_cols_, reduce_;
    std::vector<std::unique_ptr<FileInfo>> infos_;
    struct Sync;
    std::unique_ptr<Sync> sync_;
    struct Cache;
    std::unique_ptr<Cache> cache_;
    struct Ahead;
    std::unique_ptr<Ahead> ahead_;
};

// Frames stored as JPEG 2000 (above): whether this build reads them, and the bytes of one HDF5 chunk for
// (rows, cols) values as the blosc2 filter stores them, with a reversible (exact) codestream. The second is for
// tests and synthetic records; real ones are written by the detector software.
bool jpeg2000_frames_supported();
bool htj2k_float_frames_supported();
std::vector<std::uint8_t> jpeg2000_chunk(const std::uint16_t* values, int rows, int cols);

}  // namespace helirec
