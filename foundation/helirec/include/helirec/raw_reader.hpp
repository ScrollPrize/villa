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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

class RawReader {
  public:
    RawReader(std::vector<std::string> files, std::string dataset, int rows, int cols);
    ~RawReader();

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    std::size_t n_files() const { return files_.size(); }

    // How a file stores its frames (the file is inspected; a compressed file's first chunk is read).
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

    // Whole frames: out (n, rows, cols)
    void read_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, uint16_t* out,
                     int n_threads = 0);
    // Whole frames in pieces of `nrows` rows: visit(i, r0, nr, data) with data = rows [r0, r0 + nr) of frame i, in
    // row order for each frame, from the reading threads (frames in parallel, any order). `data` is that thread's
    // buffer, valid during the call (small pieces stay in its cache: the frame is traversed once).
    void visit_frames(const int32_t* file_index, const int32_t* frame_index, std::size_t n, int nrows,
                      const std::function<void(std::size_t, int, int, const uint16_t*)>& visit, int n_threads = 0);

  private:
    struct FileInfo;
    const FileInfo& info(int file);  // lazily inspects the file (thread safe)

    std::vector<std::string> files_;
    std::string dataset_;
    int rows_, cols_;
    std::vector<std::unique_ptr<FileInfo>> infos_;
    struct Sync;
    std::unique_ptr<Sync> sync_;
};

// Frames stored as JPEG 2000 (above): whether this build reads them, and the bytes of one HDF5 chunk for
// (rows, cols) values as the blosc2 filter stores them, with a reversible (exact) codestream. The second is for
// tests and synthetic records; real ones are written by the detector software.
bool jpeg2000_frames_supported();
std::vector<std::uint8_t> jpeg2000_chunk(const std::uint16_t* values, int rows, int cols);

}  // namespace helirec
