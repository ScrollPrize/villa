#pragma once
// Raw frames of a scan, addressed as the record addresses them (stretch, frame), and the one value conversion
// the checks use: line integrals from a dark subtraction and a division by the mean flat.
//
// Nothing else is applied: no defect map, no beam normalisation, no geometry, and no distortion map unless the
// caller hands one over to be tested (DetectorMap, line_integrals_mapped).
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "helirec/calibration/record.hpp"

namespace helirec {
class RawReader;
}

namespace helirec::calibration {

class Frames {
  public:
    // reduce = 1, 2, 3: every frame at a half, a quarter, an eighth, from the lower resolutions its JPEG 2000
    // codestream holds (helirec/raw_reader.hpp: value (j, k) sits at the stored pixel (j, k) 2^reduce; plain frames
    // have none and are refused when read). rows(), cols() and every row below are then the reduced frame's.
    explicit Frames(const Record& rec, int n_threads = 0, int reduce = 0);
    ~Frames();
    const Record& record() const { return *rec_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int reduce() const { return reduce_; }

    // rows [row0[i], row0[i] + nrows) of frame frame[i] of stretch `stretch` -> out (n, nrows, cols).
    // Rows outside the detector are 0.
    void read_rows(int stretch, const std::int64_t* frame, const std::int32_t* row0, std::size_t n, int nrows,
                   std::uint16_t* out);
    // Decoded chunks of compressed frames are kept between reads up to this many bytes (the default is
    // default_cache_bytes: blocks of a calibration come back to the same frames; 0: nothing is kept).
    static constexpr std::size_t default_cache_bytes = std::size_t(3) << 30;
    void set_cache_bytes(std::size_t bytes);
    void cache_counts(std::uint64_t& hits, std::uint64_t& decoded) const;
    // Decoding ahead (RawReader): workers of the reader's own, and the rows of a read_rows that is coming. Frames
    // that are not on disk are left to the read to report.
    void set_prefetch_threads(int threads);
    void prefetch_rows(int stretch, std::int64_t first, std::int64_t count, int row0, int nrows);
    // frames [first, first + count) of a stretch, whole
    void read(int stretch, std::int64_t first, std::int64_t count, std::uint16_t* out);
    // visit(frame, data): every frame in [first, first + count), whole, from the reading threads in any order
    void visit(int stretch, std::int64_t first, std::int64_t count,
               const std::function<void(std::int64_t, const std::uint16_t*)>& visit);

    // How the frames of a stretch are stored, from its first file on disk (helirec/raw_reader.hpp). A frame not
    // on disk is never read: read_rows, read and visit throw for it.
    struct Stored {
        bool on_disk = false;       // false: no file of the stretch is on disk
        bool compressed = false;
        int chunk_rows = 0;
        std::string codec;
        bool reversible = true;     // false: the stored values are not the detector's
        int bit_depth = 16;
        double bytes_per_frame = 0;
    };
    Stored stored(int stretch);
    // Every chunk of every frame on disk of the stretch located in its file, without reading the values: the
    // number of frames found. Throws if a frame of a file on disk was not written.
    std::int64_t locate_all(int stretch);

  private:
    void locate(int stretch, std::int64_t frame, std::int32_t& file, std::int32_t& index) const;
    const Record* rec_;
    int rows_ = 0, cols_ = 0, n_threads_ = 0, reduce_ = 0;
    std::unique_ptr<RawReader> reader_;
    std::vector<std::vector<std::int32_t>> piece_file_;  // per stretch, per piece: index into the reader's files
};

// Mean and variance over the frames of one stretch (a dark or a flat series), per pixel
struct SeriesStats {
    std::vector<double> mean, variance;  // (rows, cols); variance is the unbiased sample variance over frames
    std::vector<double> frame_mean;      // mean raw value of each frame
    std::int64_t n = 0;
};
SeriesStats series_stats(Frames& frames, int stretch);

// dark: mean of the scan's dark frames. gain: mean of all its flat frames (every flat series) minus dark.
struct FlatField {
    int rows = 0, cols = 0;
    std::vector<float> dark, gain;
    std::vector<SeriesStats> series;  // statistics of each dark and flat stretch, in record order
    std::vector<int> series_stretch;  // the stretch index of each
};
FlatField make_flat_field(Frames& frames);

// -ln((raw - dark) / gain) for rows [row0[i], row0[i] + nrows) x columns [col0, col0 + ncols) of n frames;
// raw: (n, nrows, cols) as returned by Frames::read_rows; out: (n, nrows, ncols). `flip_cols`: columns reversed.
void line_integrals(const FlatField& ff, const std::uint16_t* raw, const std::int32_t* row0, std::size_t n, int nrows,
                    int col0, int ncols, bool flip_cols, float* out);

// A detector map to be tested: pixel (c, v) looks along the ray at the ideal position (c + Eu(c, v), v + Ev(c, v)),
// Eu = sum_k eu[k] x^i[k] y^j[k] and Ev likewise, x = (c - c0) / cs, y = (v - v0) / vs.
struct DetectorMap {
    double c0 = 0, cs = 1, v0 = 0, vs = 1;
    std::vector<int> i, j;
    std::vector<double> eu, ev;
    // no term, or every coefficient zero: nothing is displaced, and blocks are cut from the frames as stored (a
    // resampling at whole pixels returns the pixels: the same values, for the time of reading them)
    bool empty() const {
        for (double e : eu)
            if (e != 0.0) return false;
        for (double e : ev)
            if (e != 0.0) return false;
        return true;
    }
    void displacement(double c, double v, double& du, double& dv) const;
    // the detector position (c, v) whose ray is the ideal position (x, y)
    void source(double x, double y, double& c, double& v) const;
};

// The line integrals on the ideal grid of `map`: out(i, r, c) is the value at ideal row row0 + r and ideal column
// col0 + c (col0 + ncols - 1 - c with flip_cols) of frame frame[i], read from the detector position the map puts
// there by cubic convolution (Keys, a = -1/2) of the line integrals; positions are held inside the detector.
// With an empty map this is read_rows followed by line_integrals.
void line_integrals_mapped(Frames& frames, const FlatField& ff, const DetectorMap& map, int stretch, const std::int64_t* frame,
                           int row0, std::size_t n, int nrows, int col0, int ncols, bool flip_cols, float* out);

// b(q, r, c) <- b(q + shear (c - (ncols - 1) / 2), r, c), cubic along the frames (held inside the block): frames of
// a block read at an angle that changes with the column, as the divergence of the beam makes opposite views do.
void shear_frames(float* block, std::size_t n, int nrows, int ncols, double shear);

// out(q, r, c) <- wide(q, r, pad + c + s) (pad + ncols - 1 - c + s with flip_cols), cubic along the columns, with
// s = shift[q] + (per_row ? per_row[q] (row0 + r) : 0): a block cut `pad` columns wider on both sides, each frame
// read `shift[q]` columns further on. What a sample displaced on the rotation stage does to a tile of a mosaic
// against another: its content sways along the columns once per turn; and with rows exposed one after another,
// each row at its own angle, by an amount that is a line in the row (per_row, with row0 the block's first row
// counted from the row the frame's angle belongs to). |s| must stay below pad - 2.
void shift_columns(const float* wide, std::size_t n, int nrows, int ncols, int pad, const double* shift, bool flip_cols, float* out,
                   const double* per_row = nullptr, double row0 = 0);

}  // namespace helirec::calibration
