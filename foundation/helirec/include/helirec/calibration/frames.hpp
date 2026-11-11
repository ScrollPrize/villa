#pragma once
// Raw frames of a scan, addressed as the record addresses them (stretch, frame), and the one value conversion
// the checks use: line integrals from a dark subtraction and a division by the mean flat.
//
// Nothing else is applied: no distortion map, no defect map, no beam normalisation, no geometry.
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "helirec/calibration/record.hpp"

namespace helirec {
class RawReader;
}

namespace helirec::calibration {

class Frames {
  public:
    explicit Frames(const Record& rec, int n_threads = 0);
    ~Frames();
    const Record& record() const { return *rec_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // rows [row0[i], row0[i] + nrows) of frame frame[i] of stretch `stretch` -> out (n, nrows, cols).
    // Rows outside the detector are 0.
    void read_rows(int stretch, const std::int64_t* frame, const std::int32_t* row0, std::size_t n, int nrows,
                   std::uint16_t* out);
    // frames [first, first + count) of a stretch, whole
    void read(int stretch, std::int64_t first, std::int64_t count, std::uint16_t* out);
    // visit(frame, data): every frame in [first, first + count), whole, from the reading threads in any order
    void visit(int stretch, std::int64_t first, std::int64_t count,
               const std::function<void(std::int64_t, const std::uint16_t*)>& visit);

  private:
    void locate(int stretch, std::int64_t frame, std::int32_t& file, std::int32_t& index) const;
    const Record* rec_;
    int rows_ = 0, cols_ = 0, n_threads_ = 0;
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

}  // namespace helirec::calibration
