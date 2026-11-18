#pragma once
// A mosaic reconstructed as one slice by one filtered backprojection (whole-canvas FBP).
//
// A mosaic is helical scans of one sample at several places on the rotation stage (tiles): the rotation axis falls
// on the same detector column in every tile, and each tile records the rays within a detector's width of its own
// axis. A tile reconstructed alone (tile_reconstructor, reconstruction.hpp) ramp-filters rows that are cut at the
// frame's edges with sample on either side. Here the tiles' rows are first joined:
//
//   the canvas       the slice of the whole mosaic, (2 half + 1) pixels a side, the anchor tile's rotation axis at
//                    its middle pixel `half`: pixel (row, col) = (half + Y, half + X)
//   a tile's place   its rotation axis passes through the canvas at (X, Y) = (ax, ay): with the joint geometry
//                    model's place d of the sample on the rotation stage (geometry_calibration.hpp), (ax, ay) =
//                    (-d[0], d[1]) in pixels of the frames read
//   one sinogram     at the angle a (the exact backprojector's: u = X cos a + Y sin a + rot_center) the tile's
//                    column c records the canvas column  c - rot_center_tile + half + ax cos a + ay sin a.
//                    Every tile's rows of the slice (the -log values and their measured weights, as its own pipeline
//                    grids them: detector map, rows' timing, projection filters) are delayed by the fraction of that
//                    offset exactly (shift_rows) and added, value times weight and weight, into one sinogram as wide as
//                    the canvas. A ray recorded from the other side (a + pi, mirrored about the axis) is the same
//                    ray: the two halves of the turn are folded onto each other. Dividing by the summed weights gives
//                    the weighted mean of every recording of every ray; a ray that no tile recorded takes the
//                    nearest recorded value of its row.
//   levels           the tiles' -log values differ by a level each (flat-field and beam drift between tiles
//                    recorded hours apart): one additive level per tile and slice, the anchor's 0, by least squares
//                    over the rays two tiles both recorded at one angle.
//   one FBP          the sinogram's rows are whole, so one ramp filter (with the ramp's window, sino.hpp) and one exact
//                    backprojection (kernels.hpp) of half the turn give the slice in attenuation per cm. No row is
//                    cut inside the sample.
//
// What it leaves out: the beam's divergence (a tile's fan, which its own slice takes: ReconConfig::fan_per_column)
// and the tiles' rotations against each other; the tiles' angles of one bin are taken as equal (their weighted mean:
// MosaicCanvas::angle_spread says by how much they differ).
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vc_tomography/common.hpp"
#include "vc_tomography/pipeline.hpp"

namespace vc_tomography {

// ---------------------------------------------------------------- kernels
// tile_rows: out_w[r, c] = w[r, c] gain[c], out_s[r, c] = s[r, c] out_w[r, c] (0 where the weight is not above 0);
//            s, w: (n_rows, n); out_s, out_w: (n_rows, stride), stride >= n (shift_row_stride(n) for shift_rows)
// place_rows: canvas[r, first[r] + c] += scale x rows[r, c] for c < n, where that column lies in [0, width);
//            nonneg: values not above 0 add nothing (weights after a band-limited shift). rows: (n_rows, stride),
//            canvas: (n_rows, width)
// fold_opposite: S (n_rows, width), n_rows even, the rows over one full turn and the axis at (width - 1) / 2: S[r, j]
//            and S[r + n_rows / 2, width - 1 - j], the two recordings of one ray, both become their sum
namespace cpu {
void tile_rows(const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w);
void place_rows(const float* rows, const std::int32_t* first, float scale, bool nonneg, int n_rows, int n, int stride, float* canvas, int width);
void fold_opposite(float* S, int n_rows, int width);
}  // namespace cpu
#ifdef VC_TOMOGRAPHY_WITH_CUDA
namespace cuda {   // device pointers (first too)
void tile_rows(const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w, void* stream = nullptr);
void place_rows(const float* rows, const std::int32_t* first, float scale, bool nonneg, int n_rows, int n, int stride, float* canvas, int width,
                void* stream = nullptr);
void fold_opposite(float* S, int n_rows, int width, void* stream = nullptr);
}  // namespace cuda
#endif

// ---------------------------------------------------------------- the canvas
struct CanvasTile {
    double rot_center = 0;   // the column of the tile's rows its rotation axis falls on
    double ax = 0, ay = 0;   // where that axis passes through the canvas, from its middle [px]
    int cols = 0;            // the columns of the tile's rows
};
struct CanvasConfig {
    Device device = Device::CPU;
    int n_angles = 0;        // the rows of every tile's sinogram: the bins of one turn (even)
    int half = 0;            // the canvas has 2 half + 1 pixels a side (0: what the tiles reach)
    double pixel_size_m = 0;
    // the exact backprojector and the ramp's window, as ReconConfig has them
    std::string bp_interp = "bspline3";
    int bp_oversample = 2;
    std::string ramp_filter_name = "ramlak";
    std::vector<float> ramp_filter_par;
    // columns at either end of a tile's rows over which its weight falls to nothing (raised cosine): the ends of a
    // frame hold what a filter of the projections (a phase filter) made of the cut
    int taper = 0;
    bool levels = true;      // one additive level per tile and slice from the rays two tiles share (false: none)
    int level_step = 25;     // every level_step-th angle is compared
    float mend_threshold = 1e-4f;   // as ReconConfig::mend_threshold, of the canvas' summed weights
    int anchor = 0;          // the tile whose level is 0
};
// What a slice was made of, kept on request (host): the sinogram (n_angles, width) as it is ramp-filtered (folded,
// divided, mended), its summed weights (folded), and the half turn of rows as they are backprojected.
struct CanvasDebug {
    std::vector<float> sino, weights, filtered, angles;
};

class MosaicCanvas {
  public:
    MosaicCanvas(CanvasConfig config, std::vector<CanvasTile> tiles);
    ~MosaicCanvas();
    int size() const;        // pixels a side
    int half() const;
    const CanvasConfig& config() const;
    // a chunk of n_slices slices: empty sums
    void begin(int n_slices);
    // Tile `tile`'s rows of slice y of the chunk: (n_angles, cols) -log values and their measured weights (0: not
    // recorded), in the memory of the canvas' device, and the angle of every row (host; the exact backprojector's).
    void add(int tile, int y, const float* rows, const float* weights, const float* angles);
    // slice y of the chunk -> out (y1 - y0, x1 - x0) (host), 0 outside the tiles' discs; roi null: the whole canvas.
    // Once a slice is asked for, nothing more is added to it.
    void slice(int y, float* out, const SliceRoi* roi = nullptr, CanvasDebug* debug = nullptr);
    // of slice y, once made: every tile's level, and the largest difference of a tile's angle of a bin from the
    // angle the bin is backprojected at [bins]
    const std::vector<double>& levels(int y) const;
    double angle_spread(int y) const;

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------- a mosaic from its frames
struct MosaicOptions {
    Device device = Device::CUDA;
    // reconstruction parameters over the defaults (as tile_reconstructor's): filters, backprojector=fbp(...) with
    // its interpolation, oversampling and ramp window, z_interp, x_interp, beam_divergence is not taken (above)
    std::string params = "{}";
    int reduce = 0;                 // 1, 2, 3: frames read at a half, a quarter, an eighth
    std::string projection_store;   // STORE.zarr: every tile's frames (and darks and flats where its files are not on disk) from its group
    std::string cache_dir;          // the tiles' reduced darks and flats are kept there
    int taper = 200;                // CanvasConfig::taper in stored pixels
    bool levels = true;
    // every tile's reconstructor kept between chunks (faster: no tile is set up twice; each holds its share of the
    // device, which at full resolution is more than a device has for many tiles); false: one tile's at a time
    bool keep_tiles = false;
    std::function<void(const std::string&)> log;
};
struct MosaicTile {
    std::string name, master;
    double ax = 0, ay = 0, height = 0, rot_center = 0;   // canvas place and axis [px of the frames read]; height [stored rows]
    std::pair<int, int> doable{0, 0};
};

// The mosaic of a joint geometry calibration of its tiles (vc_tomography_calibration geometry's file, "model":
// "joint"): every scan of it with its place d and height against the anchor (parameters.scans[k].d, .height: what
// the step's solve of the tiles against each other gives, solve_tiles; a file that leaves every place at nothing is
// refused). The masters are the
// calibration's, or those of the same folders under `projections` (a calibration may name them elsewhere).
class MosaicReconstruction {
  public:
    MosaicReconstruction(const std::string& projections, const std::string& geometry_text, MosaicOptions options);
    ~MosaicReconstruction();
    std::pair<int, int> doable() const;    // slices [lo, hi) every tile can give; slice h is the same height of the sample in all
    int size() const;                      // canvas pixels a side
    double pixel_size_m() const;           // of a canvas pixel
    const std::vector<MosaicTile>& tiles() const;
    std::string info_json() const;         // doable, size, half, n_angles, pixel_size_m, reduce, row_origin, tiles, parameters
    // slices h_lo .. h_hi - 1 -> out (h_hi - h_lo, y1 - y0, x1 - x0); debug: one entry per slice
    void reconstruct(int h_lo, int h_hi, float* out, const SliceRoi* roi = nullptr, std::vector<CanvasDebug>* debug = nullptr);
    // of the last chunk, per slice: every tile's level; the angles' spread [bins]
    const std::vector<std::vector<double>>& levels() const;
    const std::vector<double>& angle_spread() const;

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace vc_tomography
