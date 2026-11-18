#pragma once
// A mosaic reconstructed as one slice by full-field filtered backprojection (whole-canvas FBP).
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
// The folded blend described above takes each bin's weighted mean angle. With rebin_angles, each tile is rebinned for its
// column fan at fixed canvas column. Uniform grids use the first tile's phase and fold the two halves.
// Measured mode completes and backprojects each native direction. The default exposure mode mixes the ordinary
// and travel-weighted fine frequency bands on that grid for each coverage disc, then crossfades overlapping discs.
// All modes use horizontal rays and fixed tile poses.
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
// One turn of measured row centre angles, sorted in output-bin units, with periodic
// neighbours at either end. lower[b] brackets integer bin b. fan[c] is the signed
// shift in input bins needed to reach the target ray direction, modulo one turn.
struct CanvasView { double bin; int row; };
struct CanvasRebin {
    const CanvasView* views;
    const int* lower;
    const double* fan;
    const double* measured;
};
// Geometry of a native view and of a detector column for the single angular gather.
struct CanvasPose { int first; double fraction, projection, tangent; };
struct CanvasFan { double bins, cos_minus_one, sin_angle; };
struct CanvasTarget { int first; double fraction, bin; };
// Opposite-side stencil at the mirrored canvas column, and the native view's
// full-turn quadrature relative to the existing half-turn ramp scale.
struct CanvasOpposite {
    int row[6];
    float coefficient[6], fraction, quadrature;
    bool lanczos;
};
struct CanvasGather {
    const CanvasView* views;
    const int* lower;
    const CanvasPose* poses;
    const CanvasFan* fan;
    const CanvasTarget* targets;
    bool lanczos;
};
// Source rows and column offsets for two-pass interpolation at fixed canvas column.
struct CanvasBracket {
    int lo, hi, dx_lo, dx_hi;
    float fraction;
};
// Footprint in absolute canvas coordinates. fade is the width of its inner cosine taper.
struct CanvasDisc { double x, y, radius, fade; };
namespace cpu {
void place_exposure(const float* signal, const float* weight, const std::int32_t* first, const float* quality, float level,
                    int rows, int n, int stride, float* sums, float* support, int width);
void blend_region(const CanvasDisc* discs, int count, int focus, SliceRoi region, SliceRoi output,
                  const float* image, float* result);
void complete_views(const CanvasOpposite* plan, const float* sums, const float* weights, int rows, int width, float* out, float* support);
void pad_measured(const CanvasOpposite* plan, const float* sino, int rows, int width, int padded, int split, float* out);
void gather_rows(CanvasGather grid, const float* s, const float* w, int rows, int n, int stride, float* out_s, float* out_w);
void rebin_placed_rows(const CanvasBracket* brackets, const float* s, const float* w, int n_rows, int n, int stride, float* out_s, float* out_w);
void rebin_tile_rows(CanvasRebin grid, const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w);
void tile_rows(const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w);
void place_rows(const float* rows, const std::int32_t* first, float scale, bool nonneg, int n_rows, int n, int stride, float* canvas, int width);
void fold_opposite(float* S, int n_rows, int width);
}  // namespace cpu
#ifdef VC_TOMOGRAPHY_WITH_CUDA
namespace cuda {
void place_exposure(const float* signal, const float* weight, const std::int32_t* first, const float* quality, float level,
                    int rows, int n, int stride, float* sums, float* support, int width, void* stream = nullptr);
void blend_region(const CanvasDisc* discs, int count, int focus, SliceRoi region, SliceRoi output,
                  const float* image, float* result, void* stream = nullptr);
void complete_views(const CanvasOpposite* plan, const float* sums, const float* weights, int rows, int width, float* out, float* support, void* stream = nullptr);
void pad_measured(const CanvasOpposite* plan, const float* sino, int rows, int width, int padded, int split, float* out, void* stream = nullptr);
void gather_rows(CanvasGather grid, const float* s, const float* w, int rows, int n, int stride, float* out_s, float* out_w, void* stream = nullptr);
void rebin_placed_rows(const CanvasBracket* brackets, const float* s, const float* w, int n_rows, int n, int stride, float* out_s, float* out_w, void* stream = nullptr);
void rebin_tile_rows(CanvasRebin grid, const float* s, const float* w, const float* gain, int n_rows, int n, int stride, float* out_s, float* out_w, void* stream = nullptr);
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
    std::vector<float> fan;  // calibrated fan angle [rad] for each column, or empty for parallel rays
    double exposure_gate_rad = 0;  // angle swept during one exposure; zero when unknown
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
    bool rebin_angles = true;  // interpolate onto a common physical angle grid
    std::string rebin_kernel = "exposure";  // two_pass, linear, lanczos3, measured or exposure
    double exposure_fade = 600;  // coverage-disc taper in canvas pixels
};
// On request: completed sinogram and support (n_angles, width); filtered rows and
// their backprojection angles (n_angles in measured and exposure modes, n_angles/2 after folding). Exposure reports the ordinary full-field rows.
struct CanvasDebug {
    std::vector<float> sino, weights, filtered, angles;
    double opposite_only_fraction = 0;  // fraction of supported samples supplied only by the other side
};

class MosaicCanvas {
  public:
    MosaicCanvas(CanvasConfig config, std::vector<CanvasTile> tiles);
    ~MosaicCanvas();
    int size() const;        // pixels a side
    bool exposure_weighted() const;  // true when the requested mode has gates for all tiles
    int half() const;
    const CanvasConfig& config() const;
    // a chunk of n_slices slices: empty sums
    void begin(int n_slices);
    // Optional geometry prepass for measured mode. Supply one vector per tile before add.
    // Without a prepass, the first added tile supplies the target directions.
    void add_view_angles(int y, const float* angles);
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
    // its interpolation, oversampling and ramp window, z_interp, x_interp. rebin_angles takes beam_divergence.
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
    bool rebin_angles = true;
    std::string rebin_kernel = "exposure";
};
struct MosaicTile {
    std::string name, master;
    double ax = 0, ay = 0, height = 0, rot_center = 0;   // canvas place and axis [px of the frames read]; height [stored rows]
    std::pair<int, int> doable{0, 0};
    double exposure_gate_rad = 0;
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
    int max_chunk() const;                 // maximum number of slices to group with the current working set
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
