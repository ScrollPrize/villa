#pragma once
// The intensity calibration of a helical acquisition of scans moved sideways (two, or more; helirec-calibration
// intensity; python/helirec/calibration.py binds a pair's): the darks and flats reduced as the reference (nx2etf)
// reduces them, the weight map and its horizontal apodisation (prepare_weights_double), the detector's blur field
// from the noise of the flats, and the placement of the scans in the frame they are merged in, from one axis and a
// vector per scan after the first (the geometry calibration's merged layout: helirec/geometry.hpp). Below it, its kernels: the scipy-compatible Gaussian filter and the borders of the
// weight maps, the mean of frames, detector images at reduced resolution, the noise spectra of tiles and the blur
// field's fit. CPU (OpenMP); the data sizes are small. (The geometry itself is the geometry calibration's.)
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "helirec/calibration/scan.hpp"
#include "helirec/calibration/solve.hpp"
#include "helirec/common.hpp"
#include "helirec/geometry.hpp"

namespace helirec {
class RawReader;
}

namespace helirec::calibration {

// scipy.ndimage.gaussian_filter1d semantics: kernel exp(-x^2 / 2 sigma^2) normalised (double),
// radius = int(truncate * sigma + 0.5), mode "reflect" (d c b a | a b c d) or "nearest",
// accumulated in double; data shape (n0, n1, n2), filtered along `axis`.
template <class T>
void gaussian_filter1d(const T* in, T* out, std::size_t n0, std::size_t n1, std::size_t n2, int axis, double sigma,
                       const std::string& mode = "reflect", double truncate = 4.0);

// Horizontal apodisation of a weight map (nabuxx EtfScan::check_weight), float32 with std::erf(float):
// left(x) = 0.5 (1 + erf((x - (wl/2 + shl)) / max(1, wl/5))), right(x) = 0.5 (1 - erf((x - (Lx - (wr/2 + shr))) / max(1, wr/5)))
std::vector<float> apodisation_mask(int Lx, float w_left, float w_right, float min_val, int sh_left = 0, int sh_right = 0);

// Vertical border of prepare_weights_double: (1 - erf((z - L + m + cb)/w))/2 * (1 + erf((z - m - ct)/w))/2,
// m = round_half_even(tw / 2), w = tw / 5, float32 arguments, erf evaluated in double (scipy 'f' loop)
std::vector<float> weight_border_v(int L, double transition_width, int crop_top = 0, int crop_bottom = 0);

// ---------------------------------------------------------------- darks, flats and weights
// The mean of n frames read through `reader` (uint16 frames), as nx2etf reduces darks and flats: every frame taken
// to float32 and added in the order given, the sum divided by n. With `dark` and `control_A` (one ring current
// per frame), each frame F enters as dark + (F - dark) nominal_A / control_A: a flat at the nominal current.
// progress(k): called as k more frames are done. Returns (rows, cols) of the reader.
std::vector<float> mean_frames(RawReader& reader, const int32_t* file_index, const int32_t* frame_index, std::size_t n,
                               const float* dark = nullptr, const double* control_A = nullptr, double nominal_A = 0,
                               const std::function<void(std::size_t)>& progress = nullptr);

// prepare_weights_double: the mean of all the flats given (each (rows, cols), added in order) over its maximum,
// smoothed by a Gaussian of `sigma` along both axes ("reflect"), times the vertical border (weight_border_v).
std::vector<float> weight_map(const std::vector<const float*>& flats, int rows, int cols, double transition_width_vertical = 15.0,
                              double sigma = 10.0, int crop_top = 0, int crop_bottom = 0);

// The rows the beam lights, [first, end): around the brightest row of a flat less its dark (each row's mean over
// the columns), the rows that reach `share` of it. A beam cut by slits along the axis lights a part of the
// detector (Diamond I12 at PHerc. 172: rows 252 to 1905 of 2160, the rows beyond reading 5 to 8 % of the
// brightest); where every row is lit (BM18: the first and last row at half the brightest) it is every row.
// The geometry calibration keeps its comparisons inside them, the intensity calibration its weights.
inline constexpr double kLitShare = 0.12;
std::array<int, 2> lit_rows(const float* flat_less_dark, int rows, int cols, double share = kLitShare);

// The horizontal apodisation widths (left, right) of the main scan and of the companion, as
// prepare_weights_double --is_tetra writes them: on the main scan a slope beyond the rotation axis
// (min(2 transition_width_horizontal, distance from the axis to the edge)) on the axis' side and a border taper
// towards the companion; on the companion a border taper on the side facing the main scan.
// est_cor_main_offset: the axis' column from the middle of the main scan; companion_dx: where the companion lies
// (below nothing: on the left, the axis then in the main scan's right half, as the reference has it; above: on
// the right, the axis in its left half, the same widths mirrored).
struct ApodisationWidths {
    double main_left = 0, main_right = 0, companion_left = 0, companion_right = 0;
};
ApodisationWidths apodisation_widths(int n_cols, double est_cor_main_offset, double companion_dx,
                                     double transition_width_horizontal = 1000.0, double transition_width_borders = 50.0);

// Images of the detector (darks, flats, weights: n images of (rows, cols)) as RawReader(reduce=) gives the frames
// stored as JPEG 2000 in chunks of chunk_rows rows: `reduce` levels of the irreversible (9/7) wavelet's low band,
// each chunk of rows an image of its own, sample (j, k) at the stored pixel (j, k) 2^reduce. A flat reduced so
// divides a frame's fixed pattern out as the full-resolution flat does; a mean of 2^reduce pixels would not. A
// last chunk that reaches beyond the frame holds zeros there (as the detector software stores it), which change
// the frame's last two or three reduced rows, here as in the frames. Computed in double.
// Returns n images of (rows >> reduce, cols halved `reduce` times, rounding up); out_rows, out_cols receive that.
std::vector<float> at_resolution(const double* images, std::size_t n, int rows, int cols, int reduce, int chunk_rows,
                                 int* out_rows = nullptr, int* out_cols = nullptr);

// ---------------------------------------------------------------- detector blur from the noise of the flats
// Power spectra of images on a lattice of tiles. img: (n, rows, cols); the tile at (ys[iy], xs[ix]) of every image is
// multiplied by a Hann window (numpy.hanning along both axes) and transformed; the squared magnitudes are added over
// the images into power (ny, nx, tile, tile / 2 + 1): rows of the vertical frequency in FFT order (i / tile, from
// tile / 2 on (i - tile) / tile), columns the horizontal frequency j / tile.
void tile_power_spectra(const float* img, std::size_t n, std::size_t rows, std::size_t cols, int tile, const int* ys,
                        int ny, const int* xs, int nx, double* power);

// The detector's blur field s(r) = A r + c from the noise power spectra of flats on tiles (tile_power_spectra of
// differences of flats).
// The part of the noise that has passed through the optics is blurred like the signal: above the white floor (the
// mean of the frequencies beyond 0.42 cycles / pixel in both directions) and normalised below 0.11 cycles / pixel, the
// noise power of the tile at r is taken as B(f) sinc^2(s(r) . f), B the spectrum where there is no blur (one unknown
// per frequency bin, eliminated by least squares): a uniform blur along s. Bins of 1 / 16 cycles per pixel between
// 0.05 and f_max; above about 0.3 the blurred noise is a few percent of a floor that is not white enough to carry it.
// field: [a11, a12, a21, a22, c1, c2], s = (a11 X + a12 Y + c1, a21 X + a22 Y + c2) pixels along the columns and rows
// at (X, Y) = (column, row) of the tile's centre from the detector's centre in units of 1000 pixels; the noise does
// not see the sign of s: a11 >= 0. rms_none, rms_fit: residual of the normalised spectra without and with the field.
struct DetectorBlurField {
    FitSummary solver;
    double field[6] = {0, 0, 0, 0, 0, 0};
    double rms_none = 0, rms_fit = 0;
};
DetectorBlurField detector_blur_field_fit(const double* power, int ny, int nx, int tile, const int* ys, const int* xs,
                           std::size_t rows, std::size_t cols, double f_max = 0.30);

// ---------------------------------------------------------------- the calibration of a scan pair
// Bump when anything that changes the cached results changes: it is part of every fingerprint.
inline constexpr int kIntensityCalibrationVersion = 3;
// nx2etf takes every flat to this ring current [A]
inline constexpr double kNominalCurrentA = 0.2;

// A scan's darks and flats as nx2etf stores them: the mean of the darks, and per flat series the mean of its frames
// each taken to the nominal ring current (mean_frames).
struct DarksFlats {
    int rows = 0, cols = 0;
    std::vector<float> dark;   // (rows, cols)
    std::vector<float> flats;  // (series, rows, cols)
    std::size_t series() const { return rows && cols ? flats.size() / (static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) : 0; }
};
// progress(k): called as k more of the scan's dark and flat frames are done
DarksFlats reduce_darks_flats(const Scan& scan, const std::function<void(std::size_t)>& progress = nullptr);
// The same kept in cache_dir ("": nothing kept) as darks_flats_<scan>.bin (helirec's array file: io/array_file.hpp),
// valid for the dark and flat files it was made from (path, size, time of change), the frames taken from them and
// the flats' ring current. Reducing a scan's 1200 frames takes most of a minute; reading them back a fraction of a second.
DarksFlats reduced_darks_flats(const Scan& scan, const std::string& cache_dir, const std::function<void(const std::string&)>& log = nullptr);

// Where a scan lies in the merged frame: its column 0 at merged column shift_x - x_readjustment, its rows
// z_readjustment rows from the first scan's.
struct ScanPlacement {
    int shift_x = 0;
    double x_readjustment = 0, z_readjustment = 0;
};
// nabuxx check_companion: the first scan and every other at its vector (rows, columns) from it, the outermost scan
// at column 0 of the merged frame. anchor: where the first scan's column 0 lies; width: the merged frame's columns.
struct MergedPlacement {
    std::vector<ScanPlacement> scans;
    int width = 0;
    double anchor = 0;
};
MergedPlacement placement_from_vectors(const std::vector<std::array<double, 2>>& vectors, int cols);
MergedPlacement placement_from_vector(const std::array<double, 2>& vector, int cols);  // a pair

// The detector's blur field of a scan, from the noise of its first flat series (flat_noise_spectra, then
// detector_blur_field_fit). It is kept (`field`) only if it explains the noise spectra (a residual below
// kBlurMinGain of the residual without it) and reaches kBlurMinPx somewhere on the detector; otherwise `field` is
// empty and the reconstruction does not use one.
inline constexpr int kBlurTile = 128;
inline constexpr int kBlurMinFlats = 20;      // fewer flats in the series do not carry the noise spectra
inline constexpr double kBlurMinPx = 0.5;
inline constexpr double kBlurMinGain = 0.97;
struct DetectorBlur {
    std::vector<double> field;       // [a11, a12, a21, a22, c1, c2] (DetectorBlurField), or none
    bool was_measured = false;       // false: too few flats or tiles (reason)
    std::vector<double> measured;    // the field as fitted, kept or not
    double rms_none = 0, rms_fit = 0, max_blur_px = 0;
    std::array<double, 2> zero{};    // the detector position (column, row) where the field is zero
    std::array<int, 2> tiles{};      // (rows, columns) of the lattice of tiles
    int n_flats = 0;
    FitSummary solver;
    std::string scan, reason;
};
// Noise power spectra of the scan's first flat series on a lattice of tiles of `tile` pixels (tile_power_spectra):
// of the differences of consecutive flats, each less the dark and divided by its mean in float32 (the beam's profile
// and the fixed pattern drop out, the noise stays). Returns power (ny, nx, tile, tile / 2 + 1) and the tiles' origins.
struct FlatNoise {
    std::vector<double> power;
    std::vector<int> ys, xs;
    int n_flats = 0;
};
// lit: the rows the beam lights, [first, end) (lit_rows): the tiles lie inside them, since a tile the beam does not
// light holds no noise of the beam's (nothing named, end <= first: every row).
FlatNoise flat_noise_spectra(const Scan& scan, const float* dark, int n_flats = 200, int tile = kBlurTile, std::array<int, 2> lit = {0, 0});
DetectorBlur measure_detector_blur(const Scan& scan, const float* dark, int n_flats = 200,
                                   const std::function<void(const std::string&)>& log = nullptr, std::array<int, 2> lit = {0, 0});

// sha256 over what the calibration of a pair depends on: its version, the scans' identities, metadata, frame
// geometry and trajectories, and the raw files they read (path, size, time of change). A replaced acquisition (even
// under the same name) or another pair gives another fingerprint.
std::string acquisition_fingerprint(const Scan& main, const Scan& companion);
std::string acquisition_fingerprint(const std::vector<const Scan*>& scans);  // the same of any number of scans

// A scan's darks and flats as a projection store holds them at the level `reduce` (helirec/projection_store.hpp:
// float images, taken through the frames' low-pass when the store was made), reduced as reduce_darks_flats reduces
// the detector's: the mean of the darks, and per flat series the mean of its frames each taken to the nominal ring
// current. tile_group: STORE.zarr/<tile>. rows, cols of the result: the level's.
DarksFlats store_darks_flats(const Scan& scan, const std::string& tile_group, int reduce);

// The scans in the order of the geometry calibration: the first (the main scan) has the rotation axis in its field,
// the others (companions) lie beside it, each further from the axis, all on one side of it; or at the columns of
// one of those and another height (a grid of circular scans: the rows of scans overlap along the axis, and where
// two hold a height both enter by their weights, which the weight map's vertical border tapers).
struct IntensityCalibration {
    std::string main, companion;           // the names of the first two scans (companion: "" for a scan alone)
    std::vector<std::string> scans;        // every scan's name
    int rows = 0, cols = 0;                // of the frames it is of (a projection store's level: that level's)
    std::string projection_store;          // the store its darks and flats are from ("": the detector's files)
    int reduce = 0;                        // and the level
    double cor_main_detector = 0;          // the rotation axis in the main scan's columns
    std::array<double, 2> vector{};        // (rows, columns) from the main scan to the second
    std::vector<std::array<double, 2>> vectors;             // from the main scan to every scan after it
    double rot_center = 0;                 // the rotation axis in the merged frame
    int merged_width = 0;
    std::vector<ScanPlacement> placement;                   // per scan
    std::vector<std::array<double, 2>> apodisation;         // (left, right) widths, per scan
    std::vector<double> estimated_cor_from_motor_pix;       // per scan
    geometry::MergedLayout layout;         // what the frame was laid out from
    DetectorBlur detector_blur;
    std::string fingerprint;               // acquisition_fingerprint of the scans
    int version = kIntensityCalibrationVersion;

    std::vector<DarksFlats> reduced;                  // per scan
    // The rows the beam lights in every scan, [first, end) (lit_rows of each scan's flats less its dark): the
    // weight map is cropped to them (weight_map's crop_top, crop_bottom). Every row where every row is lit.
    std::array<int, 2> lit{0, 0};
    std::vector<float> weight_map;                    // of all the flats (the last scan's first, as the reference orders a pair's)
    std::vector<std::vector<float>> weight;           // per scan: the weight map times the scan's horizontal apodisation

    // calibration.json: everything but the arrays
    std::string json() const;
};

// The intensity calibration in the merged frame of `layout` (geometry::Calibration::merged_layout). cache_dir ("":
// nothing kept): calibration_arrays.bin (the darks and flats reduced and the weight map) and calibration.json (the
// rest, the blur field among it), both keyed by acquisition_fingerprint; a cache of another acquisition or version
// is ignored and replaced. The blur field is the detector's and is kept whatever the merged frame. progress: show
// progress bars for the reduction.
// projection_store, reduce: the darks and flats are a projection store's at the level `reduce` (store_darks_flats;
// each scan's group is projection_store_tile_group of its sample name), and everything is of that level's frames:
// `layout` is then given in the level's pixels, the weight map's and the apodisation's widths are divided with the
// pixels, nothing is kept in cache_dir, and no blur field is measured (the noise of reduced flats does not hold it).
IntensityCalibration calibrate_intensity(const std::vector<const Scan*>& scans, const geometry::MergedLayout& layout, const std::string& cache_dir,
                                         const std::function<void(const std::string&)>& log = nullptr, bool progress = false,
                                         const std::string& projection_store = "", int reduce = 0);
IntensityCalibration calibrate_intensity(const Scan& main, const Scan& companion, const geometry::MergedLayout& layout, const std::string& cache_dir,
                                         const std::function<void(const std::string&)>& log = nullptr, bool progress = false);  // a pair

}  // namespace helirec::calibration
