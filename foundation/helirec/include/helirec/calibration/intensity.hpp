#pragma once
#include "helirec/calibration/solve.hpp"
// The intensity calibration's kernels (python/helirec/calibration.py): the scipy-compatible Gaussian filter and the
// borders of the weight maps, and the detector's blur field from the noise of the flats. CPU (OpenMP); the data
// sizes are small. (The geometry is the geometry calibration's: helirec/calibration/, python/helirec/geometry.py.)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "helirec/common.hpp"

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

// The horizontal apodisation widths (left, right) of the main scan and of the companion, as
// prepare_weights_double --is_tetra writes them: on the main scan a slope beyond the rotation axis
// (min(2 transition_width_horizontal, distance from the axis to the edge)) on the axis' side and a border taper
// towards the companion; on the companion a border taper on the side facing the main scan.
// est_cor_main_offset: the axis' column from the middle of the main scan; companion_dx: where the companion lies.
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

}  // namespace helirec::calibration
