#pragma once
// Calibration kernels: rotation-axis merits, HA/TA overlap
// correlation, and the scipy-compatible Gaussian filter used by the weight maps
// and the overlap filtering. CPU (OpenMP); the data sizes are small.
#include <cstddef>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::calib {

// scipy.ndimage.gaussian_filter1d semantics: kernel exp(-x^2 / 2 sigma^2) normalised (double),
// radius = int(truncate * sigma + 0.5), mode "reflect" (d c b a | a b c d) or "nearest",
// accumulated in double; data shape (n0, n1, n2), filtered along `axis`.
template <class T>
void gaussian_filter1d(const T* in, T* out, std::size_t n0, std::size_t n1, std::size_t n2, int axis, double sigma,
                       const std::string& mode = "reflect", double truncate = 4.0);

// nabuxx build_all_merits with its effective argument roles:
// band = G_wide(s) - G_narrow(s), norm = G_narrow(|s|), float kernels (radius ceil(3 sigma)),
// clamp-to-edge; for overlap o = ovl_min + k/ovs:  p = W - 1 - (x + o - W)  (mirror of line 1)
// merit[l, k] = sum w0 w1(p) |b0 - b1(p)| / (n0 + n1(p)) / sum w0 w1(p)   (+inf -> nearest finite)
// sig, w: (2, n_lines, W)  -> merits, energies: (n_lines, n_steps)
void build_all_merits(const float* sig, const float* w, std::size_t n_lines, std::size_t W, int ovl_min, int ovl_max,
                      int ovs, float sigma_wide, float sigma_narrow, float* merits, float* energies);

// L1 loss of img0[k, y, x] vs img1[k, y - dy, x - dx], weighted by w0 * w1, 1-pixel margin
double loss_integer(const double* img0, const double* w0, const double* img1, const double* w1, std::size_t n,
                    std::size_t Y, std::size_t X, int dy, int dx);
// same with a bilinear (clamped) sample of img1 / w1 at (y - dyF, x - dxF)
double loss_subpixel(const double* img0, const double* w0, const double* img1, const double* w1, std::size_t n,
                     std::size_t Y, std::size_t X, double dyF, double dxF);

struct OffsetResult {
    int best_dy = 0, best_dx = 0;
    double best_loss = 0;
    std::vector<double> loss_map;  // (2 ry + 1, 2 rx + 1)
    double sub_dy = 0, sub_dx = 0, sub_loss = 0;
};
// integer search dy in init_dy +- ry, dx in init_dx +- rx (strict <, dy-major), then sub-pixel
// refinement on a (2 range / step + 1)^2 grid around the integer optimum
OffsetResult find_offset(const double* img0, const double* w0, const double* img1, const double* w1, std::size_t n,
                         std::size_t Y, std::size_t X, int init_dy, int init_dx, int ry, int rx, double sub_range = 1.5,
                         double sub_step = 0.25);

// Opposite projections of a parallel beam are mirror images about the rotation axis; in a helical scan the second is
// also displaced along the rows by the sample's travel in between. a, b: (rows, cols) line integrals (-log of the
// flat-fielded frames) of two projections half a turn apart. The columns axis + u of a are compared with the columns
// axis - u of b, u = -half_width .. half_width (linear interpolation along the columns), both less their Gaussian
// blur of `sigma` pixels (rows and columns): what is left is the object's fine structure. Returns the shift s (rows:
// what is at row v of a is at row v + s of b) with the largest normalised correlation among the whole shifts within
// `radius` of `expected`, refined by a parabola through its neighbours, and that correlation; on_edge: the largest
// was the first or last shift tried (no maximum inside the range: not a measurement). `margin` rows and columns at
// the borders of the compared strips are left out.
struct RowShift {
    double shift = 0, correlation = 0;
    bool on_edge = false;
};
RowShift mirror_row_shift(const float* a, const float* b, std::size_t rows, std::size_t cols, double axis,
                          int half_width, double expected, int radius, double sigma = 6.0, int margin = 20);

// Horizontal apodisation of a weight map (nabuxx EtfScan::check_weight), float32 with std::erf(float):
// left(x) = 0.5 (1 + erf((x - (wl/2 + shl)) / max(1, wl/5))), right(x) = 0.5 (1 - erf((x - (Lx - (wr/2 + shr))) / max(1, wr/5)))
std::vector<float> apodisation_mask(int Lx, float w_left, float w_right, float min_val, int sh_left = 0, int sh_right = 0);

// Vertical border of prepare_weights_double: (1 - erf((z - L + m + cb)/w))/2 * (1 + erf((z - m - ct)/w))/2,
// m = round_half_even(tw / 2), w = tw / 5, float32 arguments, erf evaluated in double (scipy 'f' loop)
std::vector<float> weight_border_v(int L, double transition_width, int crop_top = 0, int crop_bottom = 0);

// ---------------------------------------------------------------- registration of two frames on tiles
// Displacement of image b's content from image a's on a lattice of tiles: tiles of `tile` x `tile` pixels every
// `step`, `margin` pixels from the borders; each tile less its mean under a Hann window; the largest value of the
// cross-correlation within `search` pixels, refined by a parabola through its neighbours along each axis. Returns
// (n, 5): the tile's centre row and column, dv, du (what is at (r, c) of a is at (r + dv, c + du) of b) and the
// normalised correlation there; dv, du are NaN (correlation 0) where the largest value lies on the edge of the
// search range.
std::vector<double> tile_shifts(const float* a, const float* b, std::size_t rows, std::size_t cols, int tile, int step,
                                int margin, int search);

// ---------------------------------------------------------------- detector smear from the noise of the flats
// Power spectra of images on a lattice of tiles. img: (n, rows, cols); the tile at (ys[iy], xs[ix]) of every image is
// multiplied by a Hann window (numpy.hanning along both axes) and transformed; the squared magnitudes are added over
// the images into power (ny, nx, tile, tile / 2 + 1): rows of the vertical frequency in FFT order (i / tile, from
// tile / 2 on (i - tile) / tile), columns the horizontal frequency j / tile.
void tile_power_spectra(const float* img, std::size_t n, std::size_t rows, std::size_t cols, int tile, const int* ys,
                        int ny, const int* xs, int nx, double* power);

// The detector's smear field s(r) = A r + c from the noise power spectra of flats on tiles (tile_power_spectra of
// differences of flats).
// The part of the noise that has passed through the optics is smeared like the signal: above the white floor (the
// mean of the frequencies beyond 0.42 cycles / pixel in both directions) and normalised below 0.11 cycles / pixel, the
// noise power of the tile at r is taken as B(f) sinc^2(s(r) . f), B the spectrum where there is no smear (one unknown
// per frequency bin, eliminated by least squares): a uniform smear along s. Bins of 1 / 16 cycles per pixel between
// 0.05 and f_max; above about 0.3 the smeared noise is a few percent of a floor that is not white enough to carry it.
// field: [a11, a12, a21, a22, c1, c2], s = (a11 X + a12 Y + c1, a21 X + a22 Y + c2) pixels along the columns and rows
// at (X, Y) = (column, row) of the tile's centre from the detector's centre in units of 1000 pixels; the noise does
// not see the sign of s: a11 >= 0. rms_none, rms_fit: residual of the normalised spectra without and with the field.
struct SmearField {
    double field[6] = {0, 0, 0, 0, 0, 0};
    double rms_none = 0, rms_fit = 0;
};
SmearField smear_field_fit(const double* power, int ny, int nx, int tile, const int* ys, const int* xs,
                           std::size_t rows, std::size_t cols, double f_max = 0.30);

}  // namespace helirec::calib
