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

// Horizontal apodisation of a weight map (nabuxx EtfScan::check_weight), float32 with std::erf(float):
// left(x) = 0.5 (1 + erf((x - (wl/2 + shl)) / max(1, wl/5))), right(x) = 0.5 (1 - erf((x - (Lx - (wr/2 + shr))) / max(1, wr/5)))
std::vector<float> apodisation_mask(int Lx, float w_left, float w_right, float min_val, int sh_left = 0, int sh_right = 0);

// Vertical border of prepare_weights_double: (1 - erf((z - L + m + cb)/w))/2 * (1 + erf((z - m - ct)/w))/2,
// m = round_half_even(tw / 2), w = tw / 5, float32 arguments, erf evaluated in double (scipy 'f' loop)
std::vector<float> weight_border_v(int L, double transition_width, int crop_top = 0, int crop_bottom = 0);

}  // namespace helirec::calib
