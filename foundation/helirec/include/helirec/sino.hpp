#pragma once
// Sinogram-stage kernels between the gridded projections and backprojection:
// -log, half-acquisition rebalance,
// Ram-Lak ramp filter with opposite-ray ("mirror edges") padding.
#include <utility>
#include <vector>

#include "helirec/detector_map.hpp"
#include "helirec/fourier.hpp"

namespace helirec {

// ------------------------------------------------------------- -log
// x <- -ln(min(max(x, clip_min), clip_max))
namespace cpu {
void minus_log(float* x, std::size_t n, float clip_min = 1e-6f, float clip_max = 10.f);
// x[i] *= w[i]
void multiply(float* x, const float* w, std::size_t n);
// x[i] += s * y[i]
void add_scaled(float* x, const float* y, float s, std::size_t n);
// t[i] = x[i] * (w2[i] - w[i])
void weight_difference(const float* x, const float* w, const float* w2, float* t, std::size_t n);
// x[i] = x[i] * w[i] + t[i]
void multiply_add(float* x, const float* w, const float* t, std::size_t n);
}  // namespace cpu
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void minus_log(float* x, std::size_t n, float clip_min = 1e-6f, float clip_max = 10.f, void* stream = nullptr);
void multiply(float* x, const float* w, std::size_t n, void* stream = nullptr);
void add_scaled(float* x, const float* y, float s, std::size_t n, void* stream = nullptr);
void weight_difference(const float* x, const float* w, const float* w2, float* t, std::size_t n, void* stream = nullptr);
void multiply_add(float* x, const float* w, const float* t, std::size_t n, void* stream = nullptr);
}  // namespace cuda
#endif

// ------------------------------------------------------------- band-dependent redundancy weights
// When the recordings of a ray differ in sharpness (PipelineConfig::sharpness_weighting), they are combined with two sets
// of weights: the usual ones below the band [f1, f2] cycles per pixel, weights that favour the sharper recording
// above it. With X the result of the usual weights and Y that of the others, the combination is X + HP (Y - X), HP
// a raised cosine from 0 at f1 to 1 at f2: both sets sum to one for every ray, so every band is an unbiased average.
// The high-pass as a Fourier filter (fourier.hpp): ndim 2 on frames (h, w), on the frequency magnitude in the
// frame; ndim 1 along rows of w samples. Edge padding.
FourierFilterSpec band_filter_spec(int ndim, int h, int w, double f1, double f2);
// The detector's squared response at `f` cycles per pixel along the columns, for a uniform blur whose width is
// blur[0] X + blur[1] Y + blur[4] pixels at (X, Y) = (column, row) from the detector's centre in units of 1000
// pixels (the first row and the constant of the blur field s = A r + c); 0 beyond the response's first zero.
// (rows, cols), floored at 1e-6.
std::vector<float> blur_response(const std::vector<double>& blur, double f, int rows, int cols);

// ------------------------------------------------------------- rebalance
// R, W: (n_proj, n_rows, n_cols), grid angles theta_i = 2 pi i / n_proj over one full turn.
// Fills empty-angle holes in R and W (mid-row test), then replaces W with the balanced weights
//   W_i / (W_i + M(W at theta_i + pi))       M: mirror about rot_center (interpolated for fractional reflection)
// Every measured line is counted once: itself plus its opposite view, the partner angle taken periodically, so the
// two balanced weights of an exact opposite pair sum to 1 (partition of unity).
// legacy (the reference mode): the exact nabuxx rebalance, including its endpoint quirks (theta + k pi for
// k = +-1..+-3 located without wrapping: angle 0 has no partner at pi - pi, the n_proj endpoint maps to the last
// angle, and angle 0 gets an extra full-turn contribution).
// Returns the number of filled hole runs.
namespace cpu {
int rebalance(float* R, float* W, int n_proj, int n_rows, int n_cols, float rot_center, bool legacy = false);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
// device R, W
int rebalance(float* R, float* W, int n_proj, int n_rows, int n_cols, float rot_center, bool legacy = false,
              void* stream = nullptr);
// the three steps of rebalance, for sinograms sharing their weights (parameter sweeps):
// empty-angle runs of W ({-1, -1} alone: every angle is empty), hole filling of any stack, balanced weights
std::vector<std::pair<int, int>> rebalance_holes(const float* W, int n_proj, int n_rows, int n_cols, void* stream = nullptr);
void fill_holes(float* X, const std::vector<std::pair<int, int>>& holes, int n_proj, int n_rows, int n_cols,
                void* stream = nullptr);
void balance_weights(float* W, int n_proj, int n_rows, int n_cols, float rot_center, bool legacy = false,
                     void* stream = nullptr);
}
#endif

// ------------------------------------------------------------- ramp filter
struct RampGeometry {
    int n_bins = 0, n_proj = 0;
    int npad = 0, pad_l = 0;
    float rot_center = 0.f;
    int two_rots = 0, my_rot = 0, R = 0, switch_pos = 0;
    bool legacy = false;   // padding reflection about lround(2 rot_center) (reference) instead of 2 rot_center
};
// padded length: 2 * smallest 2^a 3^b 5^c 7^d >= n_bins (a in [2,15], b<=9, c<=6, d<=5) * pad_factor
RampGeometry ramp_geometry(int n_bins, int n_proj, float rot_center, int pad_factor = 1, bool legacy = false);
// Ram-Lak half spectrum (npad/2+1) in cycles per pixel (h[0] = 1/4, h[odd n] = -1/(pi n)^2), times `scale`.
// scale = the angular quadrature weight: for n_proj views over a full turn with pair-balanced weights (an opposite
// pair sums to 1, rebalance) attenuation needs 2 pi / n_proj (ramp_scale); the reference uses pi / n_proj, i.e.
// half the attenuation (legacy reference units).
std::vector<float> ramlak_half_spectrum(int npad, float scale);
float ramp_scale(int n_proj, bool legacy);
// Ramp filters (the FBP ramp and gridrec): `ramp` at x cycles per pixel (|x| <= 1/2) times the window `name`.
//   ramlak       1 (the ramp itself)
//   shepp        sinc(x)
//   cosine       cos(pi x)
//   hann         (1 + cos(pi x / p0)) / 2            zero beyond the cutoff p0
//   hamming      0.54 + 0.46 cos(pi x / p0)          zero beyond p0
//   parzen       (1 - |x| / p0)^3                    zero beyond p0
//   butterworth  1 / (1 + (x / p0)^(2 p1))           half amplitude at p0, order p1
//   none         no ramp either: the value is 1 (plain backprojection)
// par = {p0, p1}: the cutoff p0 in cycles per pixel (default 0.5, Nyquist) and the Butterworth order p1 (default 8);
// text "p0:p1": ramp_filter_par. Hann and Hamming are tomopy 1.0's: since 1.1 tomopy's cos(2 pi x / p0) puts their
// minimum at p0 / 2 (half Nyquist by default) and returns to 1 at p0, and tomopy's windows are not cut beyond p0.
std::vector<std::string> ramp_filter_names();
std::vector<float> ramp_filter_par(const std::string& text);   // "0.5:8", "0.5/8", "[0.5, 8]" or a single value
float ramp_windowed(const std::string& name, float x, const std::vector<float>& par, float ramp);
// 1D FourierFilterSpec on padded rows of length npad (no further padding): the Ram-Lak spectrum x window x scale
FourierFilterSpec ramp_filter_spec(const RampGeometry& g, float scale, const std::string& filter_name = "ramlak",
                                   const std::vector<float>& filter_par = {});

// sino (n_proj, n_bins) -> padded (n_proj, npad): own row, opposite (theta+pi) row
// mirrored about the axis (offset-matched near the axis), constant edge elsewhere. The mirror is x -> 2 rot_center - x,
// linearly interpolated (g.legacy: x -> lround(2 rot_center) - x, integer samples, as the reference)
namespace cpu {
void ramp_pad(const float* sino, float* padded, const RampGeometry& g);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void ramp_pad(const float* sino, float* padded, const RampGeometry& g, void* stream = nullptr);
}
#endif

// crop padded (n_proj, npad) rows to the first n_bins samples: out (n_proj, n_bins)
namespace cpu {
void ramp_crop(const float* padded, float* out, const RampGeometry& g);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void ramp_crop(const float* padded, float* out, const RampGeometry& g, void* stream = nullptr);
}
#endif

// ------------------------------------------------------------- exposure arc
// In a continuous scan the detector integrates while the sample turns: each view is the sinogram averaged over an arc
// of `arc` grid angles (a box along the angle; at radius r a tangential blur of r x arc x step). Its inverse
// 1 / sinc(arc f) is bounded (arc 0.91: 1.43 at the angular Nyquist frequency) and well represented by a short
// symmetric kernel: arc_deblur_taps returns g[0..radius] (the Fourier series of the inverse, unit sum), and
// angular_fir applies  out[j, x] = sum_n g[|n|] in[(j + n) mod n_proj, x]  (in != out). With `support` (same shape),
// only where every tap has support > 0; elsewhere out = in.
std::vector<float> arc_deblur_taps(double arc, int radius = 4);
namespace cpu {
void angular_fir(const float* in, float* out, int n_proj, int n_cols, const float* g, int radius,
                 const float* support = nullptr);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void angular_fir(const float* in, float* out, int n_proj, int n_cols, const float* g, int radius,
                 const float* support = nullptr, void* stream = nullptr);
}
#endif

// ------------------------------------------------------------- oversampling
// band-limited upsampling of rows (FFT zero padding to upsample_fft_length(n), Nyquist bin split):
// out (n_rows, n * os), out[r, i] = row r at position i / os
int upsample_fft_length(int n);
namespace cpu {
void upsample_rows(const float* in, float* out, int n_rows, int n, int os);
// rows -> B-spline coefficients (degree 3 or 5; in may equal out)
void bspline_prefilter(const float* in, float* out, int n_rows, int n, int degree);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void upsample_rows(const float* in, float* out, int n_rows, int n, int os, void* stream = nullptr);
void bspline_prefilter(const float* in, float* out, int n_rows, int n, int degree, void* stream = nullptr);
}
#endif

// ------------------------------------------------------------- band-limited shift
// Rows delayed by `shift` samples, in place, without loss: sample i of a row becomes the row's band-limited value at
// position i - shift (a phase ramp on its discrete Fourier transform; a kernel of a few taps cannot do it: at half a
// sample lanczos3 passes 0.83 at 0.36 cycles per sample, 0.33 at 0.45). `rows` is (n_rows, shift_row_stride(n)): n
// samples per row, the rest work space. The transform has upsample_fft_length(n) samples, the ones beyond n a
// raised-cosine bridge from the row's last sample to its first, so that the periodic extension is continuous. For
// an even length the Nyquist bin, which a real row cannot delay by a fraction, is scaled by cos(pi shift).
// shifts (n_rows; CUDA: a device pointer): row r is delayed by shift + shifts[r].
int shift_row_stride(int n);
namespace cpu {
void shift_rows(float* rows, int n_rows, int n, float shift, const float* shifts = nullptr);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void shift_rows(float* rows, int n_rows, int n, float shift, void* stream = nullptr, const float* shifts = nullptr);
}
#endif

// ------------------------------------------------------------- band-limited resampling
// Rows read at positions of their own, in place, without loss: sample j of row r becomes the value at position
// p(r, j) of the row's band-limited interpolant, the one shift_rows delays (the same storage, transform length,
// bridge and treatment of the Nyquist bin: p = j - shift is shift_rows up to rounding):
//   G(p) = (1 / L) (A_0 + 2 Re sum_{0 < k < L / 2} A_k exp(2 pi i k p / L) + Re A_{L/2} cos(pi p)),
// A the transform of the row bridged to L = upsample_fft_length(n) samples (periodic in p). Evaluated as a
// non-uniform transform (type 2): the row's transform, divided by the transform of a compact kernel, is laid in
// one of resample_oversampling x L bins and transformed back; each value is the sum of the resample_taps nearest
// samples of that, weighted by the kernel. The kernel is a Kaiser-Bessel window, I0(beta sqrt(1 - z^2)) / I0(beta),
// z = distance / (taps / 2), as a polynomial of degree resample_degree on each of its taps' intervals (through the
// window at the interval's Chebyshev nodes: within 2e-11 of it); the transform it is divided by is that of those
// polynomials, so that the kernel's aliases are all that is lost: at most 4e-9 of the amplitude of a frequency, at
// any frequency up to Nyquist (10 taps, oversampling 2, beta = 23; a row of n frequencies in phase could add them
// up, random ones do not). The row's transform is taken in single precision, as shift_rows takes it, and its
// rounding is what the result's error is made of (some 1e-7 of the amplitude, 5e-7 at most over the rows tested);
// all that follows is in double precision: the kernel's transform falls by 3.7 from the middle of the band to its
// end, and the sum takes that back by cancellation.
constexpr int resample_oversampling = 2;
constexpr int resample_taps = 10;
constexpr int resample_degree = 10;
constexpr double resample_beta = 23.0;
constexpr int resample_block = 8;   // positions of a row computed together (CPU: in its vector unit; CUDA: by one thread)
static_assert(resample_taps % 2 == 0 && resample_degree >= 2, "resample_values takes the taps in mirrored pairs");
// The kernel: coefficient [q * (resample_taps / 2) + i] of t^q of tap i < resample_taps / 2, t = 2 f - 1, where the
// taps are the oversampled samples first + i, first = ceil(u - taps / 2) for the position u, and
// f = first - (u - taps / 2) in [0, 1); tap resample_taps - 1 - i is the same polynomial of -t (the kernel is even).
const std::vector<double>& resample_kernel();
// the factors of the transform's bins 0 .. L / 2 before they are transformed back: 1 / (L x the kernel's transform),
// half of it at the Nyquist bin of an even L
std::vector<double> resample_deconvolution(int n);

// One row of resample_rows read through a detector map (detector_map.hpp): sample j is the value at the ideal column
// X = j - shift of the row whose ideal row is y at column slope_col and y + row_slope (c - slope_col) at detector
// column c, i.e. the row's value at the detector column p with
//   p + Eu(p, v) + col_scale (p - scale_col) = X,   v + Ev(p, v) + row_scale (v - scale_row) = y + row_slope (p - slope_col)
struct ResampleRow {
    double y = 0;
    float shift = 0;
    float row_slope = 0;
    float col_scale = 0;
    float row_scale = 0;
};
struct ResampleGeometry {
    DetectorMapView map;   // (pointer space = backend)
    float slope_col = 0, scale_col = 0, scale_row = 0;
};
namespace cpu {
// positions: (n_rows, n), in samples of the row (finite; a sample whose position is not becomes NaN)
void resample_rows(float* rows, int n_rows, int n, const double* positions);
// row: (n_rows)
void resample_rows(float* rows, int n_rows, int n, const ResampleRow* row, const ResampleGeometry& g);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
// (positions, row: device pointers)
void resample_rows(float* rows, int n_rows, int n, const double* positions, void* stream = nullptr);
void resample_rows(float* rows, int n_rows, int n, const ResampleRow* row, const ResampleGeometry& g, void* stream = nullptr);
}
#endif

}  // namespace helirec
