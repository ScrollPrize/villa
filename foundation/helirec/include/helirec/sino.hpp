#pragma once
// Sinogram-stage kernels between the gridded projections and backprojection:
// -log, half-acquisition rebalance,
// Ram-Lak ramp filter with opposite-ray ("mirror edges") padding.
#include <utility>
#include <vector>

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
// When the recordings of a ray differ in sharpness (PipelineConfig::detector_smear), they are combined with two sets
// of weights: the usual ones below the band [f1, f2] cycles per pixel, weights that favour the sharper recording
// above it. With X the result of the usual weights and Y that of the others, the combination is X + HP (Y - X), HP
// a raised cosine from 0 at f1 to 1 at f2: both sets sum to one for every ray, so every band is an unbiased average.
// The high-pass as a Fourier filter (fourier.hpp): ndim 2 on frames (h, w), on the frequency magnitude in the
// frame; ndim 1 along rows of w samples. Edge padding.
FourierFilterSpec band_filter_spec(int ndim, int h, int w, double f1, double f2);
// The detector's squared response at `f` cycles per pixel along the columns, for a uniform smear whose width is
// smear[0] X + smear[1] Y + smear[4] pixels at (X, Y) = (column, row) from the detector's centre in units of 1000
// pixels (the first row and the constant of the smear field s = A r + c); 0 beyond the response's first zero.
// (rows, cols), floored at 1e-6.
std::vector<float> smear_response(const std::vector<double>& smear, double f, int rows, int cols);

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

}  // namespace helirec
