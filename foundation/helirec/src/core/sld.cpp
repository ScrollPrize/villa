#include "helirec/sld.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <mutex>
#include <limits>
#include <numeric>
#include <vector>

#include "common/sld_core.hpp"
#include "helirec/filters.hpp"
#include "helirec/fourier.hpp"

namespace helirec {

namespace {
std::vector<std::complex<double>> real_fft(std::vector<double>& data, int h, int w) {
    std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(h) * (w / 2 + 1));
    fftw_plan plan;
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        plan = fftw_plan_dft_r2c_2d(h, w, data.data(), reinterpret_cast<fftw_complex*>(spectrum.data()), FFTW_ESTIMATE);
    }
    if (!plan) throw Error("sld: FFT planning failed");
    fftw_execute(plan);
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        fftw_destroy_plan(plan);
    }
    return spectrum;
}
}  // namespace

struct ScintillatorGlare::Impl {
    int det_rows = 0, det_cols = 0, rows = 0, cols = 0, bin = 1;
    int border_x = 0, border_y = 0;
    bool bm18 = false;
    std::unique_ptr<FourierFilter> filter;   // S on the zero-padded map grid
    std::vector<float> maps;                 // work: glare maps of correct()

    void configure_bm18(const SldParams& p) {
        if (p.bin != 1) throw Error("sld: BM18 full-resolution correction requires bin=1");
        if (p.stride < 1) throw Error("sld: stride must be >= 1");
        if (p.border_x < 0 || p.border_y < 0) throw Error("sld: illuminated borders must be non-negative");
        auto fractions = p.fractions.empty() ? std::vector<double>{p.fraction} : p.fractions;
        auto lengths = p.lengths.empty() ? std::vector<double>{p.length} : p.lengths;
        if (fractions.size() != lengths.size()) throw Error("sld: fraction and length lists must have equal lengths");
        double max_length = 0, mass = 1;
        for (std::size_t k = 0; k < fractions.size(); ++k) {
            if (!std::isfinite(fractions[k]) || fractions[k] < 0)
                throw Error("sld: fractions must be finite and non-negative");
            if (!std::isfinite(lengths[k]) || lengths[k] <= 0)
                throw Error("sld: lengths must be finite and positive");
            max_length = std::max(max_length, lengths[k]);
            mass += fractions[k];
        }
        if (!std::isfinite(mass)) throw Error("sld: total tail weight is not finite");
        bm18 = true;
        border_x = p.border_x;
        border_y = p.border_y;
        // Nabuxx nr_26_10_27: illuminated edge extension, then 2*max(length)
        // dark pixels; extra FFT padding only at the bottom/right. Full-resolution
        // filtering retains both frequency bands without a truncated HF kernel.
        auto padded_size = [&](int n, int border) {
            const double size = n + 2.0 * border + 2.0 * std::floor(2.0 * max_length);
            if (size > std::numeric_limits<int>::max() / 2)
                throw Error("sld: requested padding is too large");
            return static_cast<int>(size);
        };
        FourierFilterSpec sp;
        sp.ph = next_smooth_power(padded_size(rows, border_y));
        sp.pw = next_smooth_power(padded_size(cols, border_x));
        sp.h = rows + 2 * border_y;
        sp.w = cols + 2 * border_x;
        sp.pad_top = sp.pad_left = static_cast<int>(2.0 * max_length);
        sp.mode = PadMode::Zero;
        std::vector<double> kernel(static_cast<std::size_t>(sp.ph) * sp.pw, 0.0);
        std::vector<double> tail(kernel.size());
        for (std::size_t k = 0; k < fractions.size(); ++k) {
            if (fractions[k] == 0) continue;
            // Subtract the nearest non-origin distance in the exponent to avoid
            // underflow for sub-pixel tails; normalization cancels the factor.
            for (int r = 0; r < sp.ph; ++r)
                for (int c = 0; c < sp.pw; ++c) {
                    const double radius = std::hypot(std::min(r, sp.ph - r), std::min(c, sp.pw - c));
                    tail[static_cast<std::size_t>(r) * sp.pw + c] = radius > 0 ?
                        std::exp(-(radius - 1.0) / lengths[k]) / (radius + 1e-5) : 0.0;
                }
            const double sum = std::accumulate(tail.begin(), tail.end(), 0.0);
            if (!(sum > 0)) throw Error("sld: no non-origin samples in diffusion kernel");
            for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] += fractions[k] * tail[i] / sum;
        }
        kernel[0] = 1.0;
        const auto K = real_fft(kernel, sp.ph, sp.pw);
        sp.filter.resize(K.size());
        for (std::size_t i = 0; i < K.size(); ++i) {
            // Do not silently regularize a singular or non-positive response.
            if (!(K[i].real() > 1e-8 * mass)) throw Error("sld: diffusion PSF is not safely invertible");
            sp.filter[i] = static_cast<float>(1.0 - mass / K[i].real());
        }
        filter = cpu::make_fourier_filter(sp);
    }

    // mean of each bin x bin cell (edge cells: of the pixels they hold), for the cells of detector rows [ra, rb)
    // (ra a multiple of bin); row(r, v): detector row r -> v (det_cols). Each cell sums its pixels in double, row by
    // row.
    template <class Row>
    void bin_rows(Row&& row, int ra, int rb, float* out) const {
        std::vector<float> v(det_cols);
        std::vector<double> s(cols);
        for (int i = ra / bin; i * bin < rb; ++i) {
            const int r0 = i * bin, r1 = std::min(det_rows, r0 + bin);
            std::fill(s.begin(), s.end(), 0.0);
            for (int r = r0; r < r1; ++r) {
                row(r, v.data());
                for (int j = 0; j < cols; ++j) {
                    const int c0 = j * bin, c1 = std::min(det_cols, c0 + bin);
                    double a = s[j];
                    for (int c = c0; c < c1; ++c) a += v[c];
                    s[j] = a;
                }
            }
            for (int j = 0; j < cols; ++j) {
                const int c0 = j * bin, c1 = std::min(det_cols, c0 + bin);
                out[static_cast<std::size_t>(i) * cols + j] = static_cast<float>(s[j] / ((r1 - r0) * (c1 - c0)));
            }
        }
    }
};

ScintillatorGlare::ScintillatorGlare(int det_rows, int det_cols, const SldParams& p) : impl_(std::make_unique<Impl>()) {
    if (det_rows < 1 || det_cols < 1) throw Error("sld: detector dimensions must be positive");
    if (p.bin < 1 || p.bin > 64) throw Error("sld: bin must be 1..64");
    auto& I = *impl_;
    I.det_rows = det_rows;
    I.det_cols = det_cols;
    I.bin = p.bin;
    I.rows = (det_rows + p.bin - 1) / p.bin;
    I.cols = (det_cols + p.bin - 1) / p.bin;
    if (p.bm18) {
        I.configure_bm18(p);
        return;
    }
    if (!p.fractions.empty() || !p.lengths.empty() || p.border_x || p.border_y)
        throw Error("sld: multiple tails and illuminated borders require the BM18 profile");
    if (!(p.fraction > 0 && p.fraction < 1)) throw Error("sld: fraction must be in (0, 1)");
    if (!(p.length >= 1) || !std::isfinite(p.length)) throw Error("sld: length must be finite and >= 1 pixel");
    // zero padding: three damping lengths on every side (the tail there is < 2 % of its value at one length)
    const int b = p.bin, pad = static_cast<int>(std::ceil(3.0 * p.length / b));
    FourierFilterSpec sp;
    sp.ndim = 2;
    sp.h = I.rows;
    sp.w = I.cols;
    sp.ph = roundfft(I.rows + 2 * pad);
    sp.pw = roundfft(I.cols + 2 * pad);
    sp.pad_top = (sp.ph - I.rows) / 2;
    sp.pad_left = (sp.pw - I.cols) / 2;
    sp.mode = PadMode::Zero;
    // the tail between binned images: the detector-pixel tail t(r) = exp(-r / l) / r (0 at r = 0) coupled between
    // two bin x bin cells, i.e. correlated with the triangle (b - |u|)(b - |v|) / b^2; then unit sum
    const int ph = sp.ph, pw = sp.pw;
    std::vector<double> t(static_cast<std::size_t>(ph) * pw);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ph; ++i) {
        const int dy = i <= ph / 2 ? i : i - ph;
        for (int j = 0; j < pw; ++j) {
            const int dx = j <= pw / 2 ? j : j - pw;
            double s = 0;
            for (int u = -(b - 1); u < b; ++u)
                for (int v = -(b - 1); v < b; ++v) {
                    const double r = std::hypot(static_cast<double>(dy) * b + u, static_cast<double>(dx) * b + v);
                    if (r > 0) s += (b - std::abs(u)) * (b - std::abs(v)) * std::exp(-r / p.length) / r;
                }
            t[static_cast<std::size_t>(i) * pw + j] = s;
        }
    }
    double sum = 0;
    for (double v : t) sum += v;
    for (double& v : t) v /= sum;
    const auto T = real_fft(t, ph, pw);
    sp.filter.resize(T.size());
    for (std::size_t k = 0; k < T.size(); ++k) {   // t is even: its transform is real
        const double th = T[k].real();
        sp.filter[k] = static_cast<float>(p.fraction * th / (1.0 + p.fraction * th));
    }
    I.filter = cpu::make_fourier_filter(sp);
}

ScintillatorGlare::~ScintillatorGlare() = default;
int ScintillatorGlare::rows() const { return impl_->rows; }
int ScintillatorGlare::cols() const { return impl_->cols; }
int ScintillatorGlare::bin() const { return impl_->bin; }

void ScintillatorGlare::bin_raw(const uint16_t* data, int r0, int nr, const float* dark, float div, float* out) const {
    const auto& I = *impl_;
    if (r0 % I.bin || r0 < 0 || r0 + nr > I.det_rows || (nr % I.bin && r0 + nr != I.det_rows))
        throw Error("sld: bin_raw takes whole rows of cells");
    I.bin_rows([&](int r, float* v) {
        const uint16_t* f = data + static_cast<std::size_t>(r - r0) * I.det_cols;
        const float* d = dark + static_cast<std::size_t>(r) * I.det_cols;
        for (int c = 0; c < I.det_cols; ++c) v[c] = (static_cast<float>(f[c]) - d[c]) / div;
    }, r0, r0 + nr, out);
}

void ScintillatorGlare::filter(float* maps, int n) {
    auto& I = *impl_;
    if (!I.bm18) {
        I.filter->apply(maps, maps, n);
        return;
    }
    const int h = I.rows + 2 * I.border_y, w = I.cols + 2 * I.border_x;
    std::vector<float> extended(static_cast<std::size_t>(h) * w);
    for (int f = 0; f < n; ++f) {
        float* frame = maps + static_cast<std::size_t>(f) * I.rows * I.cols;
        for (int r = 0; r < h; ++r) {
            const float* row = frame + static_cast<std::size_t>(std::clamp(r - I.border_y, 0, I.rows - 1)) * I.cols;
            for (int c = 0; c < w; ++c) extended[static_cast<std::size_t>(r) * w + c] = row[std::clamp(c - I.border_x, 0, I.cols - 1)];
        }
        I.filter->apply(extended.data(), extended.data(), 1);
        for (int r = 0; r < I.rows; ++r)
            std::copy_n(extended.data() + static_cast<std::size_t>(r + I.border_y) * w + I.border_x,
                        I.cols, frame + static_cast<std::size_t>(r) * I.cols);
    }
}

void ScintillatorGlare::from_frames(const float* frames, int n, float* out) {
    const auto& I = *impl_;
    const std::size_t plane = static_cast<std::size_t>(I.det_rows) * I.det_cols, cell = static_cast<std::size_t>(I.rows) * I.cols;
#pragma omp parallel for schedule(static)
    for (int f = 0; f < n; ++f) {
        const float* fr = frames + f * plane;
        I.bin_rows([&](int r, float* v) { std::copy(fr + static_cast<std::size_t>(r) * I.det_cols, fr + static_cast<std::size_t>(r + 1) * I.det_cols, v); },
                   0, I.det_rows, out + f * cell);
    }
    filter(out, n);
}

void ScintillatorGlare::correct(float* frames, int n) {
    auto& I = *impl_;
    const std::size_t plane = static_cast<std::size_t>(I.det_rows) * I.det_cols, cell = static_cast<std::size_t>(I.rows) * I.cols;
    const float inv_bin = 1.f / static_cast<float>(I.bin);
    for (int f = 0; f < n; ++f) {   // one frame at a time: bounded work memory
        I.maps.resize(cell);
        from_frames(frames + f * plane, 1, I.maps.data());
        float* fr = frames + f * plane;
#pragma omp parallel for schedule(static)
        for (int r = 0; r < I.det_rows; ++r)
            for (int c = 0; c < I.det_cols; ++c)
            {
                float& value = fr[static_cast<std::size_t>(r) * I.det_cols + c];
                value -= sld_core::glare_sample(I.maps.data(), I.rows, I.cols, inv_bin, r, c);
                if (I.bm18) value = std::max(value, 0.f);  // full deconvolution of flats, as in Nabuxx
            }
    }
}

}  // namespace helirec
