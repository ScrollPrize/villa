#include "helirec/sld.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <mutex>
#include <vector>

#include "common/sld_core.hpp"
#include "helirec/filters.hpp"
#include "helirec/fourier.hpp"

namespace helirec {

struct ScintillatorGlare::Impl {
    int det_rows = 0, det_cols = 0, rows = 0, cols = 0, bin = 1;
    std::unique_ptr<FourierFilter> filter;   // S on the zero-padded map grid
    std::vector<float> maps;                 // work: glare maps of correct()

    // mean of each bin x bin cell (edge cells: of the pixels they hold)
    template <class Pixel>
    void bin_frame(Pixel&& pixel, float* out) const {
        for (int i = 0; i < rows; ++i) {
            const int r0 = i * bin, r1 = std::min(det_rows, r0 + bin);
            for (int j = 0; j < cols; ++j) {
                const int c0 = j * bin, c1 = std::min(det_cols, c0 + bin);
                double s = 0;
                for (int r = r0; r < r1; ++r)
                    for (int c = c0; c < c1; ++c) s += pixel(static_cast<std::size_t>(r) * det_cols + c);
                out[static_cast<std::size_t>(i) * cols + j] = static_cast<float>(s / ((r1 - r0) * (c1 - c0)));
            }
        }
    }
};

ScintillatorGlare::ScintillatorGlare(int det_rows, int det_cols, const SldParams& p) : impl_(std::make_unique<Impl>()) {
    if (!(p.fraction > 0 && p.fraction < 1)) throw Error("sld: fraction must be in (0, 1)");
    if (!(p.length >= 1)) throw Error("sld: length must be >= 1 pixel");
    if (p.bin < 1 || p.bin > 64) throw Error("sld: bin must be 1..64");
    auto& I = *impl_;
    I.det_rows = det_rows;
    I.det_cols = det_cols;
    I.bin = p.bin;
    I.rows = (det_rows + p.bin - 1) / p.bin;
    I.cols = (det_cols + p.bin - 1) / p.bin;
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
    const int ph = sp.ph, pw = sp.pw, nc = pw / 2 + 1;
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
    std::vector<std::complex<double>> T(static_cast<std::size_t>(ph) * nc);
    fftw_plan plan;
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        plan = fftw_plan_dft_r2c_2d(ph, pw, t.data(), reinterpret_cast<fftw_complex*>(T.data()), FFTW_ESTIMATE);
    }
    fftw_execute(plan);
    {
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        fftw_destroy_plan(plan);
    }
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

void ScintillatorGlare::from_raw(const uint16_t* raw, const float* dark, const float* div, int n, float* out) {
    const auto& I = *impl_;
    const std::size_t plane = static_cast<std::size_t>(I.det_rows) * I.det_cols, cell = static_cast<std::size_t>(I.rows) * I.cols;
#pragma omp parallel for schedule(static)
    for (int f = 0; f < n; ++f) {
        const uint16_t* fr = raw + f * plane;
        const float d = div[f];
        I.bin_frame([&](std::size_t k) { return (static_cast<float>(fr[k]) - dark[k]) / d; }, out + f * cell);
    }
    I.filter->apply(out, out, n);
}

void ScintillatorGlare::from_frames(const float* frames, int n, float* out) {
    const auto& I = *impl_;
    const std::size_t plane = static_cast<std::size_t>(I.det_rows) * I.det_cols, cell = static_cast<std::size_t>(I.rows) * I.cols;
#pragma omp parallel for schedule(static)
    for (int f = 0; f < n; ++f) {
        const float* fr = frames + f * plane;
        I.bin_frame([&](std::size_t k) { return fr[k]; }, out + f * cell);
    }
    I.filter->apply(out, out, n);
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
                fr[static_cast<std::size_t>(r) * I.det_cols + c] -= sld_core::glare_sample(I.maps.data(), I.rows, I.cols, inv_bin, r, c);
    }
}

}  // namespace helirec
