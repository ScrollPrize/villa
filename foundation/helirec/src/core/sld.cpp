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

void ScintillatorGlare::filter(float* maps, int n) { impl_->filter->apply(maps, maps, n); }

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
                fr[static_cast<std::size_t>(r) * I.det_cols + c] -= sld_core::glare_sample(I.maps.data(), I.rows, I.cols, inv_bin, r, c);
    }
}

}  // namespace helirec
