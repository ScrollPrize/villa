// CPU FourierFilter on FFTW (single precision). Frames (2D) or rows (1D) are
// distributed over OpenMP threads; each thread owns aligned work buffers and
// all threads share one plan (fftwf_execute_dft_* is thread safe).
#include <fftw3.h>
#include <omp.h>

#include <mutex>
#include <algorithm>
#include <vector>

#include "helirec/fourier.hpp"

namespace helirec {
std::mutex& fftw_planner_mutex() {
    static std::mutex m;
    return m;
}
}  // namespace helirec

namespace helirec::cpu {

namespace {

std::mutex& planner_mutex() { return fftw_planner_mutex(); }

struct FftwBuf {
    float* real = nullptr;
    fftwf_complex* cplx = nullptr;
    FftwBuf(std::size_t nr, std::size_t nc)
        : real(fftwf_alloc_real(nr)), cplx(fftwf_alloc_complex(nc)) {
        if (!real || !cplx) throw Error("fftw allocation failed");
    }
    ~FftwBuf() { fftwf_free(real); fftwf_free(cplx); }
    FftwBuf(const FftwBuf&) = delete;
    FftwBuf& operator=(const FftwBuf&) = delete;
};

class FftwFourierFilter final : public FourierFilter {
  public:
    explicit FftwFourierFilter(const FourierFilterSpec& s) : FourierFilter(s) {
        validate(spec_);
        const auto& p = spec_;
        rows_ = p.ndim == 2 ? p.ph : 1;
        nc_ = static_cast<std::size_t>(p.pw / 2 + 1);
        FftwBuf tmp(static_cast<std::size_t>(rows_) * p.pw, rows_ * nc_);
        std::lock_guard<std::mutex> lock(planner_mutex());
        if (p.ndim == 2) {
            fwd_ = fftwf_plan_dft_r2c_2d(p.ph, p.pw, tmp.real, tmp.cplx, FFTW_ESTIMATE);
            inv_ = fftwf_plan_dft_c2r_2d(p.ph, p.pw, tmp.cplx, tmp.real, FFTW_ESTIMATE);
        } else {
            fwd_ = fftwf_plan_dft_r2c_1d(p.pw, tmp.real, tmp.cplx, FFTW_ESTIMATE);
            inv_ = fftwf_plan_dft_c2r_1d(p.pw, tmp.cplx, tmp.real, FFTW_ESTIMATE);
        }
        if (!fwd_ || !inv_) throw Error("fftw planning failed");
        norm_ = 1.f / static_cast<float>(static_cast<std::size_t>(rows_) * p.pw);
    }
    ~FftwFourierFilter() override {
        std::lock_guard<std::mutex> lock(planner_mutex());
        fftwf_destroy_plan(fwd_);
        fftwf_destroy_plan(inv_);
    }

    void apply(const float* in, float* out, int n_frames, void*) override {
        const auto& p = spec_;
        // unit of work: a whole frame (2D) or one row (1D)
        const std::ptrdiff_t n_items = p.ndim == 2 ? n_frames : static_cast<std::ptrdiff_t>(n_frames) * p.h;
        if (n_items == 0) return;
        // A full BM18 detector FFT can require hundreds of MB per worker.
        // Do not allocate idle worker buffers when filtering only one frame.
        const int workers = static_cast<int>(std::min<std::ptrdiff_t>(omp_get_max_threads(), n_items));
#pragma omp parallel num_threads(workers)
        {
            FftwBuf buf(static_cast<std::size_t>(rows_) * p.pw, rows_ * nc_);
#pragma omp for schedule(dynamic)
            for (std::ptrdiff_t item = 0; item < n_items; ++item) {
                const float* src = in + item * static_cast<std::ptrdiff_t>(p.ndim == 2 ? p.h * p.w : p.w);
                float* dst = out + item * static_cast<std::ptrdiff_t>(p.ndim == 2 ? p.h * p.w : p.w);
                // pad
                for (int r = 0; r < rows_; ++r) {
                    const int sr = p.ndim == 2 ? pad_source_index(r - p.pad_top, p.h, p.mode) : 0;
                    float* prow = buf.real + static_cast<std::size_t>(r) * p.pw;
                    if (sr < 0) {
                        std::fill(prow, prow + p.pw, 0.f);
                        continue;
                    }
                    const float* srow = src + static_cast<std::size_t>(sr) * p.w;
                    for (int c = 0; c < p.pw; ++c) {
                        const int sc = pad_source_index(c - p.pad_left, p.w, p.mode);
                        prow[c] = sc < 0 ? 0.f : srow[sc];
                    }
                }
                fftwf_execute_dft_r2c(fwd_, buf.real, buf.cplx);
                const std::size_t nf = rows_ * nc_;
                if (p.filter_imag.empty()) {
                    for (std::size_t k = 0; k < nf; ++k) {
                        const float f = p.filter[k] * norm_;
                        buf.cplx[k][0] *= f;
                        buf.cplx[k][1] *= f;
                    }
                } else {
                    for (std::size_t k = 0; k < nf; ++k) {
                        const float fr = p.filter[k] * norm_, fi = p.filter_imag[k] * norm_;
                        const float re = buf.cplx[k][0], im = buf.cplx[k][1];
                        buf.cplx[k][0] = re * fr - im * fi;
                        buf.cplx[k][1] = re * fi + im * fr;
                    }
                }
                fftwf_execute_dft_c2r(inv_, buf.cplx, buf.real);
                // crop
                const int out_rows = p.ndim == 2 ? p.h : 1;
                for (int r = 0; r < out_rows; ++r) {
                    const float* prow = buf.real + static_cast<std::size_t>(r + (p.ndim == 2 ? p.pad_top : 0)) * p.pw + p.pad_left;
                    std::copy(prow, prow + p.w, dst + static_cast<std::size_t>(r) * p.w);
                }
            }
        }
    }

  private:
    fftwf_plan fwd_ = nullptr, inv_ = nullptr;
    int rows_ = 1;
    std::size_t nc_ = 0;
    float norm_ = 1.f;
};

}  // namespace

std::unique_ptr<FourierFilter> make_fourier_filter(const FourierFilterSpec& spec) {
    return std::make_unique<FftwFourierFilter>(spec);
}

}  // namespace helirec::cpu
