// CUDA FourierFilter on cuFFT: pad kernel -> batched R2C -> multiply -> C2R -> crop kernel.
// Frames are processed in batches to bound the padded work buffer.
#include <cufft.h>

#include <algorithm>

#include "cuda_utils.cuh"
#include "helirec/fourier.hpp"

namespace helirec::cuda {

namespace {

void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error ") + std::to_string(r) + " in " + what);
}

// one thread per padded sample; `rows` = padded rows per item (ph, or 1 for 1D rows)
__global__ void pad_kernel(const float* __restrict__ in, float* __restrict__ out, int n_items, int rows,
                           int pw, int h, int w, int pad_top, int pad_left, PadMode mode, bool two_d) {
    const size_t total = static_cast<size_t>(n_items) * rows * pw;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int c = static_cast<int>(k % pw);
        const int r = static_cast<int>((k / pw) % rows);
        const size_t item = k / (static_cast<size_t>(pw) * rows);
        const int sr = two_d ? pad_source_index(r - pad_top, h, mode) : 0;
        const int sc = pad_source_index(c - pad_left, w, mode);
        const float* src = in + item * static_cast<size_t>(two_d ? h * w : w);
        out[k] = (sr < 0 || sc < 0) ? 0.f : src[static_cast<size_t>(sr) * w + sc];
    }
}

__global__ void multiply_kernel(cufftComplex* __restrict__ data, const float* __restrict__ filter,
                                const float* __restrict__ filter_imag, size_t per_item, size_t total, float norm) {
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float fr = filter[k % per_item] * norm;
        if (!filter_imag) {
            data[k].x *= fr;
            data[k].y *= fr;
        } else {
            const float fi = filter_imag[k % per_item] * norm;
            const float re = data[k].x, im = data[k].y;
            data[k].x = __fsub_rn(__fmul_rn(re, fr), __fmul_rn(im, fi));
            data[k].y = __fadd_rn(__fmul_rn(re, fi), __fmul_rn(im, fr));
        }
    }
}

__global__ void multiply_copy_kernel(const cufftComplex* __restrict__ src, cufftComplex* __restrict__ dst,
                                     const float* __restrict__ filter, const float* __restrict__ filter_imag,
                                     size_t per_item, size_t total, float norm) {
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float fr = filter[k % per_item] * norm;
        const float re = src[k].x, im = src[k].y;
        if (!filter_imag) {
            dst[k] = make_cuFloatComplex(re * fr, im * fr);
        } else {
            const float fi = filter_imag[k % per_item] * norm;
            dst[k] = make_cuFloatComplex(__fsub_rn(__fmul_rn(re, fr), __fmul_rn(im, fi)),
                                         __fadd_rn(__fmul_rn(re, fi), __fmul_rn(im, fr)));
        }
    }
}

__global__ void crop_kernel(const float* __restrict__ in, float* __restrict__ out, int n_items, int rows,
                            int pw, int out_rows, int w, int pad_top, int pad_left) {
    const size_t total = static_cast<size_t>(n_items) * out_rows * w;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int c = static_cast<int>(k % w);
        const int r = static_cast<int>((k / w) % out_rows);
        const size_t item = k / (static_cast<size_t>(w) * out_rows);
        out[k] = in[(item * rows + r + pad_top) * pw + c + pad_left];
    }
}

unsigned grid_for(size_t n) { return static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535u * 16u)); }

class CufftFourierFilter final : public FourierFilter {
  public:
    explicit CufftFourierFilter(const FourierFilterSpec& s) : FourierFilter(s) {
        validate(spec_);
        rows_ = spec_.ndim == 2 ? spec_.ph : 1;
        nc_ = static_cast<size_t>(spec_.pw / 2 + 1);
        d_filter_.upload(spec_.filter.data(), spec_.filter.size());
        if (!spec_.filter_imag.empty()) d_filter_imag_.upload(spec_.filter_imag.data(), spec_.filter_imag.size());
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
        // batch: ~256 MB of padded real data per pass
        const size_t item_bytes = static_cast<size_t>(rows_) * spec_.pw * sizeof(float);
        batch_ = static_cast<int>(std::max<size_t>(1, (256u << 20) / item_bytes));
        norm_ = 1.f / static_cast<float>(static_cast<size_t>(rows_) * spec_.pw);
    }
    ~CufftFourierFilter() override { destroy_plans(); }

  protected:
    void on_filter_changed() override {
        d_filter_.upload(spec_.filter.data(), spec_.filter.size());
        if (!spec_.filter_imag.empty()) d_filter_imag_.upload(spec_.filter_imag.data(), spec_.filter_imag.size());
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
    }

  public:

    void apply(const float* in, float* out, int n_frames, void* stream) override {
        const auto& p = spec_;
        cudaStream_t s = as_stream(stream);
        const int items_total = p.ndim == 2 ? n_frames : n_frames * p.h;
        const size_t in_item = static_cast<size_t>(p.ndim == 2 ? p.h * p.w : p.w);
        for (int i0 = 0; i0 < items_total; i0 += batch_) {
            const int n = std::min(batch_, items_total - i0);
            ensure_plans(n, s);
            const size_t n_real = static_cast<size_t>(n) * rows_ * p.pw;
            const size_t n_cplx = static_cast<size_t>(n) * rows_ * nc_;
            d_real_.resize(std::max(d_real_.size(), n_real));
            d_cplx_.resize(std::max(d_cplx_.size(), n_cplx));
            pad_kernel<<<grid_for(n_real), 256, 0, s>>>(in + i0 * in_item, d_real_.data(), n, rows_, p.pw,
                                                        p.h, p.w, p.pad_top, p.pad_left, p.mode, p.ndim == 2);
            HELIREC_CUDA_CHECK(cudaGetLastError());
            cufft_check(cufftExecR2C(fwd_, d_real_.data(), d_cplx_.data()), "R2C");
            multiply_kernel<<<grid_for(n_cplx), 256, 0, s>>>(d_cplx_.data(), d_filter_.data(),
                                                             spec_.filter_imag.empty() ? nullptr : d_filter_imag_.data(),
                                                             rows_ * nc_, n_cplx, norm_);
            HELIREC_CUDA_CHECK(cudaGetLastError());
            cufft_check(cufftExecC2R(inv_, d_cplx_.data(), d_real_.data()), "C2R");
            const int out_rows = p.ndim == 2 ? p.h : 1;
            const size_t n_out = static_cast<size_t>(n) * out_rows * p.w;
            crop_kernel<<<grid_for(n_out), 256, 0, s>>>(d_real_.data(), out + i0 * in_item, n, rows_, p.pw,
                                                         out_rows, p.w, p.ndim == 2 ? p.pad_top : 0, p.pad_left);
            HELIREC_CUDA_CHECK(cudaGetLastError());
        }
        HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
    }

    std::size_t spectrum_floats() const override {
        if (spec_.ndim != 2) throw Error("two-stage filtering: 2D filters only");
        return static_cast<std::size_t>(rows_) * nc_ * 2;
    }
    void forward(const float* in, float* spectrum, int n_frames, void* stream) override {
        const auto& p = spec_;
        cudaStream_t s = as_stream(stream);
        const size_t in_item = static_cast<size_t>(p.h) * p.w, per = static_cast<size_t>(rows_) * nc_;
        auto* spec = reinterpret_cast<cufftComplex*>(spectrum);
        (void)spectrum_floats();
        for (int i0 = 0; i0 < n_frames; i0 += batch_) {
            const int n = std::min(batch_, n_frames - i0);
            ensure_plans(n, s);
            const size_t n_real = static_cast<size_t>(n) * rows_ * p.pw;
            d_real_.resize(std::max(d_real_.size(), n_real));
            pad_kernel<<<grid_for(n_real), 256, 0, s>>>(in + i0 * in_item, d_real_.data(), n, rows_, p.pw, p.h, p.w,
                                                        p.pad_top, p.pad_left, p.mode, true);
            HELIREC_CUDA_CHECK(cudaGetLastError());
            cufft_check(cufftExecR2C(fwd_, d_real_.data(), spec + i0 * per), "R2C");
        }
    }
    void inverse(const float* spectrum, float* out, int n_frames, void* stream) override {
        const auto& p = spec_;
        cudaStream_t s = as_stream(stream);
        const size_t out_item = static_cast<size_t>(p.h) * p.w, per = static_cast<size_t>(rows_) * nc_;
        const auto* spec = reinterpret_cast<const cufftComplex*>(spectrum);
        (void)spectrum_floats();
        for (int i0 = 0; i0 < n_frames; i0 += batch_) {
            const int n = std::min(batch_, n_frames - i0);
            ensure_plans(n, s);
            const size_t n_real = static_cast<size_t>(n) * rows_ * p.pw, n_cplx = static_cast<size_t>(n) * per;
            d_real_.resize(std::max(d_real_.size(), n_real));
            d_cplx_.resize(std::max(d_cplx_.size(), n_cplx));
            multiply_copy_kernel<<<grid_for(n_cplx), 256, 0, s>>>(
                spec + i0 * per, d_cplx_.data(), d_filter_.data(),
                spec_.filter_imag.empty() ? nullptr : d_filter_imag_.data(), per, n_cplx, norm_);
            HELIREC_CUDA_CHECK(cudaGetLastError());
            cufft_check(cufftExecC2R(inv_, d_cplx_.data(), d_real_.data()), "C2R");
            const size_t n_out = static_cast<size_t>(n) * out_item;
            crop_kernel<<<grid_for(n_out), 256, 0, s>>>(d_real_.data(), out + i0 * out_item, n, rows_, p.pw, p.h, p.w,
                                                         p.pad_top, p.pad_left);
            HELIREC_CUDA_CHECK(cudaGetLastError());
        }
    }

  private:
    // plans by batch size (full passes and the remainder pass of a frame count keep their plans)
    void ensure_plans(int n, cudaStream_t s) {
        Plans* hit = nullptr;
        for (auto& p : plans_)
            if (p.n == n) { hit = &p; break; }
        if (!hit) {
            if (plans_.size() >= 4) {
                cufftDestroy(plans_.front().fwd);
                cufftDestroy(plans_.front().inv);
                plans_.erase(plans_.begin());
            }
            Plans p;
            p.n = n;
            if (spec_.ndim == 2) {
                int dims[2] = {spec_.ph, spec_.pw};
                cufft_check(cufftPlanMany(&p.fwd, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, n), "plan R2C");
                cufft_check(cufftPlanMany(&p.inv, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, n), "plan C2R");
            } else {
                int dims[1] = {spec_.pw};
                cufft_check(cufftPlanMany(&p.fwd, 1, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, n), "plan R2C");
                cufft_check(cufftPlanMany(&p.inv, 1, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, n), "plan C2R");
            }
            plans_.push_back(p);
            hit = &plans_.back();
        }
        fwd_ = hit->fwd;
        inv_ = hit->inv;
        cufft_check(cufftSetStream(fwd_, s), "set stream");
        cufft_check(cufftSetStream(inv_, s), "set stream");
    }
    void destroy_plans() {
        for (auto& p : plans_) {
            cufftDestroy(p.fwd);
            cufftDestroy(p.inv);
        }
        plans_.clear();
    }
    struct Plans { int n = 0; cufftHandle fwd = 0, inv = 0; };
    std::vector<Plans> plans_;

    DeviceBuffer<float> d_filter_, d_filter_imag_, d_real_;
    DeviceBuffer<cufftComplex> d_cplx_;
    cufftHandle fwd_ = 0, inv_ = 0;
    int batch_ = 1, rows_ = 1;
    size_t nc_ = 0;
    float norm_ = 1.f;
};

}  // namespace

std::unique_ptr<FourierFilter> make_fourier_filter(const FourierFilterSpec& spec) {
    return std::make_unique<CufftFourierFilter>(spec);
}

}  // namespace helirec::cuda
