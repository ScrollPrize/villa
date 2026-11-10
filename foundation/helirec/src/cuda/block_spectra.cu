// CUDA block cross-spectra (cuFFT 3D transforms) and affine resampling.
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/block_spectra_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/block_spectra.hpp"

namespace helirec::cuda {

namespace core = block_spectra_core;

namespace {

void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error ") + std::to_string(r) + " in " + what);
}

// per (z, y) row of a block of edge e: sums over x of w v and w v^2 (doubles; the host adds the rows in order)
__global__ void row_sums_kernel(const float* __restrict__ v, const float* __restrict__ w, int e, double* __restrict__ out) {
    const size_t rows = static_cast<size_t>(e) * e;
    for (size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; r < rows;
         r += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float* row = v + r * e;
        double s1 = 0, s2 = 0;
        for (int x = 0; x < e; ++x) {
            const double t = row[x];
            s1 += w[x] * t;
            s2 += w[x] * t * t;
        }
        out[2 * r] = s1;
        out[2 * r + 1] = s2;
    }
}

__global__ void window_kernel(const float* __restrict__ v, const float* __restrict__ w, int e, float mean,
                              float* __restrict__ out) {
    const size_t total = static_cast<size_t>(e) * e * e;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(k % e), y = static_cast<int>((k / e) % e), z = static_cast<int>(k / (static_cast<size_t>(e) * e));
        out[k] = (v[k] - mean) * (w[z] * w[y] * w[x]);
    }
}

// the frequencies of a's grid out of the (m, m, m / 2 + 1) transform, in a's layout (n, n, n / 2 + 1), scaled
__global__ void crop_kernel(const cufftComplex* __restrict__ src, cufftComplex* __restrict__ dst, int n, int m, float scale) {
    const int nc = n / 2 + 1, mc = m / 2 + 1;
    const size_t total = static_cast<size_t>(n) * n * nc;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int c = static_cast<int>(k % nc), iy = static_cast<int>((k / nc) % n), iz = static_cast<int>(k / (static_cast<size_t>(nc) * n));
        const int kz = core::signed_index(iz, n), ky = core::signed_index(iy, n);
        const int jz = kz < 0 ? kz + m : kz, jy = ky < 0 ? ky + m : ky;
        const cufftComplex s = src[(static_cast<size_t>(jz) * m + jy) * mc + c];
        dst[k] = make_cuFloatComplex(s.x * scale, s.y * scale);
    }
}

// A conj B on the box |kz|, |ky| <= kb, 0 <= kx <= kb, normalised as the tables
__global__ void box_kernel(const cufftComplex* __restrict__ A, const cufftComplex* __restrict__ B, int n, int kb,
                           double norm, float* __restrict__ out) {
    const int side = 2 * kb + 1, nx = kb + 1, nc = n / 2 + 1;
    const size_t total = static_cast<size_t>(side) * side * nx;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int c = static_cast<int>(k % nx), v = static_cast<int>((k / nx) % side), u = static_cast<int>(k / (static_cast<size_t>(nx) * side));
        const int iz = (u - kb + n) % n, iy = (v - kb + n) % n;
        const size_t i = (static_cast<size_t>(iz) * n + iy) * nc + c;
        const cufftComplex a = A[i], b = B[i];
        out[2 * k] = static_cast<float>(norm * (static_cast<double>(a.x) * b.x + static_cast<double>(a.y) * b.y));
        out[2 * k + 1] = static_cast<float>(norm * (static_cast<double>(a.y) * b.x - static_cast<double>(a.x) * b.y));
    }
}

// Thread t sums the frequencies [t, t + 1) * per of the half spectrum into its own table part[t] (n_bins, 3): a fixed
// partition, added in a fixed order by reduce_kernel, so the sums do not depend on scheduling.
__global__ void table_kernel(const cufftComplex* __restrict__ A, const cufftComplex* __restrict__ B, int n, size_t per,
                             int n_parts, int n_bins, int thetas, double shell_scale, const double* __restrict__ tan2,
                             double dz, double dy, double dx, double* __restrict__ part) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_parts) return;
    const int nc = n / 2 + 1;
    const size_t total = static_cast<size_t>(n) * n * nc;
    double* p = part + static_cast<size_t>(t) * n_bins * 3;
    const bool shifted = dz != 0 || dy != 0 || dx != 0;
    const double w0 = 2 * M_PI / n;
    const size_t end = min(total, (static_cast<size_t>(t) + 1) * per);
    for (size_t k = static_cast<size_t>(t) * per; k < end; ++k) {
        const int c = static_cast<int>(k % nc), iy = static_cast<int>((k / nc) % n), iz = static_cast<int>(k / (static_cast<size_t>(nc) * n));
        const int kz = core::signed_index(iz, n), ky = core::signed_index(iy, n);
        const double w = core::half_weight(kz, ky, c, n);
        if (w == 0) continue;
        const double ar = A[k].x, ai = A[k].y, br = B[k].x, bi = B[k].y;
        double re = ar * br + ai * bi;
        if (shifted) {
            const double im = ai * br - ar * bi, phi = -w0 * (kz * dz + ky * dy + c * dx);
            re = re * cos(phi) - im * sin(phi);
        }
        double* q = p + static_cast<size_t>(core::bin_of(kz, ky, c, shell_scale, tan2, thetas)) * 3;
        q[0] += w * (ar * ar + ai * ai);
        q[1] += w * (br * br + bi * bi);
        q[2] += w * re;
    }
}

__global__ void reduce_kernel(const double* __restrict__ part, int n_parts, int n_values, double* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) return;
    double s = 0;
    for (int t = 0; t < n_parts; ++t) s += part[static_cast<size_t>(t) * n_values + i];
    out[i] = s;
}

__global__ void resample_kernel(const float* __restrict__ src, int sz, int sy, int sx, const double* __restrict__ m,
                                float* __restrict__ out, int nz, int ny, int nx, int interp) {
    const size_t total = static_cast<size_t>(nz) * ny * nx;
    for (size_t q = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; q < total;
         q += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(q % nx), j = static_cast<int>((q / nx) % ny), k = static_cast<int>(q / (static_cast<size_t>(nx) * ny));
        // (as the CPU: the row's base, then the step along x)
        const double bz = m[0] * k + m[1] * j + m[3], by = m[4] * k + m[5] * j + m[7], bx = m[8] * k + m[9] * j + m[11];
        out[q] = core::sample_volume(src, sz, sy, sx, bz + m[2] * i, by + m[6] * i, bx + m[10] * i, interp);
    }
}

unsigned grid_for(size_t n) { return static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535u * 16u)); }

class CufftBlockCorrelator final : public BlockCorrelator {
  public:
    explicit CufftBlockCorrelator(const BlockSpectraSpec& s) : BlockCorrelator(s) {
        n_ = spec_.n;
        m_ = spec_.n * spec_.oversample;
        nc_ = n_ / 2 + 1;
        wa_ = block_spectra::hann(n_, 1);
        wb_ = block_spectra::hann(n_, spec_.oversample);
        const std::vector<double> tan2 = block_spectra::theta_tan2(spec_);
        double s2 = 0;
        for (float w : wa_) s2 += static_cast<double>(w) * w;
        norm_ = 1.0 / (std::pow(static_cast<double>(n_), 3) * s2 * s2 * s2);
        d_wa_.upload(wa_.data(), wa_.size());
        d_wb_.upload(wb_.data(), wb_.size());
        d_tan2_.upload(tan2.data(), tan2.size());
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
        const size_t na = static_cast<size_t>(n_) * n_ * nc_;
        d_real_.resize(static_cast<size_t>(m_) * m_ * m_);
        d_A_.resize(na);
        d_B_.resize(na);
        if (m_ != n_) d_Bfull_.resize(static_cast<size_t>(m_) * m_ * (m_ / 2 + 1));
        d_rows_.resize(static_cast<size_t>(m_) * m_ * 2);
        rows_.resize(d_rows_.size());
        parts_ = static_cast<int>(std::min<size_t>(2048, na));
        per_ = (na + parts_ - 1) / parts_;
        d_part_.resize(static_cast<size_t>(parts_) * spec_.n_bins() * 3);
        d_table_.resize(static_cast<size_t>(spec_.n_bins()) * 3);
        table_.resize(d_table_.size());
        cufft_check(cufftPlan3d(&fwd_a_, n_, n_, n_, CUFFT_R2C), "plan R2C");
        if (m_ != n_) cufft_check(cufftPlan3d(&fwd_b_, m_, m_, m_, CUFFT_R2C), "plan R2C");
    }
    ~CufftBlockCorrelator() override {
        if (fwd_a_) cufftDestroy(fwd_a_);
        if (fwd_b_) cufftDestroy(fwd_b_);
    }

    void compute(const float* a, const float* b, BlockSpectra& out, const double* shift) override {
        window(a, n_, d_wa_, wa_, out.mean_a, out.var_a);
        cufft_check(cufftExecR2C(fwd_a_, d_real_.data(), d_A_.data()), "R2C");
        window(b, m_, d_wb_, wb_, out.mean_b, out.var_b);
        if (m_ == n_) {
            cufft_check(cufftExecR2C(fwd_a_, d_real_.data(), d_B_.data()), "R2C");
        } else {
            cufft_check(cufftExecR2C(fwd_b_, d_real_.data(), d_Bfull_.data()), "R2C");
            const int f = spec_.oversample;
            crop_kernel<<<grid_for(d_B_.size()), 256>>>(d_Bfull_.data(), d_B_.data(), n_, m_, 1.f / static_cast<float>(f * f * f));
            HELIREC_CUDA_CHECK(cudaGetLastError());
        }
        out.shift = {0, 0, 0};
        out.peak = 0;
        if (shift) out.shift = {shift[0], shift[1], shift[2]};
        table(out);
        if (spec_.align && !shift) {
            const int kb = block_spectra::align_box(spec_), side = 2 * kb + 1, nx = kb + 1;
            const size_t nbox = static_cast<size_t>(side) * side * nx;
            d_box_.resize(nbox * 2);
            box_.resize(nbox * 2);
            box_kernel<<<grid_for(nbox), 256>>>(d_A_.data(), d_B_.data(), n_, kb, norm_, d_box_.data());
            HELIREC_CUDA_CHECK(cudaGetLastError());
            d_box_.download(box_.data());
            block_spectra::align(spec_, box_.data(), out.aa, out.bb, out.shift, out.peak);
            table(out);
        }
    }

  private:
    void window(const float* v, int e, const DeviceBuffer<float>& d_w, const std::vector<float>& w, double& mean,
                double& var) {
        const size_t rows = static_cast<size_t>(e) * e;
        row_sums_kernel<<<grid_for(rows), 256>>>(v, d_w.data(), e, d_rows_.data());
        HELIREC_CUDA_CHECK(cudaGetLastError());
        HELIREC_CUDA_CHECK(cudaMemcpy(rows_.data(), d_rows_.data(), rows * 2 * sizeof(double), cudaMemcpyDeviceToHost));
        double t0 = 0, t1 = 0, t2 = 0, ws = 0;
        for (int x = 0; x < e; ++x) ws += w[x];
        for (int z = 0; z < e; ++z) {   // as the CPU: rows of a plane, then the planes, in order
            double a1 = 0, a2 = 0;
            for (int y = 0; y < e; ++y) {
                a1 += w[y] * rows_[2 * (static_cast<size_t>(z) * e + y)];
                a2 += w[y] * rows_[2 * (static_cast<size_t>(z) * e + y) + 1];
            }
            t1 += w[z] * a1;
            t2 += w[z] * a2;
        }
        t0 = ws * ws * ws;
        mean = t1 / t0;
        var = t2 / t0 - mean * mean;
        window_kernel<<<grid_for(rows * e), 256>>>(v, d_w.data(), e, static_cast<float>(mean), d_real_.data());
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }

    void table(BlockSpectra& out) {
        const int nb = spec_.n_bins();
        HELIREC_CUDA_CHECK(cudaMemset(d_part_.data(), 0, d_part_.size() * sizeof(double)));
        table_kernel<<<grid_1d(parts_, 64), 64>>>(d_A_.data(), d_B_.data(), n_, per_, parts_, nb, spec_.thetas,
                                                   2.0 * spec_.shells / n_, d_tan2_.data(), out.shift[0], out.shift[1],
                                                   out.shift[2], d_part_.data());
        HELIREC_CUDA_CHECK(cudaGetLastError());
        reduce_kernel<<<grid_1d(nb * 3, 64), 64>>>(d_part_.data(), parts_, nb * 3, d_table_.data());
        HELIREC_CUDA_CHECK(cudaGetLastError());
        d_table_.download(table_.data());
        out.aa.resize(nb);
        out.bb.resize(nb);
        out.ab.resize(nb);
        for (int i = 0; i < nb; ++i) {
            out.aa[i] = table_[3 * i] * norm_;
            out.bb[i] = table_[3 * i + 1] * norm_;
            out.ab[i] = table_[3 * i + 2] * norm_;
        }
    }

    int n_ = 0, m_ = 0, parts_ = 0;
    size_t nc_ = 0, per_ = 0;
    double norm_ = 1;
    std::vector<float> wa_, wb_, box_;
    std::vector<double> rows_, table_;
    DeviceBuffer<float> d_wa_, d_wb_, d_real_, d_box_;
    DeviceBuffer<double> d_tan2_, d_rows_, d_part_, d_table_;
    DeviceBuffer<cufftComplex> d_A_, d_B_, d_Bfull_;
    cufftHandle fwd_a_ = 0, fwd_b_ = 0;
};

}  // namespace

std::unique_ptr<BlockCorrelator> make_block_correlator(const BlockSpectraSpec& spec) {
    return std::make_unique<CufftBlockCorrelator>(spec);
}

void resample_affine(const float* src, int sz, int sy, int sx, const double* m, float* out, int nz, int ny, int nx,
                     int interp, void* stream) {
    cudaStream_t s = as_stream(stream);
    DeviceBuffer<double> d_m;
    d_m.upload(m, 12, s);
    const size_t total = static_cast<size_t>(nz) * ny * nx;
    resample_kernel<<<grid_for(total), 256, 0, s>>>(src, sz, sy, sx, d_m.data(), out, nz, ny, nx, interp);
    HELIREC_CUDA_CHECK(cudaGetLastError());
    HELIREC_CUDA_CHECK(cudaStreamSynchronize(s));
}

void resample_affine_host(const float* src, int sz, int sy, int sx, const double* m, float* out, int nz, int ny, int nx,
                          int interp) {
    DeviceBuffer<float> d_src;
    d_src.upload(src, static_cast<size_t>(sz) * sy * sx);
    resample_affine(d_src.data(), sz, sy, sx, m, out, nz, ny, nx, interp, nullptr);
}

}  // namespace helirec::cuda
