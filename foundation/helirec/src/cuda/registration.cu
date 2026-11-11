// The array work of helirec::calibration's registration on CUDA: cuFFT transforms and kernels for the windows,
// the displacement of the padded block and the sums over the cross spectrum. The estimator that drives it is
// the same code as on the CPU (src/calibration/registration.cpp, registration_core.hpp); every step here has
// its counterpart in CpuBackend there and follows the same arithmetic. Blocks arrive from the host.
#include <cufft.h>

#include <cmath>
#include <complex>
#include <vector>

#include "calibration/registration_core.hpp"
#include "cuda_utils.cuh"

namespace helirec::calibration {

namespace {

using helirec::cuda::DeviceBuffer;
using helirec::cuda::PinnedBuffer;

void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error ") + std::to_string(r) + " in " + what);
}

constexpr unsigned kBlock = 256;
unsigned grid_for(size_t n) { return static_cast<unsigned>(std::min<size_t>((n + kBlock - 1) / kBlock, 65535u * 16u)); }

// products without contraction into fused multiply-adds, as the CPU code is compiled
__device__ inline cufftComplex cmul(cufftComplex a, cufftComplex b) {
    return make_cuFloatComplex(__fsub_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)), __fadd_rn(__fmul_rn(a.x, b.y), __fmul_rn(a.y, b.x)));
}
__device__ inline cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b) {
    return make_cuDoubleComplex(__dsub_rn(__dmul_rn(a.x, b.x), __dmul_rn(a.y, b.y)), __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x)));
}
__device__ inline double bin_frequency(int k, int n) { return static_cast<double>(k <= n / 2 ? k : k - n) / n; }

// out[frame * n1 + row] = sum over the row's columns of w2[col] * src (w2 null: plain sum); one thread per row
__global__ void row_sums_kernel(const float* __restrict__ src, size_t frame_stride, size_t row_stride, int n0, int n1, int n2,
                                const float* __restrict__ w2, double* __restrict__ out) {
    const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= static_cast<size_t>(n0) * n1) return;
    const float* p = src + (idx / n1) * frame_stride + (idx % n1) * row_stride;
    double s = 0;
    if (w2)
        for (int k = 0; k < n2; ++k) s += static_cast<double>(w2[k]) * p[k];
    else
        for (int k = 0; k < n2; ++k) s += p[k];
    out[idx] = s;
}

// mean[frame] = sum over rows of w1[row] * rowsum / norm (w1 null: plain sum); one thread per frame
__global__ void frame_means_kernel(const double* __restrict__ rowsum, int n0, int n1, const float* __restrict__ w1, double norm,
                                   float* __restrict__ mean) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n0) return;
    double s = 0;
    for (int j = 0; j < n1; ++j) s += (w1 ? static_cast<double>(w1[j]) : 1.0) * rowsum[static_cast<size_t>(i) * n1 + j];
    mean[i] = static_cast<float>(s / norm);
}

// dst (n0, n1, n2) = (src - mean[frame]) * w0[frame] * w1[row] * w2[col]; src with the given strides
__global__ void window_kernel(const float* __restrict__ src, size_t frame_stride, size_t row_stride, float* __restrict__ dst, int n0,
                              int n1, int n2, const float* __restrict__ mean, const float* __restrict__ w0, const float* __restrict__ w1,
                              const float* __restrict__ w2) {
    const size_t total = static_cast<size_t>(n0) * n1 * n2;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(idx % n2);
        const int j = static_cast<int>((idx / n2) % n1);
        const size_t i = idx / (static_cast<size_t>(n1) * n2);
        const float wj = w0[i] * w1[j];
        dst[idx] = (src[i * frame_stride + j * row_stride + k] - mean[i]) * wj * w2[k];
    }
}

// in place: x = (x - mean[frame]) * t1[row]
__global__ void centre_kernel(float* __restrict__ x, int n0, int n1, int n2, const float* __restrict__ mean, const float* __restrict__ t1) {
    const size_t total = static_cast<size_t>(n0) * n1 * n2;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>((idx / n2) % n1);
        const size_t i = idx / (static_cast<size_t>(n1) * n2);
        x[idx] = (x[idx] - mean[i]) * t1[j];
    }
}

// in place: x *= t0[frame] * t2[col]
__global__ void taper_kernel(float* __restrict__ x, int n0, int n1, int n2, const float* __restrict__ t0, const float* __restrict__ t2) {
    const size_t total = static_cast<size_t>(n0) * n1 * n2;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int k = static_cast<int>(idx % n2);
        const size_t i = idx / (static_cast<size_t>(n1) * n2);
        x[idx] *= t0[i] * t2[k];
    }
}

// in place, spectra of the frames (n0, n1, nc): bin (frame, k1, k2) *= e[frame * n1 + k1]
__global__ void row_phase_kernel(cufftComplex* __restrict__ spec, int n0, int n1, int nc, const cufftComplex* __restrict__ e) {
    const size_t total = static_cast<size_t>(n0) * n1 * nc;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x)
        spec[idx] = cmul(spec[idx], e[idx / nc]);
}

// dst = src * e0[k0] e1[k1] e2[k2], the phasors multiplied in double as on the CPU
__global__ void displace_kernel(const cufftComplex* __restrict__ src, cufftComplex* __restrict__ dst, int n0, int n1, int nc,
                                const cufftDoubleComplex* __restrict__ e0, const cufftDoubleComplex* __restrict__ e1,
                                const cufftDoubleComplex* __restrict__ e2) {
    const size_t total = static_cast<size_t>(n0) * n1 * nc;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int k2 = static_cast<int>(idx % nc);
        const int k1 = static_cast<int>((idx / nc) % n1);
        const size_t k0 = idx / (static_cast<size_t>(n1) * nc);
        const cufftDoubleComplex e = cmul(cmul(e0[k0], e1[k1]), e2[k2]);
        dst[idx] = cmul(src[idx], make_cuFloatComplex(static_cast<float>(e.x), static_cast<float>(e.y)));
    }
}

// One thread per line (k0, k1) of the half spectrum. conjugate: a = conj(a) * w (block a); otherwise c = aw * c
// (the cross spectrum). energy[line] = sum over k2 of mult * w * |x|^2 of the spectrum as it came in.
__global__ void weigh_kernel(cufftComplex* __restrict__ x, const cufftComplex* __restrict__ aw, const float* __restrict__ weight, int lines, int nc,
                             bool conjugate, double* __restrict__ energy) {
    const int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= lines) return;
    cufftComplex* p = x + static_cast<size_t>(line) * nc;
    const float* w = weight + static_cast<size_t>(line) * nc;
    double e = 0;
    for (int k2 = 0; k2 < nc; ++k2) {
        const float norm = __fadd_rn(__fmul_rn(p[k2].x, p[k2].x), __fmul_rn(p[k2].y, p[k2].y));
        e += (k2 == 0 ? 1.0 : 2.0) * w[k2] * norm;
        if (conjugate) p[k2] = make_cuFloatComplex(p[k2].x * w[k2], -p[k2].y * w[k2]);
        else p[k2] = cmul(aw[static_cast<size_t>(line) * nc + k2], p[k2]);
    }
    energy[line] = e;
}

// One thread per line (k0, k1): the line's part of r, its gradient and Hessian at a displacement whose phasors
// are e0, e1, e2. out[line * 10 ...] = v, g0, g1, g2, h00, h01, h02, h11, h12, h22.
__global__ void correlation_kernel(const cufftComplex* __restrict__ C, int n0, int n1, int n2, int nc, const cufftDoubleComplex* __restrict__ e0,
                                   const cufftDoubleComplex* __restrict__ e1, const cufftDoubleComplex* __restrict__ e2, double* __restrict__ out) {
    const int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= n0 * n1) return;
    const int k0 = line / n1, k1 = line % n1;
    const double two_pi = 6.283185307179586476925286766559;
    const double q0 = two_pi * bin_frequency(k0, n0), q1 = two_pi * bin_frequency(k1, n1);
    const cufftDoubleComplex e01 = cmul(e0[k0], e1[k1]);
    const cufftComplex* pc = C + static_cast<size_t>(line) * nc;
    double sr = 0, si = 0, sr2 = 0, si2 = 0, sr22 = 0, si22 = 0;
    for (int k2 = 0; k2 < nc; ++k2) {
        const double mult = (k2 == 0) ? 1.0 : 2.0;
        const double q2 = two_pi * bin_frequency(k2, n2);
        const cufftDoubleComplex t = cmul(make_cuDoubleComplex(pc[k2].x, pc[k2].y), e2[k2]);
        sr += mult * t.x;
        si += mult * t.y;
        sr2 += mult * q2 * t.x;
        si2 += mult * q2 * t.y;
        sr22 += mult * q2 * q2 * t.x;
        si22 += mult * q2 * q2 * t.y;
    }
    const double re = sr * e01.x - si * e01.y, im = sr * e01.y + si * e01.x;
    const double re2 = sr2 * e01.x - si2 * e01.y, im2 = sr2 * e01.y + si2 * e01.x;
    const double re22 = sr22 * e01.x - si22 * e01.y;
    double* o = out + static_cast<size_t>(line) * 10;
    o[0] = re;
    o[1] = -q0 * im;
    o[2] = -q1 * im;
    o[3] = -im2;
    o[4] = -q0 * q0 * re;
    o[5] = -q0 * q1 * re;
    o[6] = -q0 * re2;
    o[7] = -q1 * q1 * re;
    o[8] = -q1 * re2;
    o[9] = -re22;
}

// the correlation at the displacements -s .. s of each axis (wrapped), packed
__global__ void gather_kernel(const float* __restrict__ r, int n0, int n1, int n2, int s0, int s1, int s2, float* __restrict__ out) {
    const int w0 = 2 * s0 + 1, w1 = 2 * s1 + 1, w2 = 2 * s2 + 1;
    const size_t total = static_cast<size_t>(w0) * w1 * w2;
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d2 = static_cast<int>(idx % w2) - s2;
        const int d1 = static_cast<int>((idx / w2) % w1) - s1;
        const int d0 = static_cast<int>(idx / (static_cast<size_t>(w1) * w2)) - s0;
        out[idx] = r[(static_cast<size_t>((d0 + n0) % n0) * n1 + (d1 + n1) % n1) * n2 + (d2 + n2) % n2];
    }
}

std::vector<cufftDoubleComplex> to_device_phasors(const std::vector<std::complex<double>>& e) {
    std::vector<cufftDoubleComplex> out(e.size());
    for (std::size_t i = 0; i < e.size(); ++i) out[i] = make_cuDoubleComplex(e[i].real(), e[i].imag());
    return out;
}

class CudaBackend final : public RegistrationBackend {
  public:
    CudaBackend(const RegistrationGeometry& g, const RegistrationOptions& opt, const float* weights) : g_(g), opt_(opt) {
        r_.resize(g_.N);
        aw_.resize(g_.NC);
        c_.resize(g_.NC);
        cs_.resize(g_.NC);
        weight_.upload(weights, g_.NC);
        rbig_.resize(g_.NB);
        cbig_.resize(g_.NBC);
        cwork_.resize(g_.NBC);
        w0_.upload(g_.w0.data(), g_.w0.size());
        w1_.upload(g_.w1.data(), g_.w1.size());
        w2_.upload(g_.w2.data(), g_.w2.size());
        const std::size_t lines = static_cast<std::size_t>(std::max(g_.n[0], g_.L[0])) * std::max(g_.n[1], g_.L[1]);
        rowsum_.resize(lines);
        mean_.resize(static_cast<std::size_t>(g_.L[0]));
        partial_.resize(static_cast<std::size_t>(g_.n[0]) * g_.n[1] * 10);
        host_partial_.resize(partial_.size());
        // the weighted number of samples of a frame under the analysis window, summed as on the CPU
        double weight = 0;
        for (int j = 0; j < g_.n[1]; ++j)
            for (int k = 0; k < g_.n[2]; ++k) weight += static_cast<double>(g_.w1[j]) * g_.w2[k];
        window_norm_ = weight;
        // transforms: the inner block and the padded block (3D, or 2D for a single frame), and the padded
        // block's frames one by one (2D, batched) for the displacement of rows. One work area serves all.
        std::size_t work = 0;
        forward_ = plan(g_.n, 1, CUFFT_R2C, work);
        inverse_ = plan(g_.n, 1, CUFFT_C2R, work);
        big_forward_ = plan(g_.L, 1, CUFFT_R2C, work);
        big_inverse_ = plan(g_.L, 1, CUFFT_C2R, work);
        const int frame[3] = {1, g_.L[1], g_.L[2]};
        frames_forward_ = plan(frame, g_.L[0], CUFFT_R2C, work);
        frames_inverse_ = plan(frame, g_.L[0], CUFFT_C2R, work);
        work_.resize(work);
        for (cufftHandle h : {forward_, inverse_, big_forward_, big_inverse_, frames_forward_, frames_inverse_})
            cufft_check(cufftSetWorkArea(h, work_.data()), "cufftSetWorkArea");
        HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CudaBackend() override {
        for (cufftHandle h : {forward_, inverse_, big_forward_, big_inverse_, frames_forward_, frames_inverse_}) cufftDestroy(h);
    }

    float* staging_a() override {
        stage_a_.ensure(g_.N);
        return stage_a_.data();
    }
    float* staging_b() override {
        stage_b_.ensure(g_.NB);
        return stage_b_.data();
    }

    double set_a(const float* a) override {
        const int* n = g_.n;
        // the padded block's buffer is free until set_b: the raw block goes there
        HELIREC_CUDA_CHECK(cudaMemcpy(rbig_.data(), a, g_.N * sizeof(float), cudaMemcpyHostToDevice));
        prepare(rbig_.data(), static_cast<std::size_t>(n[1]) * n[2], n[2]);
        cufft_check(cufftExecR2C(forward_, r_.data(), aw_.data()), "cufftExecR2C");
        const int lines = n[0] * n[1];
        weigh_kernel<<<cuda::grid_1d(lines, 64), 64>>>(aw_.data(), nullptr, weight_.data(), lines, g_.nc, true, partial_.data());
        return sum_partial(lines);
    }

    void set_b(const float* b, const double* row_shift) override {
        const int* L = g_.L;
        double largest = 0;
        if (row_shift)
            for (int q = 0; q < L[0]; ++q) largest = std::max(largest, std::fabs(row_shift[q]));
        std::vector<float> t0, t1, t2;
        g_.big_tapers(largest, t0, t1, t2);
        t0_.upload(t0.data(), t0.size());
        t1_.upload(t1.data(), t1.size());
        t2_.upload(t2.data(), t2.size());
        HELIREC_CUDA_CHECK(cudaMemcpy(rbig_.data(), b, g_.NB * sizeof(float), cudaMemcpyHostToDevice));
        // per frame: the mean removed and the row taper
        const std::size_t frame = static_cast<std::size_t>(L[1]) * L[2];
        const int rows = L[0] * L[1];
        row_sums_kernel<<<cuda::grid_1d(rows, 64), 64>>>(rbig_.data(), frame, L[2], L[0], L[1], L[2], nullptr, rowsum_.data());
        frame_means_kernel<<<cuda::grid_1d(L[0], 64), 64>>>(rowsum_.data(), L[0], L[1], nullptr, static_cast<double>(frame), mean_.data());
        centre_kernel<<<grid_for(g_.NB), kBlock>>>(rbig_.data(), L[0], L[1], L[2], mean_.data(), t1_.data());
        if (row_shift && largest > 0) {
            // each frame's rows displaced by its own amount: a phase along k1 of the frame's spectrum
            std::vector<cufftComplex> e(static_cast<std::size_t>(L[0]) * L[1]);
            const double norm = 1.0 / (static_cast<double>(L[1]) * L[2]);
            for (int q = 0; q < L[0]; ++q) {
                const std::vector<std::complex<double>> p = phasors(L[1], L[1], row_shift[q]);
                for (int k = 0; k < L[1]; ++k)
                    e[static_cast<std::size_t>(q) * L[1] + k] = make_cuFloatComplex(static_cast<float>(p[k].real() * norm), static_cast<float>(p[k].imag() * norm));
            }
            row_phase_.upload(e.data(), e.size());
            cufft_check(cufftExecR2C(frames_forward_, rbig_.data(), cwork_.data()), "cufftExecR2C");
            row_phase_kernel<<<grid_for(g_.NBC), kBlock>>>(cwork_.data(), L[0], L[1], g_.Lc, row_phase_.data());
            cufft_check(cufftExecC2R(frames_inverse_, cwork_.data(), rbig_.data()), "cufftExecC2R");
        }
        taper_kernel<<<grid_for(g_.NB), kBlock>>>(rbig_.data(), L[0], L[1], L[2], t0_.data(), t2_.data());
        cufft_check(cufftExecR2C(big_forward_, rbig_.data(), cbig_.data()), "cufftExecR2C");
    }

    double cross(const double d[3]) override {
        const int* L = g_.L;
        const int* n = g_.n;
        upload_phasors(phasors(L[0], L[0], d[0]), phasors(L[1], L[1], d[1]), phasors(L[2], g_.Lc, d[2]));
        displace_kernel<<<grid_for(g_.NBC), kBlock>>>(cbig_.data(), cwork_.data(), L[0], L[1], g_.Lc, e0_.data(), e1_.data(), e2_.data());
        cufft_check(cufftExecC2R(big_inverse_, cwork_.data(), rbig_.data()), "cufftExecC2R");
        prepare(rbig_.data() + (static_cast<std::size_t>(g_.m[0]) * L[1] + g_.m[1]) * L[2] + g_.m[2], static_cast<std::size_t>(L[1]) * L[2], L[2]);
        cufft_check(cufftExecR2C(forward_, r_.data(), c_.data()), "cufftExecR2C");
        const int lines = n[0] * n[1];
        weigh_kernel<<<cuda::grid_1d(lines, 64), 64>>>(c_.data(), aw_.data(), weight_.data(), lines, g_.nc, false, partial_.data());
        return sum_partial(lines);
    }

    bool peak(const int s[3], int best[3]) override {
        const int* n = g_.n;
        // the transform may overwrite its input: a copy of the cross spectrum
        HELIREC_CUDA_CHECK(cudaMemcpy(cs_.data(), c_.data(), g_.NC * sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
        cufft_check(cufftExecC2R(inverse_, cs_.data(), r_.data()), "cufftExecC2R");
        const int w0 = 2 * s[0] + 1, w1 = 2 * s[1] + 1, w2 = 2 * s[2] + 1;
        const std::size_t total = static_cast<std::size_t>(w0) * w1 * w2;
        window_.ensure(total);
        host_window_.resize(total);
        gather_kernel<<<grid_for(total), kBlock>>>(r_.data(), n[0], n[1], n[2], s[0], s[1], s[2], window_.data());
        HELIREC_CUDA_CHECK(cudaMemcpy(host_window_.data(), window_.data(), total * sizeof(float), cudaMemcpyDeviceToHost));
        float top = -INFINITY;
        best[0] = best[1] = best[2] = 0;
        std::size_t idx = 0;
        for (int d0 = -s[0]; d0 <= s[0]; ++d0)
            for (int d1 = -s[1]; d1 <= s[1]; ++d1)
                for (int d2 = -s[2]; d2 <= s[2]; ++d2, ++idx)
                    if (host_window_[idx] > top) top = host_window_[idx], best[0] = d0, best[1] = d1, best[2] = d2;
        bool at_edge = false;
        for (int j = 0; j < 3; ++j)
            if (s[j] > 0 && std::abs(best[j]) == s[j]) at_edge = true;
        return at_edge;
    }

    void correlation(const double at[3], double& value, double g[3], double H[3][3]) override {
        const int* n = g_.n;
        std::vector<std::complex<double>> e0(n[0]), e1(n[1]), e2(g_.nc);
        for (int k = 0; k < n[0]; ++k) e0[k] = std::polar(1.0, 2 * M_PI * frequency(k, n[0]) * at[0]);
        for (int k = 0; k < n[1]; ++k) e1[k] = std::polar(1.0, 2 * M_PI * frequency(k, n[1]) * at[1]);
        for (int k = 0; k < g_.nc; ++k) e2[k] = std::polar(1.0, 2 * M_PI * frequency(k, n[2]) * at[2]);
        upload_phasors(e0, e1, e2);
        const int lines = n[0] * n[1];
        correlation_kernel<<<cuda::grid_1d(lines, 64), 64>>>(c_.data(), n[0], n[1], n[2], g_.nc, e0_.data(), e1_.data(), e2_.data(), partial_.data());
        HELIREC_CUDA_CHECK(cudaMemcpy(host_partial_.data(), partial_.data(), static_cast<std::size_t>(lines) * 10 * sizeof(double), cudaMemcpyDeviceToHost));
        double s[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int line = 0; line < lines; ++line)
            for (int k = 0; k < 10; ++k) s[k] += host_partial_[static_cast<std::size_t>(line) * 10 + k];
        value = s[0];
        g[0] = s[1], g[1] = s[2], g[2] = s[3];
        H[0][0] = s[4], H[0][1] = H[1][0] = s[5], H[0][2] = H[2][0] = s[6], H[1][1] = s[7], H[1][2] = H[2][1] = s[8], H[2][2] = s[9];
    }

  private:
    // a 3D transform of dims (2D if dims[0] is 1), `batch` of them; `work` grows to the largest work area needed
    static cufftHandle plan(const int dims[3], int batch, cufftType type, std::size_t& work) {
        cufftHandle h;
        cufft_check(cufftCreate(&h), "cufftCreate");
        cufft_check(cufftSetAutoAllocation(h, 0), "cufftSetAutoAllocation");
        int d[3] = {dims[0], dims[1], dims[2]};
        const int rank = dims[0] == 1 ? 2 : 3;
        std::size_t size = 0;
        cufft_check(cufftMakePlanMany(h, rank, d + (3 - rank), nullptr, 1, 0, nullptr, 1, 0, type, batch, &size), "cufftMakePlanMany");
        work = std::max(work, size);
        return h;
    }

    // r_ = (src - weighted mean of its frame) * window; src with the given strides
    void prepare(const float* src, std::size_t frame_stride, std::size_t row_stride) {
        const int* n = g_.n;
        const int rows = n[0] * n[1];
        row_sums_kernel<<<cuda::grid_1d(rows, 64), 64>>>(src, frame_stride, row_stride, n[0], n[1], n[2], w2_.data(), rowsum_.data());
        frame_means_kernel<<<cuda::grid_1d(n[0], 64), 64>>>(rowsum_.data(), n[0], n[1], w1_.data(), window_norm_, mean_.data());
        window_kernel<<<grid_for(g_.N), kBlock>>>(src, frame_stride, row_stride, r_.data(), n[0], n[1], n[2], mean_.data(), w0_.data(), w1_.data(), w2_.data());
    }

    double sum_partial(int lines) {
        HELIREC_CUDA_CHECK(cudaMemcpy(host_partial_.data(), partial_.data(), static_cast<std::size_t>(lines) * sizeof(double), cudaMemcpyDeviceToHost));
        double e = 0;
        for (int line = 0; line < lines; ++line) e += host_partial_[line];
        return e;
    }

    void upload_phasors(const std::vector<std::complex<double>>& e0, const std::vector<std::complex<double>>& e1, const std::vector<std::complex<double>>& e2) {
        const auto d0 = to_device_phasors(e0), d1 = to_device_phasors(e1), d2 = to_device_phasors(e2);
        e0_.ensure(d0.size());
        e1_.ensure(d1.size());
        e2_.ensure(d2.size());
        HELIREC_CUDA_CHECK(cudaMemcpy(e0_.data(), d0.data(), d0.size() * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        HELIREC_CUDA_CHECK(cudaMemcpy(e1_.data(), d1.data(), d1.size() * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        HELIREC_CUDA_CHECK(cudaMemcpy(e2_.data(), d2.data(), d2.size() * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    }

    RegistrationGeometry g_;
    RegistrationOptions opt_;
    double window_norm_ = 1;
    DeviceBuffer<float> r_, rbig_, weight_, w0_, w1_, w2_, t0_, t1_, t2_, mean_, window_;
    DeviceBuffer<cufftComplex> aw_, c_, cs_, cbig_, cwork_, row_phase_;
    DeviceBuffer<cufftDoubleComplex> e0_, e1_, e2_;
    DeviceBuffer<double> rowsum_, partial_;
    DeviceBuffer<unsigned char> work_;
    std::vector<double> host_partial_;
    std::vector<float> host_window_;
    PinnedBuffer<float> stage_a_, stage_b_;
    cufftHandle forward_ = 0, inverse_ = 0, big_forward_ = 0, big_inverse_ = 0, frames_forward_ = 0, frames_inverse_ = 0;
};

}  // namespace

std::unique_ptr<RegistrationBackend> make_cuda_registration(const RegistrationGeometry& geom, const RegistrationOptions& opt, const float* weights) {
    return std::make_unique<CudaBackend>(geom, opt, weights);
}

}  // namespace helirec::calibration
