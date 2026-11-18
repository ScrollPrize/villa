#include <algorithm>
#include <vector>

#include "common/accumulate_core.hpp"
#include "cuda_utils.cuh"
#include "vc_tomography/accumulate.hpp"
#include "vc_tomography/sino.hpp"
#include "../common/elementwise_core.hpp"

namespace vc_tomography::cuda {

using namespace vc_tomography::accum_core;

namespace {

// threads: x over merged columns, y over rows t, z over grid angles of the block
// (the scan's view is read from device memory: a kernel that takes a view of this size by value and hands it on by
// reference copies it to the stack of every thread, which cost several times the gather itself)
__global__ void accumulate_kernel(const AccumScanView* __restrict__ view, const int32_t* __restrict__ offsets, const int32_t* __restrict__ proj,
                                  const int8_t* __restrict__ role, int L, int H, int NX, AccumParams p,
                                  float* __restrict__ R, float* __restrict__ RW) {
    const AccumScanView& s = *view;
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int jl = blockIdx.z;
    if (X >= NX || t >= H || jl >= L) return;
    const size_t o = (static_cast<size_t>(jl) * H + t) * NX + X;
    float a = R[o], b = RW[o];
    for (int e = offsets[jl]; e < offsets[jl + 1]; ++e) contribute(s, proj[e], role[e], t, X, p, a, b);
    R[o] = a;
    RW[o] = b;
}

// the vertically interpolated values of the block's projections and their weights (wt: of the scan's weight map; wt2:
// of a second map, weight2; null: none): threads x over detector columns, y over rows t, z over the projections (slots)
// finite: the rows are to be read exactly (a value that is not finite becomes 0: see cpu::accumulate)
__global__ void column_values_kernel(const AccumScanView* __restrict__ view, const int32_t* __restrict__ slot_proj, int H, int stride, bool finite,
                                     float non_zero, float* __restrict__ out, const float* __restrict__ weight2,
                                     float* __restrict__ wt, float* __restrict__ wt2) {
    const AccumScanView& s = *view;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= s.det_cols || t >= H) return;
    const size_t row = static_cast<size_t>(blockIdx.z) * H + t;
    const size_t o = row * s.det_cols + c;
    column_cache_entry(s, weight2, slot_proj[blockIdx.z], t, c, non_zero, finite, out[row * stride + c], wt ? wt + o : nullptr,
                       wt2 ? wt2 + o : nullptr);
}

// the same for measured pixels only (measured_value): each detector column's measured value and its weight
__global__ void measured_values_kernel(const AccumScanView* __restrict__ view, const int32_t* __restrict__ slot_proj, int H, int stride, AccumParams p,
                                       float* __restrict__ out, float* __restrict__ wt) {
    const AccumScanView& s = *view;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= s.det_cols || t >= H) return;
    const size_t o = (static_cast<size_t>(blockIdx.z) * H + t) * stride + c;
    measured_value(s, slot_proj[blockIdx.z], t, c, p, out[o], wt[o]);
}

// every such row made ready for an exact read (measured_row_for_exact_read): a thread per row
__global__ void measured_rows_kernel(float* __restrict__ val, float* __restrict__ wt, int n_rows, int n, int stride, int guard) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n_rows) measured_row_for_exact_read(val + static_cast<size_t>(q) * stride, wt + static_cast<size_t>(q) * stride, n, guard);
}

// the delay of every cached row for the frames' lateral registration (column_delay): (n_slots, H)
__global__ void column_delay_kernel(const AccumScanView* __restrict__ view, const int32_t* __restrict__ slot_proj, int n_slots, int H, float* __restrict__ out) {
    const AccumScanView& s = *view;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n_slots * H) out[q] = column_delay(s, slot_proj[q / H], q % H);
}

// the rows of resample_rows for the cached rows (resample_row): (n_slots, H)
__global__ void resample_row_kernel(const AccumScanView* __restrict__ view, const int32_t* __restrict__ slot_proj, int n_slots, int H, ResampleRow* __restrict__ out) {
    const AccumScanView& s = *view;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n_slots * H) out[q] = resample_row(s, slot_proj[q / H], q % H);
}

__global__ void lum_sums_kernel(const float* R, const float* RW, int L, int H, int NX, int cx, const float* gx,
                                const float* gz, int radius, float* A, float* B) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= L * H) return;
    window_sum(R, RW, H, NX, q / H, q % H, cx, gx, gz, radius, A[q], B[q]);
}

__global__ void lum_apply_kernel(float* R, const float* RW, const float* Wp, const float* Aha, const float* Bha,
                                 const float* Atot, const float* Btot, int LH, int NX, float rep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y + 65535 * blockIdx.z;  // rows of (L * H) may exceed gridDim.y's limit
    if (x >= NX || q >= LH) return;
    const float d = luminosity_diff(Aha[q], Bha[q], Atot[q], Btot[q]);
    const float dfin = __fdiv_rn(__fmul_rn(d, __fadd_rn(rep, 100.0f)), 200.0f);
    const size_t o = static_cast<size_t>(q) * NX + x;
    R[o] = __fadd_rn(R[o], __fsub_rn(__fmul_rn(d, Wp[o]), __fmul_rn(dfin, RW[o])));
}

__global__ void mend_kernel(float* R, const float* RW, int LH, int NX, float thr, int x0, int x1) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= LH) return;
    mend_row(R + static_cast<size_t>(q) * NX, RW + static_cast<size_t>(q) * NX, NX, thr, x0, x1);
}

__global__ void support_kernel(float* R, float* RW, int LH, int NX, float thr) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < LH) measured_support(R + static_cast<size_t>(q) * NX, RW + static_cast<size_t>(q) * NX, NX, thr);
}
__global__ void fill_measured_kernel(float* R, const float* RW, int L, int H, int NX, float empty) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= L * H) return;
    const size_t o = static_cast<size_t>(q / H) * H * NX;
    fill_measured_row(R + o, RW + o, H, NX, q % H, empty);
}

template <class T>
DeviceBuffer<T> upload(const std::vector<T>& v, cudaStream_t s) {
    DeviceBuffer<T> d;
    d.upload(v.data(), v.size(), s);
    return d;
}

}  // namespace

// `v` into a device buffer kept between calls (grow-only)
template <class T>
const T* load(DeviceBuffer<T>& d, const std::vector<T>& v, cudaStream_t s) {
    d.ensure(std::max<std::size_t>(1, v.size()));
    if (!v.empty()) VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s));
    return d.data();
}

void accumulate(const AccumScanView& view, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                void* stream, const float* weight2, float* R2, float* RW2) {
    cudaStream_t st = as_stream(stream);
    const int L = c.g1 - c.g0;
    if (L <= 0 || p.H <= 0 || p.NX <= 0 || c.proj.empty()) return;
    // (work buffers of the calling thread, kept between blocks: every call finishes before it returns)
    thread_local DeviceBuffer<int32_t> d_off, d_proj, d_slot, d_slot_proj;
    thread_local DeviceBuffer<int8_t> d_role;
    thread_local DeviceBuffer<float> d_col, d_col_wt, d_col_w, d_col_w2, d_delay;
    thread_local DeviceBuffer<ResampleRow> d_row;
    thread_local DeviceBuffer<AccumScanView> d_view;
    thread_local ThreadScratch scratch(d_view, d_off, d_proj, d_slot, d_slot_proj, d_role, d_col, d_col_wt, d_col_w, d_col_w2, d_delay, d_row);
    AccumScanView s = view;
    // the view as the kernels read it: slot i of d_view takes the view as it stands (the launches are queued, so each
    // state of the view a kernel is given keeps its own slot, on the host as well until the copies have been made)
    d_view.ensure(3);
    AccumScanView states[3];
    const auto on_device = [&](int i) {
        states[i] = s;
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_view.data() + i, &states[i], sizeof s, cudaMemcpyHostToDevice, st));
        return static_cast<const AccumScanView*>(d_view.data() + i);
    };
    // the vertically interpolated values of the block's projections, once each (they are the same at every column
    // tap of every pixel a sample lands on); measured-only gridding interpolates per corner instead, unless the scan
    // has a lateral registration or the lateral part of a map: then its measured values are read exactly as well
    // (AccumScanView::col_wt)
    const ColumnCachePlan cache = column_cache_plan(s, c, p);
    const bool measured_exact = cache.measured_exact;
    if (cache.use) {
        const std::vector<int32_t>& slot = cache.slot;
        const std::vector<int32_t>& slot_proj = cache.slot_proj;
        const std::size_t n_slots = slot_proj.size();
        const bool shift = cache.shift;
        const int stride = cache.stride;
        d_col.ensure(n_slots * p.H * stride);
        s.col_slot = load(d_slot, slot, st);
        dim3 block(128, 1, 1), grid(grid_1d(s.det_cols, 128), p.H, static_cast<unsigned>(n_slots));
        const int32_t* sp = load(d_slot_proj, slot_proj, st);
        const int n_rows = static_cast<int>(n_slots) * p.H;
        const AccumScanView* dv = on_device(0);
        if (measured_exact) {
            d_col_wt.ensure(n_slots * p.H * stride);
            measured_values_kernel<<<grid, block, 0, st>>>(dv, sp, p.H, stride, p, d_col.data(), d_col_wt.data());
            VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
            measured_rows_kernel<<<(n_rows + 63) / 64, 64, 0, st>>>(d_col.data(), d_col_wt.data(), n_rows, s.det_cols, stride, s.col_guard);
        } else {   // (with the rows' weights, of both weight maps: the gather below reads them)
            const std::size_t n_w = n_slots * p.H * static_cast<std::size_t>(s.det_cols);
            d_col_w.ensure(n_w);
            if (weight2) d_col_w2.ensure(n_w);
            column_values_kernel<<<grid, block, 0, st>>>(dv, sp, p.H, stride, shift, p.non_zero, d_col.data(), weight2, d_col_w.data(),
                                                         weight2 ? d_col_w2.data() : nullptr);
        }
        VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
        if (shift && s.map_cols) {   // each ideal column from its detector column: the detector map, the frames' scale
            d_row.ensure(n_rows);
            resample_row_kernel<<<(n_rows + 255) / 256, 256, 0, st>>>(dv, sp, static_cast<int>(n_slots), p.H, d_row.data());
            VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
            ResampleGeometry g;
            g.map = s.map;
            g.slope_col = s.slope_col;
            g.scale_col = s.scale_col;
            g.scale_row = s.scale_row;
            resample_rows(d_col.data(), n_rows, s.det_cols, d_row.data(), g, st);
        } else if (shift && s.col_offset) {   // each row by its own amount: the frames' lateral registration
            d_delay.ensure(n_rows);
            column_delay_kernel<<<(n_rows + 255) / 256, 256, 0, st>>>(dv, sp, static_cast<int>(n_slots), p.H, d_delay.data());
            VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
            shift_rows(d_col.data(), n_rows, s.det_cols, s.fract_x, st, d_delay.data());
        } else if (shift) {
            shift_rows(d_col.data(), n_rows, s.det_cols, s.fract_x, st);
        }
        s.col_val = d_col.data();
        s.col_rows = p.H;
        s.col_stride = stride;
        s.col_shifted = shift ? 1 : 0;
        if (measured_exact) s.col_wt = d_col_wt.data();
        else s.col_w = d_col_w.data();
    }
    const int32_t* off = load(d_off, c.offsets, st);
    const int32_t* proj = load(d_proj, c.proj, st);
    const int8_t* role = load(d_role, c.role, st);
    dim3 block(128, 1, 1), grid(grid_1d(p.NX, 128), p.H, L);
    accumulate_kernel<<<grid, block, 0, st>>>(on_device(1), off, proj, role, L, p.H, p.NX, p, R, RW);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    if (weight2) {   // the same contributions with the second weight map
        s.weight = weight2;
        if (s.col_w) s.col_w = d_col_w2.data();
        accumulate_kernel<<<grid, block, 0, st>>>(on_device(2), off, proj, role, L, p.H, p.NX, p, R2, RW2);
        VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    }
    VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamSynchronize(st));
}

void luminosity_sums(const float* R, const float* RW, int L, int H, int NX, int cx, const LuminosityKernel& k,
                     float* A, float* B, void* stream) {
    cudaStream_t st = as_stream(stream);
    auto gx = upload(k.gx, st);
    auto gz = upload(k.gz, st);
    lum_sums_kernel<<<grid_1d(L * H, 128), 128, 0, st>>>(R, RW, L, H, NX, cx, gx.data(), gz.data(), k.radius, A, B);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamSynchronize(st));
}

void luminosity_apply(float* R, const float* RW, const float* W_prev, const float* A_ha, const float* B_ha,
                      const float* A_tot, const float* B_tot, int L, int H, int NX, float rep, void* stream) {
    const int LH = L * H;
    dim3 grid((NX + 255) / 256, std::min(LH, 65535), (LH + 65534) / 65535);
    lum_apply_kernel<<<grid, 256, 0, as_stream(stream)>>>(R, RW, W_prev, A_ha, B_ha, A_tot, B_tot, L * H, NX, rep);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

void add_into(float* R, const float* A, std::size_t n, void* stream) { each(n, elementwise::AddInto{R, A}, as_stream(stream)); }
void divide(float* R, const float* RW, std::size_t n, void* stream) { each(n, elementwise::Divide{R, RW}, as_stream(stream)); }

void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float thr, int x0, int x1, void* stream) {
    mend_kernel<<<grid_1d(L * H, 128), 128, 0, as_stream(stream)>>>(R, RW, L * H, NX, thr, x0, x1);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
}

void prepare_measured(float* R, float* RW, int L, int H, int NX, float thr, bool fill, float empty, void* stream) {
    support_kernel<<<grid_1d(L * H, 128), 128, 0, as_stream(stream)>>>(R, RW, L * H, NX, thr);
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    if (fill) {
        fill_measured_kernel<<<grid_1d(L * H, 128), 128, 0, as_stream(stream)>>>(R, RW, L, H, NX, empty);
        VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace vc_tomography::cuda
