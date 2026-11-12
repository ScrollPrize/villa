#include <algorithm>
#include <cmath>
#include <numeric>

#include "common/accumulate_core.hpp"
#include "common/interp_core.hpp"
#include "helirec/accumulate.hpp"
#include "helirec/sino.hpp"

namespace helirec {

ContributorList make_contributors(const int32_t* j0, int n_proj, int n_grid, int g0, int g1, bool upper_bin) {
    ContributorList c;
    c.g0 = g0;
    c.g1 = g1;
    const int L = g1 - g0;
    std::vector<std::vector<std::pair<int32_t, int8_t>>> lists(L);
    for (int k = 0; k < n_proj; ++k) {  // ascending projection order
        const int a = j0[k], b = (j0[k] + 1) % n_grid;
        if (a >= g0 && a < g1) lists[a - g0].push_back({k, 0});  // j0 == n_grid is dropped (reference)
        if (upper_bin && b >= g0 && b < g1) lists[b - g0].push_back({k, 1});
    }
    c.offsets.resize(L + 1, 0);
    for (int i = 0; i < L; ++i) c.offsets[i + 1] = c.offsets[i] + static_cast<int32_t>(lists[i].size());
    c.proj.reserve(c.offsets[L]);
    c.role.reserve(c.offsets[L]);
    for (auto& l : lists)
        for (auto [k, r] : l) {
            c.proj.push_back(k);
            c.role.push_back(r);
        }
    return c;
}

void contributor_slots(const ContributorList& c, int n_proj, std::vector<int32_t>& slot, std::vector<int32_t>& slot_proj) {
    slot.assign(static_cast<std::size_t>(n_proj), -1);
    for (int32_t k : c.proj) slot[k] = 0;
    slot_proj.clear();
    for (int k = 0; k < n_proj; ++k)
        if (slot[k] == 0) {
            slot[k] = static_cast<int32_t>(slot_proj.size());
            slot_proj.push_back(k);
        }
}

void set_x_kernel(AccumScanView& v, int interp, bool exact) {
    v.x_taps = 0;
    v.x_exact = 0;
    if (v.fract_x == 0.f || interp == interp_core::Linear) return;
    v.x_exact = exact ? 1 : 0;
    if (interp != interp_core::Cubic && interp != interp_core::Lanczos3)
        throw Error("x_interp must be linear, cubic or lanczos3");
    const int T = interp == interp_core::Cubic ? 4 : 6;
    double w[8], sum = 0;
    for (int i = 0; i < T; ++i) {   // tap i: column c0 - T / 2 + i, at  i - T / 2 + fract_x  from the source position
        const float d = static_cast<float>(i - T / 2) + v.fract_x;
        w[i] = interp == interp_core::Cubic ? interp_core::keys(d) : interp_core::lanczos3(d);
        sum += w[i];
    }
    for (int i = 0; i < T; ++i) v.x_w[i] = static_cast<float>(w[i] / sum);
    v.x_taps = T;
}

int z_kernel_context(int interp) {
    if (interp == interp_core::Linear) return 0;
    if (interp == interp_core::Cubic) return 1;
    if (interp == interp_core::Lanczos3) return 2;
    throw Error("z_interp must be linear, cubic or lanczos3");
}

std::vector<float> z_kernel_weights(const std::vector<float>& fract_v, int interp) {
    const int T = 2 + 2 * z_kernel_context(interp);
    if (T == 2) return {};
    std::vector<float> out(fract_v.size() * T);
    for (std::size_t k = 0; k < fract_v.size(); ++k) {
        double w[8], sum = 0;
        for (int i = 0; i < T; ++i) {   // tap i: row t - (T / 2 - 1) + i, at  i - (T / 2 - 1) - a  from the position t + a
            const float d = static_cast<float>(i - (T / 2 - 1)) - fract_v[k];
            w[i] = interp == interp_core::Cubic ? interp_core::keys(d) : interp_core::lanczos3(d);
            sum += w[i];
        }
        for (int i = 0; i < T; ++i) out[k * T + i] = static_cast<float>(w[i] / sum);
    }
    return out;
}

std::vector<float> z_kernel_lut(int interp, int n) {
    std::vector<float> fr(n + 1);
    for (int i = 0; i <= n; ++i) fr[i] = static_cast<float>(i) / static_cast<float>(n);
    return z_kernel_weights(fr, interp);
}

LuminosityKernel luminosity_kernel() {
    LuminosityKernel k;
    const int r = k.radius;
    const float sigma = r / 2.f;
    k.gx.resize(2 * r + 1);
    k.gz.resize(2 * r + 1);
    for (int d = -r; d <= r; ++d) {
        k.gx[d + r] = std::exp(-0.5f * d * d / (sigma * sigma));
        k.gz[d + r] = std::exp(-0.5f * d * d / (sigma * sigma));
    }
    return k;
}

namespace cpu {

using namespace helirec::accum_core;

void accumulate(const AccumScanView& view, const ContributorList& c, const AccumParams& p, float* R, float* RW,
                const float* weight2, float* R2, float* RW2) {
    const int L = c.g1 - c.g0;
    AccumScanView s = view;
    // the vertically interpolated values of the block's projections, once each (they are the same at every column
    // tap of every pixel a sample lands on); measured-only gridding interpolates per corner instead
    thread_local std::vector<float> col;
    std::vector<int32_t> slot, slot_proj;
    if (!p.measured_only && !s.col_val && !c.proj.empty()) {
        contributor_slots(c, s.n_proj, slot, slot_proj);
        const int n_slots = static_cast<int>(slot_proj.size()), W = s.det_cols;
        const bool shift = s.x_exact && (s.fract_x != 0.f || s.col_offset || s.map_cols);
        const int stride = shift ? shift_row_stride(W) : W;
        col.resize(static_cast<std::size_t>(n_slots) * p.H * stride);
        float* const values = col.data();   // (the calling thread's buffer: `col` names another in each worker)
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
        for (int q = 0; q < n_slots; ++q)
            for (int t = 0; t < p.H; ++t) {
                float* out = values + (static_cast<std::size_t>(q) * p.H + t) * stride;
                for (int x = 0; x < W; ++x) out[x] = column_interp(s, slot_proj[q], t, x);
            }
        if (shift && s.map_cols) {   // each ideal column from its detector column: the detector map, the frames' scale
            std::vector<ResampleRow> row(static_cast<std::size_t>(n_slots) * p.H);
            for (int q = 0; q < n_slots; ++q)
                for (int t = 0; t < p.H; ++t) row[static_cast<std::size_t>(q) * p.H + t] = resample_row(s, slot_proj[q], t);
            ResampleGeometry g;
            g.map = s.map;
            g.slope_col = s.slope_col;
            g.scale_col = s.scale_col;
            g.scale_row = s.scale_row;
            resample_rows(values, n_slots * p.H, W, row.data(), g);
        } else if (shift && s.col_offset) {   // each row by its own amount: the frames' lateral registration
            std::vector<float> delay(static_cast<std::size_t>(n_slots) * p.H);
            for (int q = 0; q < n_slots; ++q)
                for (int t = 0; t < p.H; ++t) delay[static_cast<std::size_t>(q) * p.H + t] = column_delay(s, slot_proj[q], t);
            shift_rows(values, n_slots * p.H, W, s.fract_x, delay.data());
        } else if (shift) {
            shift_rows(values, n_slots * p.H, W, s.fract_x);
        }
        s.col_val = values;
        s.col_slot = slot.data();
        s.col_rows = p.H;
        s.col_stride = stride;
        s.col_shifted = shift ? 1 : 0;
    }
    for (int pass = 0; pass < (weight2 ? 2 : 1); ++pass) {
        if (pass == 1) s.weight = weight2;
        float* const Rp = pass ? R2 : R;
        float* const RWp = pass ? RW2 : RW;
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
        for (int jl = 0; jl < L; ++jl)
            for (int t = 0; t < p.H; ++t) {
                float* r = Rp + (static_cast<std::size_t>(jl) * p.H + t) * p.NX;
                float* w = RWp + (static_cast<std::size_t>(jl) * p.H + t) * p.NX;
                for (int X = 0; X < p.NX; ++X) {
                    float a = r[X], b = w[X];
                    for (int e = c.offsets[jl]; e < c.offsets[jl + 1]; ++e)
                        contribute(s, c.proj[e], c.role[e], t, X, p, a, b);
                    r[X] = a;
                    w[X] = b;
                }
            }
    }
}

void luminosity_sums(const float* R, const float* RW, int L, int H, int NX, int cx, const LuminosityKernel& k,
                     float* A, float* B) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < L; ++j)
        for (int t = 0; t < H; ++t)
            window_sum(R, RW, H, NX, j, t, cx, k.gx.data(), k.gz.data(), k.radius, A[j * H + t], B[j * H + t]);
}

void luminosity_apply(float* R, const float* RW, const float* W_prev, const float* A_ha, const float* B_ha,
                      const float* A_tot, const float* B_tot, int L, int H, int NX, float rep) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < L; ++j)
        for (int t = 0; t < H; ++t) {
            const int q = j * H + t;
            const float d = luminosity_diff(A_ha[q], B_ha[q], A_tot[q], B_tot[q]);
            const float dfin = (d * (rep + 100.0f)) / 200.0f;
            const std::size_t o = static_cast<std::size_t>(q) * NX;
            for (int x = 0; x < NX; ++x) R[o + x] += d * W_prev[o + x] - dfin * RW[o + x];
        }
}

void divide(float* R, const float* RW, std::size_t n) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) R[i] = RW[i] > 0 ? R[i] / RW[i] : 0.f;
}

void mend_null_weights(float* R, const float* RW, int L, int H, int NX, float thr, int x0, int x1) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < L; ++j)
        for (int t = 0; t < H; ++t) {
            const std::size_t o = (static_cast<std::size_t>(j) * H + t) * NX;
            mend_row(R + o, RW + o, NX, thr, x0, x1);
        }
}

void prepare_measured(float* R, float* RW, int L, int H, int NX, float thr, bool fill, float empty) {
#pragma omp parallel for
    for (int q = 0; q < L * H; ++q) {
        const std::size_t o = static_cast<std::size_t>(q) * NX;
        measured_support(R + o, RW + o, NX, thr);
    }
    if (!fill) return;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < L; ++j)
        for (int t = 0; t < H; ++t) {
            const std::size_t o = static_cast<std::size_t>(j) * H * NX;
            fill_measured_row(R + o, RW + o, H, NX, t, empty);
        }
}

}  // namespace cpu
}  // namespace helirec
