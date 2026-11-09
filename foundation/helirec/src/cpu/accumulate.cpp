#include <algorithm>
#include <cmath>
#include <numeric>

#include "common/accumulate_core.hpp"
#include "helirec/accumulate.hpp"

namespace helirec {

ContributorList make_contributors(const int32_t* j0, int n_proj, int n_grid, int g0, int g1) {
    ContributorList c;
    c.g0 = g0;
    c.g1 = g1;
    const int L = g1 - g0;
    std::vector<std::vector<std::pair<int32_t, int8_t>>> lists(L);
    for (int k = 0; k < n_proj; ++k) {  // ascending projection order
        const int a = j0[k], b = (j0[k] + 1) % n_grid;
        if (a >= g0 && a < g1) lists[a - g0].push_back({k, 0});  // j0 == n_grid is dropped (reference)
        if (b >= g0 && b < g1) lists[b - g0].push_back({k, 1});
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

void accumulate(const AccumScanView& s, const ContributorList& c, const AccumParams& p, float* R, float* RW) {
    const int L = c.g1 - c.g0;
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int jl = 0; jl < L; ++jl)
        for (int t = 0; t < p.H; ++t) {
            float* r = R + (static_cast<std::size_t>(jl) * p.H + t) * p.NX;
            float* w = RW + (static_cast<std::size_t>(jl) * p.H + t) * p.NX;
            for (int X = 0; X < p.NX; ++X) {
                float a = r[X], b = w[X];
                for (int e = c.offsets[jl]; e < c.offsets[jl + 1]; ++e)
                    contribute(s, c.proj[e], c.role[e], t, X, p.NX, p.non_zero, p.border, a, b);
                r[X] = a;
                w[X] = b;
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

}  // namespace cpu
}  // namespace helirec
