#pragma once
// Eikonal phase retrieval (EPR), local second-order closure: the per-pixel pieces shared by the CPU and CUDA
// backends (Mirone et al., "Eikonal phase retrieval", J. Synchrotron Rad. 32 (2025); "Fast eikonal phase retrieval
// for high-throughput beamlines", arXiv:2601.22793, eq. 1).
//
// Homogeneous object (phase = (delta/beta) / 2 x ln I), intensity I in the contact plane, detector pixels as the
// unit of length, A = delta_beta lambda L / (4 pi px^2) (the ray shift in pixels is A grad ln I):
//
//   F(I) = I - A lap I                                         Paganin's (TIE) model
//          + A^2 / 2 [div(D g) + 1/2 div(I grad |g|^2)]        second order in the ray shift:
//                                                              d_i d_j (I g_i g_j), g = grad ln I, D = div(I g)
//          - A^2 / delta_beta^2 div(I grad Q)                  diffraction term, Q = lap(sqrt I) / sqrt I
//
// The first-order term uses the compact 5-point Laplacian: it is exactly the operator the Paganin filter of
// filters.cpp inverts, so without the second-order terms the Paganin result is the fixed point of the iteration
// (cpu/epr.cpp, cuda/epr.cu). g is a central difference, clamped to gmax (a ray shift limit, for stability where
// the local expansion no longer holds). Indices are clamped at the frame edges.
#include <cmath>
#include <cstddef>

#include "hd_math.hpp"

namespace helirec::epr_core {

struct Coef {
    float A;        // delta_beta lambda L / (4 pi px^2)
    float half_A2;  // A^2 / 2
    float press;    // A^2 / delta_beta^2
    float gmax;     // limit of |grad ln I| = max ray shift / A; <= 0: none
    float eps;      // floor of intensities
};

HR_HD int clampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }
HR_HD float floor_eps(float v, float eps) { return v > eps ? v : eps; }  // also maps NaN to eps

HR_HD std::size_t at(int y, int x, int W) { return static_cast<std::size_t>(y) * W + x; }

// pass 1: g = clamped central gradient of ln I, q = lap(sqrt I) / sqrt I    (I >= eps > 0)
HR_HD void gradient(const float* I, int H, int W, int y, int x, const Coef& c, float& gx, float& gy, float& q) {
    const float ic = I[at(y, x, W)];
    const float il = I[at(y, clampi(x - 1, W), W)], ir = I[at(y, clampi(x + 1, W), W)];
    const float iu = I[at(clampi(y - 1, H), x, W)], id = I[at(clampi(y + 1, H), x, W)];
    float ax = 0.5f * (logf(ir) - logf(il)), ay = 0.5f * (logf(id) - logf(iu));
    const float m2 = ax * ax + ay * ay;
    if (c.gmax > 0.f && m2 > c.gmax * c.gmax) {
        const float s = c.gmax / sqrtf(m2);
        ax *= s;
        ay *= s;
    }
    gx = ax;
    gy = ay;
    const float sc = sqrtf(ic);
    q = (sqrtf(il) + sqrtf(ir) + sqrtf(iu) + sqrtf(id) - 4.f * sc) / sc;
}

// pass 2: D = div(I g), central differences
HR_HD float flux_divergence(const float* I, const float* gx, const float* gy, int H, int W, int y, int x) {
    const std::size_t l = at(y, clampi(x - 1, W), W), r = at(y, clampi(x + 1, W), W);
    const std::size_t u = at(clampi(y - 1, H), x, W), d = at(clampi(y + 1, H), x, W);
    return 0.5f * (I[r] * gx[r] - I[l] * gx[l]) + 0.5f * (I[d] * gy[d] - I[u] * gy[u]);
}

// pass 3: the model's detector intensity F(I) at (y, x)
HR_HD float forward(const float* I, const float* gx, const float* gy, const float* q, const float* D, int H, int W,
                    int y, int x, const Coef& c) {
    const std::size_t o = at(y, x, W);
    const std::size_t l = at(y, clampi(x - 1, W), W), r = at(y, clampi(x + 1, W), W);
    const std::size_t u = at(clampi(y - 1, H), x, W), d = at(clampi(y + 1, H), x, W);
    const float ic = I[o], il = I[l], ir = I[r], iu = I[u], id = I[d];
    const float lap = il + ir + iu + id - 4.f * ic;
    // div(D g), central
    const float a = 0.5f * (D[r] * gx[r] - D[l] * gx[l]) + 0.5f * (D[d] * gy[d] - D[u] * gy[u]);
    // div(I grad v), compact with face-averaged I, for v = |g|^2 and v = q
    const float wl = 0.5f * (ic + il), wr = 0.5f * (ic + ir), wu = 0.5f * (ic + iu), wd = 0.5f * (ic + id);
    const float gc = gx[o] * gx[o] + gy[o] * gy[o];
    const float cg = wr * (gx[r] * gx[r] + gy[r] * gy[r] - gc) + wl * (gx[l] * gx[l] + gy[l] * gy[l] - gc) +
                     wd * (gx[d] * gx[d] + gy[d] * gy[d] - gc) + wu * (gx[u] * gx[u] + gy[u] * gy[u] - gc);
    const float b = wr * (q[r] - q[o]) + wl * (q[l] - q[o]) + wd * (q[d] - q[o]) + wu * (q[u] - q[o]);
    return ic - c.A * lap + c.half_A2 * (a + 0.5f * cg) - c.press * b;
}

}  // namespace helirec::epr_core
