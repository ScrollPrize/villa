// GHBP geometry and hierarchy (host). Mirrors nabuxx ghbp_cuda.cu
// build_conebeam_geometry / build_geometry_and_hierarchy, including its
// float/double choices, so CPU and CUDA kernels reproduce nabuxx.
#include <algorithm>
#include <cmath>
#include <limits>

#include "helirec/ghbp.hpp"

namespace helirec {

namespace {

void mat3_mul(const float A[9], const float B[9], float C[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
}

void mat3_inv(const float M[9], float Minv[9]) {
    const float a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], i = M[8];
    const float A = (e * i - f * h), B = -(d * i - f * g), C = (d * h - e * g);
    const float invdet = 1.0f / (a * A + b * B + c * C);
    Minv[0] = A * invdet;
    Minv[1] = (c * h - b * i) * invdet;
    Minv[2] = (b * f - c * e) * invdet;
    Minv[3] = B * invdet;
    Minv[4] = (a * i - c * g) * invdet;
    Minv[5] = (c * d - a * f) * invdet;
    Minv[6] = C * invdet;
    Minv[7] = (b * g - a * h) * invdet;
    Minv[8] = (a * e - b * d) * invdet;
}

void rot2d(float a, float out[9]) {
    const float ca = std::cos(a), sa = std::sin(a);  // float precision, as nabuxx
    const float m[9] = {ca, sa, 0.f, -sa, ca, 0.f, 0.f, 0.f, 1.f};
    std::copy(m, m + 9, out);
}

void ainvt(int N, int gh, int gw, float scale, float out[9]) {
    const float T[9] = {1, 0, -0.5f * (gh - 1), 0, 1, -0.5f * (gw - 1), 0, 0, 1};
    const float A[9] = {((N - 1) / float(gh - 1)) * scale, 0, 0, 0, ((N - 1) / float(gw - 1)) * scale, 0, 0, 0, 1};
    mat3_mul(A, T, out);
}

float angle_of_mean(const float* a, int count) {
    double xr = 0.0, xi = 0.0;
    for (int i = 0; i < count; ++i) {
        xr += std::cos(a[i]);  // float overload, promoted to double sum (as nabuxx)
        xi += std::sin(a[i]);
    }
    return static_cast<float>(std::atan2(xi, xr));
}

std::vector<float> group_means(const std::vector<float>& a, int rf, int n_out) {
    const int n = static_cast<int>(a.size());
    const int m = (n / rf) * rf;
    std::vector<float> out;
    for (int g = 0; g < m; g += rf) out.push_back(angle_of_mean(a.data() + g, rf));
    if (m < n) out.push_back(angle_of_mean(a.data() + m, n - m));
    out.resize(static_cast<std::size_t>(n_out));
    return out;
}

GhbpLevel build_level(int N, int gh, int gw, const std::vector<float>& angles, int legs, float scale, int red) {
    if (gh <= 1 || gw <= 1) throw Error("GHBP: degenerate grid level");
    GhbpLevel L;
    L.h = gh;
    L.w = gw;
    L.red = red;
    const int ng = static_cast<int>(angles.size());
    L.g = (ng + legs - 1) / legs;
    const int padded = L.g * legs;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    L.T.assign(static_cast<std::size_t>(padded) * 6, nan);
    L.Tinv.assign(static_cast<std::size_t>(padded) * 6, nan);
    float A[9], R[9], M[9], Mi[9];
    ainvt(N, gh, gw, scale, A);
    for (int i = 0; i < ng; ++i) {
        rot2d(angles[i], R);
        mat3_mul(R, A, M);
        mat3_inv(M, Mi);
        std::copy(M, M + 6, L.T.begin() + 6 * i);
        std::copy(Mi, Mi + 6, L.Tinv.begin() + 6 * i);
    }
    return L;
}

}  // namespace

GhbpPlan make_ghbp_plan(const float* angles, int n_angles, int n_bins, const GhbpParams& p,
                        const float* axis_correction) {
    if (n_angles <= 0 || n_bins <= 0 || p.grid_size <= 1) throw Error("GHBP: invalid sizes");
    if (p.legs < 1 || p.reduction_steps < 1) throw Error("GHBP: legs and reduction_steps must be >= 1");
    GhbpPlan plan;
    plan.p = p;
    plan.n_angles = n_angles;
    plan.n_bins = n_bins;
    const int N = p.grid_size;

    // nabuxx receives -A; all its angle arithmetic is on that array
    std::vector<float> passed(n_angles);
    for (int k = 0; k < n_angles; ++k) passed[k] = -angles[k];

    // projection matrices P @ R(-w), w = -passed = A  (double, stored float)
    const double d = 1.0e6 * static_cast<double>(static_cast<float>(p.axis_source_meters)) /
                     static_cast<double>(p.voxel_size_microns);
    const double t = static_cast<double>(p.rot_center);
    plan.setups.resize(static_cast<std::size_t>(n_angles) * 8);
    for (int k = 0; k < n_angles; ++k) {
        const float w = -passed[k];
        const float rck = p.rot_center + (axis_correction ? axis_correction[k] : 0.0f);
        const double Pm[12] = {0.0, -t / d, 1.0, rck, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0 / d, 0.0, 1.0};
        const double cw = std::cos(w), sw = std::sin(w);  // float overloads, widened (as nabuxx)
        const double Rm[16] = {1, 0, 0, 0, 0, cw, sw, 0, 0, -sw, cw, 0, 0, 0, 0, 1};
        float* dst = &plan.setups[static_cast<std::size_t>(k) * 8];
        const int rows[2] = {0, 2};
        for (int ri = 0; ri < 2; ++ri) {
            const int r = rows[ri];
            for (int c = 0; c < 4; ++c)
                dst[ri * 4 + c] = static_cast<float>(Pm[r * 4 + 0] * Rm[0 * 4 + c] + Pm[r * 4 + 1] * Rm[1 * 4 + c] +
                                                      Pm[r * 4 + 2] * Rm[2 * 4 + c] + Pm[r * 4 + 3] * Rm[3 * 4 + c]);
        }
    }

    // hierarchy
    const auto [mn, mx] = std::minmax_element(passed.begin(), passed.end());
    float ang_range = std::abs(*mx - *mn) / static_cast<float>(n_angles);
    const int rf = std::max(1, static_cast<int>(std::ceil(std::pow(static_cast<double>(n_angles), 1.0 / p.reduction_steps))));
    ang_range *= static_cast<float>(rf);
    int ngt = static_cast<int>(std::ceil(n_angles / static_cast<double>(rf)));
    const int w0 = static_cast<int>(std::rint(2.0f * N * p.whf_x));
    const int h0 = static_cast<int>(std::ceil(ang_range * N * p.whf_y));

    std::vector<float> means = group_means(passed, rf, ngt);
    plan.levels.push_back(build_level(N, h0, w0, means, p.legs, 1.0f, rf));
    for (int iter = 0; iter < 100; ++iter) {
        if (static_cast<int>(std::ceil(ngt / static_cast<double>(rf))) < 20) break;
        ang_range *= static_cast<float>(rf);
        ngt = static_cast<int>(std::ceil(ngt / static_cast<double>(rf)));
        const int h = static_cast<int>(std::ceil(static_cast<double>(ang_range) * N * static_cast<double>(p.whf_y)));
        means = group_means(means, rf, ngt);
        plan.levels.push_back(build_level(N, h, w0, means, p.legs, 1.0f, rf));
    }
    plan.levels.push_back(build_level(N, N, N, std::vector<float>(p.legs, 0.0f), p.legs, 1.0f / p.fac, std::max(1, ngt)));
    return plan;
}

}  // namespace helirec
