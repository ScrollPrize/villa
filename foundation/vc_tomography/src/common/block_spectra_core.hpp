#pragma once
// Shared by the CPU and CUDA block cross-spectra and affine resampling (block_spectra.hpp).
#include <cmath>

#include "hd_math.hpp"
#include "interp_core.hpp"

namespace helirec::block_spectra_core {

// signed frequency index of transform index i of an even n: [-n / 2, n / 2)
HR_HD int signed_index(int i, int n) { return i < n / 2 ? i : i - n; }

// Bin (shell * thetas + theta) of the frequency index (kz, ky, kx). shell_scale = 2 shells / n; tan2: tan^2 of the
// upper edge of each theta bin but the last (theta from the z axis: theta >= edge <=> kr^2 >= kz^2 tan^2 edge).
// Integer arithmetic and single rounded operations only: the same bin on every backend.
HR_HD int bin_of(int kz, int ky, int kx, double shell_scale, const double* tan2, int thetas) {
    const long long r2 = static_cast<long long>(ky) * ky + static_cast<long long>(kx) * kx;
    const long long z2 = static_cast<long long>(kz) * kz;
    const int shell = static_cast<int>(sqrt(static_cast<double>(r2 + z2)) * shell_scale);
    int t = 0;
    while (t < thetas - 1 && static_cast<double>(r2) >= static_cast<double>(z2) * tan2[t]) ++t;
    return shell * thetas + t;
}

// One frequency of the half spectrum: its sums' weight (conjugate half: 2; kx = 0: 1; a Nyquist component: 0).
HR_HD double half_weight(int kz, int ky, int kx, int n) {
    const int h = n / 2;
    if (kz == -h || ky == -h || kx == h) return 0.0;
    return kx == 0 ? 1.0 : 2.0;
}

// -------------------------------------------------------------------------------------------- resampling
HR_HD int resample_taps(int interp) {
    return interp == interp_core::Linear ? 2 : (interp == interp_core::Cubic ? 4 : 6);
}

// weights of the taps floor(p) + lo + t, t = 0 .. taps - 1 (lo = 1 - taps / 2), for the fraction f = p - floor(p)
HR_HD void resample_weights(int interp, float f, float* w) {
    if (interp == interp_core::Linear) {
        w[0] = 1.f - f;
        w[1] = f;
    } else if (interp == interp_core::Cubic) {
        w[0] = interp_core::keys(f + 1.f);
        w[1] = interp_core::keys(f);
        w[2] = interp_core::keys(1.f - f);
        w[3] = interp_core::keys(2.f - f);
    } else {
        float sum = 0.f;
        for (int t = 0; t < 6; ++t) {
            w[t] = interp_core::lanczos3(f - static_cast<float>(t - 2));
            sum += w[t];
        }
        for (int t = 0; t < 6; ++t) w[t] /= sum;
    }
}

// the (nz, ny, nx) volume v at (pz, py, px); 0 outside it
HR_HD float sample_volume(const float* v, int nz, int ny, int nx, double pz, double py, double px, int interp) {
    const int T = resample_taps(interp), lo = 1 - T / 2;
    const double fz = floor(pz), fy = floor(py), fx = floor(px);
    // (far outside: nothing to sum, and the casts below stay in range)
    if (!(fz > -8.0 && fz < nz + 8.0 && fy > -8.0 && fy < ny + 8.0 && fx > -8.0 && fx < nx + 8.0)) return 0.f;
    float wz[6], wy[6], wx[6];
    resample_weights(interp, static_cast<float>(pz - fz), wz);
    resample_weights(interp, static_cast<float>(py - fy), wy);
    resample_weights(interp, static_cast<float>(px - fx), wx);
    const int z0 = static_cast<int>(fz) + lo, y0 = static_cast<int>(fy) + lo, x0 = static_cast<int>(fx) + lo;
    float acc = 0.f;
    for (int a = 0; a < T; ++a) {
        const int z = z0 + a;
        if (z < 0 || z >= nz) continue;
        float plane = 0.f;
        for (int b = 0; b < T; ++b) {
            const int y = y0 + b;
            if (y < 0 || y >= ny) continue;
            const float* row = v + (static_cast<size_t>(z) * ny + y) * nx;
            float line = 0.f;
            for (int c = 0; c < T; ++c) {
                const int x = x0 + c;
                if (x >= 0 && x < nx) line += wx[c] * row[x];
            }
            plane += wy[b] * line;
        }
        acc += wz[a] * plane;
    }
    return acc;
}

}  // namespace helirec::block_spectra_core
