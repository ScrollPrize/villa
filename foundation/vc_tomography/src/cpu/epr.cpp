// Eikonal phase retrieval, CPU: Paganin estimate, then preconditioned fixed-point iterations against the
// second-order model of src/common/epr_core.hpp.
#include <algorithm>
#include <vector>

#include "common/epr_core.hpp"
#include "helirec/filters.hpp"

namespace helirec::cpu {

void epr_frames(float* frames, int n_frames, int H, int W, const EprParams& p, FourierFilter& paganin) {
    using namespace epr_core;
    const float A = static_cast<float>(p.shift_px2);
    const Coef c{A, 0.5f * A * A, static_cast<float>(p.shift_px2 * p.shift_px2 / (p.delta_beta * p.delta_beta)),
                 p.max_shift > 0.f ? p.max_shift / A : 0.f, p.eps};
    const std::size_t plane = static_cast<std::size_t>(H) * W;
    // frames per pass: six work planes each, about 1 GB in all
    const int batch = static_cast<int>(std::clamp<std::size_t>((std::size_t(1) << 30) / (6 * plane * sizeof(float)), 1,
                                                               static_cast<std::size_t>(std::max(1, n_frames))));
    const std::size_t cap = static_cast<std::size_t>(batch) * plane;
    std::vector<float> I(cap), gx, gy, q, D, r;
    if (p.iters > 0) {
        gx.resize(cap);
        gy.resize(cap);
        q.resize(cap);
        D.resize(cap);
        r.resize(cap);
    }
    for (int f0 = 0; f0 < n_frames; f0 += batch) {
        const int nb = std::min(batch, n_frames - f0);
        float* det = frames + static_cast<std::size_t>(f0) * plane;
        const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(nb) * static_cast<std::ptrdiff_t>(plane);
        paganin.apply(det, I.data(), nb);
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t k = 0; k < total; ++k) I[k] = floor_eps(I[k], c.eps);
        for (int it = 0; it < p.iters; ++it) {
#pragma omp parallel for collapse(2) schedule(static)
            for (int f = 0; f < nb; ++f)
                for (int y = 0; y < H; ++y) {
                    const std::size_t base = static_cast<std::size_t>(f) * plane;
                    for (int x = 0; x < W; ++x) {
                        const std::size_t k = base + at(y, x, W);
                        gradient(I.data() + base, H, W, y, x, c, gx[k], gy[k], q[k]);
                    }
                }
#pragma omp parallel for collapse(2) schedule(static)
            for (int f = 0; f < nb; ++f)
                for (int y = 0; y < H; ++y) {
                    const std::size_t base = static_cast<std::size_t>(f) * plane;
                    for (int x = 0; x < W; ++x)
                        D[base + at(y, x, W)] = flux_divergence(I.data() + base, gx.data() + base, gy.data() + base, H, W, y, x);
                }
#pragma omp parallel for collapse(2) schedule(static)
            for (int f = 0; f < nb; ++f)
                for (int y = 0; y < H; ++y) {
                    const std::size_t base = static_cast<std::size_t>(f) * plane;
                    for (int x = 0; x < W; ++x) {
                        const std::size_t k = base + at(y, x, W);
                        r[k] = floor_eps(det[k], c.eps) - forward(I.data() + base, gx.data() + base, gy.data() + base,
                                                                  q.data() + base, D.data() + base, H, W, y, x, c);
                    }
                }
            paganin.apply(r.data(), r.data(), nb);
#pragma omp parallel for schedule(static)
            for (std::ptrdiff_t k = 0; k < total; ++k) I[k] = floor_eps(I[k] + p.relax * r[k], c.eps);
        }
        std::copy(I.begin(), I.begin() + total, det);
    }
}

}  // namespace helirec::cpu
