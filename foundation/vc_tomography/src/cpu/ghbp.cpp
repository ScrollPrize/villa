// CPU GHBP. Per-pixel arithmetic is shared with the CUDA kernels (common/ghbp_core.hpp).
#include <algorithm>
#include <vector>

#include "common/ghbp_core.hpp"
#include "helirec/ghbp.hpp"

namespace helirec::cpu {

using namespace helirec::ghbp_core;

void ghbp(const GhbpPlan& plan, const float* sino, float* out) {
    const auto& P = plan.p;
    const int N = P.grid_size, nL = static_cast<int>(plan.levels.size());
    std::vector<std::vector<float>> buf(nL - 1);
    for (int l = 0; l + 1 < nL; ++l) {
        const auto& L = plan.levels[l];
        buf[l].resize(static_cast<std::size_t>(L.g) * L.h * L.w);
    }
    std::fill(out, out + static_cast<std::size_t>(N) * N, 0.f);

    for (int leg = 0; leg < P.legs; ++leg) {
        {  // level 0
            const auto& L = plan.levels[0];
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
            for (int gi = 0; gi < L.g; ++gi)
                for (int a = 0; a < L.h; ++a) {
                    const float* T = &L.T[static_cast<std::size_t>(leg * L.g + gi) * 6];
                    float* row = &buf[0][(static_cast<std::size_t>(gi) * L.h + a) * L.w];
                    const int first = (leg * L.g + gi) * L.red;
                    for (int b = 0; b < L.w; ++b) {
                        float Y, X;
                        grid_point(T, a, b, Y, X);
                        row[b] = hd::mul(P.scale_factor,
                                         level0_value(plan.setups.data(), sino, plan.n_bins, plan.n_angles, first,
                                                      L.red, Y, X));
                    }
                }
        }
        for (int l = 1; l < nL; ++l) {  // aggregation; the final level accumulates the legs
            const auto& L = plan.levels[l];
            const auto& Lp = plan.levels[l - 1];
            const bool last = l == nL - 1;
            float* dst = last ? out : buf[l].data();
            const float* src = buf[l - 1].data();
            const float* Tinv_leg = &Lp.Tinv[static_cast<std::size_t>(leg * Lp.g) * 6];
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
            for (int gi = 0; gi < L.g; ++gi)
                for (int a = 0; a < L.h; ++a) {
                    const float* T = &L.T[static_cast<std::size_t>(leg * L.g + gi) * 6];
                    if (hd::is_nan(T[0])) continue;
                    float* row = dst + (static_cast<std::size_t>(gi) * L.h + a) * L.w;
                    for (int b = 0; b < L.w; ++b) {
                        float Y, X;
                        grid_point(T, a, b, Y, X);
                        const float v = aggregate_value(Tinv_leg, src, Lp.w, Lp.h, Lp.g, gi, L.red, Y, X);
                        row[b] = (last && leg > 0) ? hd::add(row[b], v) : v;
                    }
                }
        }
    }
    if (P.clip_outer_circle) {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (!in_circle(i, j, N)) out[static_cast<std::size_t>(i) * N + j] = 0.f;
    }
}

}  // namespace helirec::cpu
