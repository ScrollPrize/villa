#include <algorithm>
#include <cmath>
#include <vector>

#include "common/interp_core.hpp"
#include "helirec/band.hpp"

namespace helirec::cpu {

namespace {
inline int lower(float v, int n) { return std::min(static_cast<int>(std::floor(v)), n - 2); }
}  // namespace

void band_mark(const float* y, const float* x, std::size_t a, std::size_t b, const BandBox& box, uint8_t* mask) {
    for (std::size_t k = a; k < b; ++k) {
        const int iy = lower(y[k], box.n_grid) - box.y0, ix = lower(x[k], box.n_grid) - box.x0;
        uint8_t* m = mask + static_cast<std::size_t>(iy) * box.w + ix;
        m[0] = m[1] = m[box.w] = m[box.w + 1] = 1;
    }
}

std::size_t band_compact(const uint8_t* mask, std::size_t n, int32_t* list) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (mask[i]) list[c++] = static_cast<int32_t>(i);
    return c;
}

namespace {
template <int K>
void backproject_list_k(const float* sino, const std::vector<float>& cs, const std::vector<float>& sn, int n_angles,
                        int n_det, const int32_t* list, std::size_t count, const BandBox& box, const BackprojectParams& p,
                        float c, float r, float* img) {
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t k = 0; k < count; ++k) {
        const int li = list[k];
        const float x = static_cast<float>(li % box.w + box.x0) - c;
        const float y = static_cast<float>(li / box.w + box.y0) - c;
        float acc = 0.f;
        if (!p.clip_outer_circle || x * x + y * y <= r * r)
            for (int a = 0; a < n_angles; ++a)
                acc += interp_core::sample_row<K>(sino + static_cast<std::size_t>(a) * n_det, n_det,
                                                  (x * cs[a] + y * sn[a] + p.rot_center) * p.u_scale);
        img[li] = acc * p.scale;
    }
}
}  // namespace

void backproject_list(const float* sino, const float* angles, int n_angles, int n_det, const int32_t* list,
                      std::size_t count, const BandBox& box, const BackprojectParams& p, float* img) {
    const float c = p.center >= 0.f ? p.center : 0.5f * static_cast<float>(box.n_grid - 1);
    const float r = p.clip_radius >= 0.f ? p.clip_radius : c;
    std::vector<float> cs(n_angles), sn(n_angles);
    for (int a = 0; a < n_angles; ++a) {
        cs[a] = std::cos(angles[a]);
        sn[a] = std::sin(angles[a]);
    }
    switch (p.interp) {
        case interp_core::Linear: backproject_list_k<interp_core::Linear>(sino, cs, sn, n_angles, n_det, list, count, box, p, c, r, img); break;
        case interp_core::Cubic: backproject_list_k<interp_core::Cubic>(sino, cs, sn, n_angles, n_det, list, count, box, p, c, r, img); break;
        case interp_core::Lanczos3: backproject_list_k<interp_core::Lanczos3>(sino, cs, sn, n_angles, n_det, list, count, box, p, c, r, img); break;
        case interp_core::BSpline3: backproject_list_k<interp_core::BSpline3>(sino, cs, sn, n_angles, n_det, list, count, box, p, c, r, img); break;
        case interp_core::BSpline5: backproject_list_k<interp_core::BSpline5>(sino, cs, sn, n_angles, n_det, list, count, box, p, c, r, img); break;
        default: throw Error("backproject_list: unknown interpolation");
    }
}

void band_sample(const float* img_lo, const float* img_hi, const float* fz, const float* y, const float* x,
                 std::size_t a, std::size_t b, const BandBox& box, float* out) {
#pragma omp parallel for schedule(static)
    for (std::size_t k = a; k < b; ++k) {
        const int iy = lower(y[k], box.n_grid), ix = lower(x[k], box.n_grid);
        const float wy = y[k] - static_cast<float>(iy), wx = x[k] - static_cast<float>(ix);
        const std::size_t o = static_cast<std::size_t>(iy - box.y0) * box.w + (ix - box.x0);
        auto bil = [&](const float* im) {
            const float a0 = im[o] + wx * (im[o + 1] - im[o]);
            const float a1 = im[o + box.w] + wx * (im[o + box.w + 1] - im[o + box.w]);
            return a0 + wy * (a1 - a0);
        };
        const float lo = bil(img_lo), hi = bil(img_hi);
        out[k] = lo + fz[k] * (hi - lo);
    }
}

}  // namespace helirec::cpu
