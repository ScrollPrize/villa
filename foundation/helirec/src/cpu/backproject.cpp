#include <cmath>
#include <vector>

#include "common/interp_core.hpp"
#include "helirec/kernels.hpp"

namespace helirec {

int interp_from_name(const std::string& name) {
    if (name == "linear") return interp_core::Linear;
    if (name == "cubic") return interp_core::Cubic;
    if (name == "lanczos3") return interp_core::Lanczos3;
    if (name == "bspline3" || name == "bspline") return interp_core::BSpline3;
    if (name == "bspline5") return interp_core::BSpline5;
    throw Error("unknown interpolation '" + name + "' (linear, cubic, lanczos3, bspline3, bspline5)");
}

namespace cpu {

namespace {
template <int K>
void backproject_k(const float* sino, const std::vector<float>& cs, const std::vector<float>& sn, float* slice,
                   int n_angles, int n_det, int ny, int nx, const BackprojectParams& p, float c, float r) {
#pragma omp parallel for schedule(dynamic, 4)
    for (int iy = 0; iy < ny; ++iy) {
        const float y = static_cast<float>(iy + p.y0) - c;
        for (int ix = 0; ix < nx; ++ix) {
            const float x = static_cast<float>(ix + p.x0) - c;
            float acc = 0.f;
            if (!p.clip_outer_circle || x * x + y * y <= r * r)
                for (int a = 0; a < n_angles; ++a)
                    acc += interp_core::sample_row<K>(sino + static_cast<std::size_t>(a) * n_det, n_det,
                                                      (x * cs[a] + y * sn[a] + p.rot_center) * p.u_scale);
            slice[static_cast<std::size_t>(iy) * nx + ix] = acc * p.scale;
        }
    }
}
}  // namespace

void backproject(const float* sino, const float* angles, float* slice, int n_angles, int n_det,
                 int ny, int nx, const BackprojectParams& p) {
    const float c = p.center >= 0.f ? p.center : 0.5f * static_cast<float>(std::max(nx, ny) - 1);
    const float r = p.clip_radius >= 0.f ? p.clip_radius : c;
    std::vector<float> cs(n_angles), sn(n_angles);
    for (int a = 0; a < n_angles; ++a) {
        cs[a] = std::cos(angles[a]);
        sn[a] = std::sin(angles[a]);
    }
    switch (p.interp) {
        case interp_core::Linear: backproject_k<interp_core::Linear>(sino, cs, sn, slice, n_angles, n_det, ny, nx, p, c, r); break;
        case interp_core::Cubic: backproject_k<interp_core::Cubic>(sino, cs, sn, slice, n_angles, n_det, ny, nx, p, c, r); break;
        case interp_core::Lanczos3: backproject_k<interp_core::Lanczos3>(sino, cs, sn, slice, n_angles, n_det, ny, nx, p, c, r); break;
        case interp_core::BSpline3: backproject_k<interp_core::BSpline3>(sino, cs, sn, slice, n_angles, n_det, ny, nx, p, c, r); break;
        case interp_core::BSpline5: backproject_k<interp_core::BSpline5>(sino, cs, sn, slice, n_angles, n_det, ny, nx, p, c, r); break;
        default: throw Error("backproject: unknown interpolation");
    }
}

}  // namespace cpu
}  // namespace helirec
