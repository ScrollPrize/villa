#include "../common/flatfield_core.hpp"
#include "vc_tomography/kernels.hpp"

namespace vc_tomography::cpu {

void flatfield(const float* raw, const float* dark, const float* flat, float* out,
               std::size_t n_proj, std::size_t n_pix, const FlatFieldParams& p) {
    const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(n_proj * n_pix);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t k = 0; k < total; ++k) {
        const std::size_t i = static_cast<std::size_t>(k) % n_pix;
        out[k] = flatfield_core::value(raw[k], dark[i], flat[i], p.threshold, p.take_log);
    }
}

}  // namespace vc_tomography::cpu
