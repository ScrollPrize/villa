#include <algorithm>
#include <cmath>

#include "helirec/kernels.hpp"

namespace helirec::cpu {

void flatfield(const float* raw, const float* dark, const float* flat, float* out,
               std::size_t n_proj, std::size_t n_pix, const FlatFieldParams& p) {
    const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(n_proj * n_pix);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t k = 0; k < total; ++k) {
        const std::size_t i = static_cast<std::size_t>(k) % n_pix;
        float v = (raw[k] - dark[i]) / (flat[i] - dark[i]);
        v = std::max(v, p.threshold);  // also maps NaN (0/0) to threshold
        out[k] = p.take_log ? -std::log(v) : v;
    }
}

}  // namespace helirec::cpu
