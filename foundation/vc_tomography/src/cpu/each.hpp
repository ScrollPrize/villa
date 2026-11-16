#pragma once
#include <cstddef>

namespace vc_tomography::cpu {

// f(i) for every i in [0, n), on OpenMP's threads
template <class F>
void each(std::size_t n, const F& f) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) f(static_cast<std::size_t>(i));
}

}  // namespace vc_tomography::cpu
