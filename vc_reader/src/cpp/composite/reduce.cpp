// CPU reductions of sampled layer stacks.
#include "reducer.hpp"

#include <stdexcept>

namespace vcr::composite {

void detail::checkLayers(std::size_t layers, const Options& options)
{
    if (options.method == Method::Median && layers > kMaxMedianLayers)
        throw std::invalid_argument("median composites take at most 256 layers");
}

void reduce(const float* values, std::size_t layers, std::size_t n, const Options& options, float* out)
{
    detail::checkLayers(layers, options);
    for (std::size_t p = 0; p < n; ++p) {
        detail::Reducer r(options);
        for (std::size_t k = 0; k < layers && !r.done; ++k)
            r.add(values[k * n + p]);
        out[p] = r.result();
    }
}

}  // namespace vcr::composite
