#pragma once
#include "../core.hpp"

namespace vcr::transforms {
// Nearest resize is an integer ratio: floor((x + .5) * input / output),
// or floor(x * input / output) for legacy sampling. Floating matrix/offset
// evaluation can round exact ties differently on CPU and CUDA (128 -> 109).
VCR_HD inline std::size_t resize_nearest_index(std::size_t x, std::size_t input,
                                              std::size_t output, bool legacy) {
    const auto limit = std::numeric_limits<std::size_t>::max();
    std::size_t index;
    if (output <= limit/2 && input <= limit/(2*output)) {
        index = legacy ? x*input/output : (2*x+1)*input/(2*output);
    } else {
        // Only enormous axes whose cross-product would overflow size_t need
        // the wider floating coordinate path.
        const double q = (double(x)+(legacy ? 0. : .5))*double(input)/double(output);
        index = q >= double(input-1) ? input-1 : std::size_t(q);
    }
    return index < input ? index : input-1;
}
} // namespace vcr::transforms
