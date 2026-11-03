#pragma once
#include "common.cuh"
namespace vcr::transforms::cuda {
// Match scalar CPU parameter sampling, including rejection of biased residues.
// The caller handles an empty population before requesting an index.
__device__ inline std::size_t random_index(Random& random,std::size_t n) {
    const auto threshold=(std::uint64_t(0)-n)%n;
    std::uint64_t value;
    do { value=random.next(); } while (value<threshold);
    return value%n;
}
}
