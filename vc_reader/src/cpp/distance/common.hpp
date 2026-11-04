#pragma once
#include <vc_reader/distance_transform.hpp>
#include <limits>
#include <stdexcept>
namespace vcr::distance::detail {
inline std::size_t validate(const void* input, void* output, const std::vector<std::size_t>& shape) {
    if (shape.size() != 2 && shape.size() != 3) throw std::invalid_argument("expected a 2D or 3D grid");
    std::size_t n=1;
    for (auto d : shape) {
        if (n && d > std::numeric_limits<std::size_t>::max()/n) throw std::invalid_argument("grid dimensions overflow");
        n *= d;
    }
    if (n && (!input || !output)) throw std::invalid_argument("null grid input/output");
    return n;
}
#if defined(__CUDACC__)
#define VCR_DISTANCE_HD __host__ __device__
#else
#define VCR_DISTANCE_HD
#endif
// In-place forward/backward minimum scan; each line is independent.
VCR_DISTANCE_HD inline void line(std::uint8_t* f, std::size_t extent, std::size_t stride) {
    int previous=255;
    for (std::size_t q=0;q<extent;++q) {
        const int v=f[q*stride]; previous = v < previous+1 ? v : previous+1;
        f[q*stride]=static_cast<std::uint8_t>(previous);
    }
    previous=255;
    for (std::size_t q=extent;q-->0;) {
        const int v=f[q*stride]; previous = v < previous+1 ? v : previous+1;
        f[q*stride]=static_cast<std::uint8_t>(previous);
    }
}
#undef VCR_DISTANCE_HD
}
