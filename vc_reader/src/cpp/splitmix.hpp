#pragma once
// SplitMix64 (Steele, Lea and Flood, 2014), the counter-based generator behind
// seeded vc_reader code on CPU and CUDA. The stream of seed s has outputs
// mix(s + k * increment) for k = 1, 2, ...; any output can be computed directly.
#include <cstdint>

#if defined(__CUDACC__)
#define VCR_SPLITMIX_HD __host__ __device__ inline
#else
#define VCR_SPLITMIX_HD inline
#endif

namespace vcr::splitmix {
constexpr std::uint64_t increment = UINT64_C(0x9e3779b97f4a7c15);

VCR_SPLITMIX_HD std::uint64_t mix(std::uint64_t z)
{
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// Advances `state` by one step and returns that step's output.
VCR_SPLITMIX_HD std::uint64_t next(std::uint64_t& state) { return mix(state += increment); }
}  // namespace vcr::splitmix
