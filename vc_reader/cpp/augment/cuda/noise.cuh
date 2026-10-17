#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
// SplitMix is indexed by each Box-Muller pair, preserving the CPU RNG stream.
// Each thread uses the same two uniforms, with the same precision as CPU.
__device__ inline double uniform_at(std::uint64_t state, std::uint64_t draw) {
    auto z = state + UINT64_C(0x9e3779b97f4a7c15) * (draw + 1);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (double((z ^ (z >> 31)) >> 11) + .5) * 0x1.0p-53;
}
template<class T> __global__ void noise_kernel(T* data, std::size_t n, double sigma, std::uint64_t seed, bool rician) {
    EACH_INDEX(rician ? n : (n + 1) / 2) {
        const T u = from_double<T>(uniform_at(seed, 2*i));
        const T angle = from_double<T>(6.2831853071795864769 * uniform_at(seed, 2*i+1));
        T radius, a, b;
        if constexpr (std::is_same_v<T, float>) {
            radius = sqrtf(T(-2) * logf(u)); a = radius * cosf(angle); b = radius * sinf(angle);
        } else { radius = sqrt(T(-2) * log(u)); a = radius * cos(angle); b = radius * sin(angle); }
        if (rician) { double re = data[i] + sigma * a, im = sigma * b; data[i] = T(sqrt(re*re + im*im)); }
        else {
            data[2*i] = from_double<T>(to_double(data[2*i]) + sigma*to_double(a));
            if (2*i+1 < n) data[2*i+1] = from_double<T>(to_double(data[2*i+1]) + sigma*to_double(b));
        }
    }
}

} // namespace vcr::augment::cuda
