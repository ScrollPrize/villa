#pragma once
#include "channel.cuh"
#include "../random_sampling.hpp"
namespace vcr::augment::cuda {
// SplitMix is indexed by each Box-Muller pair, preserving the CPU RNG stream.
// Each thread uses the same two uniforms, with the same precision as CPU.
__device__ inline double uniform_at(std::uint64_t state, std::uint64_t draw) {
    return volume_detail::uniform(state, draw);
}
template<class T> __device__ void noise_kernel(KernelIndex vcr_launch, T* data, std::size_t n, double sigma, std::uint64_t seed, bool rician) {
    EACH_INDEX(rician ? n : (n + 1) / 2) {
        const T u = from_double<T>(uniform_at(seed, 2*i));
        const T angle = T(6.2831853071795864769) * from_double<T>(uniform_at(seed, 2*i+1));
        T radius, a, b;
        if constexpr (std::is_same_v<T, float>) {
            radius = sqrtf(T(-2) * logf(u)); a = radius * cosf(angle); b = radius * sinf(angle);
        } else { radius = sqrt(T(-2) * log(u)); a = radius * cos(angle); b = radius * sin(angle); }
        const auto add = [&](T x,T noise) {
            T y=x+T(sigma)*noise;
            if constexpr(std::is_same_v<T,float>) if(!isfinite(y)) y=T(widen(x)+sigma*widen(noise));
            return y;
        };
        if (rician) {
            const T re=add(data[i],a), im=T(sigma)*b;
            // hypot rescales its operands, avoiding overflow of re*re + im*im.
            data[i]=hypot(re,im);
        } else {
            data[2*i]=add(data[2*i],a);
            if(2*i+1<n)data[2*i+1]=add(data[2*i+1],b);
        }
    }
}
struct noise_kernel_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { noise_kernel(index,args...); }
};

} // namespace vcr::augment::cuda
