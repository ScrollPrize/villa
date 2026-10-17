#pragma once
#include "common.cuh"
namespace vcr::augment::cuda {
struct Partial { double sum, weight, low, high; };
__device__ inline void reduce(Partial* values) {
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) {
            auto& a = values[threadIdx.x]; const auto& b = values[threadIdx.x+offset];
            a.sum += b.sum; a.weight += b.weight;
            a.low = fmin(a.low,b.low); a.high = fmax(a.high,b.high);
        }
        __syncthreads();
    }
}
} // namespace vcr::augment::cuda
