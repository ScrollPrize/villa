#pragma once
#include <cuda_runtime.h>
namespace vcr::augment::cuda {
template<int Metrics, class Real> __device__ void block_reduce(Real (&values)[Metrics][256]) {
    const int t = threadIdx.x;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (t < offset) {
            #pragma unroll
            for (int m = 0; m < Metrics; ++m) {
                if constexpr (Metrics == 3) values[m][t] += values[m][t+offset];
                else if (m == 0) values[m][t] += values[m][t+offset];
                else values[m][t] = fmax(values[m][t], values[m][t+offset]);
            }
        }
        __syncthreads();
    }
}
template<int Metrics, bool Mean = false, class Real> __global__ void finish_reduce(const Real* partial, Real* stats, int blocks, std::size_t count = 1) {
    __shared__ Real values[Metrics][256];
    const int t = threadIdx.x;
    #pragma unroll
    for (int m = 0; m < Metrics; ++m) {
        Real value = 0;
        for (int i = t; i < blocks; i += 256) {
            if constexpr (Metrics == 3) value += partial[i*Metrics+m];
            else if (m == 0) value += partial[i*Metrics+m];
            else value = fmax(value, partial[i*Metrics+m]);
        }
        values[m][t] = value;
    }
    block_reduce(values);
    if (t == 0) for (int m = 0; m < Metrics; ++m) {
        if constexpr (Mean) stats[m] = values[m][0]/Real(count);
        else stats[m] = values[m][0];
    }
}
}
