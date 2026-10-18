#pragma once
#include "resident.cuh"
#include "../separable_filter.hpp"

namespace vcr::augment::cuda::filter_detail {
using namespace vcr::augment::filter_detail;
// Correlation along one axis with shared taps/boundary rules; accumulates in T.
template<class T> __global__ void correlate_kernel(const T* in, T* out, Layout g, int axis, const T* taps, int radius, Boundary edge) {
    EACH_INDEX(g.size) {
        const auto x = (long long)g.coord(i,axis); const auto stride = g.stride[axis];
        const auto base = i - std::size_t(x)*stride;
        T sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            const auto j = neighbour(x+k,(long long)g.shape[axis],edge);
            if (j >= 0) sum += T(taps[k+radius])*in[base+std::size_t(j)*stride];
        }
        out[i] = sum;
    }
}
// In-place separable filter through one scratch buffer of the same size.
template<class T> void separable_cuda(T*& data, T*& scratch, const Grid& grid, const std::vector<Taps>& taps, Boundary edge) {
    Layout g(grid);
    for (std::size_t a = 0; a < taps.size(); ++a) {
        if (taps[a].radius == 0 && taps[a].weights[0] == 1) continue;
        Buffer<T> weights(std::vector<T>(taps[a].weights.begin(), taps[a].weights.end()));
        correlate_kernel<<<blocks(g.size),256,0,vcr::device::stream()>>>(data,scratch,g,int(a),weights.ptr,taps[a].radius,edge);
        check(cudaGetLastError()); std::swap(data,scratch);
    }
}
} // namespace vcr::augment::cuda::filter_detail
