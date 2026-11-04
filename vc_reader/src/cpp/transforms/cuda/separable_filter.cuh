#pragma once
#include "resident.cuh"
#include "../shared/separable_filter.hpp"

namespace vcr::transforms::cuda::filter_detail {
using namespace vcr::transforms::filter_detail;
template<class T> __device__ void convolution(KernelIndex vcr_launch, const T* src, T* dst, Layout g, int axis, const T* weights, int radius, Boundary mirror) {
    EACH_INDEX(g.size) {
        const auto x = g.coord(i, axis), base = i - x * g.stride[axis];
        T sum = 0;
        for (int j = -radius; j <= radius; ++j) {
            const auto at = neighbour(static_cast<long long>(x)+j, g.shape[axis], mirror);
            if (at >= 0) sum += weights[j+radius] * src[base + std::size_t(at) * g.stride[axis]];
        }
        dst[i] = sum;
    }
}
struct convolution_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { convolution(index,args...); }
};
// Extend a tile once, then reuse its samples for every tap. The contiguous
// axis uses a row; other axes use 32 adjacent lines so global loads coalesce.
// This removes per-output/per-tap boundary division and redundant global reads.
template<class T, class Index, bool Contiguous, class Weights>
__device__ void convolution_tile(KernelIndex vcr_launch, const T* src, T* dst, Index extent, Index stride,
                                Index lines, const Weights& weights, int radius, Boundary edge) {
    extern __shared__ __align__(16) unsigned char storage[];
    T* tile = reinterpret_cast<T*>(storage);
    constexpr unsigned width = Contiguous ? 128 : 8;
    constexpr unsigned lanes = Contiguous ? 1 : 32;
    const Index chunks = (extent - 1) / width + 1;
    const std::size_t groups = (std::size_t(lines) - 1) / lanes + 1;
    const std::size_t count = groups * chunks;
    for (std::size_t block = vcr_launch.block; block < count; block += vcr_launch.blocks) {
        const Index tile_index = Index(block);
        const Index start = (tile_index % chunks) * width;
        const Index first_line = (tile_index / chunks) * lanes;
        const unsigned loaded = (width + 2 * radius) * lanes;
        for (unsigned j = threadIdx.x; j < loaded; j += blockDim.x) {
            const Index line = first_line + j % lanes;
            const long long x = static_cast<long long>(start) + j / lanes - radius;
            if (line < lines) {
                const auto mapped = neighbour(x, extent, edge);
                const Index base = (line / stride) * extent * stride + line % stride;
                tile[j] = mapped < 0 ? T(0) : src[base + Index(mapped) * stride];
            } else tile[j] = 0;
        }
        __syncthreads();
        const Index line = first_line + threadIdx.x % lanes;
        const Index x = start + threadIdx.x / lanes;
        if (line < lines && x < extent) {
            T sum = 0;
            for (int j = 0; j <= 2 * radius; ++j)
                sum += weights[j] * tile[threadIdx.x + j * lanes];
            const Index base = (line / stride) * extent * stride + line % stride;
            dst[base + x * stride] = sum;
        }
        __syncthreads();
    }
}
template<class T, class Index, bool Contiguous> struct convolution_tile_launch3 {
    static constexpr bool inline_arguments = true;
    template<class... Args> __device__ void operator()(KernelIndex index, const Args&... args) const { convolution_tile<T,Index,Contiguous>(index,args...); }
};
template<class T, class Index, class Weights> void launch_convolution_tile(const T* src, T* dst, const Grid& g,
                                                                         int axis, Weights weights, int radius, Boundary edge) {
    const Index extent = Index(g.shape[axis]), stride = Index(g.stride[axis]);
    const Index lines = Index(g.size / extent);
    const bool contiguous = stride == 1;
    const auto count = contiguous ? std::size_t(lines) * ((extent - 1) / 128 + 1)
                                 : ((std::size_t(lines) - 1) / 32 + 1) * ((extent - 1) / 8 + 1);
    const unsigned blocks = unsigned(std::min<std::size_t>(count,65535));
    if (contiguous)
        launch(blocks,128,(128+2*radius)*sizeof(T),vcr::device::stream(), convolution_tile_launch3<T,Index,true>{}, src,dst,extent,stride,lines,weights,radius,edge);
    else
        launch(blocks,256,(8+2*radius)*32*sizeof(T),vcr::device::stream(), convolution_tile_launch3<T,Index,false>{}, src,dst,extent,stride,lines,weights,radius,edge);
}
template<class T> struct InlineWeights {
    T values[17]{};
    __device__ T operator[](int i) const { return values[i]; }
};
template<class T> void convolve_axis(const T* source, T* dest, const Grid& grid, int axis,
                                     const std::vector<T>& host, Boundary edge) {
    const Layout g(grid);
    const int radius = int(host.size()/2);
    // Bound shared memory independently of the tap count. Wide filters retain the
    // general implementation; large arrays retain full-width indexing.
    if (radius <= 32) {
        const auto tiled=[&](auto weights) {
            // Leave headroom for the final partial tile's inactive threads.
            if (g.size <= 0xffffff00ull)
                launch_convolution_tile<T,unsigned>(source,dest,grid,axis,weights,radius,edge);
            else launch_convolution_tile<T,std::size_t>(source,dest,grid,axis,weights,radius,edge);
        };
        if (radius <= 8) {
            // Short filters fit in the launch packet, avoiding a separate
            // device allocation and host upload for every sample and axis.
            InlineWeights<T> weights;
            std::copy(host.begin(),host.end(),weights.values);
            tiled(weights);
        } else {
            Buffer<T> weights(host);
            tiled(weights.ptr);
        }
    } else {
        Buffer<T> weights(host);
        launch(blocks(g.size),256,0,vcr::device::stream(), convolution_launch{}, source,dest,g,axis,weights.ptr,radius,edge);
    }
    check(cudaGetLastError());
}
// In-place separable filter through one scratch buffer of the same size.
template<class T> void separable_cuda(T*& data, T*& scratch, const Grid& grid, const std::vector<Taps>& taps, Boundary edge) {
    for (std::size_t a = 0; a < taps.size(); ++a) {
        if (taps[a].radius == 0 && taps[a].weights[0] == 1) continue;
        convolve_axis(data,scratch,grid,int(a),std::vector<T>(taps[a].weights.begin(), taps[a].weights.end()),edge);
        std::swap(data,scratch);
    }
}
} // namespace vcr::transforms::cuda::filter_detail
