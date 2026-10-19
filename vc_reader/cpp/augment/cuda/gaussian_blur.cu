#include "gaussian.cuh"
namespace vcr::augment::cuda {
template<class T> __device__ void convolution(KernelIndex vcr_launch, const T* src, T* dst, Layout g, int axis, const T* weights, int radius, Boundary mirror) {
    EACH_INDEX(g.size) {
        const auto x = g.coord(i, axis), base = i - x * g.stride[axis];
        T sum = 0;
        for (int j = -radius; j <= radius; ++j)
            sum += weights[j+radius] * src[base + boundary(static_cast<long long>(x)+j, g.shape[axis], mirror) * g.stride[axis]];
        dst[i] = sum;
    }
}
struct convolution_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { convolution(index,args...); }
};
// Extend a tile once, then reuse its samples for every tap. The contiguous
// axis uses a row; other axes use 32 adjacent lines so global loads coalesce.
// This removes per-output/per-tap boundary division and redundant global reads.
template<class T, class Index, bool Contiguous>
__device__ void convolution_tile(KernelIndex vcr_launch, const T* src, T* dst, Index extent, Index stride,
                                Index lines, const T* weights, int radius, Boundary edge) {
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
                const Index mapped = x >= 0 && x < extent ? Index(x) : Index(boundary(x,extent,edge));
                const Index base = (line / stride) * extent * stride + line % stride;
                tile[j] = src[base + mapped * stride];
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
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { convolution_tile<T,Index,Contiguous>(index,args...); }
};
template<class T, class Index> void launch_convolution_tile(const T* src, T* dst, const Grid& g,
                                                           int axis, const T* weights, int radius, Boundary edge) {
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
template<class T> void blur_impl(View<T> data, View<T> output, const Grid& grid, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) {
    require(sigmas.size() == grid.shape.size() && truncate > 0 && std::isfinite(truncate), "invalid Gaussian parameters");
    Layout g(grid); std::vector<std::vector<T>> kernels(g.dim); unsigned passes = 0;
    for (int a = 0; a < g.dim; ++a) {
        require(sigmas[a] >= 0 && std::isfinite(sigmas[a]) && sigmas[a]*truncate < 100000, "invalid Gaussian sigma");
        if (!sigmas[a]) continue;
        kernels[a] = gaussian_kernel<T>(sigmas[a],truncate,full);
        passes += kernels[a].size() > 1;
    }
    if (!passes) {
        if (output.ptr != data.ptr) copy_device(output,data);
        return;
    }
    // For separate output, choose the first buffer so the last pass lands in
    // output. Local smoothing/unsharp masking need neither an input copy nor
    // the odd-pass result copy required by an in-place blur.
    const bool inplace = data.ptr == output.ptr;
    Buffer<T> tmp(inplace || passes > 1 ? grid.size : 0);
    T* source = data.ptr;
    T* dest = inplace || passes % 2 == 0 ? tmp.ptr : output.ptr;
    for (int a = 0; a < g.dim; ++a) {
        const auto& host = kernels[a];
        if (host.empty()) continue;
        if (host.size() == 1) continue;
        Buffer<T> weights(host); const int radius = int(host.size()/2);
        // Bound shared memory independently of sigma. Wide filters retain the
        // general implementation; large arrays retain full-width indexing.
        if (radius <= 32) {
            // Leave headroom for the final partial tile's inactive threads.
            if (g.size <= 0xffffff00ull)
                launch_convolution_tile<T,unsigned>(source,dest,grid,a,weights.ptr,radius,mirror);
            else launch_convolution_tile<T,std::size_t>(source,dest,grid,a,weights.ptr,radius,mirror);
        } else launch(blocks(g.size),256,0,vcr::device::stream(), convolution_launch{}, source,dest,g,a,weights.ptr,radius,mirror);
        check(cudaGetLastError()); source = dest;
        dest = dest == output.ptr ? tmp.ptr : output.ptr;
    }
    if (source != output.ptr) copy_device(output,View<T>{source,data.size});
}
void blur(View<float> data, const Grid& g, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) { blur_impl(data,data,g,sigmas,truncate,mirror,full); }
void blur(View<double> data, const Grid& g, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) { blur_impl(data,data,g,sigmas,truncate,mirror,full); }
void blur_into(View<float> data, View<float> output, const Grid& g, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) { blur_impl(data,output,g,sigmas,truncate,mirror,full); }
void blur_into(View<double> data, View<double> output, const Grid& g, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) { blur_impl(data,output,g,sigmas,truncate,mirror,full); }
void apply_cuda(const GaussianBlur& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        blur(v,g,c.values,t.truncate,t.boundary,t.full_width);
    });
}
} // namespace vcr::augment::cuda
