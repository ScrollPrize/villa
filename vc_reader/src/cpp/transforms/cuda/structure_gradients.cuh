#pragma once
#include "separable_filter.cuh"
#include "../cpu/structure_tensor.hpp"

namespace vcr::transforms::cuda::structure_detail {
using namespace structure_tensor_detail;
using filter_detail::separable_cuda;
template<class T, class S> __device__ void scale_copy(KernelIndex vcr_launch, const S* in, T* out, std::size_t n, double scale) {
    EACH_INDEX(n) out[i] = T(double(in[i])*scale);
}
struct scale_copy_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { scale_copy(index,args...); }
};

// Reused by structure-tensor output and iterative diffusion. Volume buffers and
// derivative taps are allocated once; all filtering stays on the caller's stream.
template<class T> struct Gradients {
    Buffer<T> base, scratch, local;
    std::vector<Buffer<T>> gradient;
    std::vector<T*> pointers;
    std::unique_ptr<Buffer<T*>> table;
    std::vector<std::vector<Taps>> taps;
    std::vector<Taps> smoothing;
    Gradients(const StructureTensor& t, const Grid& grid)
        : base(grid.size), scratch(grid.size), local(grid.size),
          taps(gradient_taps(t,grid.shape.size())) {
        if (t.derivative == Derivative::Pavel && t.sigma > 0)
            smoothing.assign(grid.shape.size(),torch_gaussian(t.sigma));
        for (std::size_t a = 0; a < grid.shape.size(); ++a) {
            gradient.emplace_back(grid.size); pointers.push_back(gradient.back().ptr);
        }
        table = std::make_unique<Buffer<T*>>(pointers);
    }
    template<class S> void compute(const S* input, const StructureTensor& t, const Grid& grid) {
        const auto n = grid.size;
        launch(blocks(n),256,0,vcr::device::stream(),scale_copy_launch{},input,base.ptr,n,t.scale);
        T* data = base.ptr; T* tmp = scratch.ptr;
        separable_cuda(data,tmp,grid,smoothing,t.boundary);
        for (std::size_t a = 0; a < gradient.size(); ++a) {
            check(copy_async(gradient[a].ptr,data,n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            T* g = gradient[a].ptr; T* sc = local.ptr;
            separable_cuda(g,sc,grid,taps[a],t.boundary);
            if (g != gradient[a].ptr)
                check(copy_async(gradient[a].ptr,g,n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
        }
    }
};
}
