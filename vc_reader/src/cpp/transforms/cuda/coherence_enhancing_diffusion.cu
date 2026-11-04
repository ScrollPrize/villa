#include "channel.cuh"
#include "structure_gradients.cuh"
#include "../shared/coherence_enhancing_diffusion.hpp"

namespace vcr::transforms::cuda {
namespace {
template<class T> struct Product {
    __device__ void operator()(KernelIndex vcr_launch, const T* a, const T* b, T* out, std::size_t n) const {
        EACH_INDEX(n) out[i] = a[i]*b[i];
    }
};
template<class T> struct Tensor {
    __device__ void operator()(KernelIndex vcr_launch, T* tensor, std::size_t n, int d, T lambda, T exponent, T alpha) const {
        EACH_INDEX(n) diffusion_detail::diffusion_tensor(tensor,n,i,d,lambda,exponent,alpha);
    }
};
template<class T, class Grid> struct Flux {
    __device__ void operator()(KernelIndex vcr_launch, const T* input, const T* tensor, T* flux, Grid g) const {
        EACH_INDEX(g.size) diffusion_detail::flux(input,tensor,flux,g,i,g.dim);
    }
};
template<class T, class Grid> struct Update {
    __device__ void operator()(KernelIndex vcr_launch, T* input, const T* flux, Grid g, T dt) const {
        EACH_INDEX(g.size) input[i] = diffusion_detail::update(input,flux,g,i,g.dim,dt);
    }
};
}

void apply_cuda(const CoherenceEnhancingDiffusion& t, Sample& s, const Parameters& p, Session& session) {
    if (t.num_steps == 0) return;
    apply_channels(t,s,p,session,[](const auto& t, auto data, const Grid& g, const ChannelPlan&) {
        using T = typename decltype(data)::value_type;
        using namespace structure_tensor_detail;
        volume_detail::with_grid(g.shape,[&](const auto& layout) {
            using Grid = std::decay_t<decltype(layout)>;
            const auto st = diffusion_detail::tensor_options(t);
            const int d = int(g.shape.size()), count = component_count(d);
            const auto n = g.size;
            structure_detail::Gradients<T> gradients(st,g);
            const auto integration = integration_taps(st,d);
            Buffer<T> tensor(count*n), scratch(n), flux(d*n);
            for (int step = 0; step < t.num_steps; ++step) {
                gradients.compute(data.ptr,st,g);
                for (int c = 0; c < count; ++c) {
                    int a,b; component_axes(c,d,a,b);
                    T* value = tensor.ptr+c*n; T* work = scratch.ptr;
                    launch(blocks(n),256,0,vcr::device::stream(),Product<T>{},gradients.pointers[a],gradients.pointers[b],value,n);
                    filter_detail::separable_cuda(value,work,g,integration,st.boundary);
                    if (value != tensor.ptr+c*n)
                        check(copy_async(tensor.ptr+c*n,value,n*sizeof(T),cudaMemcpyDeviceToDevice,vcr::device::stream()));
                }
                launch(blocks(n),256,0,vcr::device::stream(),Tensor<T>{},tensor.ptr,n,d,T(t.lambda),T(t.exponent),T(t.alpha));
                launch(blocks(n),256,0,vcr::device::stream(),Flux<T,Grid>{},data.ptr,tensor.ptr,flux.ptr,layout);
                launch(blocks(n),256,0,vcr::device::stream(),Update<T,Grid>{},data.ptr,flux.ptr,layout,T(t.step_size));
            }
            check(cudaGetLastError());
        });
    });
}
}
