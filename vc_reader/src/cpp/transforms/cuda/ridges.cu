#include "channel.cuh"
#include "../cpu/ridges.hpp"
namespace vcr::transforms::cuda {
namespace {
template<class T> __device__ void gradient(KernelIndex vcr_launch,const T* in,T* out,Layout g,int a) {
    EACH_INDEX(g.size)out[i]=ridge_detail::derivative(in,i,g.coord(i,a),g.shape[a],g.stride[a]);
}
struct gradient_launch {template<class... A> __device__ void operator()(KernelIndex i,A... a)const{gradient(i,a...);}};
template<class T> __device__ void response(KernelIndex vcr_launch,const T* h,T* out,std::size_t n,int d,double sigma,double gamma,double b1,double b2) {
    EACH_INDEX(n) {
        double m[9];for(int c=0;c<structure_tensor_detail::component_count(d);++c){int a,b;structure_tensor_detail::component_axes(c,d,a,b);m[a*d+b]=m[b*d+a]=h[c*n+i];}
        out[i]=T(ridge_detail::response(m,d,sigma,gamma,b1,b2));
    }
}
struct response_launch {template<class... A> __device__ void operator()(KernelIndex i,A... a)const{response(i,a...);}};
}
void apply_cuda(const Ridges& t,Sample& s,const Parameters& p,Session& session) {
    apply_channels(t,s,p,session,[](const auto& t,auto data,const Grid& g,const ChannelPlan&) {
        using T=typename decltype(data)::value_type;const int d=int(g.shape.size());
        for(auto n:g.shape)require(n>=3,"Ridges needs at least three samples on each spatial axis");
        Buffer<T> first(g.size),h(structure_tensor_detail::component_count(d)*g.size);
        for(int q=0;q<d;++q) {
            launch(blocks(g.size),256,0,vcr::device::stream(),gradient_launch{},data.ptr,first.ptr,Layout(g),q);
            for(int c=0;c<structure_tensor_detail::component_count(d);++c) {
                int a,b;structure_tensor_detail::component_axes(c,d,a,b);if(b!=q)continue;
                launch(blocks(g.size),256,0,vcr::device::stream(),gradient_launch{},first.ptr,h.ptr+c*g.size,Layout(g),a);
            }
        }
        launch(blocks(g.size),256,0,vcr::device::stream(),response_launch{},h.ptr,data.ptr,g.size,d,t.sigma,t.gamma,t.beta1,t.beta2);
        check(cudaGetLastError());
    });
}
}
