#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void sharpen_kernel(const T* src, T* dst, Layout g, T strength, bool clip, const Statistics* stats) {
    EACH_INDEX(g.size) {
        T laplace = T(2*g.dim)*src[i];
        for (int a = 0; a < g.dim; ++a) {
            const auto x = g.coord(i,a), step = g.stride[a];
            laplace -= src[x ? i-step : i];
            laplace -= src[x+1 < g.shape[a] ? i+step : i];
        }
        T value = src[i] + strength * laplace;
        dst[i] = clip ? T(clamp_value(value,stats->low,stats->high)) : value;
    }
}
void apply_cuda(const Sharpen& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto&, auto v, const Grid& g, const ChannelPlan& c) {
        using V = typename decltype(v)::value_type;
        auto stats = c.values[1] ? bounds(v) : Buffer<Statistics>(0); Buffer<V> out(v.size);
        sharpen_kernel<<<blocks(g.size),256,0,vcr::device::stream()>>>(v.ptr,out.ptr,Layout(g),V(c.values[0]),bool(c.values[1]),stats.ptr);
        check(cudaGetLastError()); copy_device(v,View<V>{out.ptr,v.size});
    });
}
} // namespace vcr::augment::cuda
