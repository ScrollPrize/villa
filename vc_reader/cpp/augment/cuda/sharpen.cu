#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void sharpen_kernel(const T* src, T* dst, Layout g, T strength, bool clip, T low, T high) {
    EACH_INDEX(g.size) {
        T laplace = T(2*g.dim)*src[i];
        for (int a = 0; a < g.dim; ++a) {
            const auto x = g.coord(i,a), step = g.stride[a];
            laplace -= src[x ? i-step : i];
            laplace -= src[x+1 < g.shape[a] ? i+step : i];
        }
        T value = src[i] + strength * laplace;
        dst[i] = clip ? T(clamp_value(value,low,high)) : value;
    }
}
void apply_cuda(const Sharpen& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto&, auto& v, const Grid& g, const ChannelPlan& c) {
        using V = typename std::decay_t<decltype(v)>::value_type;
        auto bounds = std::minmax_element(v.begin(),v.end()); Buffer<V> data(v), out(v.size());
        sharpen_kernel<<<blocks(g.size),256>>>(data.ptr,out.ptr,Layout(g),V(c.values[0]),bool(c.values[1]),*bounds.first,*bounds.second);
        check(cudaGetLastError()); out.download(v.data());
    });
}

} // namespace vcr::augment::cuda
