#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void finish_gamma(T* data, std::size_t n, double m, double scale, double offset, bool retain, bool invert) {
    EACH_INDEX(n) {
        T x = data[i];
        if (retain) x = T((x-m)*scale+offset);
        if (invert) x = -x;
        data[i] = x;
    }
}

template<class T> __global__ void gamma_kernel(T* data, std::size_t n, double low, double range, double gamma, bool invert) {
    EACH_INDEX(n) {
        double x = invert ? -data[i] : data[i];
        data[i] = T(power<T>((x-low)/fmax(range,1e-7),gamma)*range+low);
    }
}
void apply_cuda(const Gamma& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto&, auto& v, const Grid& g, const ChannelPlan& c) {
        using V = typename std::decay_t<decltype(v)>::value_type;
        Buffer<V> data(v);
        bool invert = bool(c.values[1]), retain = bool(c.values[2]);
        if (invert) for (auto& x : v) x = -x;
        const double before_mean = retain ? mean(v) : 0;
        const double before_std = retain ? deviation(v,before_mean) : 0;
        auto bounds = std::minmax_element(v.begin(),v.end());
        gamma_kernel<<<blocks(g.size),256>>>(data.ptr,g.size,double(*bounds.first),double(*bounds.second)-double(*bounds.first),c.values[0],invert);
        check(cudaGetLastError());
        double after_mean = 0, scale = 0;
        if (retain) { data.download(v.data()); after_mean = mean(v); scale = before_std/std::max(deviation(v,after_mean),1e-7); }
        finish_gamma<<<blocks(g.size),256>>>(data.ptr,g.size,after_mean,scale,before_mean,retain,invert);
        check(cudaGetLastError()); data.download(v.data());
    });
}
} // namespace vcr::augment::cuda
