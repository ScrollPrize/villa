#include "gaussian.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void convolution(const T* src, T* dst, Layout g, int axis, const T* weights, int radius, Boundary mirror) {
    EACH_INDEX(g.size) {
        const auto x = g.coord(i, axis), base = i - x * g.stride[axis];
        T sum = 0;
        for (int j = -radius; j <= radius; ++j)
            sum += weights[j+radius] * src[base + boundary(static_cast<long long>(x)+j, g.shape[axis], mirror) * g.stride[axis]];
        dst[i] = sum;
    }
}
template<class T> void blur_impl(View<T> data, const Grid& grid, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) {
    require(sigmas.size() == grid.shape.size() && truncate > 0 && std::isfinite(truncate), "invalid Gaussian parameters");
    Buffer<T> tmp(grid.size); Layout g(grid); T* source = data.ptr; T* dest = tmp.ptr;
    for (int a = 0; a < g.dim; ++a) {
        require(sigmas[a] >= 0 && std::isfinite(sigmas[a]) && sigmas[a]*truncate < 100000, "invalid Gaussian sigma");
        if (!sigmas[a]) continue;
        auto host = gaussian_kernel<T>(sigmas[a],truncate,full); Buffer<T> weights(host);
        convolution<<<blocks(g.size),256,0,vcr::device::stream()>>>(source,dest,g,a,weights.ptr,int(host.size()/2),mirror);
        check(cudaGetLastError()); std::swap(source,dest);
    }
    if (source != data.ptr) copy_device(data,View<T>{source,data.size});
}
void blur(View<float> data, const Grid& g, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) { blur_impl(data,g,sigmas,truncate,mirror,full); }
void blur(View<double> data, const Grid& g, const std::vector<double>& sigmas, double truncate, Boundary mirror, bool full) { blur_impl(data,g,sigmas,truncate,mirror,full); }
void apply_cuda(const GaussianBlur& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        blur(v,g,c.values,t.truncate,t.boundary,true);
    });
}
} // namespace vcr::augment::cuda
