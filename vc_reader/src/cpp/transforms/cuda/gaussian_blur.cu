#include "gaussian.cuh"
#include "separable_filter.cuh"
namespace vcr::transforms::cuda {
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
        filter_detail::convolve_axis(source,dest,grid,a,host,mirror);
        source = dest;
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
} // namespace vcr::transforms::cuda
