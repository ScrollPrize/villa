#include "voxel_transform.cuh"
#include "sort_keys.cuh"
namespace vcr::transforms::cuda {
namespace {
template<class T> __device__ void clip_bounds(KernelIndex vcr_launch, const T* sorted, std::size_t n, double lower, double upper, double* bounds) {
    const auto percentile = [&](double q) {
        const double pos = q / 100 * (n - 1); const auto lo = std::size_t(pos), hi = lo + 1 < n ? lo + 1 : lo;
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    };
    bounds[0] = percentile(lower); bounds[1] = percentile(upper);
}
struct clip_bounds_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { clip_bounds(index,args...); }
};
template<class T> __device__ void clip_values(KernelIndex vcr_launch, const T* input, T* output, std::size_t n, const double* bounds) {
    EACH_INDEX(n) output[i] = fmin(bounds[1],fmax(bounds[0],double(input[i])));
}
struct clip_values_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { clip_values(index,args...); }
};
template<class T> __device__ void retain_deviation(KernelIndex vcr_launch, T* output, std::size_t n, const Statistics* before, const Statistics* after) {
    const auto a = *after;
    if (a.deviation > 1e-8) { EACH_INDEX(n) output[i] = (output[i]-a.mean)/a.deviation*before->deviation+a.mean; }
}
struct retain_deviation_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { retain_deviation(index,args...); }
};
}
void apply_cuda(const ClipOutliers& t, Sample& s, const Parameters& p, Session& session) {
    with_voxel_channels(t,s,p,session,[&](const Grid&,const ChannelPlan& c,auto input,auto output,const auto*,int*) {
        using T=typename decltype(input)::value_type;
        Buffer<T> keys(input.size);Buffer<double> bounds(2); copy_device(View<T>{keys.ptr,keys.size},input); sort_keys(keys,keys.size);
        launch(1,1,0,vcr::device::stream(), clip_bounds_launch{}, keys.ptr,keys.size,c.values[0],c.values[1],bounds.ptr);
        launch(blocks(input.size),256,0,vcr::device::stream(), clip_values_launch{}, input.ptr,output.ptr,input.size,bounds.ptr);
        if (c.values[2]) {
            auto before = statistics(input,true), after = statistics(output,true);
            launch(blocks(input.size),256,0,vcr::device::stream(), retain_deviation_launch{}, output.ptr,output.size,before.ptr,after.ptr);
        }
        check(cudaGetLastError());
    });
}
}
