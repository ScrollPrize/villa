#include "voxel_transform.cuh"
#include "sort_keys.cuh"
namespace vcr::augment::cuda {
namespace {
template<class T> __global__ void clip_bounds(const T* sorted, std::size_t n, double lower, double upper, double* bounds) {
    const auto percentile = [&](double q) {
        const double pos = q / 100 * (n - 1); const auto lo = std::size_t(pos), hi = lo + 1 < n ? lo + 1 : lo;
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    };
    bounds[0] = percentile(lower); bounds[1] = percentile(upper);
}
template<class T> __global__ void clip_values(const T* input, T* output, std::size_t n, const double* bounds) {
    EACH_INDEX(n) output[i] = fmin(bounds[1],fmax(bounds[0],double(input[i])));
}
template<class T> __global__ void retain_deviation(T* output, std::size_t n, const Statistics* before, const Statistics* after) {
    const auto a = *after;
    if (a.deviation > 1e-8) { EACH_INDEX(n) output[i] = (output[i]-a.mean)/a.deviation*before->deviation+a.mean; }
}
}
void apply_cuda(const ClipOutliers& t, Sample& s, const Parameters& p, Session& session) {
    with_voxel_channels(t,s,p,session,[&](const Grid&,const ChannelPlan& c,auto input,auto output,const auto*,int*) {
        using T=typename decltype(input)::value_type;
        Buffer<T> keys(input.size);Buffer<double> bounds(2); copy_device(View<T>{keys.ptr,keys.size},input); sort_keys(keys,keys.size);
        clip_bounds<<<1,1,0,vcr::device::stream()>>>(keys.ptr,keys.size,c.values[0],c.values[1],bounds.ptr);
        clip_values<<<blocks(input.size),256,0,vcr::device::stream()>>>(input.ptr,output.ptr,input.size,bounds.ptr);
        if (c.values[2]) {
            auto before = statistics(input,true), after = statistics(output,true);
            retain_deviation<<<blocks(input.size),256,0,vcr::device::stream()>>>(output.ptr,output.size,before.ptr,after.ptr);
        }
        check(cudaGetLastError());
    });
}
}
