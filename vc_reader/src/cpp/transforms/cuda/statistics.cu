#include "resident.cuh"
#include "statistics.cuh"
#include "channel_statistics.cuh"
namespace vcr::transforms::cuda {
template<class T, bool Variance> __device__ void collect(KernelIndex vcr_launch, View<T> input, Partial* output, const float* weights, const Statistics* stats) {
    __shared__ Partial values[256];
    Partial p{0,0,INFINITY,-INFINITY};
    // Versioned on the uniform weights test. Unweighted, the weight stays +0
    // exactly as when +0 was added per element.
    const auto each = [&](auto weighted) {
        EACH_INDEX(input.size) {
            const double x = to_double(input.ptr[i]);
            if constexpr (Variance) p.sum += (x-stats->mean)*(x-stats->mean);
            else {
                if constexpr (decltype(weighted)::value) { p.sum += x*weights[i]; p.weight += weights[i]; }
                else p.sum += x;
                p.low = fmin(p.low,x); p.high = fmax(p.high,x);
            }
        }
    };
    if (weights) each(std::true_type{}); else each(std::false_type{});
    values[threadIdx.x] = p; reduce(values);
    if (!threadIdx.x) output[vcr_launch.block] = values[0];
}
template<class T, bool Variance> struct collect_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { collect<T,Variance>(index,args...); }
};
__device__ void finish_stats(KernelIndex vcr_launch, const Partial* input, unsigned count, std::size_t n, bool weighted, bool variance, Statistics* output) {
    __shared__ Partial values[256]; Partial p{0,0,INFINITY,-INFINITY};
    for (unsigned i = threadIdx.x; i < count; i += 256) {
        auto q = input[i]; p.sum += q.sum; p.weight += q.weight;
        p.low = fmin(p.low,q.low); p.high = fmax(p.high,q.high);
    }
    values[threadIdx.x] = p; reduce(values);
    if (!threadIdx.x) {
        auto q = values[0];
        if (variance) output->deviation = n > 1 ? sqrt(q.sum/(n-1)) : 0;
        else *output = {q.sum/(weighted ? q.weight+1e-8 : double(n)),q.low,q.high,0};
    }
}
struct finish_stats_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { finish_stats(index,args...); }
};
template<class T> Buffer<Statistics> stats_impl(View<T> data, bool deviation, const float* weights) {
    const unsigned count = std::min(blocks(data.size),1024u);
    Buffer<Partial> partial(count); Buffer<Statistics> result(1);
    launch(count,256,0,vcr::device::stream(), collect_launch2<T,false>{}, data,partial.ptr,weights,nullptr);
    launch(1,256,0,vcr::device::stream(), finish_stats_launch{}, partial.ptr,count,data.size,weights != nullptr,false,result.ptr);
    if (deviation) {
        launch(count,256,0,vcr::device::stream(), collect_launch2<T,true>{}, data,partial.ptr,nullptr,result.ptr);
        launch(1,256,0,vcr::device::stream(), finish_stats_launch{}, partial.ptr,count,data.size,false,true,result.ptr);
    }
    check(cudaGetLastError()); return result;
}
__device__ void magnitude_flag(KernelIndex vcr_launch, View<float> input, float limit, int* flags, int bit) {
    bool exceeded = false;
    EACH_INDEX(input.size) exceeded |= fabsf(input.ptr[i]) > limit;
    // Ordinary images perform no atomics. Exceptional blocks publish once.
    const bool any = __syncthreads_or(exceeded);
    if (!threadIdx.x && any) atomicOr(flags,bit);
}
struct magnitude_flag_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { magnitude_flag(index,args...); }
};
void flag_magnitude(View<float> input, double limit, int* flags, int bit) {
    // Round down so comparing float inputs preserves the double threshold.
    float threshold = float(limit);
    if (double(threshold) > limit) threshold = std::nextafter(threshold,-INFINITY);
    launch(blocks(input.size),256,0,vcr::device::stream(), magnitude_flag_launch{}, input,threshold,flags,bit);
    check(cudaGetLastError());
}
template<class T> Buffer<Statistics> bounds_impl(View<T> input) {
    Buffer<Statistics> result(1); summarize_bounds(input,result.ptr); return result;
}
Buffer<Statistics> bounds(View<float> v) { return bounds_impl(v); }
Buffer<Statistics> bounds(View<double> v) { return bounds_impl(v); }
Buffer<Statistics> statistics(View<float> v, bool d, const float* w) { return stats_impl(v,d,w); }
Buffer<Statistics> statistics(View<double> v, bool d, const float* w) { return stats_impl(v,d,w); }
} // namespace vcr::transforms::cuda
