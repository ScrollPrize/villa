#include "resident.cuh"
#include "statistics.cuh"
namespace vcr::augment::cuda {
template<class T, bool Variance> __global__ void collect(View<T> input, Partial* output, const double* weights, const Statistics* stats) {
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
    if (!threadIdx.x) output[blockIdx.x] = values[0];
}
__global__ void finish_stats(const Partial* input, unsigned count, std::size_t n, bool weighted, bool variance, Statistics* output) {
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
template<class T> Buffer<Statistics> stats_impl(View<T> data, bool deviation, const double* weights) {
    const unsigned count = std::min(blocks(data.size),1024u);
    Buffer<Partial> partial(count); Buffer<Statistics> result(1);
    collect<T,false><<<count,256,0,vcr::device::stream()>>>(data,partial.ptr,weights,nullptr);
    finish_stats<<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,data.size,weights != nullptr,false,result.ptr);
    if (deviation) {
        collect<T,true><<<count,256,0,vcr::device::stream()>>>(data,partial.ptr,nullptr,result.ptr);
        finish_stats<<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,data.size,false,true,result.ptr);
    }
    check(cudaGetLastError()); return result;
}
Buffer<Statistics> statistics(View<float> v, bool d, const double* w) { return stats_impl(v,d,w); }
Buffer<Statistics> statistics(View<double> v, bool d, const double* w) { return stats_impl(v,d,w); }
} // namespace vcr::augment::cuda
