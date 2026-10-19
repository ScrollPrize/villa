#include "local_field.cuh"
namespace vcr::augment::cuda {
namespace {
// Long weighted sums need accumulation headroom, including signed images and
// finite float32 values whose sum exceeds FLT_MAX. Coefficients stay float32.
struct WeightedSum { double sum, weight; };
__device__ void reduce_weighted(WeightedSum* values) {
    __syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(threadIdx.x<stride) {
            values[threadIdx.x].sum+=values[threadIdx.x+stride].sum;
            values[threadIdx.x].weight+=values[threadIdx.x+stride].weight;
        }
        __syncthreads();
    }
}
template<class T> __device__ void collect_local_mean(KernelIndex vcr_launch, View<T> input, LocalFieldView field, WeightedSum* partial) {
    WeightedSum value{0,0};
    EACH_INDEX(input.size) {
        const double weight=to_double(field[i]);
        value.sum+=to_double(input.ptr[i])*weight; value.weight+=weight;
    }
    __shared__ WeightedSum shared[256]; shared[threadIdx.x]=value; reduce_weighted(shared);
    if(!threadIdx.x)partial[vcr_launch.block]=shared[0];
}
struct collect_local_mean_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { collect_local_mean(index,args...); }
};
__device__ void finish_local_mean(KernelIndex vcr_launch, const WeightedSum* partial, unsigned count, Statistics* output) {
    WeightedSum value{0,0};
    for(unsigned i=threadIdx.x;i<count;i+=256) {
        value.sum+=partial[i].sum; value.weight+=partial[i].weight;
    }
    __shared__ WeightedSum shared[256]; shared[threadIdx.x]=value; reduce_weighted(shared);
    if(!threadIdx.x)*output={shared[0].sum/(shared[0].weight+1e-8),0,0,0};
}
struct finish_local_mean_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { finish_local_mean(index,args...); }
};
template<class T> Buffer<Statistics> local_stats_impl(View<T> input, LocalFieldView field) {
    const unsigned count=std::min(blocks(input.size),1024u);
    Buffer<WeightedSum> partial(count); Buffer<Statistics> result(1);
    launch(count,256,0,vcr::device::stream(), collect_local_mean_launch{}, input,field,partial.ptr);
    launch(1,256,0,vcr::device::stream(), finish_local_mean_launch{}, partial.ptr,count,result.ptr);
    check(cudaGetLastError()); return result;
}
}
Buffer<Statistics> local_statistics(View<float> input, LocalFieldView field) { return local_stats_impl(input,field); }
Buffer<Statistics> local_statistics(View<double> input, LocalFieldView field) { return local_stats_impl(input,field); }
} // namespace vcr::augment::cuda
