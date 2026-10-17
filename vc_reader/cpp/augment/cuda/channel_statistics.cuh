#pragma once
#include "resident.cuh"
#include "../histogram.hpp"
namespace vcr::augment::cuda {
// Welford moments use double: squared deviations of finite float32 values can
// exceed FLT_MAX, even when the final standard deviation is representable.
// Min/max-only reductions below stay in the input dtype. Both use fixed trees.
struct ChannelMoment {
    double low,high,mean,m2;
    unsigned long long count;
};
template<bool Moments> __device__ inline ChannelMoment merge_moments(ChannelMoment a,ChannelMoment b) {
    if(!a.count)return b;if(!b.count)return a;
    a.low=fmin(a.low,b.low);a.high=fmax(a.high,b.high);
    if constexpr(Moments) {
        const double delta=b.mean-a.mean,weight=double(b.count)/double(a.count+b.count);
        a.mean+=delta*weight;
        a.m2+=b.m2+delta*delta*(double(a.count)*weight);
    }
    a.count+=b.count;return a;
}
template<bool Moments> __device__ inline void reduce_moments(ChannelMoment* shared) {
    __syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)shared[threadIdx.x]=merge_moments<Moments>(shared[threadIdx.x],shared[threadIdx.x+stride]);__syncthreads();}
}
template<class T,bool Moments> __global__ void collect_channel_moments(View<T> input,ChannelMoment* partial) {
    ChannelMoment value{INFINITY,-INFINITY,0,0,0};
    EACH_INDEX(input.size) {
        const double x=double(input.ptr[i]);value.low=fmin(value.low,x);value.high=fmax(value.high,x);++value.count;
        if constexpr(Moments){const double delta=x-value.mean;value.mean+=delta/double(value.count);value.m2+=delta*(x-value.mean);}
    }
    __shared__ ChannelMoment shared[256];shared[threadIdx.x]=value;reduce_moments<Moments>(shared);
    if(!threadIdx.x)partial[blockIdx.x]=shared[0];
}
template<bool Moments> __global__ void finish_channel_moments(const ChannelMoment* partial,unsigned count,volume_detail::Stats* result) {
    ChannelMoment value{INFINITY,-INFINITY,0,0,0};
    for(unsigned i=threadIdx.x;i<count;i+=256)value=merge_moments<Moments>(value,partial[i]);
    __shared__ ChannelMoment shared[256];shared[threadIdx.x]=value;reduce_moments<Moments>(shared);
    if(!threadIdx.x){const auto q=shared[0];*result={q.low,q.high,q.mean,Moments?sqrt(fmax(0.,q.m2)/double(q.count)):0.};}
}
template<class T> struct ChannelBounds {T low,high;};
template<class T> __device__ void reduce_bounds(ChannelBounds<T>* shared){
    __syncthreads();
    for(int stride=128;stride;stride/=2){
        if(threadIdx.x<stride){auto& a=shared[threadIdx.x];const auto b=shared[threadIdx.x+stride];a.low=a.low<b.low?a.low:b.low;a.high=a.high>b.high?a.high:b.high;}
        __syncthreads();
    }
}
template<class T> __global__ void collect_channel_bounds(View<T> input,ChannelBounds<T>* partial){
    ChannelBounds<T> value{std::numeric_limits<T>::max(),std::numeric_limits<T>::lowest()};
    EACH_INDEX(input.size){const T x=input.ptr[i];value.low=value.low<x?value.low:x;value.high=value.high>x?value.high:x;}
    __shared__ ChannelBounds<T> shared[256];shared[threadIdx.x]=value;reduce_bounds(shared);
    if(!threadIdx.x)partial[blockIdx.x]=shared[0];
}
template<class T> __global__ void finish_channel_bounds(const ChannelBounds<T>* partial,unsigned count,volume_detail::Stats* result){
    ChannelBounds<T> value{std::numeric_limits<T>::max(),std::numeric_limits<T>::lowest()};
    for(unsigned i=threadIdx.x;i<count;i+=256){const auto b=partial[i];value.low=value.low<b.low?value.low:b.low;value.high=value.high>b.high?value.high:b.high;}
    __shared__ ChannelBounds<T> shared[256];shared[threadIdx.x]=value;reduce_bounds(shared);
    if(!threadIdx.x)*result={double(shared[0].low),double(shared[0].high),0,0};
}
template<class T> void summarize_channel(View<T> data,volume_detail::Stats* output,bool moments) {
    const unsigned count=std::min(blocks(data.size),1024u);
    if(moments){
        Buffer<ChannelMoment> partial(count);
        collect_channel_moments<T,true><<<count,256,0,vcr::device::stream()>>>(data,partial.ptr);
        finish_channel_moments<true><<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,output);
    }else{
        Buffer<ChannelBounds<T>> partial(count);
        collect_channel_bounds<<<count,256,0,vcr::device::stream()>>>(data,partial.ptr);
        finish_channel_bounds<<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,output);
    }
    check(cudaGetLastError());
}
}
