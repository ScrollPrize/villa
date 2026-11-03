#pragma once
#include "resident.cuh"
#include "../shared/histogram.hpp"
namespace vcr::transforms::cuda {
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
template<class T,bool Moments> __device__ void collect_channel_moments(KernelIndex vcr_launch, View<T> input,ChannelMoment* partial) {
    ChannelMoment value{INFINITY,-INFINITY,0,0,0};
    EACH_INDEX(input.size) {
        const double x=double(input.ptr[i]);value.low=fmin(value.low,x);value.high=fmax(value.high,x);++value.count;
        if constexpr(Moments){const double delta=x-value.mean;value.mean+=delta/double(value.count);value.m2+=delta*(x-value.mean);}
    }
    __shared__ ChannelMoment shared[256];shared[threadIdx.x]=value;reduce_moments<Moments>(shared);
    if(!threadIdx.x)partial[vcr_launch.block]=shared[0];
}
template<class T,bool Moments> struct collect_channel_moments_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { collect_channel_moments<T,Moments>(index,args...); }
};
template<bool Moments> __device__ void finish_channel_moments(KernelIndex vcr_launch, const ChannelMoment* partial,unsigned count,volume_detail::Stats* result) {
    ChannelMoment value{INFINITY,-INFINITY,0,0,0};
    for(unsigned i=threadIdx.x;i<count;i+=256)value=merge_moments<Moments>(value,partial[i]);
    __shared__ ChannelMoment shared[256];shared[threadIdx.x]=value;reduce_moments<Moments>(shared);
    if(!threadIdx.x){const auto q=shared[0];*result={q.low,q.high,q.mean,Moments?sqrt(fmax(0.,q.m2)/double(q.count)):0.};}
}
template<bool Moments> struct finish_channel_moments_launch1 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { finish_channel_moments<Moments>(index,args...); }
};
template<class T> struct ChannelBounds {T low,high;};
template<class T> __device__ void reduce_bounds(ChannelBounds<T>* shared){
    __syncthreads();
    for(int stride=128;stride;stride/=2){
        if(threadIdx.x<stride){auto& a=shared[threadIdx.x];const auto b=shared[threadIdx.x+stride];a.low=a.low<b.low?a.low:b.low;a.high=a.high>b.high?a.high:b.high;}
        __syncthreads();
    }
}
template<class T> __device__ void collect_channel_bounds(KernelIndex vcr_launch, View<T> input,ChannelBounds<T>* partial){
    ChannelBounds<T> value{std::numeric_limits<T>::max(),std::numeric_limits<T>::lowest()};
    EACH_INDEX(input.size){const T x=input.ptr[i];value.low=value.low<x?value.low:x;value.high=value.high>x?value.high:x;}
    __shared__ ChannelBounds<T> shared[256];shared[threadIdx.x]=value;reduce_bounds(shared);
    if(!threadIdx.x)partial[vcr_launch.block]=shared[0];
}
struct collect_channel_bounds_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { collect_channel_bounds(index,args...); }
};
template<class T,class Result> __device__ void finish_channel_bounds(KernelIndex vcr_launch, const ChannelBounds<T>* partial,unsigned count,Result* result){
    ChannelBounds<T> value{std::numeric_limits<T>::max(),std::numeric_limits<T>::lowest()};
    for(unsigned i=threadIdx.x;i<count;i+=256){const auto b=partial[i];value.low=value.low<b.low?value.low:b.low;value.high=value.high>b.high?value.high:b.high;}
    __shared__ ChannelBounds<T> shared[256];shared[threadIdx.x]=value;reduce_bounds(shared);
    if(!threadIdx.x){*result={};result->low=double(shared[0].low);result->high=double(shared[0].high);}
}
struct finish_channel_bounds_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { finish_channel_bounds(index,args...); }
};
// Both channel transforms and intensity transforms consume these bounds.
// The reduction stays in the image dtype; only the two final scalars widen.
template<class T,class Result> void summarize_bounds(View<T> data,Result* output) {
    const unsigned count=std::min(blocks(data.size),1024u);
    Buffer<ChannelBounds<T>> partial(count);
    launch(count,256,0,vcr::device::stream(), collect_channel_bounds_launch{}, data,partial.ptr);
    launch(1,256,0,vcr::device::stream(), finish_channel_bounds_launch{}, partial.ptr,count,output);
    check(cudaGetLastError());
}
template<class T> void summarize_channel(View<T> data,volume_detail::Stats* output,bool moments) {
    const unsigned count=std::min(blocks(data.size),1024u);
    if(moments){
        Buffer<ChannelMoment> partial(count);
        launch(count,256,0,vcr::device::stream(), collect_channel_moments_launch2<T,true>{}, data,partial.ptr);
        launch(1,256,0,vcr::device::stream(), finish_channel_moments_launch1<true>{}, partial.ptr,count,output);
    }else{
        summarize_bounds(data,output);
    }
    check(cudaGetLastError());
}
}
