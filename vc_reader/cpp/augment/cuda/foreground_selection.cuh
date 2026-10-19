#pragma once
#include "../foreground_selection.hpp"
#include "resident.cuh"
#include "scan.cuh"
namespace vcr::augment::cuda {
using ForegroundCount=unsigned long long;
template<class T> __device__ void foreground_flags(KernelIndex vcr_launch, const T* mask,std::size_t n,double threshold,ForegroundCount* prefix){EACH_INDEX(n)prefix[i]=double(mask[i])>threshold;}
struct foreground_flags_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { foreground_flags(index,args...); }
};
template<class T> void foreground_prefix(const T* mask,std::size_t n,double threshold,Buffer<ForegroundCount>& prefix) {
    launch(blocks(n),256,0,vcr::device::stream(), foreground_flags_launch{}, mask,n,threshold,prefix.ptr);check(cudaGetLastError());
    inclusive_sum(prefix,n);
}
__device__ inline void select_foreground(KernelIndex vcr_launch, const ForegroundCount* prefix,std::size_t n,std::uint64_t seed,ForegroundCount* selected,std::size_t count,bool holes) {
    EACH_INDEX(n) {
        const auto rank=prefix[i];if(!rank || (i && rank==prefix[i-1]))continue;
        for(std::size_t h=0;h<count;++h) {
            const auto current_seed=holes?volume_detail::mix(seed+h):seed;
            // Reservoir sampling is exactly the last accepted foreground voxel.
            if(volume_detail::uniform(current_seed,i)<1./double(rank))atomicMax(selected+h,ForegroundCount(i+1));
        }
    }
}
struct select_foreground_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { select_foreground(index,args...); }
};
inline void choose_foreground_indices(const Buffer<ForegroundCount>& prefix,std::uint64_t seed,Buffer<ForegroundCount>& selected,bool holes) {
    if(!selected.size)return;
    check(memset_async(selected.ptr,0,selected.size*sizeof(ForegroundCount),vcr::device::stream()));
    launch(blocks(prefix.size),256,0,vcr::device::stream(), select_foreground_launch{}, prefix.ptr,prefix.size,seed,selected.ptr,selected.size,holes);check(cudaGetLastError());
}
}
