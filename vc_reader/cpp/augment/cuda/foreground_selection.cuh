#pragma once
#include "../foreground_selection.hpp"
#include "resident.cuh"
#include <cub/device/device_scan.cuh>
namespace vcr::augment::cuda {
using ForegroundCount=unsigned long long;
template<class T> __global__ void foreground_flags(const T* mask,std::size_t n,double threshold,ForegroundCount* prefix){EACH_INDEX(n)prefix[i]=double(mask[i])>threshold;}
template<class T> void foreground_prefix(const T* mask,std::size_t n,double threshold,Buffer<ForegroundCount>& prefix) {
    foreground_flags<<<blocks(n),256,0,vcr::device::stream()>>>(mask,n,threshold,prefix.ptr);check(cudaGetLastError());
    std::size_t bytes=0;check(cub::DeviceScan::InclusiveSum(nullptr,bytes,prefix.ptr,prefix.ptr,n,vcr::device::stream()));Buffer<std::byte> scratch(bytes);
    check(cub::DeviceScan::InclusiveSum(scratch.ptr,bytes,prefix.ptr,prefix.ptr,n,vcr::device::stream()));
}
__global__ inline void select_foreground(const ForegroundCount* prefix,std::size_t n,std::uint64_t seed,ForegroundCount* selected,std::size_t count,bool holes) {
    EACH_INDEX(n) {
        const auto rank=prefix[i];if(!rank || (i && rank==prefix[i-1]))continue;
        for(std::size_t h=0;h<count;++h) {
            const auto current_seed=holes?volume_detail::mix(seed+h):seed;
            // Reservoir sampling is exactly the last accepted foreground voxel.
            if(volume_detail::uniform(current_seed,i)<1./double(rank))atomicMax(selected+h,ForegroundCount(i+1));
        }
    }
}
inline void choose_foreground_indices(const Buffer<ForegroundCount>& prefix,std::uint64_t seed,Buffer<ForegroundCount>& selected,bool holes) {
    if(!selected.size)return;
    check(cudaMemsetAsync(selected.ptr,0,selected.size*sizeof(ForegroundCount),vcr::device::stream()));
    select_foreground<<<blocks(prefix.size),256,0,vcr::device::stream()>>>(prefix.ptr,prefix.size,seed,selected.ptr,selected.size,holes);check(cudaGetLastError());
}
}
