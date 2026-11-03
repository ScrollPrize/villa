#pragma once
#include "resident.cuh"
#include "scan.cuh"
namespace vcr::transforms::cuda {
// Stable compaction preserves scan-order RNG indexing without a serial scan.
struct SelectionFlags {
    template<class Index> __device__ void operator()(KernelIndex vcr_launch,const std::uint8_t* flags,Index* prefix,std::size_t n) const {
        EACH_INDEX(n) prefix[i]=flags[i]!=0;
    }
};
struct SelectionIndices {
    template<class Index> __device__ void operator()(KernelIndex vcr_launch,const std::uint8_t* flags,const Index* prefix,Index* selected,Index* count,std::size_t n) const {
        EACH_INDEX(n) if (flags[i]) selected[prefix[i]-1]=Index(i);
        if (!vcr_launch.block && !threadIdx.x) *count=n?prefix[n-1]:0;
    }
};
template<class Index> void select_flagged_indices(const Buffer<std::uint8_t>& flags,Buffer<Index>& selected,Buffer<Index>& count){
    Buffer<Index> prefix(flags.size);
    launch(blocks(flags.size),256,0,vcr::device::stream(),SelectionFlags{},flags.ptr,prefix.ptr,flags.size);
    inclusive_sum(prefix,flags.size);
    launch(std::max(1u,blocks(flags.size)),256,0,vcr::device::stream(),SelectionIndices{},flags.ptr,prefix.ptr,selected.ptr,count.ptr,flags.size);
    check(cudaGetLastError());
}
}
