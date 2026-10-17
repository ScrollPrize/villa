#pragma once
#include "resident.cuh"
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
namespace vcr::augment::cuda {
// Stable compaction preserves scan-order RNG indexing without a serial scan.
template<class Index> void select_flagged_indices(const Buffer<std::uint8_t>& flags,Buffer<Index>& selected,Buffer<Index>& count){
    thrust::counting_iterator<Index> indices(0);std::size_t bytes=0;
    check(cub::DeviceSelect::Flagged(nullptr,bytes,indices,flags.ptr,selected.ptr,count.ptr,flags.size,vcr::device::stream()));Buffer<std::byte> scratch(bytes);
    check(cub::DeviceSelect::Flagged(scratch.ptr,bytes,indices,flags.ptr,selected.ptr,count.ptr,flags.size,vcr::device::stream()));
}
}
