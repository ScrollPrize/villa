#pragma once
#include "common.cuh"
#include <cub/device/device_radix_sort.cuh>
namespace vcr::augment::cuda {
template<class K> void sort_keys(Buffer<K>& keys, std::size_t n) {
    Buffer<K> sorted(n); std::size_t bytes = 0;
    check(cub::DeviceRadixSort::SortKeys(nullptr,bytes,keys.ptr,sorted.ptr,n,0,int(sizeof(K)*8),vcr::device::stream()));
    Buffer<std::byte> scratch(bytes);
    check(cub::DeviceRadixSort::SortKeys(scratch.ptr,bytes,keys.ptr,sorted.ptr,n,0,int(sizeof(K)*8),vcr::device::stream()));
    std::swap(keys.ptr,sorted.ptr);
}
}
