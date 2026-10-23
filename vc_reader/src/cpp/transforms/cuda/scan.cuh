#pragma once
#include "common.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
namespace vcr::transforms::cuda {
struct ScanTiles {
    template<class T> __device__ void operator()(KernelIndex vcr_launch, T* data, std::size_t n, T* totals) const {
        using Scan = cub::BlockScan<T,256>;
        __shared__ typename Scan::TempStorage storage;
        const auto tiles = (n+255)/256;
        for (std::size_t tile=vcr_launch.block;tile<tiles;tile+=vcr_launch.blocks) {
            const auto i=tile*256+threadIdx.x;
            const T value=i<n?data[i]:T(0);
            T prefix, total;
            Scan(storage).InclusiveSum(value,prefix,total);
            if (i<n) data[i]=prefix;
            if (!threadIdx.x) totals[tile]=total;
            __syncthreads();
        }
    }
};
struct ScanOffsets {
    template<class T> __device__ void operator()(KernelIndex vcr_launch,T* data,std::size_t n,const T* totals) const {
        EACH_INDEX(n) if (i>=256) data[i]+=totals[i/256-1];
    }
};
template<class T> void inclusive_sum(Buffer<T>& data,std::size_t n) {
    if (!n) return;
    if(!BatchQueue::current) {
        std::size_t bytes=0;
        check(cub::DeviceScan::InclusiveSum(nullptr,bytes,data.ptr,data.ptr,n,vcr::device::stream()));
        Buffer<std::byte> scratch(bytes);
        check(cub::DeviceScan::InclusiveSum(scratch.ptr,bytes,data.ptr,data.ptr,n,vcr::device::stream()));
        return;
    }
    const auto tiles=(n+255)/256;
    Buffer<T> totals(tiles);
    launch(unsigned(std::min<std::size_t>(tiles,65535)),256,0,vcr::device::stream(),ScanTiles{},data.ptr,n,totals.ptr);
    if (tiles>1) {
        inclusive_sum(totals,tiles);
        launch(blocks(n),256,0,vcr::device::stream(),ScanOffsets{},data.ptr,n,totals.ptr);
    }
    check(cudaGetLastError());
}
}
