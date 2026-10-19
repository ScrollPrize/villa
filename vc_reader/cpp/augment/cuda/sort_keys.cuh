#pragma once
#include "common.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
namespace vcr::augment::cuda {
// Sort by (sample, value) so large samples use all SMs. A segmented
// radix sort assigns too little parallel work to a handful of large volumes.
// The integer sample tag is independent of the image's floating-point dtype.
template<class K> struct BatchSortKey { K value; unsigned sample; };
struct BatchSortDecomposer {
    template<class K> __host__ __device__ auto operator()(BatchSortKey<K>& key) const {
        return ::cuda::std::tie(key.sample,key.value);
    }
};
struct PackSortKeys {
    template<class K> __device__ void operator()(KernelIndex vcr_launch,const K* input,BatchSortKey<K>* output,std::size_t n,unsigned sample) const {
        EACH_INDEX(n) output[i]={input[i],sample};
    }
};
struct UnpackSortKeys {
    template<class K> __device__ void operator()(KernelIndex vcr_launch,const BatchSortKey<K>* input,K* output,std::size_t n) const {
        EACH_INDEX(n) output[i]=input[i].value;
    }
};
template<class K> struct SortJob {
    K* input;
    K* output;
    std::size_t size;
    static void run(const std::vector<const SortJob*>& jobs) {
        if (jobs.size()==1) {
            const auto& j=*jobs.front();std::size_t bytes=0;
            check(cub::DeviceRadixSort::SortKeys(nullptr,bytes,j.input,j.output,j.size,0,int(sizeof(K)*8),vcr::device::stream()));
            Buffer<std::byte> scratch(bytes);
            check(cub::DeviceRadixSort::SortKeys(scratch.ptr,bytes,j.input,j.output,j.size,0,int(sizeof(K)*8),vcr::device::stream()));
            return;
        }
        std::vector<std::size_t> offsets{0};
        for (const auto* job:jobs) offsets.push_back(offsets.back()+job->size);
        Buffer<BatchSortKey<K>> packed(offsets.back()), sorted(offsets.back());
        {
            BatchQueue copies(jobs.size());
            for (std::size_t i=0;i<jobs.size();++i) {
                copies.lane(i);
                launch(blocks(jobs[i]->size),256,0,vcr::device::stream(),PackSortKeys{},jobs[i]->input,packed.ptr+offsets[i],jobs[i]->size,unsigned(i));
            }
            copies.flush();
        }
        unsigned sample_bits=0;
        for(auto n=jobs.size()-1;n;n>>=1) ++sample_bits;
        const int end_bit=int(sizeof(K)*8+sample_bits);
        std::size_t bytes=0;
        check(cub::DeviceRadixSort::SortKeys(nullptr,bytes,packed.ptr,sorted.ptr,offsets.back(),BatchSortDecomposer{},0,end_bit,vcr::device::stream()));
        Buffer<std::byte> scratch(bytes);
        check(cub::DeviceRadixSort::SortKeys(scratch.ptr,bytes,packed.ptr,sorted.ptr,offsets.back(),BatchSortDecomposer{},0,end_bit,vcr::device::stream()));
        {
            BatchQueue copies(jobs.size());
            for (std::size_t i=0;i<jobs.size();++i) {
                copies.lane(i);
                launch(blocks(jobs[i]->size),256,0,vcr::device::stream(),UnpackSortKeys{},sorted.ptr+offsets[i],jobs[i]->output,jobs[i]->size);
            }
            copies.flush();
        }
    }
};
template<class K> void sort_keys(Buffer<K>& keys, std::size_t n) {
    Buffer<K> sorted(n);
    SortJob<K> job{keys.ptr,sorted.ptr,n};
    if (BatchQueue::current) BatchQueue::current->append(job);
    else SortJob<K>::run({&job});
    std::swap(keys.ptr,sorted.ptr);
}
}
