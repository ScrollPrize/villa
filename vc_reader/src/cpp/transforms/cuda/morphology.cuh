#pragma once
#include "../cpu/labels.hpp"
#include "resident.cuh"
#include <map>
#include <mutex>
// Binary morphology shared by CUDA label transforms. Voxels outside the grid
// count as `outside`: false matches the CPU default, true its neutral border.
namespace vcr::transforms::cuda {
struct Offset { int v[3]; };
template<class T, class O> __device__ void morphology_kernel(KernelIndex vcr_launch, const T* input, O* output, Layout g, const Offset* footprint, std::size_t count, bool dilate, bool outside = false) {
    EACH_INDEX(g.size) {
        bool result = !dilate;
        for (std::size_t j = 0; j < count; ++j) {
            bool valid = true; std::size_t at = 0;
            for (int a = 0; a < g.dim; ++a) {
                auto x = static_cast<long long>(g.coord(i,a)) + (dilate ? -footprint[j].v[a] : footprint[j].v[a]);
                if (x < 0 || x >= static_cast<long long>(g.shape[a])) valid = false; else at += x*g.stride[a];
            }
            bool value = valid ? input[at] != 0 : outside;
            if (dilate ? value : !value) { result = dilate; break; }
        }
        output[i] = O(result);
    }
}
struct morphology_kernel_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { morphology_kernel(index,args...); }
};
// Device copies of footprints, uploaded once per device and kept for the process: a
// batch of samples would otherwise stage the same small pageable upload per sample.
struct DeviceFootprint { const Offset* ptr; std::size_t size; };
inline DeviceFootprint footprint_offsets(const Footprint& footprint) {
    static std::mutex mutex;
    static std::map<std::pair<int,std::vector<int>>,Offset*> uploaded;
    int device = 0; check(cudaGetDevice(&device));
    std::vector<int> key; key.reserve(3*footprint.size());
    for (const auto& offset:footprint) key.insert(key.end(),offset.begin(),offset.end());
    std::lock_guard lock(mutex);
    auto& ptr = uploaded[{device,key}];
    if (!ptr && !footprint.empty()) {
        std::vector<Offset> offsets(footprint.size());
        for (std::size_t i = 0; i < offsets.size(); ++i) std::copy(footprint[i].begin(),footprint[i].end(),offsets[i].v);
        void* storage = nullptr;
        check(cudaMalloc(&storage,offsets.size()*sizeof(Offset)));
        check(cudaMemcpy(storage,offsets.data(),offsets.size()*sizeof(Offset),cudaMemcpyHostToDevice));
        ptr = static_cast<Offset*>(storage);
    }
    return {ptr,footprint.size()};
}
} // namespace vcr::transforms::cuda
