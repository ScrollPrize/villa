#pragma once
#include "../labels.hpp"
#include "resident.cuh"
// Binary morphology shared by CUDA label transforms. Voxels outside the grid
// count as `outside`: false matches the CPU default, true its neutral border.
namespace vcr::augment::cuda {
struct Offset { int v[3]; };
template<class T, class O> __global__ void morphology_kernel(const T* input, O* output, Layout g, const Offset* footprint, std::size_t count, bool dilate, bool outside = false) {
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
inline Buffer<Offset> footprint_offsets(const Footprint& footprint) {
    std::vector<Offset> offsets(footprint.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) std::copy(footprint[i].begin(),footprint[i].end(),offsets[i].v);
    return Buffer<Offset>(offsets);
}
} // namespace vcr::augment::cuda
