#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace vcr::display {
struct SlabLayer { const std::uint8_t *values, *coverage; std::size_t valueStride, coverageStride; };
struct SlabProjection {
    bool perspective = false;
    float m00=1, m01=0, m10=0, m11=1;
    const float* layerOffsets = nullptr; // pairs, one per layer
    float pos[3]{}, rayBase[3]{}, e1OverF[3]{}, e2OverF[3]{};
    float centerU=0, centerV=0;
    const float* layerNum = nullptr;
};
struct Slab {
    const SlabLayer* layers = nullptr;
    int count=0, rows=0, cols=0;
    int left=0, top=0, width=0, height=0;
    SlabProjection projection;
    const std::uint32_t* colors = nullptr; // 256 packed ARGB entries
    const float* opacity = nullptr; // 256 entries
    float lighting = 0;
};
std::array<float,256> opacityLut(float low,float high,float opacity,float gamma,std::uint8_t cutoff,float segmentLength);
// Output is packed RGB and byte coverage, contiguous. The slab's pointers refer to CPU memory.
void compositeSlab(const Slab& slab,std::uint8_t* rgb,std::uint8_t* coverage);
#ifdef VC_READER_CUDA
namespace cuda {
std::array<float,256> opacityLut(float low,float high,float opacity,float gamma,std::uint8_t cutoff,float segmentLength);
// All slab pointers, including layers and palettes, refer to device memory.
void compositeSlab(const Slab& slab,std::uint8_t* rgb,std::uint8_t* coverage);
}
#endif
}
