#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace vcr::components {
// Label the nonzero foreground of a contiguous 2D YX or 3D ZYX uint8 grid.
// Background is zero. Each component's label is its smallest flat voxel index
// plus one (stable across CPU/CUDA, not consecutive). At most UINT32_MAX voxels.
// fullConnectivity selects 8/26 neighbors; false selects 4/6 neighbors.
void label(const std::uint8_t* input,std::uint32_t* output,
           const std::vector<std::size_t>& shape,bool fullConnectivity=true);
#ifdef VC_READER_CUDA
namespace cuda {
void label(const std::uint8_t* input,std::uint32_t* output,
           const std::vector<std::size_t>& shape,bool fullConnectivity=true);
// Device buffers; work completes on vc_reader's current stream before returning.
void labelDevice(const std::uint8_t* input,std::uint32_t* output,
                 const std::vector<std::size_t>& shape,bool fullConnectivity=true);
}
#endif
}
