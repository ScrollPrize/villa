#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vcr::distance {
// Exact Manhattan distance to the nearest zero in a contiguous binary grid.
// Any nonzero is foreground. Supports 2D YX and 3D ZYX. Distances saturate at
// 255, including grids with no zero. Outside the grid is not a feature.
// Input/output may alias. Empty dimensions are accepted; invalid shapes throw.
void manhattan(const std::uint8_t* input, std::uint8_t* output,
               const std::vector<std::size_t>& shape);
#ifdef VC_READER_CUDA
namespace cuda {
// Host arrays, including transfers; completes before return.
void manhattan(const std::uint8_t* input, std::uint8_t* output,
               const std::vector<std::size_t>& shape);
// Device arrays on the calling thread's device and vc_reader stream.
void manhattanDevice(const std::uint8_t* input, std::uint8_t* output,
                     const std::vector<std::size_t>& shape);
}
#endif
}
