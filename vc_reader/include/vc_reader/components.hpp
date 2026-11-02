#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace vcr::components {
// Connected components of the nonzero voxels of a C-order 2D (Y, X) or 3D
// (Z, Y, X) mask. Writes labels 1..count, numbered in scan order of each
// component's first voxel (as scipy.ndimage.label), and 0 to background, and
// returns count. connectivity is 4 or 8 in 2D, 6, 18 or 26 in 3D. threads = 0
// uses every hardware thread; labels do not depend on it. Throws if count
// would exceed INT32_MAX.
std::size_t label(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
                  std::int32_t* labels, std::size_t threads = 0);
#ifdef VC_READER_CUDA
namespace cuda {
// The same labels for device mask and label pointers, on the calling thread's
// CUDA device and current vc_reader stream; returns once count is known.
std::size_t label(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
                  std::int32_t* labels);
}
#endif
}
