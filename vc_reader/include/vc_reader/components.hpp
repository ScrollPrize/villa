#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace vcr::components {
// Connected components of the nonzero voxels of a C-order 2D (Y, X) or 3D
// (Z, Y, X) mask. Writes labels 1..count, numbered in scan order of each
// component's first voxel (as scipy.ndimage.label), and 0 to background, and
// returns count. connectivity is 4 or 8 in 2D, 6, 18 or 26 in 3D; on a 2D mask
// 6 means 4 and 18 or 26 mean 8, as cc3d reads them. threads = 0
// uses every hardware thread; labels do not depend on it. Throws if count
// would exceed INT32_MAX.
std::size_t label(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
                  std::int32_t* labels, std::size_t threads = 0);
// scipy.ndimage.binary_fill_holes: `filled` is the mask plus every background
// component not connected to the array border. `connectivity` is the background's
// (4 or 8 in 2D, 6, 18 or 26 in 3D; 0 for scipy's default structure, 4 or 6;
// 3D values on a 2D mask as for label). mask and filled may alias. threads as for label.
void fillHoles(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
               std::uint8_t* filled, std::size_t threads = 0);
#ifdef VC_READER_CUDA
namespace cuda {
// The same labels for device mask and label pointers, on the calling thread's
// CUDA device and current vc_reader stream; returns once count is known.
std::size_t label(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
                  std::int32_t* labels);
// fillHoles for device mask and output pointers.
void fillHoles(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
               std::uint8_t* filled);
}
#endif
}
