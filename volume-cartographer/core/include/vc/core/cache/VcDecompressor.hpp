#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ChunkData.hpp"
#include "ChunkKey.hpp"

namespace vc {
class VcDataset;
}

namespace vc::cache {

// Create a DecompressFn that uses VcDataset's decompressor.
//
// The returned function decompresses raw bytes (read from disk or network)
// into a ChunkData containing the voxel values. Handles uint8 and uint16
// datasets, with optional uint16->uint8 downsampling (divide by 257).
//
// If the input has the VC3D video codec magic header, it is decoded with
// video_decode() instead of the normal zarr decompression path. This
// allows transparent reading from both original and recompressed formats.
//
// datasets: one VcDataset* per pyramid level (index = level).
// The dataset is used to determine chunk shape, dtype, and decompressor.
// Pointers must remain valid for the lifetime of the returned function.
DecompressFn makeVcDecompressor(const std::vector<vc::VcDataset*>& datasets);

// Convenience overload for a single dataset (level 0 only).
DecompressFn makeVcDecompressor(vc::VcDataset* ds);

// Create a RecompressFn that recompresses chunks using a video codec.
// The function first decompresses the original data using the dataset's
// decompressor, then re-encodes it with the specified video codec.
//
// codecType: 0=H264, 1=H265, 2=AV1
// qp: quantization parameter (0-51)
RecompressFn makeVideoRecompressor(
    const std::vector<vc::VcDataset*>& datasets,
    int codecType = 0, int qp = 26);

}  // namespace vc::cache
