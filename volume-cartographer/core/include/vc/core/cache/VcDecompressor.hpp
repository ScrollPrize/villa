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
// datasets: one VcDataset* per pyramid level (index = level).
// The dataset is used to determine chunk shape, dtype, and decompressor.
// Pointers must remain valid for the lifetime of the returned function.
DecompressFn makeVcDecompressor(const std::vector<vc::VcDataset*>& datasets);

// Convenience overload for a single dataset (level 0 only).
DecompressFn makeVcDecompressor(vc::VcDataset* ds);

}  // namespace vc::cache
