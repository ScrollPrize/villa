#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vcr {
namespace zarr { struct OpenedArray; }

// Normal XYZ and relative eigengap (4 floats) at each XYZ point of a 3-D uint8 or float32
// array in its voxels; invalid support gives zeros. Reads the 109-cubed context around each.
void sheetNormals(zarr::OpenedArray& array, const double* xyz, std::size_t count, float* out);

// Calls the same implementation as Python's sheet_normals. Centers are local
// XYZ in [48,51), followed by validity; patches are contiguous B*109^3.
void sheetNormalsCpu(std::span<const uint8_t> patches, std::span<const std::array<float, 4>> centers,
                     std::span<std::array<float, 4>> out);
void sheetNormalsCpu(std::span<const float> patches, std::span<const std::array<float, 4>> centers,
                     std::span<std::array<float, 4>> out);

// Shared Zarr v3 -> v2 metadata normalization for native and Python opening.
std::string normalizeZarrMetadata(const std::string& json);
} // namespace vcr
