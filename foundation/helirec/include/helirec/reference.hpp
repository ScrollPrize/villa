#pragma once
// The mask of a reference volume on the reconstruction's grid: a masked uint8 volume of the same scan (e.g. the
// published one), whose zero voxels are zero in the uint8 output, so that its mask need not be redone. Any pyramid
// level of the reference serves (`scale`): a coarse level is a small read and marks the same area to within its voxel
// size. Read with zarr.hpp: a local or http(s):// / s3:// Zarr v2 array or OME-Zarr pyramid, with an optional disk
// cache.
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "helirec/pipeline.hpp"

namespace helirec {

class ReferenceMask {
  public:
    // shape: the reconstruction's whole volume (z, y, x); the reference's level 0 must have it.
    // scale: the pyramid level read (0: full resolution).
    ReferenceMask(const std::string& location, int scale, const std::array<std::int64_t, 3>& shape,
                  const std::string& cache_dir = {});
    ~ReferenceMask();

    const std::string& location() const;
    int scale() const;
    std::array<int, 3> factors() const;             // volume voxels per mask voxel (z, y, x)
    std::array<std::int64_t, 3> level_shape() const;

    // The mask of volume slices [z0, z1) of the ROI. Its data is the buffer `slot` of this object: valid until the next
    // read into that slot (two slots let the next chunk's mask be read while this one is in use).
    OutputMask read(int z0, int z1, const SliceRoi& roi, int slot = 0);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
