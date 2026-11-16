#pragma once
// The mask of a reference volume on the reconstruction's grid: a masked uint8 volume of the same scan (e.g. the
// published one), whose zero voxels are zero in the uint8 output, so that its mask need not be redone. Any pyramid
// level of the reference serves (`scale`): a coarse level is a small read and marks the same area to within its voxel
// size. Read with zarr.hpp: a local or http(s):// / s3:// Zarr v2 array or OME-Zarr pyramid, with an optional disk
// cache.
//
// The reference must have the reconstruction's slices (y, x). Its number of slices may differ a little (up to
// kMaxSliceMismatch of it): a reconstruction of the same scan with another helical pitch calibration covers the same
// part of the sample with slightly more or fewer slices. The reference's slices are then spread evenly over the
// reconstruction's (slice z takes the mask of the reference slice nearest (z + 1/2) Nref / N - 1/2), which places
// the mask to within a few slices: the two ranges begin and end at the scan's limits, but the sample does not sit
// exactly proportionally in them.
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "helirec/pipeline.hpp"

namespace helirec {

class ReferenceMask {
  public:
    static constexpr double kMaxSliceMismatch = 0.02;
    // shape: the reconstruction's whole volume (z, y, x); the reference's level 0 must have its y and x, and its z
    // to within kMaxSliceMismatch.
    // scale: the pyramid level read (0: full resolution).
    ReferenceMask(const std::string& location, int scale, const std::array<std::int64_t, 3>& shape,
                  const std::string& cache_dir = {});
    ~ReferenceMask();

    const std::string& location() const;
    int scale() const;
    std::array<int, 3> factors() const;             // volume voxels per mask voxel (z, y, x)
    std::array<std::int64_t, 3> level_shape() const;
    double slice_ratio() const;                     // reference slices per slice of the volume (1: the same grid)

    // The mask of volume slices [z0, z1) of the ROI. Its data is the buffer `slot` of this object: valid until the next
    // read into that slot (two slots let the next chunk's mask be read while this one is in use).
    OutputMask read(int z0, int z1, const SliceRoi& roi, int slot = 0);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A reference volume on another grid, resampled on blocks of the reconstruction's: e.g. a finer scan of the same
// sample, registered by an affine map (the volume transforms of the open-data catalogue). Read with zarr.hpp, uint8
// or float32, any pyramid level; resampled with block_spectra.hpp resample_affine.
class RegisteredVolume {
  public:
    // affine: 3 x 4, row-major, rows (z, y, x): the reference's level-0 voxel index of the reconstruction's voxel
    // index (z, y, x, 1). level: the pyramid level read (its voxels placed by the pyramid's scales and offsets).
    RegisteredVolume(const std::string& location, int level, const std::array<double, 12>& affine,
                     const std::string& cache_dir = {});
    ~RegisteredVolume();

    const std::string& location() const;
    int level() const;
    std::array<std::int64_t, 3> level_shape() const;
    // the map from a block's sample index (k, j, i, 1) to the level's voxel index (z, y, x), 3 x 4
    std::array<double, 12> block_affine(const std::array<double, 3>& origin, double step) const;

    // The reference at the reconstruction's positions origin + step (k, j, i) (z, y, x; step < 1: finer than its
    // grid), k < shape[0] ...: out (shape), host memory, or device memory with Device::CUDA (resampled there).
    // interp: resample_interp_from_name. Returns the fraction of the reference voxels read for it that are not 0 (a
    // masked reference is 0 outside the object). Thread-safe.
    double read(const std::array<double, 3>& origin, double step, const std::array<int, 3>& shape, float* out,
                int interp, Device device = Device::CPU) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
