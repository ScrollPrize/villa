#pragma once

#include <filesystem>
#include <optional>

#include "utils/Json.hpp"

namespace vc::metadata
{

// Voxel size, in micrometers, from a volume store's own sidecar document: the
// parsed contents of `meta.json` or `metadata.json`, exactly as published and
// before any key flattening.
//
// Each recognized schema is matched by its exact path. These documents also
// record unrelated scales (a segmentation mask downsample, a crop range), and a
// search for any key that merely looks like a resolution finds those too.
//
// Recognized, in order:
//   * a top-level `voxelsize` (this codebase's own field, micrometers), or one
//     of the `voxel_size_um` / `voxelSizeUm` / `pixel_size_um` / `pixelSizeUm`
//     / `resolution_um` spellings. Only at the top level, where the field
//     describes this volume: the same names nested under `metadata` on a
//     derived store belong to its *source* volume, at an unstated pyramid
//     level.
//   * `scan.tomo.acquisition.detector.samplePixelSize`  -- ESRF/BM18 scan record
//   * `tomo.acquisition.detector.samplePixelSize`       -- the same record at the
//                                                document root, as the fused
//                                                mosaic exports write it
//   * the two above under `source.metadata`, for a derived store (surface or
//     ink prediction), scaled by 2^`source.resolution` because inference runs
//     on a downscaled level of its source volume. `source.resolution` is
//     required, not defaulted: without it the factor is unknown, and a voxel
//     size wrong by an unstated power of two is worse than none.
// `samplePixelSize` is in millimeters in all of these.
[[nodiscard]] std::optional<double> voxelSizeFromStoreMetadata(const utils::Json& doc);

// voxelSizeFromStoreMetadata() over a local volume store: its `meta.json`,
// else its `metadata.json`. For tools that read a store without constructing a
// Volume, so they agree with Volume::voxelSize() on what the store says. Never
// throws; a missing or malformed document resolves nothing.
[[nodiscard]] std::optional<double> resolveLocalStoreVoxelSize(
    const std::filesystem::path& storeRoot);

} // namespace vc::metadata
