#pragma once

#include <filesystem>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(containers, xtensor.hpp)

// Read a flat zarr array (single dataset at root, created by zarr.save_array())
// into xtensor array.
// Returns shape (depth, height, width) for 3D arrays in ZYX order.
// Throws runtime_error if file cannot be opened or format is unsupported.
xt::xarray<float> read3DZarr(const std::filesystem::path& path);

// Write a 3D xtensor array to zarr format (flat layout matching zarr.save_array()).
// Creates .zarray at root of path, not in subdirectory.
// Data should be shape (depth, height, width) in ZYX order.
void write3DZarr(const std::filesystem::path& path, const xt::xarray<float>& data);
