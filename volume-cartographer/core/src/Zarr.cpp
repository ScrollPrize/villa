#include "vc/core/util/Zarr.hpp"

#include <stdexcept>
#include <fstream>

#include <nlohmann/json.hpp>
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

xt::xarray<float> read3DZarr(const std::filesystem::path& path)
{
    namespace fs = std::filesystem;

    // Validate path exists
    if (!fs::exists(path)) {
        throw std::runtime_error("Zarr path does not exist: " + path.string());
    }

    // For a flat zarr array (created by zarr.save_array), .zarray is at root
    fs::path zarray_path = path / ".zarray";
    if (!fs::exists(zarray_path)) {
        throw std::runtime_error("Not a valid zarr array (missing .zarray): " + path.string());
    }

    // Read dimension separator from .zarray metadata
    std::string dimsep = ".";
    try {
        auto metadata = nlohmann::json::parse(std::ifstream(zarray_path));
        if (metadata.contains("dimension_separator")) {
            dimsep = metadata["dimension_separator"].get<std::string>();
        }
    } catch (...) {
        // Use default dimension separator
    }

    // Create handles: parent directory as Group, dataset name is the directory name
    z5::filesystem::handle::Group parent(path.parent_path(), z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(parent, path.filename().string(), dimsep);

    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);
    if (!ds) {
        throw std::runtime_error("Failed to open zarr dataset: " + path.string());
    }

    // Verify dtype is float32
    if (ds->getDtype() != z5::types::Datatype::float32) {
        throw std::runtime_error("Expected float32 zarr dataset, got different dtype in: " + path.string());
    }

    // Get shape and verify 3D
    const auto& shape = ds->shape();
    if (shape.size() != 3) {
        throw std::runtime_error("Expected 3D zarr array, got " + std::to_string(shape.size()) +
                                 "D in: " + path.string());
    }

    // Allocate output array
    xt::xarray<float> result = xt::empty<float>({shape[0], shape[1], shape[2]});

    // Read entire array
    z5::types::ShapeType offset = {0, 0, 0};
    z5::multiarray::readSubarray<float>(*ds, result, offset.begin());

    return result;
}

void write3DZarr(const std::filesystem::path& path, const xt::xarray<float>& data)
{
    namespace fs = std::filesystem;

    // Validate input is 3D
    if (data.dimension() != 3) {
        throw std::runtime_error("write3DZarr expects 3D array, got " +
                                 std::to_string(data.dimension()) + "D");
    }

    // Remove existing directory if present
    if (fs::exists(path)) {
        fs::remove_all(path);
    }

    // Create the zarr directory
    fs::create_directories(path);

    std::vector<size_t> shape = {data.shape(0), data.shape(1), data.shape(2)};

    // Write .zarray metadata (zarr v2 format, matches Python's zarr.save_array)
    nlohmann::json zarray_meta = {
        {"zarr_format", 2},
        {"shape", {shape[0], shape[1], shape[2]}},
        {"chunks", {shape[0], shape[1], shape[2]}},  // Single chunk
        {"dtype", "<f4"},  // little-endian float32
        {"compressor", nullptr},
        {"fill_value", 0.0},
        {"order", "C"},
        {"filters", nullptr},
        {"dimension_separator", "."}
    };

    std::ofstream zarray_file(path / ".zarray");
    if (!zarray_file) {
        throw std::runtime_error("Failed to create .zarray file: " + (path / ".zarray").string());
    }
    zarray_file << zarray_meta.dump(2);
    zarray_file.close();

    // Write chunk data (single chunk named 0.0.0)
    // xtensor stores data in row-major (C) order, write raw bytes
    std::ofstream chunk_file(path / "0.0.0", std::ios::binary);
    if (!chunk_file) {
        throw std::runtime_error("Failed to create chunk file: " + (path / "0.0.0").string());
    }
    chunk_file.write(reinterpret_cast<const char*>(data.data()),
                     data.size() * sizeof(float));
    chunk_file.close();
}
