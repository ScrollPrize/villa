// Test program to compare H264-compressed zarr with original zarr
// Usage: test_h264_zarr <h264_zarr_path> <original_zarr_path>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>

#include <nlohmann/json.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)

using shape_t = z5::types::ShapeType;

struct Stats {
    double min_diff = std::numeric_limits<double>::max();
    double max_diff = 0;
    double sum_diff = 0;
    double sum_sq_diff = 0;
    size_t count = 0;
    size_t zero_count = 0;
    size_t nonzero_h264 = 0;
    size_t nonzero_orig = 0;

    void update(uint8_t h264_val, uint8_t orig_val) {
        double diff = std::abs(static_cast<double>(h264_val) - static_cast<double>(orig_val));
        min_diff = std::min(min_diff, diff);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        sum_sq_diff += diff * diff;
        count++;
        if (h264_val == 0) zero_count++;
        if (h264_val != 0) nonzero_h264++;
        if (orig_val != 0) nonzero_orig++;
    }

    double mean() const { return count > 0 ? sum_diff / count : 0; }
    double rmse() const { return count > 0 ? std::sqrt(sum_sq_diff / count) : 0; }
    double zero_pct() const { return count > 0 ? 100.0 * zero_count / count : 0; }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <h264_zarr_path> <original_zarr_path>\n";
        std::cerr << "Example: " << argv[0]
                  << " /path/to/scroll5_openh264.zarr /path/to/scroll5.zarr\n";
        return 1;
    }

    const std::string h264_path = argv[1];
    const std::string orig_path = argv[2];

    std::cout << "H264 zarr: " << h264_path << "\n";
    std::cout << "Original zarr: " << orig_path << "\n\n";

    // Open H264 zarr
    std::cout << "Opening H264 zarr...\n";
    z5::filesystem::handle::Group h264_group(h264_path, z5::FileMode::FileMode::r);

    // Read .zarray to get dimension separator
    std::ifstream h264_zarray(h264_path + "/0/.zarray");
    auto h264_meta = nlohmann::json::parse(h264_zarray);
    std::string h264_dim_sep = h264_meta.value("dimension_separator", ".");
    std::cout << "H264 .zarray:\n" << h264_meta.dump(2) << "\n\n";

    z5::filesystem::handle::Dataset h264_ds_handle(h264_group, "0", h264_dim_sep);
    auto h264_ds = z5::filesystem::openDataset(h264_ds_handle);

    std::cout << "H264 shape: [" << h264_ds->shape()[0] << ", "
              << h264_ds->shape()[1] << ", " << h264_ds->shape()[2] << "]\n";
    std::cout << "H264 chunks: [" << h264_ds->chunking().blockShape()[0] << ", "
              << h264_ds->chunking().blockShape()[1] << ", "
              << h264_ds->chunking().blockShape()[2] << "]\n\n";

    // Open original zarr
    std::cout << "Opening original zarr...\n";
    z5::filesystem::handle::Group orig_group(orig_path, z5::FileMode::FileMode::r);

    std::ifstream orig_zarray(orig_path + "/0/.zarray");
    auto orig_meta = nlohmann::json::parse(orig_zarray);
    std::string orig_dim_sep = orig_meta.value("dimension_separator", "/");
    std::cout << "Original .zarray:\n" << orig_meta.dump(2) << "\n\n";

    z5::filesystem::handle::Dataset orig_ds_handle(orig_group, "0", orig_dim_sep);
    auto orig_ds = z5::filesystem::openDataset(orig_ds_handle);

    std::cout << "Original shape: [" << orig_ds->shape()[0] << ", "
              << orig_ds->shape()[1] << ", " << orig_ds->shape()[2] << "]\n";
    std::cout << "Original chunks: [" << orig_ds->chunking().blockShape()[0] << ", "
              << orig_ds->chunking().blockShape()[1] << ", "
              << orig_ds->chunking().blockShape()[2] << "]\n\n";

    // Test reading a single chunk from H264 zarr
    std::cout << "=== Testing single chunk read from H264 zarr ===\n";
    {
        shape_t chunk_shape = h264_ds->chunking().blockShape();
        xt::xarray<uint8_t> h264_chunk = xt::zeros<uint8_t>({chunk_shape[0], chunk_shape[1], chunk_shape[2]});

        // Read chunk at position (100, 50, 50) - should be in the middle of the volume
        shape_t offset = {6400, 3200, 4544};  // Center-ish position
        shape_t read_shape = {chunk_shape[0], chunk_shape[1], chunk_shape[2]};

        std::cout << "Reading H264 chunk at offset [" << offset[0] << ", " << offset[1] << ", " << offset[2] << "]...\n";

        auto start = std::chrono::high_resolution_clock::now();
        try {
            z5::multiarray::readSubarray<uint8_t>(*h264_ds, h264_chunk, offset.begin());
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            // Calculate statistics
            size_t zero_count = 0;
            size_t nonzero_count = 0;
            uint8_t min_val = 255, max_val = 0;
            double sum = 0;

            for (size_t z = 0; z < chunk_shape[0]; z++) {
                for (size_t y = 0; y < chunk_shape[1]; y++) {
                    for (size_t x = 0; x < chunk_shape[2]; x++) {
                        uint8_t val = h264_chunk(z, y, x);
                        if (val == 0) zero_count++;
                        else nonzero_count++;
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        sum += val;
                    }
                }
            }

            size_t total = chunk_shape[0] * chunk_shape[1] * chunk_shape[2];
            std::cout << "  Read time: " << elapsed << "s\n";
            std::cout << "  Chunk size: " << total << " voxels\n";
            std::cout << "  Zero voxels: " << zero_count << " (" << 100.0 * zero_count / total << "%)\n";
            std::cout << "  Non-zero voxels: " << nonzero_count << " (" << 100.0 * nonzero_count / total << "%)\n";
            std::cout << "  Min value: " << (int)min_val << "\n";
            std::cout << "  Max value: " << (int)max_val << "\n";
            std::cout << "  Mean value: " << sum / total << "\n";

            // Print a small sample of values
            std::cout << "  Sample values at (32,32,32): ";
            for (int i = 0; i < 10; i++) {
                std::cout << (int)h264_chunk(32, 32, 32 + i) << " ";
            }
            std::cout << "\n\n";

        } catch (const std::exception& e) {
            std::cerr << "  ERROR reading H264 chunk: " << e.what() << "\n\n";
        }
    }

    // Test reading same chunk from original zarr
    std::cout << "=== Testing single chunk read from original zarr ===\n";
    {
        shape_t orig_chunk_shape = orig_ds->chunking().blockShape();
        shape_t h264_chunk_shape = h264_ds->chunking().blockShape();
        xt::xarray<uint8_t> orig_chunk = xt::zeros<uint8_t>({h264_chunk_shape[0], h264_chunk_shape[1], h264_chunk_shape[2]});

        // Read same position from original
        shape_t offset = {6400, 3200, 4544};

        std::cout << "Reading original chunk at offset [" << offset[0] << ", " << offset[1] << ", " << offset[2] << "]...\n";

        auto start = std::chrono::high_resolution_clock::now();
        try {
            z5::multiarray::readSubarray<uint8_t>(*orig_ds, orig_chunk, offset.begin());
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            // Calculate statistics
            size_t zero_count = 0;
            size_t nonzero_count = 0;
            uint8_t min_val = 255, max_val = 0;
            double sum = 0;

            for (size_t z = 0; z < h264_chunk_shape[0]; z++) {
                for (size_t y = 0; y < h264_chunk_shape[1]; y++) {
                    for (size_t x = 0; x < h264_chunk_shape[2]; x++) {
                        uint8_t val = orig_chunk(z, y, x);
                        if (val == 0) zero_count++;
                        else nonzero_count++;
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        sum += val;
                    }
                }
            }

            size_t total = h264_chunk_shape[0] * h264_chunk_shape[1] * h264_chunk_shape[2];
            std::cout << "  Read time: " << elapsed << "s\n";
            std::cout << "  Chunk size: " << total << " voxels\n";
            std::cout << "  Zero voxels: " << zero_count << " (" << 100.0 * zero_count / total << "%)\n";
            std::cout << "  Non-zero voxels: " << nonzero_count << " (" << 100.0 * nonzero_count / total << "%)\n";
            std::cout << "  Min value: " << (int)min_val << "\n";
            std::cout << "  Max value: " << (int)max_val << "\n";
            std::cout << "  Mean value: " << sum / total << "\n";

            // Print a small sample of values
            std::cout << "  Sample values at (32,32,32): ";
            for (int i = 0; i < 10; i++) {
                std::cout << (int)orig_chunk(32, 32, 32 + i) << " ";
            }
            std::cout << "\n\n";

        } catch (const std::exception& e) {
            std::cerr << "  ERROR reading original chunk: " << e.what() << "\n\n";
        }
    }

    // Compare multiple chunks
    std::cout << "=== Comparing multiple chunks ===\n";
    Stats overall_stats;

    // Test positions - various locations in the volume
    std::vector<shape_t> test_positions = {
        {0, 0, 0},
        {64, 64, 64},
        {6400, 3200, 4544},
        {10000, 3000, 4000},
        {15000, 5000, 7000},
    };

    shape_t h264_chunk_shape = h264_ds->chunking().blockShape();

    for (const auto& offset : test_positions) {
        // Check bounds
        if (offset[0] + h264_chunk_shape[0] > orig_ds->shape()[0] ||
            offset[1] + h264_chunk_shape[1] > orig_ds->shape()[1] ||
            offset[2] + h264_chunk_shape[2] > orig_ds->shape()[2]) {
            std::cout << "Skipping offset [" << offset[0] << ", " << offset[1] << ", " << offset[2]
                      << "] - out of bounds for original\n";
            continue;
        }

        xt::xarray<uint8_t> h264_chunk = xt::zeros<uint8_t>({h264_chunk_shape[0], h264_chunk_shape[1], h264_chunk_shape[2]});
        xt::xarray<uint8_t> orig_chunk = xt::zeros<uint8_t>({h264_chunk_shape[0], h264_chunk_shape[1], h264_chunk_shape[2]});

        std::cout << "Comparing chunk at [" << offset[0] << ", " << offset[1] << ", " << offset[2] << "]...\n";

        try {
            z5::multiarray::readSubarray<uint8_t>(*h264_ds, h264_chunk, offset.begin());
            z5::multiarray::readSubarray<uint8_t>(*orig_ds, orig_chunk, offset.begin());

            Stats chunk_stats;
            for (size_t z = 0; z < h264_chunk_shape[0]; z++) {
                for (size_t y = 0; y < h264_chunk_shape[1]; y++) {
                    for (size_t x = 0; x < h264_chunk_shape[2]; x++) {
                        chunk_stats.update(h264_chunk(z, y, x), orig_chunk(z, y, x));
                        overall_stats.update(h264_chunk(z, y, x), orig_chunk(z, y, x));
                    }
                }
            }

            std::cout << "  Mean diff: " << std::fixed << std::setprecision(2) << chunk_stats.mean()
                      << ", Max diff: " << chunk_stats.max_diff
                      << ", RMSE: " << chunk_stats.rmse()
                      << ", H264 zeros: " << chunk_stats.zero_pct() << "%"
                      << ", H264 nonzero: " << chunk_stats.nonzero_h264
                      << ", Orig nonzero: " << chunk_stats.nonzero_orig << "\n";

        } catch (const std::exception& e) {
            std::cerr << "  ERROR: " << e.what() << "\n";
        }
    }

    std::cout << "\n=== Overall Statistics ===\n";
    std::cout << "Total voxels compared: " << overall_stats.count << "\n";
    std::cout << "Mean absolute difference: " << std::fixed << std::setprecision(4) << overall_stats.mean() << "\n";
    std::cout << "Max absolute difference: " << overall_stats.max_diff << "\n";
    std::cout << "RMSE: " << overall_stats.rmse() << "\n";
    std::cout << "H264 zero voxels: " << overall_stats.zero_pct() << "%\n";
    std::cout << "H264 non-zero voxels: " << overall_stats.nonzero_h264 << "\n";
    std::cout << "Original non-zero voxels: " << overall_stats.nonzero_orig << "\n";

    return 0;
}
