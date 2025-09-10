#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <atomic>
#include <chrono>
#include <mutex>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "vc/core/util/Slicing.hpp"
#include <vc/core/util/GridStore.hpp>
#include "support.hpp"
#include "vc/core/util/LifeTime.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

enum class SliceDirection { XY, XZ, YZ };

int main(int argc, char* argv[]) {
    po::options_description desc("Generate normal grids for all slices in a Zarr volume.");
    desc.add_options()
        ("help,h", "Print usage message")
        ("input,i", po::value<std::string>()->required(), "Input Zarr volume path")
        ("output,o", po::value<std::string>()->required(), "Output directory path")
        ("spiral-step", po::value<double>()->default_value(8.0), "Spiral step for resampling")
        ("grid-step", po::value<int>()->default_value(64), "Grid cell size for the GridStore");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    std::string input_path = vm["input"].as<std::string>();
    std::string output_path = vm["output"].as<std::string>();

    std::cout << "Input Zarr path: " << input_path << std::endl;
    std::cout << "Output directory: " << output_path << std::endl;

    z5::filesystem::handle::Group group_handle(input_path);
    std::unique_ptr<z5::Dataset> ds = z5::openDataset(group_handle, "0");
    if (!ds) {
        std::cerr << "Error: Could not open dataset '0' in volume '" << input_path << "'." << std::endl;
        return 1;
    }
    auto shape = ds->shape();

    double spiral_step = vm["spiral-step"].as<double>();

    fs::path output_fs_path(output_path);
    fs::create_directories(output_fs_path / "xy");
    fs::create_directories(output_fs_path / "xz");
    fs::create_directories(output_fs_path / "yz");
    fs::create_directories(output_fs_path / "xy_img");
    fs::create_directories(output_fs_path / "xz_img");
    fs::create_directories(output_fs_path / "yz_img");

    nlohmann::json metadata;
    metadata["spiral-step"] = spiral_step;
    metadata["grid-step"] = vm["grid-step"].as<int>();
    std::ofstream o(output_fs_path / "metadata.json");
    o << std::setw(4) << metadata << std::endl;

    ChunkCache cache(4llu*1024*1024*1024);

    for (SliceDirection dir : {SliceDirection::XY, SliceDirection::XZ, SliceDirection::YZ}) {
        std::atomic<size_t> processed = 0;
        std::atomic<size_t> skipped = 0;
        std::atomic<size_t> total_size = 0;
        std::atomic<size_t> total_segments = 0;
        std::atomic<size_t> total_buckets = 0;
        auto last_report_time = std::chrono::steady_clock::now();
        auto start_time = std::chrono::steady_clock::now();
        std::mutex report_mutex;

        size_t num_slices;
        std::string dir_str;

        switch (dir) {
            case SliceDirection::XY: num_slices = shape[0]; dir_str = "xy"; break;
            case SliceDirection::XZ: num_slices = shape[1]; dir_str = "xz"; break;
            case SliceDirection::YZ: num_slices = shape[2]; dir_str = "yz"; break;
        }

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_slices; ++i) {
            std::vector<size_t> slice_shape;
            cv::Vec3i offset;
            cv::Mat slice_mat;

            switch (dir) {
                case SliceDirection::XY:
                    slice_shape = {1, shape[1], shape[2]};
                    offset = {(int)i, 0, 0};
                    slice_mat = cv::Mat(shape[1], shape[2], CV_8U);
                    break;
                case SliceDirection::XZ:
                    slice_shape = {shape[0], 1, shape[2]};
                    offset = {0, (int)i, 0};
                    slice_mat = cv::Mat(shape[0], shape[2], CV_8U);
                    break;
                case SliceDirection::YZ:
                    slice_shape = {shape[0], shape[1], 1};
                    offset = {0, 0, (int)i};
                    slice_mat = cv::Mat(shape[0], shape[1], CV_8U);
                    break;
            }

            char filename[256];
            snprintf(filename, sizeof(filename), "%06zu.grid", i);
            std::string out_path = (output_fs_path / dir_str / filename).string();
            std::string tmp_path = out_path + ".tmp";

            if (fs::exists(out_path)) {
                skipped++;
                processed++;
                continue;
            }

            xt::xtensor<uint8_t, 3, xt::layout_type::column_major> slice_data = xt::zeros<uint8_t>(slice_shape);
            double read_time, skeleton_time, grid_time;
            {
                ALifeTime t;
                readArea3D(slice_data, offset, ds.get(), &cache);
                read_time = t.seconds();
            }

            for (int z = 0; z < slice_mat.rows; ++z) {
                for (int y = 0; y < slice_mat.cols; ++y) {
                    switch (dir) {
                        case SliceDirection::XY: slice_mat.at<uint8_t>(z, y) = slice_data(0, z, y); break;
                        case SliceDirection::XZ: slice_mat.at<uint8_t>(z, y) = slice_data(z, 0, y); break;
                        case SliceDirection::YZ: slice_mat.at<uint8_t>(z, y) = slice_data(z, y, 0); break;
                    }
                }
            }
            
            cv::Mat binary_slice = slice_mat > 0;

            if (cv::countNonZero(binary_slice) == 0) {
                std::ofstream ofs(out_path); // Create empty file
                processed++;
            } else {
                decltype(generate_skeleton_graph(binary_slice, vm)) skeleton_res;
                {
                    ALifeTime t;
                    skeleton_res = generate_skeleton_graph(binary_slice, vm);
                    skeleton_time = t.seconds();
                }
                auto& [skeleton_graph, skeleton_id_img] = skeleton_res;

                vc::core::util::GridStore grid_store(cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), vm["grid-step"].as<int>());
                {
                    ALifeTime t;
                    populate_normal_grid(skeleton_graph, grid_store, spiral_step);
                    grid_store.save(tmp_path);
                    fs::rename(tmp_path, out_path);
                    grid_time = t.seconds();
                }

                snprintf(filename, sizeof(filename), "%06zu.tif", i);
                cv::imwrite((output_fs_path / (dir_str + "_img") / filename).string(), binary_slice);
                
                size_t file_size = fs::file_size(out_path);
                size_t num_segments = grid_store.numSegments();
                size_t num_buckets = grid_store.numNonEmptyBuckets();

                std::cout << dir_str << " Slice " << i << ": read " << read_time << "s, skeleton " << skeleton_time << "s, grid " << grid_time << "s" << std::endl;

                total_size += file_size;
                total_segments += num_segments;
                total_buckets += num_buckets;
                processed++;
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 1) {
                std::lock_guard<std::mutex> lock(report_mutex);
                // Re-check in case another thread just reported
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 1) {
                    last_report_time = now;
                    size_t p = processed; // Read atomic once
                    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                    double slices_per_second = p / elapsed_seconds;
                    double remaining_seconds = (num_slices - p) / slices_per_second;
                    
                    int rem_min = static_cast<int>(remaining_seconds) / 60;
                    int rem_sec = static_cast<int>(remaining_seconds) % 60;

                    std::cout << dir_str << " Avg: " << p << " / " << num_slices
                                << " (" << (100.0 * p / num_slices) << "%)"
                                << ", skipped: " << skipped
                                << ", ETA: " << rem_min << "m " << rem_sec << "s"
                                << ", avg size: " << (total_size / (p - skipped))
                                << ", avg segments: " << (total_segments / (p - skipped))
                                << ", avg buckets: " << (total_buckets / (p - skipped)) << std::endl;
                }
            }
        }
    }

    std::cout << "Processing complete." << std::endl;
    return 0;
}
