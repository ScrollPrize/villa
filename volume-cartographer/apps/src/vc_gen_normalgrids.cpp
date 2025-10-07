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
#include <unordered_map>
#include <arpa/inet.h>

#include <opencv2/ximgproc.hpp>

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

static std::pair<SkeletonGraph, cv::Mat> generate_skeleton_graph(const cv::Mat& binary_slice, const po::variables_map& vm) {
    cv::Mat skeleton_img;
    cv::ximgproc::thinning(binary_slice, skeleton_img, cv::ximgproc::THINNING_GUOHALL);
    SkeletonGraph skeleton_graph = trace_skeleton_segments(skeleton_img, vm);
    return {std::move(skeleton_graph), std::move(skeleton_img)};
}

enum class SliceDirection { XY, XZ, YZ };

void run_generate(const po::variables_map& vm);
void run_convert(const po::variables_map& vm);

int main(int argc, char* argv[]) {
    po::options_description global("Global options");
    global.add_options()
        ("help,h", "Print usage message")
        ("command", po::value<std::string>(), "Command to execute (generate, convert)")
        ("subargs", po::value<std::vector<std::string>>(), "Arguments for command");

    po::positional_options_description pos;
    pos.add("command", 1).add("subargs", -1);

    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).
        options(global).
        positional(pos).
        allow_unregistered().
        run();
    
    po::store(parsed, vm);

    if (vm.count("help") || !vm.count("command")) {
        std::cout << "Usage: vc_gen_normalgrids <command> [options]\n\n"
                  << "Commands:\n"
                  << "  generate   Generate normal grids for all slices in a Zarr volume.\n"
                  << "  convert    Recursively find and convert GridStore files to the latest version.\n\n";
        return 0;
    }

    std::string cmd = vm["command"].as<std::string>();
    
    if (cmd == "generate") {
        po::options_description generate_desc("`generate` options");
        generate_desc.add_options()
            ("input,i", po::value<std::string>()->required(), "Input Zarr volume path")
            ("output,o", po::value<std::string>()->required(), "Output directory path")
            ("spiral-step", po::value<double>()->default_value(20.0), "Spiral step for resampling")
            ("grid-step", po::value<int>()->default_value(64), "Grid cell size for the GridStore");

        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        opts.erase(opts.begin()); // Erase the command
        
        po::variables_map generate_vm;
        po::store(po::command_line_parser(opts).options(generate_desc).run(), generate_vm);
        po::notify(generate_vm);
        run_generate(generate_vm);

    } else if (cmd == "convert") {
        po::options_description convert_desc("`convert` options");
        convert_desc.add_options()
            ("input,i", po::value<std::string>()->required(), "Input directory to scan for GridStore files")
            ("grid-step", po::value<int>()->default_value(64), "New grid cell size for the GridStore");

        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        opts.erase(opts.begin());

        po::variables_map convert_vm;
        po::store(po::command_line_parser(opts).options(convert_desc).run(), convert_vm);
        po::notify(convert_vm);
        run_convert(convert_vm);

    } else {
        std::cerr << "Error: Unknown command '" << cmd << "'" << std::endl;
        return 1;
    }

    return 0;
}

void run_convert(const po::variables_map& vm) {
    fs::path input_dir = vm["input"].as<std::string>();
    int new_grid_step = vm["grid-step"].as<int>();
    std::cout << "Scanning directory: " << input_dir << " with new grid step: " << new_grid_step << std::endl;

    std::vector<fs::path> grid_files;
    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".grid") {
            grid_files.push_back(entry.path());
        }
    }

    std::cout << "Found " << grid_files.size() << " grid files to process." << std::endl;

    std::atomic<size_t> converted_count = 0;
    std::atomic<size_t> skipped_count = 0;
    std::atomic<size_t> error_count = 0;
    std::atomic<size_t> processed_count = 0;

    #pragma omp parallel for
    for (size_t i = 0; i < grid_files.size(); ++i) {
        const auto& path = grid_files[i];
        try {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                #pragma omp critical
                std::cerr << "Error: Could not open file " << path << std::endl;
                error_count++;
                continue;
            }

            uint32_t magic, version;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            file.read(reinterpret_cast<char*>(&version), sizeof(version));
            magic = ntohl(magic);
            version = ntohl(version);

            if (magic != 0x56434753) { // "VCGS"
                #pragma omp critical
                std::cerr << "Warning: Skipping file with invalid magic: " << path << std::endl;
                skipped_count++;
                continue;
            }

            if (version < 3) {
                vc::core::util::GridStore old_store(path.string());
                vc::core::util::GridStore new_store(cv::Rect(0, 0, old_store.size().width, old_store.size().height), new_grid_step);
                
                auto all_paths = old_store.get_all();
                for(const auto& p : all_paths) {
                    new_store.add(*p);
                }
                new_store.meta = old_store.meta;

                std::string tmp_path = path.string() + ".tmp";
                new_store.save(tmp_path);
                fs::rename(tmp_path, path);
                converted_count++;
            } else {
                skipped_count++;
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            std::cerr << "Error processing file " << path << ": " << e.what() << std::endl;
            error_count++;
        }
        
        size_t processed = ++processed_count;
        if (processed % 100 == 0) {
            #pragma omp critical
            std::cout << "Processed " << processed << "/" << grid_files.size()
                      << " (Converted: " << converted_count
                      << ", Skipped: " << skipped_count
                      << ", Errors: " << error_count << ")" << std::endl;
        }
    }

    std::cout << "Conversion complete. Total processed: " << processed_count
              << ", Converted: " << converted_count
              << ", Skipped: " << skipped_count
              << ", Errors: " << error_count << std::endl;
}


void run_generate(const po::variables_map& vm) {
    std::string input_path = vm["input"].as<std::string>();
    std::string output_path = vm["output"].as<std::string>();

    std::cout << "Input Zarr path: " << input_path << std::endl;
    std::cout << "Output directory: " << output_path << std::endl;

    z5::filesystem::handle::Group group_handle(input_path);
    std::unique_ptr<z5::Dataset> ds = z5::openDataset(group_handle, "0");
    if (!ds) {
        std::cerr << "Error: Could not open dataset '0' in volume '" << input_path << "'." << std::endl;
        exit(1);
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

    ChunkCache cache(10llu*1024*1024*1024);

    size_t total_slices_all_dirs = shape[0] + shape[1] + shape[2];
    std::atomic<size_t> total_processed_all_dirs = 0;

    for (SliceDirection dir : {SliceDirection::XY, SliceDirection::XZ, SliceDirection::YZ}) {
        std::atomic<size_t> processed = 0;
        std::atomic<size_t> skipped = 0;
        std::atomic<size_t> total_size = 0;
        std::atomic<size_t> total_segments = 0;
        std::atomic<size_t> total_buckets = 0;
        
        struct TimingStats {
            std::atomic<size_t> count;
            std::atomic<double> total_time;
        };
        std::unordered_map<std::string, TimingStats> timings;

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
                total_processed_all_dirs++;
                continue;
            }

            xt::xtensor<uint8_t, 3, xt::layout_type::column_major> slice_data = xt::zeros<uint8_t>(slice_shape);
            
            ALifeTime t;
            readArea3D(slice_data, offset, ds.get(), &cache);
            t.mark("read");

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
                auto skeleton_res = generate_skeleton_graph(binary_slice, vm);
                t.mark("skeleton");
                auto& skeleton_graph = skeleton_res.first;

                vc::core::util::GridStore grid_store(cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), vm["grid-step"].as<int>());
                populate_normal_grid(skeleton_graph, grid_store, spiral_step);
                grid_store.save(tmp_path);
                fs::rename(tmp_path, out_path);
                t.mark("grid");

                if (i % 100 == 0) {
                    snprintf(filename, sizeof(filename), "%06zu.jpg", i);
                    cv::imwrite((output_fs_path / (dir_str + "_img") / filename).string(), binary_slice);
                }
                
                size_t file_size = fs::file_size(out_path);
                size_t num_segments = grid_store.numSegments();
                size_t num_buckets = grid_store.numNonEmptyBuckets();

                std::cout << dir_str << " Slice " << i << ": " << t.report() << std::endl;

                for(const auto& mark : t.getMarks()) {
                    timings[mark.first].count++;
                    timings[mark.first].total_time += mark.second;
                }

                total_size += file_size;
                total_segments += num_segments;
                total_buckets += num_buckets;
                processed++;
                total_processed_all_dirs++;
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 1) {
                std::lock_guard<std::mutex> lock(report_mutex);
                // Re-check in case another thread just reported
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 1) {
                    last_report_time = now;
                    size_t p = processed; // Read atomic once
                    size_t total_p = total_processed_all_dirs;
                    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                    double slices_per_second = (p > skipped) ? (p - skipped) / elapsed_seconds : 0.0;
                    if (slices_per_second == 0) slices_per_second = 1; // Avoid division by zero
                    double remaining_seconds = (total_slices_all_dirs - total_p) / slices_per_second;
                    
                    int rem_min = static_cast<int>(remaining_seconds) / 60;
                    int rem_sec = static_cast<int>(remaining_seconds) % 60;

                    std::cout << dir_str << " " << p << "/" << num_slices
                                << " | Total " << total_p << "/" << total_slices_all_dirs
                                << " (" << std::fixed << std::setprecision(1) << (100.0 * total_p / total_slices_all_dirs) << "%)"
                                << ", skipped: " << skipped
                                << ", ETA: " << rem_min << "m " << rem_sec << "s";
                    if (p > skipped) {
                        std::cout << ", avg size: " << (total_size / (p - skipped))
                                  << ", avg segments: " << (total_segments / (p - skipped))
                                  << ", avg buckets: " << (total_buckets / (p - skipped));
                    }

                    for(auto const& [key, val] : timings) {
                        if (val.count > 0) {
                            std::cout << ", avg " << key << ": " << (val.total_time / val.count) << "s";
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    std::cout << "Processing complete." << std::endl;
}
