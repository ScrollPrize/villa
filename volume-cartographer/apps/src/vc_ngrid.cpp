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

#include <omp.h>
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
#include <opencv2/ximgproc.hpp>
#include "vc/core/util/Thinning.hpp"

#include "support.hpp"
#include "vc/core/util/normalgridtools.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

enum class SliceDirection { XY, XZ, YZ };

int main(int argc, char* argv[]) {
    po::options_description desc("vc_ngrid: Generate and visualize normal grids.\n\n"
                                 "Modes:\n"
                                 "  batch-vol-gen: Generate normal grids for all slices in a Zarr volume.\n"
                                 "  vis: Visualize a normal grid from a .grid file.\n"
                                 "  find-umbilicus: Find and visualize the umbilicus point from a .grid file.\n"
                                 "  align: Align and visualize segment directions.\n\n"
                                 "Options");
   desc.add_options()
       ("help,h", "Print usage message")
       ("mode", po::value<std::string>()->required(), "Mode to operate in (batch-vol-gen, vis, find-umbilicus, align)")
       ("input,i", po::value<std::string>()->required(), "Input path (Zarr volume for batch-vol-gen, .grid file for vis/find-umbilicus/align)")
       ("output,o", po::value<std::string>()->required(), "Output path (directory for batch-vol-gen, .tif file for vis/find-umbilicus/align)")
       ("spiral-step", po::value<double>()->default_value(20.0), "Spiral step for resampling (batch-vol-gen only)")
        ("grid-step", po::value<int>()->default_value(64), "Grid cell size for the GridStore (batch-vol-gen only)")
        ("sparse-volume", po::value<int>()->default_value(4), "Process every N-th slice (batch-vol-gen only)");

    po::positional_options_description p;
    p.add("mode", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

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

    std::string mode = vm["mode"].as<std::string>();

    if (mode == "vis") {
        std::string input_grid = vm["input"].as<std::string>();
        std::string output_vis = vm["output"].as<std::string>();

        vc::core::util::GridStore normal_grid(input_grid);
        cv::Mat vis = visualize_normal_grid(normal_grid, normal_grid.size());
        cv::imwrite(output_vis, vis);
        std::cout << "Visualization saved to " << output_vis << std::endl;

    } else if (mode == "find-umbilicus") {
        std::string input_grid = vm["input"].as<std::string>();
        std::string output_vis = vm["output"].as<std::string>();

        vc::core::util::GridStore normal_grid(input_grid);
        cv::Vec2f umbilicus = vc::core::util::align_and_extract_umbilicus(normal_grid);

        cv::Mat vis = visualize_normal_grid(normal_grid, normal_grid.size());
        if (!std::isnan(umbilicus[0]) && !std::isnan(umbilicus[1])) {
            cv::Point umbilicus_pt(cvRound(umbilicus[0]), cvRound(umbilicus[1]));
            cv::line(vis, cv::Point(umbilicus_pt.x, 0), cv::Point(umbilicus_pt.x, vis.rows), cv::Scalar(0, 255, 0), 1);
            cv::line(vis, cv::Point(0, umbilicus_pt.y), cv::Point(vis.cols, umbilicus_pt.y), cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite(output_vis, vis);
        std::cout << "Umbilicus visualization saved to " << output_vis << std::endl;

    } else if (mode == "align") {
        std::string input_grid = vm["input"].as<std::string>();
        std::string output_grid = vm["output"].as<std::string>();

        vc::core::util::GridStore normal_grid(input_grid);
        vc::core::util::GridStore res(cv::Rect(0, 0, normal_grid.size().width, normal_grid.size().height), 64);


        cv::Vec2f umbilicus = vc::core::util::align_and_extract_umbilicus(normal_grid);
        align_and_filter_segments(normal_grid, res, umbilicus);

        res.save(output_grid);
        std::cout << "Aligned grid saved to " << output_grid << std::endl;

    } else if (mode == "batch-vol-gen") {
        cv::setNumThreads(0);

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
        fs::create_directories(output_fs_path / "xy_thin");
        fs::create_directories(output_fs_path / "xz_thin");
        fs::create_directories(output_fs_path / "yz_thin");
        fs::create_directories(output_fs_path / "xy_traces");
        fs::create_directories(output_fs_path / "xz_traces");
        fs::create_directories(output_fs_path / "yz_traces");

        nlohmann::json metadata;
        metadata["spiral-step"] = spiral_step;
        metadata["grid-step"] = vm["grid-step"].as<int>();
        metadata["sparse-volume"] = vm["sparse-volume"].as<int>();
        std::ofstream o(output_fs_path / "metadata.json");
        o << std::setw(4) << metadata << std::endl;

        ChunkCache cache(1llu*1024*1024*1024);

        std::atomic<size_t> total_processed_all_dirs = 0;
        std::atomic<size_t> total_skipped_all_dirs = 0;
        double work_progress_per_dir[3] = {0.0, 0.0, 0.0};
        double display_progress_per_dir[3] = {0.0, 0.0, 0.0};

        int dir_idx = 0;
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

            int num_threads = omp_get_max_threads();
            if (num_threads == 0) {
                num_threads = 1;
            }
            int sparse_volume = vm["sparse-volume"].as<int>();
            if (!sparse_volume)
                sparse_volume = 1;
            int chunk_size_tgt = num_threads*sparse_volume;

            for (size_t chunk_start = 0; chunk_start < num_slices; chunk_start += chunk_size_tgt) {
                size_t chunk_end = std::min(chunk_start + chunk_size_tgt, num_slices);
                size_t chunk_size = chunk_end - chunk_start;

                bool all_exist = true;
                for (size_t i = chunk_start; i < chunk_end; ++i) {
                    if (i % sparse_volume != 0) {
                        continue;
                    }
                    char filename[256];
                    snprintf(filename, sizeof(filename), "%06zu.grid", i);
                    std::string out_path = (output_fs_path / dir_str / filename).string();
                    if (!fs::exists(out_path)) {
                        all_exist = false;
                        break;
                    }
                }

                if (all_exist) {
                    skipped += chunk_size;
                    processed += chunk_size;
                    total_processed_all_dirs += chunk_size;
                    total_skipped_all_dirs += chunk_size;
                    continue;
                }

                std::vector<size_t> chunk_shape;
                cv::Vec3i chunk_offset;
                switch (dir) {
                    case SliceDirection::XY:
                        chunk_shape = {chunk_size, shape[1], shape[2]};
                        chunk_offset = {(int)chunk_start, 0, 0};
                        break;
                    case SliceDirection::XZ:
                        chunk_shape = {shape[0], chunk_size, shape[2]};
                        chunk_offset = {0, (int)chunk_start, 0};
                        break;
                    case SliceDirection::YZ:
                        chunk_shape = {shape[0], shape[1], chunk_size};
                        chunk_offset = {0, 0, (int)chunk_start};
                        break;
                }

                ALifeTime chunk_timer;
                xt::xtensor<uint8_t, 3, xt::layout_type::column_major> chunk_data = xt::xtensor<uint8_t, 3, xt::layout_type::column_major>::from_shape(chunk_shape);
                chunk_timer.mark("xtensor init");
                readArea3D(chunk_data, chunk_offset, ds.get(), &cache);
                chunk_timer.mark("read_chunk");

                for(const auto& mark : chunk_timer.getMarks()) {
                    timings[mark.first].count++;
                    timings[mark.first].total_time += mark.second;
                    std::cout << mark.first << " " << mark.second << std::endl;
                }


                #pragma omp parallel for schedule(dynamic)
                for (size_t i_chunk = 0; i_chunk < chunk_size; ++i_chunk) {
                    size_t i = chunk_start + i_chunk;

                    if (i % vm["sparse-volume"].as<int>() != 0) {
                        processed++;
                        total_processed_all_dirs++;
                        continue;
                    }
                    cv::Mat slice_mat;

                    switch (dir) {
                        case SliceDirection::XY:
                            slice_mat = cv::Mat(shape[1], shape[2], CV_8U);
                            for (int z = 0; z < slice_mat.rows; ++z) {
                                for (int y = 0; y < slice_mat.cols; ++y) {
                                    slice_mat.at<uint8_t>(z, y) = chunk_data(i_chunk, z, y);
                                }
                            }
                            break;
                        case SliceDirection::XZ:
                            slice_mat = cv::Mat(shape[0], shape[2], CV_8U);
                             for (int z = 0; z < slice_mat.rows; ++z) {
                                for (int y = 0; y < slice_mat.cols; ++y) {
                                    slice_mat.at<uint8_t>(z, y) = chunk_data(z, i_chunk, y);
                                }
                            }
                            break;
                        case SliceDirection::YZ:
                            slice_mat = cv::Mat(shape[0], shape[1], CV_8U);
                            for (int z = 0; z < slice_mat.rows; ++z) {
                                for (int y = 0; y < slice_mat.cols; ++y) {
                                    slice_mat.at<uint8_t>(z, y) = chunk_data(z, y, i_chunk);
                                }
                            }
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
                    total_skipped_all_dirs++;
                    continue;
                }

                ALifeTime t;
                std::vector<std::vector<cv::Point>> traces;

                char traces_filename[256];
                snprintf(traces_filename, sizeof(traces_filename), "%06zu.grid", i);
                std::string traces_path = (output_fs_path / (dir_str + "_traces") / traces_filename).string();

                if (fs::exists(traces_path)) {
                    vc::core::util::GridStore trace_store(traces_path);
                    for (const auto& segment : trace_store.get_all()) {
                        traces.push_back(*segment);
                    }
                    t.mark("traces_from_cache");
                } else {

                    cv::Mat binary_slice = slice_mat > 0;

                    if (i % 100 == 0) {
                        snprintf(filename, sizeof(filename), "%06zu.tif", i);
                        cv::imwrite((output_fs_path / (dir_str + "_img") / filename).string(), binary_slice);
                    }

                    if (cv::countNonZero(binary_slice) == 0) {
                        std::ofstream ofs(out_path); // Create empty file
                        processed++;
                        continue;
                    }


                    t.mark("prepare_slice");
                    cv::Mat thinned_slice;
                    customThinning(binary_slice, thinned_slice, &traces);
                    t.mark("thinning");

                    // vc::core::util::GridStore trace_store(cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), slice_mat.cols);
                    // for (const auto& trace : traces) {
                    //     trace_store.add(trace);
                    // }
                    // trace_store.save(traces_path);
                    // t.mark("write_trace_cache");
                }

                if (traces.empty()) {
                    std::ofstream ofs(out_path); // Create empty file for empty graphs
                    processed++;
                } else {
                    vc::core::util::GridStore grid_store(cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), vm["grid-step"].as<int>());
                    populate_normal_grid(traces, grid_store, spiral_step);

                    // if (dir == SliceDirection::XY) {
                    //     cv::Vec2f umbilicus = vc::core::util::align_and_extract_umbilicus(grid_store);
                    //     vc::core::util::GridStore aligned_grid(cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), vm["grid-step"].as<int>());
                    //     align_and_filter_segments(grid_store, aligned_grid, umbilicus);
                    //     aligned_grid.save(tmp_path);
                    // } else {
                        grid_store.save(tmp_path);
                    // }
                    
                    fs::rename(tmp_path, out_path);
                    t.mark("grid");
                    
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
                        
                        display_progress_per_dir[dir_idx] = static_cast<double>(p) / num_slices;
                        double total_display_progress = (display_progress_per_dir[0] + display_progress_per_dir[1] + display_progress_per_dir[2]) / 3.0;

                        work_progress_per_dir[dir_idx] = static_cast<double>(p - skipped) / num_slices;
                        double total_work_progress = (work_progress_per_dir[0] + work_progress_per_dir[1] + work_progress_per_dir[2]) / 3.0;

                        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                        
                        double remaining_seconds = std::numeric_limits<double>::infinity();

                        if (elapsed_seconds > 1.0 && total_work_progress > 1e-6) {
                            double total_estimated_time = elapsed_seconds / total_work_progress;
                            remaining_seconds = total_estimated_time * (1.0 - total_display_progress);
                        }

                        int rem_min = static_cast<int>(remaining_seconds) / 60;
                        int rem_sec = static_cast<int>(remaining_seconds) % 60;

                        std::cout << dir_str << " " << p << "/" << num_slices
                                    << " | Total "
                                    << " (" << std::fixed << std::setprecision(1) << (total_display_progress * 100.0) << "%)"
                                    << ", skipped: " << skipped
                                    << ", ETA: " << rem_min << "m " << rem_sec << "s";
                        
                        auto finish_time_point = std::chrono::system_clock::now() + std::chrono::seconds(static_cast<long>(remaining_seconds));
                        std::time_t finish_time = std::chrono::system_clock::to_time_t(finish_time_point);
                        std::tm* finish_tm = std::localtime(&finish_time);

                        std::cout << " (finish at " << std::put_time(finish_tm, "%H:%M:%S") << ")"
                                    << ", avg size: " << (total_size / (p - skipped))
                                    << ", avg segments: " << (total_segments / (p - skipped))
                                    << ", avg buckets: " << (total_buckets / (p - skipped));

                        for(auto const& [key, val] : timings) {
                            if (val.count > 0) {
                                double avg_time = val.total_time / val.count;
                                if (key == "read_chunk") {
                                    avg_time /= num_threads;
                                }
                                std::cout << ", avg " << key << ": " << avg_time << "s";
                            }
                        }
                        std::cout << std::endl;
                    }
                }
                }
            }

            // Mark the current direction as fully processed for the next iteration's average calculation.
            display_progress_per_dir[dir_idx] = 1.0;
            work_progress_per_dir[dir_idx] = 1.0;
            dir_idx++;
        }

        std::cout << "Processing complete." << std::endl;

    } else {
        std::cerr << "Error: Invalid mode '" << mode << "'. Must be 'batch-vol-gen' or 'vis'." << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    return 0;
}
