#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "utils/Json.hpp"
#include <fstream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <arpa/inet.h>

#include <omp.h>


#include "vc/core/types/Volume.hpp"
#include <vc/core/util/GridStore.hpp>
#include "vc/core/util/NormalGridGenerate.hpp"
#include "vc/core/util/Thinning.hpp"
#include "support.hpp"
#include "vc/core/util/LifeTime.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

enum class SliceDirection { XY, XZ, YZ };

namespace {

using Json = utils::Json;

struct DirectionMetrics {
    std::string direction;
    size_t numSlices = 0;
    size_t sampledSlices = 0;
    size_t processed = 0;
    size_t skippedExisting = 0;
    size_t unsampled = 0;
    size_t emptyBinary = 0;
    size_t emptyTrace = 0;
    size_t written = 0;
    size_t previewWrites = 0;
    size_t totalSize = 0;
    size_t totalSegments = 0;
    size_t totalBuckets = 0;
    size_t chunkSizeTarget = 0;
    size_t sourceChunksTouched = 0;
    size_t bytesPerSlice = 0;
    size_t estimatedBatchBytes = 0;
    size_t thinningCalls = 0;
    std::unordered_map<std::string, double> timingTotals;
    std::unordered_map<std::string, size_t> timingCounts;
    ThinningStats thinningStats;
};

struct RunMetrics {
    std::string inputPath;
    std::string outputPath;
    int inputLevel = 0;
    int sparseVolume = 1;
    int gridStep = 64;
    double spiralStep = 20.0;
    size_t chunkBudgetMiB = 512;
    int previewEvery = 100;
    bool verifyGridSave = false;
    int ompThreads = 1;
    size_t cacheBudgetBytes = 0;
    size_t totalSlicesAllDirs = 0;
    size_t totalProcessedAllDirs = 0;
    size_t totalSkippedAllDirs = 0;
    double totalSeconds = 0.0;
    std::vector<size_t> levelShape;
    std::vector<DirectionMetrics> directions;
};

enum class SliceTaskKind { Unsampled, Exists, Process };

struct SliceTask {
    SliceTaskKind kind = SliceTaskKind::Unsampled;
    size_t sliceIndex = 0;
    fs::path outPath;
    fs::path tmpPath;
    fs::path previewPath;
};

struct ThreadSliceStats {
    size_t processed = 0;
    size_t skippedExisting = 0;
    size_t unsampled = 0;
    size_t emptyBinary = 0;
    size_t emptyTrace = 0;
    size_t written = 0;
    size_t previewWrites = 0;
    size_t totalSize = 0;
    size_t totalSegments = 0;
    size_t totalBuckets = 0;
    size_t thinningCalls = 0;
    std::unordered_map<std::string, double> timingTotals;
    std::unordered_map<std::string, size_t> timingCounts;
    ThinningStats thinningStats;
};

struct ThreadScratch {
    std::vector<std::vector<cv::Point>> traces;
};

struct AssembledSlice {
    SliceTask task;
    size_t localSliceIndex = 0;
    cv::Mat binarySlice;
    bool anyNonZero = false;
};

static const char* direction_name(SliceDirection dir) {
    switch (dir) {
    case SliceDirection::XY: return "xy";
    case SliceDirection::XZ: return "xz";
    case SliceDirection::YZ: return "yz";
    }
    return "xy";
}

static vc::core::util::NormalGridSliceDirection to_normal_grid_direction(SliceDirection dir) {
    switch (dir) {
    case SliceDirection::XY: return vc::core::util::NormalGridSliceDirection::XY;
    case SliceDirection::XZ: return vc::core::util::NormalGridSliceDirection::XZ;
    case SliceDirection::YZ: return vc::core::util::NormalGridSliceDirection::YZ;
    }
    return vc::core::util::NormalGridSliceDirection::XY;
}

static void write_metrics_json(const fs::path& path, const RunMetrics& metrics) {
    Json out;
    out["mode"] = "generate";
    out["input"] = metrics.inputPath;
    out["output"] = metrics.outputPath;
    out["input_level"] = metrics.inputLevel;
    out["sparse_volume"] = metrics.sparseVolume;
    out["grid_step"] = metrics.gridStep;
    out["spiral_step"] = metrics.spiralStep;
    out["chunk_budget_mib"] = metrics.chunkBudgetMiB;
    out["preview_every"] = metrics.previewEvery;
    out["verify_grid_save"] = metrics.verifyGridSave;
    out["omp_threads"] = metrics.ompThreads;
    out["cache_budget_bytes"] = metrics.cacheBudgetBytes;
    {
        Json arr = Json::array();
        for (auto v : metrics.levelShape) arr.push_back(static_cast<int64_t>(v));
        out["level_shape_zyx"] = std::move(arr);
    }
    out["total_slices_all_dirs"] = metrics.totalSlicesAllDirs;
    out["total_processed_all_dirs"] = metrics.totalProcessedAllDirs;
    out["total_skipped_all_dirs"] = metrics.totalSkippedAllDirs;
    out["total_seconds"] = metrics.totalSeconds;
    out["directions"] = Json::array();

    for (const auto& dir : metrics.directions) {
        Json d;
        d["direction"] = dir.direction;
        d["num_slices"] = dir.numSlices;
        d["sampled_slices"] = dir.sampledSlices;
        d["processed"] = dir.processed;
        d["skipped_existing"] = dir.skippedExisting;
        d["unsampled"] = dir.unsampled;
        d["empty_binary"] = dir.emptyBinary;
        d["empty_trace"] = dir.emptyTrace;
        d["written"] = dir.written;
        d["preview_writes"] = dir.previewWrites;
        d["total_size"] = dir.totalSize;
        d["total_segments"] = dir.totalSegments;
        d["total_buckets"] = dir.totalBuckets;
        d["chunk_size_target"] = dir.chunkSizeTarget;
        d["source_chunks_touched"] = dir.sourceChunksTouched;
        d["bytes_per_slice"] = dir.bytesPerSlice;
        d["estimated_batch_bytes"] = dir.estimatedBatchBytes;
        d["timings"] = Json::object();
        for (const auto& [name, total] : dir.timingTotals) {
            Json t;
            t["total_seconds"] = total;
            const size_t count = dir.timingCounts.contains(name) ? dir.timingCounts.at(name) : 0;
            t["count"] = count;
            t["avg_seconds"] = count > 0 ? total / static_cast<double>(count) : 0.0;
            d["timings"][name] = t;
        }
        if (dir.thinningCalls > 0) {
            d["timings"]["thinning_detail"] = {
                {"count", dir.thinningCalls},
                {"distance_transform_seconds", dir.thinningStats.distanceTransformSeconds},
                {"seed_detection_seconds", dir.thinningStats.seedDetectionSeconds},
                {"trace_paths_seconds", dir.thinningStats.tracePathsSeconds},
                {"avg_distance_transform_seconds", dir.thinningStats.distanceTransformSeconds / static_cast<double>(dir.thinningCalls)},
                {"avg_seed_detection_seconds", dir.thinningStats.seedDetectionSeconds / static_cast<double>(dir.thinningCalls)},
                {"avg_trace_paths_seconds", dir.thinningStats.tracePathsSeconds / static_cast<double>(dir.thinningCalls)},
                {"seed_count", dir.thinningStats.seedCount},
                {"trace_count", dir.thinningStats.traceCount},
                {"trace_steps", dir.thinningStats.traceSteps},
                {"candidate_evaluations", dir.thinningStats.candidateEvaluations},
            };
        }
        out["directions"].push_back(std::move(d));
    }

    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open metrics json for writing: " + path.string());
    }
    file << out.dump(2) << "\n";
    if (!file) {
        throw std::runtime_error("Failed writing metrics json: " + path.string());
    }
}

} // namespace

void run_generate(const po::variables_map& vm);
void run_convert(const po::variables_map& vm);

static void print_usage() {
    std::cout << "vc_gen_normalgrids: Generate and manage normal grids for volume data.\n\n"
              << "Usage: vc_gen_normalgrids [command] [options]\n\n"
              << "Commands:\n"
              << "  generate   Generate normal grids for all slices in a Zarr volume (default).\n"
              << "  convert    Recursively find and convert GridStore files to the latest version.\n\n"
              << "Examples:\n"
              << "  vc_gen_normalgrids -i /path/to/volume.zarr -o /path/to/output/\n"
              << "  vc_gen_normalgrids -i vol.zarr -o out/ --sparse-volume 4\n"
              << "  vc_gen_normalgrids convert -i /path/to/grids/\n\n"
              << "Generate options:\n"
              << "  -i, --input         Input Zarr volume path (required)\n"
              << "  -o, --output        Output directory path (required)\n"
              << "  --level            Input OME-Zarr pyramid level (default: 0)\n"
              << "  --spiral-step       Spiral step for resampling paths (default: 20.0)\n"
              << "  --grid-step         Grid cell size for spatial indexing (default: 64)\n"
              << "  --sparse-volume     Process every N-th slice, 1 = all (default: 1)\n"
              << "  --chunk-budget-mib  Max chunk batch budget per direction (default: 512)\n"
              << "  --preview-every     Write preview image every N written slices, 0 disables (default: 100)\n"
              << "  --verify-grid-save  Verify GridStore save by reloading each file (default: false)\n"
              << "  --metrics-json      Write structured metrics json\n\n"
              << "Convert options:\n"
              << "  -i, --input         Input directory to scan for .grid files (required)\n"
              << "  --grid-step         New grid cell size (default: 64)\n";
}

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

    // Determine command - default to "generate" if not specified or not recognized
    std::string cmd = "generate";
    bool explicit_command = false;
    if (vm.count("command")) {
        std::string maybe_cmd = vm["command"].as<std::string>();
        if (maybe_cmd == "generate" || maybe_cmd == "convert") {
            cmd = maybe_cmd;
            explicit_command = true;
        }
        // Otherwise treat it as an option for generate (e.g., user typed -i directly)
    }

    // Show help if no args or if explicitly requested with --help only
    if (argc == 1 || (vm.count("help") && argc == 2)) {
        print_usage();
        return 0;
    }

    if (cmd == "generate") {
        po::options_description generate_desc(
            "vc_gen_normalgrids generate: Generate normal grids for all slices in a Zarr volume.\n\n"
            "Uses chunked I/O for efficient processing of large volumes. Processes slices\n"
            "in all three directions (XY, XZ, YZ) and generates .grid files containing\n"
            "traced skeleton paths with normal information.\n\n"
            "Options");
        generate_desc.add_options()
            ("help,h", "Print this help message")
            ("input,i", po::value<std::string>()->required(), "Input Zarr volume path")
            ("output,o", po::value<std::string>()->required(), "Output directory path")
            ("level", po::value<int>()->default_value(0), "Input OME-Zarr level to read")
            ("spiral-step", po::value<double>()->default_value(20.0), "Spiral step for resampling paths")
            ("grid-step", po::value<int>()->default_value(64), "Grid cell size for spatial indexing")
            ("sparse-volume", po::value<int>()->default_value(1), "Process every N-th slice (1 = all slices)")
            ("chunk-budget-mib", po::value<size_t>()->default_value(512), "Maximum chunk batch budget in MiB")
            ("preview-every", po::value<int>()->default_value(100), "Write preview image every N written slices, 0 disables")
            ("verify-grid-save", po::bool_switch()->default_value(false), "Verify GridStore files by reloading after save")
            ("metrics-json", po::value<std::string>(), "Write structured metrics json");

        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        if (explicit_command && !opts.empty()) {
            opts.erase(opts.begin()); // Erase the command only if explicitly given
        }

        // Check for help before parsing required options
        for (const auto& opt : opts) {
            if (opt == "-h" || opt == "--help") {
                std::cout << generate_desc << std::endl;
                return 0;
            }
        }

        po::variables_map generate_vm;
        try {
            po::store(po::command_line_parser(opts).options(generate_desc).run(), generate_vm);
            po::notify(generate_vm);
        } catch (const po::error& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            std::cout << generate_desc << std::endl;
            return 1;
        }
        run_generate(generate_vm);

    } else if (cmd == "convert") {
        po::options_description convert_desc(
            "vc_gen_normalgrids convert: Convert GridStore files to the latest format.\n\n"
            "Recursively scans a directory for .grid files and converts any older\n"
            "format versions to the current version.\n\n"
            "Options");
        convert_desc.add_options()
            ("help,h", "Print this help message")
            ("input,i", po::value<std::string>()->required(), "Input directory to scan for GridStore files")
            ("grid-step", po::value<int>()->default_value(64), "New grid cell size for the GridStore");

        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        if (explicit_command && !opts.empty()) {
            opts.erase(opts.begin()); // Erase the command only if explicitly given
        }

        // Check for help before parsing required options
        for (const auto& opt : opts) {
            if (opt == "-h" || opt == "--help") {
                std::cout << convert_desc << std::endl;
                return 0;
            }
        }

        po::variables_map convert_vm;
        try {
            po::store(po::command_line_parser(opts).options(convert_desc).run(), convert_vm);
            po::notify(convert_vm);
        } catch (const po::error& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            std::cout << convert_desc << std::endl;
            return 1;
        }
        run_convert(convert_vm);

    } else {
        std::cerr << "Error: Unknown command '" << cmd << "'\n\n";
        print_usage();
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
    const auto total_start = std::chrono::steady_clock::now();
    const std::string input_path = vm["input"].as<std::string>();
    const std::string output_path = vm["output"].as<std::string>();
    const int input_level = vm["level"].as<int>();
    if (input_level < 0) {
        throw std::runtime_error("--level must be >= 0");
    }

    const double spiral_step = vm["spiral-step"].as<double>();
    const int grid_step = vm["grid-step"].as<int>();
    int sparse_volume = vm["sparse-volume"].as<int>();
    if (sparse_volume < 1) sparse_volume = 1;
    const size_t chunk_budget_mib = vm["chunk-budget-mib"].as<size_t>();
    const int preview_every = vm["preview-every"].as<int>();
    if (preview_every < 0) {
        throw std::runtime_error("--preview-every must be >= 0");
    }
    const bool verify_grid_save = vm["verify-grid-save"].as<bool>();
    const std::optional<fs::path> metrics_json_path = vm.count("metrics-json")
        ? std::optional<fs::path>(fs::path(vm["metrics-json"].as<std::string>()))
        : std::nullopt;

    std::cout << "Input Zarr path: " << input_path << std::endl;
    std::cout << "Input level: " << input_level << std::endl;
    std::cout << "Output directory: " << output_path << std::endl;

    Volume input_volume{fs::path(input_path)};
    auto* input_chunks = input_volume.chunkedCache();
    const auto level_shape = input_chunks->shape(input_level);
    const auto level_chunk_shape = input_chunks->chunkShape(input_level);
    const size_t dtype_size = input_chunks->dtype() == vc::render::ChunkDtype::UInt16 ? 2 : 1;
    const std::vector<size_t> shape = {
        static_cast<size_t>(level_shape[0]),
        static_cast<size_t>(level_shape[1]),
        static_cast<size_t>(level_shape[2]),
    };

    fs::path output_fs_path(output_path);
    fs::create_directories(output_fs_path / "xy");
    fs::create_directories(output_fs_path / "xz");
    fs::create_directories(output_fs_path / "yz");
    fs::create_directories(output_fs_path / "xy_img");
    fs::create_directories(output_fs_path / "xz_img");
    fs::create_directories(output_fs_path / "yz_img");

    Json metadata;
    metadata["spiral-step"] = spiral_step;
    metadata["grid-step"] = grid_step;
    metadata["sparse-volume"] = sparse_volume;
    metadata["input-level"] = input_level;
    metadata["chunk-budget-mib"] = chunk_budget_mib;
    metadata["preview-every"] = preview_every;
    metadata["verify-grid-save"] = verify_grid_save;
    std::ofstream o(output_fs_path / "metadata.json");
    o << metadata.dump(4) << std::endl;

    int num_threads = omp_get_max_threads();
    if (num_threads == 0) num_threads = 1;

    const size_t cache_budget_bytes = 10ull * 1024ull * 1024ull * 1024ull;

    input_volume.setCacheBudget(cache_budget_bytes);

    RunMetrics run_metrics;
    run_metrics.inputPath = input_path;
    run_metrics.outputPath = output_path;
    run_metrics.inputLevel = input_level;
    run_metrics.sparseVolume = sparse_volume;
    run_metrics.gridStep = grid_step;
    run_metrics.spiralStep = spiral_step;
    run_metrics.chunkBudgetMiB = chunk_budget_mib;
    run_metrics.previewEvery = preview_every;
    run_metrics.verifyGridSave = verify_grid_save;
    run_metrics.ompThreads = num_threads;
    run_metrics.cacheBudgetBytes = cache_budget_bytes;
    run_metrics.levelShape = shape;
    run_metrics.totalSlicesAllDirs = shape[0] + shape[1] + shape[2];

    std::vector<ThreadScratch> thread_scratch(static_cast<size_t>(num_threads));
    for (auto& scratch : thread_scratch) {
        scratch.traces.reserve(256);
    }
    const std::vector<size_t> source_chunk_shape = {
        static_cast<size_t>(level_chunk_shape[0]),
        static_cast<size_t>(level_chunk_shape[1]),
        static_cast<size_t>(level_chunk_shape[2]),
    };

    for (SliceDirection dir : {SliceDirection::XY, SliceDirection::XZ, SliceDirection::YZ}) {
        DirectionMetrics dir_metrics;
        dir_metrics.direction = direction_name(dir);

        size_t num_slices = 0;
        switch (dir) {
            case SliceDirection::XY: num_slices = shape[0]; break;
            case SliceDirection::XZ: num_slices = shape[1]; break;
            case SliceDirection::YZ: num_slices = shape[2]; break;
        }
        dir_metrics.numSlices = num_slices;

        const auto batch_plan = vc::core::util::planNormalGridBatch(
            shape,
            to_normal_grid_direction(dir),
            num_threads,
            sparse_volume,
            chunk_budget_mib,
            dtype_size);
        const size_t chunk_size_tgt = std::max<size_t>(1, batch_plan.chunkSizeTarget);
        dir_metrics.chunkSizeTarget = chunk_size_tgt;
        dir_metrics.bytesPerSlice = batch_plan.bytesPerSlice;
        dir_metrics.estimatedBatchBytes = batch_plan.estimatedBatchBytes;
        const auto sampled_chunk_plans = vc::core::util::planNormalGridSampledChunks(
            shape,
            source_chunk_shape,
            to_normal_grid_direction(dir),
            sparse_volume);
        size_t sampled_slices_total = 0;
        for (const auto& chunk_plan : sampled_chunk_plans) {
            sampled_slices_total += chunk_plan.sampledSlices.size();
        }
        dir_metrics.sampledSlices = sampled_slices_total;
        dir_metrics.sourceChunksTouched = sampled_chunk_plans.size();

        size_t processed = num_slices - sampled_slices_total;
        size_t skipped_existing = 0;
        size_t unsampled = num_slices - sampled_slices_total;
        size_t total_size = 0;
        size_t total_segments = 0;
        size_t total_buckets = 0;

        auto last_report_time = std::chrono::steady_clock::now();
        auto start_time = std::chrono::steady_clock::now();
        std::atomic<size_t> written_counter{0};
        run_metrics.totalProcessedAllDirs += unsampled;

        const cv::Size slice_size = vc::core::util::normalGridSliceSize(
            shape,
            to_normal_grid_direction(dir));
        const size_t chunk_count_z = (shape[0] + source_chunk_shape[0] - 1) / source_chunk_shape[0];
        const size_t chunk_count_y = (shape[1] + source_chunk_shape[1] - 1) / source_chunk_shape[1];
        const size_t chunk_count_x = (shape[2] + source_chunk_shape[2] - 1) / source_chunk_shape[2];

        for (const auto& source_chunk_plan : sampled_chunk_plans) {
            for (size_t batch_start = 0;
                 batch_start < source_chunk_plan.sampledSlices.size();
                 batch_start += chunk_size_tgt) {
                const size_t batch_end = std::min(
                    batch_start + chunk_size_tgt,
                    source_chunk_plan.sampledSlices.size());
                const size_t batch_size = batch_end - batch_start;

                std::vector<SliceTask> tasks(batch_size);
                std::vector<AssembledSlice> assembled_slices;
                assembled_slices.reserve(batch_size);
                size_t batch_existing = 0;

                for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                    const auto& sampled =
                        source_chunk_plan.sampledSlices[batch_start + batch_index];
                    auto& task = tasks[batch_index];
                    task.sliceIndex = sampled.sliceIndex;

                    char filename[256];
                    snprintf(filename, sizeof(filename), "%06zu.grid", sampled.sliceIndex);
                    task.outPath = output_fs_path / dir_metrics.direction / filename;
                    task.tmpPath = fs::path(task.outPath.string() + ".tmp");

                    char preview_filename[256];
                    snprintf(preview_filename, sizeof(preview_filename), "%06zu.jpg", sampled.sliceIndex);
                    task.previewPath = output_fs_path / (dir_metrics.direction + "_img") / preview_filename;

                    if (fs::exists(task.outPath)) {
                        task.kind = SliceTaskKind::Exists;
                        ++batch_existing;
                        continue;
                    }

                    task.kind = SliceTaskKind::Process;
                    auto& assembled = assembled_slices.emplace_back();
                    assembled.task = task;
                    assembled.localSliceIndex = sampled.localSliceIndex;
                    assembled.binarySlice = cv::Mat::zeros(slice_size, CV_8U);
                }

                if (!assembled_slices.empty()) {
                    std::array<size_t, 3> slab_shape;
                    std::array<int, 3> slab_offset;
                    switch (dir) {
                    case SliceDirection::XY:
                        slab_shape = {source_chunk_shape[0], shape[1], shape[2]};
                        slab_offset = {
                            static_cast<int>(source_chunk_plan.sourceChunkIndex * source_chunk_shape[0]),
                            0,
                            0,
                        };
                        break;
                    case SliceDirection::XZ:
                        slab_shape = {shape[0], source_chunk_shape[1], shape[2]};
                        slab_offset = {
                            0,
                            static_cast<int>(source_chunk_plan.sourceChunkIndex * source_chunk_shape[1]),
                            0,
                        };
                        break;
                    case SliceDirection::YZ:
                        slab_shape = {shape[0], shape[1], source_chunk_shape[2]};
                        slab_offset = {
                            0,
                            0,
                            static_cast<int>(source_chunk_plan.sourceChunkIndex * source_chunk_shape[2]),
                        };
                        break;
                    }
                    // Clip slab to actual volume extents so we do not read past the end.
                    for (int axis = 0; axis < 3; ++axis) {
                        const size_t end = static_cast<size_t>(slab_offset[axis]) + slab_shape[axis];
                        if (end > shape[axis]) {
                            slab_shape[axis] = shape[axis] - static_cast<size_t>(slab_offset[axis]);
                        }
                    }

                    const auto read_start = std::chrono::steady_clock::now();
                    Array3D<uint8_t> chunk_data(slab_shape);
                    input_volume.readZYX(chunk_data, slab_offset, input_level);
                    dir_metrics.timingTotals["read_chunk"] += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - read_start).count();
                    dir_metrics.timingCounts["read_chunk"] += 1;

                    for (auto& assembled : assembled_slices) {
                        const bool any_nonzero = vc::core::util::extractBinarySliceFromChunk(
                            chunk_data,
                            to_normal_grid_direction(dir),
                            assembled.localSliceIndex,
                            assembled.binarySlice);
                        assembled.anyNonZero = assembled.anyNonZero || any_nonzero;
                    }
                }

                std::vector<ThreadSliceStats> thread_stats(static_cast<size_t>(num_threads));

                #pragma omp parallel for schedule(dynamic)
                for (size_t batch_index = 0; batch_index < assembled_slices.size(); ++batch_index) {
                    const int tid = omp_get_thread_num();
                    ThreadSliceStats& local_stats = thread_stats[static_cast<size_t>(tid)];
                    ThreadScratch& scratch = thread_scratch[static_cast<size_t>(tid)];
                    const auto& assembled = assembled_slices[batch_index];
                    const SliceTask& task = assembled.task;

                    if (!assembled.anyNonZero) {
                        std::ofstream ofs(task.outPath);
                        ++local_stats.emptyBinary;
                        ++local_stats.processed;
                        continue;
                    }

                    scratch.traces.clear();
                    const auto thinning_start = std::chrono::steady_clock::now();
                    ThinningStats thinning_stats;
                    customThinningTraceOnly(assembled.binarySlice, scratch.traces, &thinning_stats);
                    local_stats.timingTotals["thinning"] += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - thinning_start).count();
                    local_stats.timingCounts["thinning"] += 1;
                    local_stats.thinningStats.accumulate(thinning_stats);
                    ++local_stats.thinningCalls;

                    if (scratch.traces.empty()) {
                        std::ofstream ofs(task.outPath);
                        ++local_stats.emptyTrace;
                        ++local_stats.processed;
                        continue;
                    }

                    vc::core::util::GridStore grid_store(
                        cv::Rect(0, 0, assembled.binarySlice.cols, assembled.binarySlice.rows),
                        grid_step);

                    const auto populate_start = std::chrono::steady_clock::now();
                    populate_normal_grid(scratch.traces, grid_store, spiral_step);
                    local_stats.timingTotals["populate_grid"] += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - populate_start).count();
                    local_stats.timingCounts["populate_grid"] += 1;

                    const auto save_start = std::chrono::steady_clock::now();
                    grid_store.save(task.tmpPath.string(), vc::core::util::GridStore::SaveOptions{
                        .verify_reload = verify_grid_save,
                    });
                    fs::rename(task.tmpPath, task.outPath);
                    local_stats.timingTotals["save_grid"] += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - save_start).count();
                    local_stats.timingCounts["save_grid"] += 1;

                    const size_t written_index = written_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (preview_every > 0 && (written_index % static_cast<size_t>(preview_every)) == 0) {
                        const auto preview_start = std::chrono::steady_clock::now();
                        cv::imwrite(task.previewPath.string(), assembled.binarySlice);
                        local_stats.timingTotals["preview_image"] += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - preview_start).count();
                        local_stats.timingCounts["preview_image"] += 1;
                        ++local_stats.previewWrites;
                    }

                    local_stats.totalSize += fs::file_size(task.outPath);
                    local_stats.totalSegments += grid_store.numSegments();
                    local_stats.totalBuckets += grid_store.numNonEmptyBuckets();
                    ++local_stats.written;
                    ++local_stats.processed;
                }

                processed += batch_existing;
                skipped_existing += batch_existing;
                run_metrics.totalProcessedAllDirs += batch_size;
                run_metrics.totalSkippedAllDirs += batch_existing;

                for (const auto& local_stats : thread_stats) {
                    processed += local_stats.processed;
                    dir_metrics.emptyBinary += local_stats.emptyBinary;
                    dir_metrics.emptyTrace += local_stats.emptyTrace;
                    dir_metrics.written += local_stats.written;
                    dir_metrics.previewWrites += local_stats.previewWrites;
                    total_size += local_stats.totalSize;
                    total_segments += local_stats.totalSegments;
                    total_buckets += local_stats.totalBuckets;

                    for (const auto& [name, total] : local_stats.timingTotals) {
                        dir_metrics.timingTotals[name] += total;
                        dir_metrics.timingCounts[name] += local_stats.timingCounts.at(name);
                    }
                    dir_metrics.thinningStats.accumulate(local_stats.thinningStats);
                    dir_metrics.thinningCalls += local_stats.thinningCalls;
                }
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time).count() >= 1) {
                last_report_time = now;
                const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                const size_t active_processed = processed > (skipped_existing + unsampled)
                    ? (processed - skipped_existing - unsampled)
                    : 0;
                double slices_per_second = active_processed > 0 ? active_processed / elapsed_seconds : 0.0;
                if (slices_per_second == 0.0) slices_per_second = 1.0;
                const double remaining_seconds =
                    (run_metrics.totalSlicesAllDirs - run_metrics.totalProcessedAllDirs) / slices_per_second;

                int rem_min = static_cast<int>(remaining_seconds) / 60;
                int rem_sec = static_cast<int>(remaining_seconds) % 60;

                std::cout << dir_metrics.direction << " " << processed << "/" << num_slices
                          << " | Total " << run_metrics.totalProcessedAllDirs << "/" << run_metrics.totalSlicesAllDirs
                          << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * run_metrics.totalProcessedAllDirs / run_metrics.totalSlicesAllDirs) << "%)"
                          << ", skipped_existing: " << skipped_existing
                          << ", unsampled: " << unsampled
                          << ", ETA: " << rem_min << "m " << rem_sec << "s";
                if (dir_metrics.written > 0) {
                    std::cout << ", avg size: " << (total_size / dir_metrics.written)
                              << ", avg segments: " << (total_segments / dir_metrics.written)
                              << ", avg buckets: " << (total_buckets / dir_metrics.written);
                }
                for (const auto& [name, total] : dir_metrics.timingTotals) {
                    const size_t count = dir_metrics.timingCounts[name];
                    if (count > 0) {
                        std::cout << ", avg " << name << ": " << (total / static_cast<double>(count)) << "s";
                    }
                }
                std::cout << std::endl;
            }
        }

        dir_metrics.processed = processed;
        dir_metrics.skippedExisting = skipped_existing;
        dir_metrics.unsampled = unsampled;
        dir_metrics.totalSize = total_size;
        dir_metrics.totalSegments = total_segments;
        dir_metrics.totalBuckets = total_buckets;
        run_metrics.directions.push_back(std::move(dir_metrics));
    }

    run_metrics.totalSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();

    if (metrics_json_path.has_value()) {
        write_metrics_json(*metrics_json_path, run_metrics);
    }

    std::cout << "Processing complete." << std::endl;
}
