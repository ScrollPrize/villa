// vc_zarr_compress_bench: Thorough streaming C3D compression benchmark.
// Processes ALL chunks iteratively — loads each chunk, tests all presets,
// accumulates stats, then discards it. Constant memory usage regardless
// of volume size.
//
// Usage: vc_zarr_compress_bench <zarr_path> [--level N] [--jobs N]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>

#include "utils/video_codec.hpp"

namespace fs = std::filesystem;

static constexpr int NUM_PRESETS = 6;
static constexpr size_t RAW_CHUNK_SIZE = 128 * 128 * 128;

static std::vector<std::byte> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + p.string());
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static std::vector<std::byte> to_raw(const std::vector<std::byte>& data) {
    if (utils::is_c3d_compressed(std::span<const std::byte>(data))) {
        utils::VideoCodecParams dp;
        dp.type = utils::VideoCodecType::C3D;
        dp.depth = 128; dp.height = 128; dp.width = 128;
        return utils::video_decode(std::span<const std::byte>(data), RAW_CHUNK_SIZE, dp);
    }
    if (data.size() == RAW_CHUNK_SIZE) return data;
    if (data.size() < RAW_CHUNK_SIZE) {
        std::vector<std::byte> padded(RAW_CHUNK_SIZE, std::byte{0});
        std::memcpy(padded.data(), data.data(), data.size());
        return padded;
    }
    return {};
}

// Per-preset accumulator, lock-free friendly (each thread has its own copy)
struct PresetAccum {
    int count = 0;
    size_t total_raw = 0;
    size_t total_compressed = 0;
    size_t total_voxels = 0;
    size_t global_err_hist[256] = {};
    double worst_max = 0;

    // For interior/boundary split
    int count_int = 0, count_bnd = 0;
    size_t raw_int = 0, comp_int = 0, raw_bnd = 0, comp_bnd = 0;
    size_t voxels_int = 0, voxels_bnd = 0;
    size_t err_hist_int[256] = {};
    size_t err_hist_bnd[256] = {};
    double worst_max_int = 0, worst_max_bnd = 0;

    void merge(const PresetAccum& o) {
        count += o.count;
        total_raw += o.total_raw;
        total_compressed += o.total_compressed;
        total_voxels += o.total_voxels;
        for (int e = 0; e < 256; e++) global_err_hist[e] += o.global_err_hist[e];
        if (o.worst_max > worst_max) worst_max = o.worst_max;

        count_int += o.count_int; count_bnd += o.count_bnd;
        raw_int += o.raw_int; comp_int += o.comp_int;
        raw_bnd += o.raw_bnd; comp_bnd += o.comp_bnd;
        voxels_int += o.voxels_int; voxels_bnd += o.voxels_bnd;
        for (int e = 0; e < 256; e++) {
            err_hist_int[e] += o.err_hist_int[e];
            err_hist_bnd[e] += o.err_hist_bnd[e];
        }
        if (o.worst_max_int > worst_max_int) worst_max_int = o.worst_max_int;
        if (o.worst_max_bnd > worst_max_bnd) worst_max_bnd = o.worst_max_bnd;
    }
};

struct HistStats {
    double psnr, rmse, mean_err;
    double p90, p95, p99;
    double ratio;
    double worst_max;
    int count;
};

static HistStats compute_from_hist(
    const size_t* hist, size_t total_voxels,
    size_t total_raw, size_t total_compressed,
    double worst_max, int count)
{
    HistStats s{};
    s.count = count;
    s.worst_max = worst_max;
    s.ratio = total_compressed > 0 ? (double)total_raw / total_compressed : 0;

    if (total_voxels == 0) return s;

    double sum_sq = 0, sum_abs = 0;
    for (int e = 0; e < 256; e++) {
        sum_sq += (double)e * e * hist[e];
        sum_abs += (double)e * hist[e];
    }
    double mse = sum_sq / total_voxels;
    s.psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 999.0;
    s.rmse = std::sqrt(mse);
    s.mean_err = sum_abs / total_voxels;

    size_t cum = 0;
    s.p90 = 0; s.p95 = 0; s.p99 = 0;
    size_t t90 = (size_t)(total_voxels * 0.90);
    size_t t95 = (size_t)(total_voxels * 0.95);
    size_t t99 = (size_t)(total_voxels * 0.99);
    for (int e = 0; e < 256; e++) {
        cum += hist[e];
        if (cum > t90 && s.p90 == 0 && e > 0) s.p90 = e;
        if (cum > t95 && s.p95 == 0 && e > 0) s.p95 = e;
        if (cum > t99 && s.p99 == 0 && e > 0) s.p99 = e;
    }

    return s;
}

static void print_row(const char* cat, const HistStats& s) {
    if (s.count == 0) return;
    printf("  %-12s %6d  %7.2fx  %7.1f dB  %6.3f  %6.3f  %4.0f  %4.0f  %4.0f  %4.0f\n",
           cat, s.count, s.ratio, s.psnr, s.rmse, s.mean_err,
           s.p90, s.p95, s.p99, s.worst_max);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: vc_zarr_compress_bench <zarr_path> [--level N] [--jobs N]\n";
        return 1;
    }

    fs::path zarr_path = argv[1];
    int pyramid_level = 0;
    int jobs = std::max(1, (int)std::thread::hardware_concurrency() / 2);

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--level" && i + 1 < argc) pyramid_level = std::atoi(argv[++i]);
        else if (arg == "--jobs" && i + 1 < argc) jobs = std::atoi(argv[++i]);
    }

    auto lpath = zarr_path / std::to_string(pyramid_level);
    if (!fs::is_directory(lpath)) {
        std::cerr << "Level " << pyramid_level << " not found\n";
        return 1;
    }

    // Collect all chunk paths
    std::vector<fs::path> chunks;
    for (auto& entry : fs::recursive_directory_iterator(lpath)) {
        if (!entry.is_regular_file()) continue;
        auto fname = entry.path().filename().string();
        if (fname[0] == '.' || fname.find(".json") != std::string::npos) continue;
        chunks.push_back(entry.path());
    }

    printf("Processing ALL %zu chunks from level %d with %d threads\n",
           chunks.size(), pyramid_level, jobs);
    printf("Streaming mode: constant memory, all 6 presets per chunk\n\n");

    auto t0 = std::chrono::steady_clock::now();

    // Per-thread accumulators (one per preset per thread)
    std::vector<std::vector<PresetAccum>> thread_accums(jobs,
        std::vector<PresetAccum>(NUM_PRESETS));

    std::atomic<size_t> next_idx{0};
    std::atomic<int> done{0};
    std::atomic<int> errors{0};

    auto worker = [&](int tid) {
        auto& accums = thread_accums[tid];

        utils::VideoCodecParams p;
        p.type = utils::VideoCodecType::C3D;
        p.depth = 128; p.height = 128; p.width = 128;

        while (true) {
            size_t idx = next_idx.fetch_add(1);
            if (idx >= chunks.size()) break;

            std::vector<std::byte> raw;
            try {
                auto data = read_file(chunks[idx]);
                raw = to_raw(data);
            } catch (...) {
                errors.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            if (raw.empty()) { done.fetch_add(1); continue; }

            // Classify: interior (>50% nonzero) vs boundary
            size_t nonzero = 0;
            for (size_t i = 0; i < RAW_CHUNK_SIZE; i++) {
                if (raw[i] != std::byte{0}) nonzero++;
            }
            bool is_int = ((double)nonzero / RAW_CHUNK_SIZE) > 0.5;

            // Test all 6 presets on this chunk
            for (int si = 0; si < NUM_PRESETS; si++) {
                p.qp = si;

                auto compressed = utils::video_encode(
                    std::span<const std::byte>(raw), p);
                auto decoded = utils::video_decode(
                    std::span<const std::byte>(compressed), RAW_CHUNK_SIZE, p);

                auto& acc = accums[si];
                acc.count++;
                acc.total_raw += RAW_CHUNK_SIZE;
                acc.total_compressed += compressed.size();
                acc.total_voxels += RAW_CHUNK_SIZE;

                // Compute per-voxel errors into histogram
                int max_e = 0;
                for (size_t i = 0; i < RAW_CHUNK_SIZE; i++) {
                    int e = std::abs(
                        static_cast<int>(static_cast<uint8_t>(raw[i])) -
                        static_cast<int>(static_cast<uint8_t>(decoded[i])));
                    acc.global_err_hist[e]++;
                    if (e > max_e) max_e = e;
                }
                if (max_e > acc.worst_max) acc.worst_max = max_e;

                // Split stats
                if (is_int) {
                    acc.count_int++;
                    acc.raw_int += RAW_CHUNK_SIZE;
                    acc.comp_int += compressed.size();
                    acc.voxels_int += RAW_CHUNK_SIZE;
                    for (size_t i = 0; i < RAW_CHUNK_SIZE; i++) {
                        int e = std::abs(
                            static_cast<int>(static_cast<uint8_t>(raw[i])) -
                            static_cast<int>(static_cast<uint8_t>(decoded[i])));
                        acc.err_hist_int[e]++;
                    }
                    if (max_e > acc.worst_max_int) acc.worst_max_int = max_e;
                } else {
                    acc.count_bnd++;
                    acc.raw_bnd += RAW_CHUNK_SIZE;
                    acc.comp_bnd += compressed.size();
                    acc.voxels_bnd += RAW_CHUNK_SIZE;
                    for (size_t i = 0; i < RAW_CHUNK_SIZE; i++) {
                        int e = std::abs(
                            static_cast<int>(static_cast<uint8_t>(raw[i])) -
                            static_cast<int>(static_cast<uint8_t>(decoded[i])));
                        acc.err_hist_bnd[e]++;
                    }
                    if (max_e > acc.worst_max_bnd) acc.worst_max_bnd = max_e;
                }
            }

            int d = done.fetch_add(1) + 1;
            if (d % 500 == 0) {
                auto now = std::chrono::steady_clock::now();
                double secs = std::chrono::duration<double>(now - t0).count();
                fprintf(stderr, "  %d/%zu chunks (%.0f/s, %.1f min remaining)\n",
                        d, chunks.size(), d / secs,
                        (chunks.size() - d) / (d / secs) / 60.0);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int j = 0; j < jobs; j++) threads.emplace_back(worker, j);
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Merge thread accumulators
    std::vector<PresetAccum> merged(NUM_PRESETS);
    for (int j = 0; j < jobs; j++) {
        for (int si = 0; si < NUM_PRESETS; si++) {
            merged[si].merge(thread_accums[j][si]);
        }
    }

    // Print results
    printf("\nCompleted %d chunks in %.1f min (%.0f chunks/s, %d errors)\n\n",
           done.load(), elapsed / 60, done.load() / elapsed, errors.load());

    const char* labels[] = {
        "Lossless(1x)", "1/2 lossless", "1/4 lossless",
        "1/8 lossless", "1/16 lossless", "1/32 lossless"
    };

    for (int si = 0; si < NUM_PRESETS; si++) {
        auto& m = merged[si];
        printf("=== %s (shift=%d) ===\n", labels[si], si);
        printf("  %-12s %6s  %8s  %10s  %6s  %6s  %4s  %4s  %4s  %4s\n",
               "Category", "Chunks", "Ratio", "PSNR", "RMSE", "Mean", "P90", "P95", "P99", "Max");
        printf("  %s\n", std::string(88, '-').c_str());

        auto all = compute_from_hist(m.global_err_hist, m.total_voxels,
                                      m.total_raw, m.total_compressed,
                                      m.worst_max, m.count);
        auto interior = compute_from_hist(m.err_hist_int, m.voxels_int,
                                           m.raw_int, m.comp_int,
                                           m.worst_max_int, m.count_int);
        auto boundary = compute_from_hist(m.err_hist_bnd, m.voxels_bnd,
                                           m.raw_bnd, m.comp_bnd,
                                           m.worst_max_bnd, m.count_bnd);

        print_row("ALL", all);
        print_row("Interior", interior);
        print_row("Boundary", boundary);

        printf("  Size: %zu MB -> %zu MB\n\n",
               m.total_raw / 1024 / 1024, m.total_compressed / 1024 / 1024);
    }

    return 0;
}
