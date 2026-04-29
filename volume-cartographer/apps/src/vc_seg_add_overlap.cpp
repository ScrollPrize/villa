#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using Json = utils::Json;

namespace {

constexpr float kOverlapTolerance = 2.0f;

struct TargetInfo {
    std::string key;
    std::string id;
    fs::path path;
};

struct SourceScanResult {
    fs::path sourcePath;
    std::string sourceKey;
    std::string sourceId;
    std::vector<std::pair<std::string, std::string>> hits;
    size_t queriedPoints = 0;
};

struct Config {
    fs::path targetRoot;
    fs::path sourceRoot;
    size_t requestedWorkers = 0;
    size_t pointStride = 1;
    bool help = false;
};

std::string canonical_key(const fs::path& path)
{
    std::error_code ec;
    fs::path p = fs::weakly_canonical(path, ec);
    if (ec) {
        p = fs::absolute(path, ec);
    }
    if (ec) {
        p = path;
    }
    return p.lexically_normal().generic_string();
}

bool is_tifxyz_dir(const fs::path& dir)
{
    std::error_code ec;
    if (!fs::is_directory(dir, ec) || ec) {
        return false;
    }

    const fs::path metaPath = dir / "meta.json";
    if (!fs::exists(metaPath, ec) || ec) {
        return false;
    }

    try {
        Json meta = Json::parse_file(metaPath);
        return meta.value("format", std::string{"NONE"}) == "tifxyz" &&
               meta.count("bbox") &&
               fs::exists(dir / "x.tif") &&
               fs::exists(dir / "y.tif") &&
               fs::exists(dir / "z.tif");
    } catch (const std::exception& e) {
        std::cerr << "Skipping " << dir << ": failed to parse meta.json: "
                  << e.what() << std::endl;
        return false;
    }
}

std::vector<fs::path> discover_tifxyz_dirs(const fs::path& root)
{
    std::vector<fs::path> dirs;
    if (is_tifxyz_dir(root)) {
        dirs.push_back(root);
        return dirs;
    }

    std::error_code ec;
    if (!fs::is_directory(root, ec) || ec) {
        throw std::runtime_error("Not a directory: " + root.string());
    }

    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory() && is_tifxyz_dir(entry.path())) {
            dirs.push_back(entry.path());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

std::vector<SurfacePatchIndex::SurfacePtr>
load_surfaces(const std::vector<fs::path>& dirs)
{
    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    surfaces.reserve(dirs.size());
    for (const auto& dir : dirs) {
        surfaces.push_back(std::make_shared<QuadSurface>(dir));
    }
    return surfaces;
}

void print_usage(const char* argv0)
{
    std::cout << "usage: " << argv0
              << " --target <target-dir> --source <source-tifxyz-or-source-dir> [options]\n"
              << "   Builds a stride-1 surface patch index for target-dir and adds\n"
              << "   overlapping.json metadata for overlaps found from the source.\n"
              << "   source may be one tifxyz segment directory or a directory of segments.\n"
              << "\n"
              << "options:\n"
              << "   --workers N        source-scan workers; default is min(source count, hardware concurrency)\n"
              << "   --point-stride N   check one out of every N valid source points; default is 1\n"
              << "   --help             show this help\n";
}

size_t parse_positive_size(const std::string& name, const std::string& value)
{
    try {
        size_t parsedChars = 0;
        const unsigned long long parsed = std::stoull(value, &parsedChars);
        if (parsedChars != value.size() ||
            parsed == 0 ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument("invalid positive integer");
        }
        return static_cast<size_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + name + " value: " + value);
    }
}

Config parse_args(int argc, char* argv[])
{
    Config cfg;

    auto require_value = [&](int& i, const std::string& option) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cfg.help = true;
            return cfg;
        }
        if (arg == "--target") {
            cfg.targetRoot = require_value(i, arg);
        } else if (arg == "--source") {
            cfg.sourceRoot = require_value(i, arg);
        } else if (arg == "--workers") {
            cfg.requestedWorkers = parse_positive_size("workers", require_value(i, arg));
        } else if (arg == "--point-stride") {
            cfg.pointStride = parse_positive_size("point-stride", require_value(i, arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.targetRoot.empty()) {
        throw std::runtime_error("Missing required --target");
    }
    if (cfg.sourceRoot.empty()) {
        throw std::runtime_error("Missing required --source");
    }
    return cfg;
}

std::string format_duration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return "--:--";
    }

    const auto totalSeconds = static_cast<long long>(seconds + 0.5);
    const long long hours = totalSeconds / 3600;
    const long long minutes = (totalSeconds % 3600) / 60;
    const long long secs = totalSeconds % 60;

    std::ostringstream out;
    if (hours > 0) {
        out << hours << ':'
            << std::setw(2) << std::setfill('0') << minutes << ':'
            << std::setw(2) << std::setfill('0') << secs;
    } else {
        out << minutes << ':'
            << std::setw(2) << std::setfill('0') << secs;
    }
    return out.str();
}

void print_progress(size_t completed,
                    size_t total,
                    size_t queriedPoints,
                    size_t pairCount,
                    std::chrono::steady_clock::time_point start,
                    bool final)
{
    constexpr size_t width = 36;
    const size_t remaining = total > completed ? total - completed : 0;
    const double frac = total == 0 ? 1.0 : static_cast<double>(completed) / static_cast<double>(total);
    const size_t filled = std::min(width, static_cast<size_t>(frac * width + 0.5));

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double eta = completed == 0
        ? -1.0
        : elapsed * static_cast<double>(remaining) / static_cast<double>(completed);

    std::cout << '\r' << '[';
    for (size_t i = 0; i < width; ++i) {
        std::cout << (i < filled ? '#' : '-');
    }
    std::cout << "] " << completed << '/' << total
              << " done, " << remaining << " left"
              << ", eta " << format_duration(eta)
              << ", points " << queriedPoints
              << ", overlaps " << pairCount
              << "          " << std::flush;

    if (final) {
        std::cout << std::endl;
    }
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc == 1) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    try {
        const Config cfg = parse_args(argc, argv);
        if (cfg.help) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }

        const std::vector<fs::path> targetDirs = discover_tifxyz_dirs(cfg.targetRoot);
        const std::vector<fs::path> sourceDirs = discover_tifxyz_dirs(cfg.sourceRoot);

        if (targetDirs.empty()) {
            std::cerr << "No tifxyz segments found in target: " << cfg.targetRoot << std::endl;
            return EXIT_FAILURE;
        }
        if (sourceDirs.empty()) {
            std::cerr << "No tifxyz segments found in source: " << cfg.sourceRoot << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Target tifxyz segments: " << targetDirs.size() << std::endl;
        std::cout << "Source tifxyz segments: " << sourceDirs.size() << std::endl;
        std::cout << "Source point stride: " << cfg.pointStride << std::endl;

        std::vector<SurfacePatchIndex::SurfacePtr> targetSurfaces = load_surfaces(targetDirs);
        std::unordered_map<std::string, SurfacePatchIndex::SurfacePtr> targetByPath;
        std::unordered_map<const QuadSurface*, TargetInfo> targetInfoBySurface;
        targetByPath.reserve(targetSurfaces.size());
        targetInfoBySurface.reserve(targetSurfaces.size());
        for (const auto& surface : targetSurfaces) {
            const std::string key = canonical_key(surface->path);
            targetByPath.emplace(key, surface);
            targetInfoBySurface.emplace(surface.get(), TargetInfo{key, surface->id, surface->path});
        }

        std::unordered_map<std::string, fs::path> sourcePathByKey;
        sourcePathByKey.reserve(sourceDirs.size());
        for (const fs::path& sourceDir : sourceDirs) {
            sourcePathByKey.emplace(canonical_key(sourceDir), sourceDir);
        }

        SurfacePatchIndex index;
        index.setSamplingStride(1);

        const auto buildStart = std::chrono::steady_clock::now();
        index.rebuild(targetSurfaces);
        const double buildSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - buildStart).count();
        std::cout << "SurfacePatchIndex built: surfaces=" << index.surfaceCount()
                  << " patches=" << index.patchCount()
                  << " seconds=" << buildSeconds << std::endl;

        const unsigned hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
        const size_t automaticWorkers = std::min<size_t>(sourceDirs.size(), hardwareThreads);
        const size_t workerCount = cfg.requestedWorkers == 0
            ? automaticWorkers
            : std::min(sourceDirs.size(), cfg.requestedWorkers);
        std::cout << "Scanning sources with " << workerCount << " worker"
                  << (workerCount == 1 ? "" : "s") << std::endl;

        std::vector<SourceScanResult> scanResults(sourceDirs.size());
        std::vector<std::string> workerErrors;
        std::mutex workerErrorsMutex;
        std::atomic_size_t nextSource{0};
        std::atomic_size_t completedSources{0};
        std::atomic_size_t queriedPoints{0};
        std::atomic_size_t discoveredPairs{0};
        std::atomic_bool scanDone{false};
        const auto scanStart = std::chrono::steady_clock::now();

        std::thread progressThread([&]() {
            while (!scanDone.load(std::memory_order_acquire)) {
                print_progress(completedSources.load(std::memory_order_relaxed),
                               sourceDirs.size(),
                               queriedPoints.load(std::memory_order_relaxed),
                               discoveredPairs.load(std::memory_order_relaxed),
                               scanStart,
                               false);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });

        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (size_t worker = 0; worker < workerCount; ++worker) {
            workers.emplace_back([&]() {
                while (true) {
                    const size_t sourceIndex = nextSource.fetch_add(1, std::memory_order_relaxed);
                    if (sourceIndex >= sourceDirs.size()) {
                        return;
                    }

                    try {
                        QuadSurface source(sourceDirs[sourceIndex]);
                        SourceScanResult result;
                        result.sourcePath = source.path;
                        result.sourceKey = canonical_key(source.path);
                        result.sourceId = source.id;

                        std::unordered_set<const QuadSurface*> excludedTargetSurfaces;
                        excludedTargetSurfaces.reserve(32);
                        if (const auto selfIt = targetByPath.find(result.sourceKey);
                            selfIt != targetByPath.end()) {
                            excludedTargetSurfaces.insert(selfIt->second.get());
                        }
                        std::vector<const QuadSurface*> pointHits;
                        pointHits.reserve(16);
                        size_t progressPointBatch = 0;
                        size_t validPointIndex = 0;
                        for (const auto& pointRef : source.validPoints()) {
                            const size_t currentPointIndex = validPointIndex++;
                            if (currentPointIndex % cfg.pointStride != 0) {
                                continue;
                            }

                            ++result.queriedPoints;
                            ++progressPointBatch;
                            if (progressPointBatch >= 4096) {
                                queriedPoints.fetch_add(progressPointBatch, std::memory_order_relaxed);
                                progressPointBatch = 0;
                            }

                            index.locateSurfaceHits(pointRef.point,
                                                    kOverlapTolerance,
                                                    excludedTargetSurfaces,
                                                    pointHits);
                            for (const QuadSurface* hitSurface : pointHits) {
                                if (!hitSurface) {
                                    continue;
                                }

                                const auto targetIt = targetInfoBySurface.find(hitSurface);
                                if (targetIt == targetInfoBySurface.end()) {
                                    continue;
                                }

                                const TargetInfo& target = targetIt->second;
                                if (target.key == result.sourceKey) {
                                    continue;
                                }
                                if (!excludedTargetSurfaces.insert(hitSurface).second) {
                                    continue;
                                }

                                result.hits.emplace_back(target.key, target.id);
                                discoveredPairs.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        if (progressPointBatch > 0) {
                            queriedPoints.fetch_add(progressPointBatch, std::memory_order_relaxed);
                        }

                        scanResults[sourceIndex] = std::move(result);
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(workerErrorsMutex);
                        workerErrors.push_back(sourceDirs[sourceIndex].string() + ": " + e.what());
                    }
                    completedSources.fetch_add(1, std::memory_order_release);
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }
        scanDone.store(true, std::memory_order_release);
        progressThread.join();
        print_progress(completedSources.load(std::memory_order_relaxed),
                       sourceDirs.size(),
                       queriedPoints.load(std::memory_order_relaxed),
                       discoveredPairs.load(std::memory_order_relaxed),
                       scanStart,
                       true);

        if (!workerErrors.empty()) {
            throw std::runtime_error("Source scan failed for " + workerErrors.front());
        }

        std::map<std::string, std::set<std::string>> overlapByPath;
        std::unordered_set<std::string> dirtyPaths;
        auto overlapsFor = [&](const std::string& key, const fs::path& path) -> std::set<std::string>& {
            auto it = overlapByPath.find(key);
            if (it == overlapByPath.end()) {
                it = overlapByPath.emplace(key, read_overlapping_json(path)).first;
            }
            return it->second;
        };

        std::set<std::pair<std::string, std::string>> pairIds;
        for (const SourceScanResult& result : scanResults) {
            if (result.sourceKey.empty()) {
                continue;
            }

            std::set<std::string>& sourceOverlaps = overlapsFor(result.sourceKey, result.sourcePath);
            for (const auto& [targetKey, targetId] : result.hits) {
                sourceOverlaps.insert(targetId);
                dirtyPaths.insert(result.sourceKey);

                const auto targetIt = targetByPath.find(targetKey);
                if (targetIt != targetByPath.end()) {
                    std::set<std::string>& targetOverlaps = overlapsFor(targetKey, targetIt->second->path);
                    targetOverlaps.insert(result.sourceId);
                    dirtyPaths.insert(targetKey);
                }

                pairIds.insert(std::minmax(result.sourceId, targetId));
            }
        }

        for (const auto& [key, overlaps] : overlapByPath) {
            if (!dirtyPaths.count(key)) {
                continue;
            }

            const auto targetIt = targetByPath.find(key);
            if (targetIt != targetByPath.end()) {
                write_overlapping_json(targetIt->second->path, overlaps);
                std::cout << "Updated overlapping data for " << targetIt->second->id
                          << " (" << overlaps.size() << " overlaps)" << std::endl;
                continue;
            }

            const auto sourceIt = sourcePathByKey.find(key);
            if (sourceIt != sourcePathByKey.end()) {
                write_overlapping_json(sourceIt->second, overlaps);
                std::cout << "Updated overlapping data for "
                          << sourceIt->second.filename().string()
                          << " (" << overlaps.size() << " overlaps)" << std::endl;
            }
        }

        std::cout << "Queried source points: " << queriedPoints.load() << std::endl;
        std::cout << "Overlap pairs found: " << pairIds.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "vc_seg_add_overlap failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
