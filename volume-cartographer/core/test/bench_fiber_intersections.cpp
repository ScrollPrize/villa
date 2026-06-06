#include "vc/atlas/FiberIntersections.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

struct TimedFiber {
    vc::atlas::FiberPolyline fiber;
    fs::path path;
    double indexMs = 0.0;
    double searchMs = 0.0;
    size_t candidateCount = 0;
};

struct Options {
    fs::path fiberDir = "/home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers";
    fs::path lasagnaManifest;
    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    vc::atlas::FiberIntersectionCeresOptions ceres;
};

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void printUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [fiber_dir] [options]\n"
        << "\n"
        << "Options:\n"
        << "  --lasagna-manifest <path>  Required .lasagna.json for grad_mag winding distance\n"
        << "  --max-distance <vx>        Broad-phase radius, default 500\n"
        << "  --max-sample-spacing <vx>  Indexed dense point spacing, default 100\n"
        << "  --seed-stride <n>          First-pass source sample stride, default 100\n"
        << "  --cluster-arclength <vx>   Candidate arclength dedup tolerance, default 8\n"
        << "  --ceres-dedup <vx>         Final result dedup tolerance, default 4\n"
        << "  --ceres-iterations <n>     Ceres max iterations, default 50\n"
        << "  --help                     Show this help\n";
}

double parseDouble(const std::string& value, const char* name)
{
    size_t pos = 0;
    const double parsed = std::stod(value, &pos);
    if (pos != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

int parseInt(const std::string& value, const char* name)
{
    size_t pos = 0;
    const int parsed = std::stoi(value, &pos);
    if (pos != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return parsed;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    bool consumedDir = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--lasagna-manifest") {
            options.lasagnaManifest = requireValue("--lasagna-manifest");
        } else if (arg == "--max-distance") {
            options.broad.maxDistance = parseDouble(requireValue("--max-distance"), "--max-distance");
        } else if (arg == "--max-sample-spacing") {
            options.broad.maxSampleSpacing = parseDouble(requireValue("--max-sample-spacing"),
                                                         "--max-sample-spacing");
        } else if (arg == "--seed-stride") {
            options.broad.seedStride = parseInt(requireValue("--seed-stride"), "--seed-stride");
        } else if (arg == "--cluster-arclength") {
            options.broad.clusterArclength = parseDouble(requireValue("--cluster-arclength"),
                                                         "--cluster-arclength");
        } else if (arg == "--ceres-dedup") {
            options.ceres.deduplicateArclength = parseDouble(requireValue("--ceres-dedup"),
                                                             "--ceres-dedup");
        } else if (arg == "--ceres-iterations") {
            options.ceres.maxIterations = parseInt(requireValue("--ceres-iterations"),
                                                   "--ceres-iterations");
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option: " + arg);
        } else if (!consumedDir) {
            options.fiberDir = arg;
            consumedDir = true;
        } else {
            throw std::runtime_error("unexpected positional argument: " + arg);
        }
    }
    return options;
}

std::optional<cv::Vec3d> pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() < 3) {
        return std::nullopt;
    }
    cv::Vec3d p{0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        if (!value[static_cast<size_t>(i)].is_number()) {
            return std::nullopt;
        }
        p[i] = value[static_cast<size_t>(i)].get<double>();
        if (!std::isfinite(p[i])) {
            return std::nullopt;
        }
    }
    return p;
}

std::vector<vc::atlas::FiberPoint> pointsFromJson(const nlohmann::json& root)
{
    const nlohmann::json* points = nullptr;
    if (root.contains("line_points") && root["line_points"].is_array()) {
        points = &root["line_points"];
    } else if (root.contains("control_points") && root["control_points"].is_array()) {
        points = &root["control_points"];
    }
    if (!points) {
        throw std::runtime_error("missing line_points/control_points array");
    }

    std::vector<vc::atlas::FiberPoint> parsed;
    parsed.reserve(points->size());
    for (const auto& value : *points) {
        if (auto point = pointFromJson(value)) {
            parsed.push_back(vc::atlas::FiberPoint{*point, std::nullopt});
        }
    }
    return parsed;
}

vc::atlas::FiberPolyline readFiberJson(const fs::path& path, uint64_t fallbackId)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    const nlohmann::json root = nlohmann::json::parse(in);

    vc::atlas::FiberPolyline fiber;
    fiber.id = root.value("id", fallbackId);
    fiber.generation = root.value("generation", uint64_t{1});
    fiber.points = pointsFromJson(root);
    if (fiber.points.size() < 2) {
        throw std::runtime_error(path.string() + " has fewer than 2 finite points");
    }
    return fiber;
}

std::vector<TimedFiber> loadFibers(const fs::path& dir)
{
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("not a directory: " + dir.string());
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    std::vector<TimedFiber> fibers;
    fibers.reserve(paths.size());
    uint64_t fallbackId = 1;
    for (const auto& path : paths) {
        fibers.push_back(TimedFiber{readFiberJson(path, fallbackId++), path});
    }
    return fibers;
}

const TimedFiber* findFiber(const std::vector<TimedFiber>& fibers, uint64_t id)
{
    const auto it = std::find_if(fibers.begin(), fibers.end(), [id](const TimedFiber& item) {
        return item.fiber.id == id;
    });
    return it == fibers.end() ? nullptr : &*it;
}

double fiberLength(const vc::atlas::FiberPolyline& fiber)
{
    double length = 0.0;
    for (size_t i = 1; i < fiber.points.size(); ++i) {
        const cv::Vec3d d = fiber.points[i].position - fiber.points[i - 1].position;
        const double step = std::sqrt(std::max(0.0, d.dot(d)));
        if (std::isfinite(step)) {
            length += step;
        }
    }
    return length;
}

double arclengthFraction(const vc::atlas::FiberPolyline& fiber, double arclength)
{
    const double length = fiberLength(fiber);
    if (!(length > 0.0) || !std::isfinite(length) || !std::isfinite(arclength)) {
        return 0.0;
    }
    return std::clamp(arclength / length, 0.0, 1.0);
}

std::string fiberLabel(const TimedFiber* fiber, uint64_t fallbackId)
{
    if (!fiber) {
        return std::to_string(fallbackId);
    }
    return fiber->path.filename().string() + "(" + std::to_string(fiber->fiber.id) + ")";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseOptions(argc, argv);
        auto fibers = loadFibers(options.fiberDir);
        if (fibers.empty()) {
            throw std::runtime_error("no .json fibers found in " + options.fiberDir.string());
        }
        if (options.lasagnaManifest.empty()) {
            throw std::runtime_error("--lasagna-manifest is required for winding-distance results");
        }

        vc::lasagna::LasagnaDataset dataset =
            vc::lasagna::LasagnaDataset::open(options.lasagnaManifest);
        vc::lasagna::LasagnaNormalSampler windingSampler(dataset);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "fiber_dir: " << options.fiberDir << "\n";
        std::cout << "lasagna_manifest: " << options.lasagnaManifest << "\n";
        std::cout << "broad_phase: maxDistance=" << options.broad.maxDistance
                  << " maxSampleSpacing=" << options.broad.maxSampleSpacing
                  << " seedStride=" << options.broad.seedStride
                  << " clusterArclength=" << options.broad.clusterArclength << "\n";
        std::cout << "ceres: maxIterations=" << options.ceres.maxIterations
                  << " deduplicateArclength=" << options.ceres.deduplicateArclength << "\n";

        std::cout << "\nfibers:\n";
        for (const auto& item : fibers) {
            std::cout << "  id=" << item.fiber.id
                      << " generation=" << item.fiber.generation
                      << " points=" << item.fiber.points.size()
                      << " file=" << item.path.filename().string() << "\n";
        }

        vc::atlas::FiberSpatialIndex index;
        const auto indexStart = Clock::now();
        for (auto& item : fibers) {
            const auto start = Clock::now();
            index.upsertCommitted(item.fiber);
            const auto end = Clock::now();
            item.indexMs = elapsedMs(start, end);
        }
        const double totalIndexMs = elapsedMs(indexStart, Clock::now());

        std::cout << "\nindex_creation:\n";
        std::cout << "  total_ms=" << totalIndexMs << "\n";
        for (const auto& item : fibers) {
            std::cout << "  fiber_id=" << item.fiber.id
                      << " ms=" << item.indexMs << "\n";
        }

        std::cout << "\nsource_search:\n";
        for (auto& item : fibers) {
            const auto start = Clock::now();
            auto candidates = index.candidatesForFiber(item.fiber, options.broad);
            const auto end = Clock::now();
            item.searchMs = elapsedMs(start, end);
            item.candidateCount = candidates.size();
            std::cout << "  source_id=" << item.fiber.id
                      << " ms=" << item.searchMs
                      << " candidates=" << item.candidateCount << "\n";
        }

        std::vector<vc::atlas::FiberPolyline> fiberPolylines;
        std::vector<uint64_t> fiberIds;
        fiberPolylines.reserve(fibers.size());
        fiberIds.reserve(fibers.size());
        for (const auto& item : fibers) {
            fiberPolylines.push_back(item.fiber);
            fiberIds.push_back(item.fiber.id);
        }

        vc::atlas::FiberSpatialIndex resultIndex;
        const auto finalStart = Clock::now();
        std::vector<vc::atlas::FiberIntersectionResult> allResults =
            vc::atlas::searchFiberIntersections(fiberPolylines,
                                                fiberIds,
                                                fiberIds,
                                                resultIndex,
                                                nullptr,
                                                options.broad,
                                                options.ceres,
                                                &windingSampler);
        const double finalMs = elapsedMs(finalStart, Clock::now());

        std::sort(allResults.begin(), allResults.end(), [](const auto& a, const auto& b) {
            if (a.windingDistance != b.windingDistance) return a.windingDistance < b.windingDistance;
            if (a.sourceFiberId != b.sourceFiberId) return a.sourceFiberId < b.sourceFiberId;
            return a.targetArclength < b.targetArclength;
        });

        std::cout << "\nfinal_results:\n";
        std::cout << "  count=" << allResults.size() << " shared_search_ms=" << finalMs << "\n";
        for (size_t i = 0; i < allResults.size(); ++i) {
            const auto& result = allResults[i];
            const auto* source = findFiber(fibers, result.sourceFiberId);
            const auto* target = findFiber(fibers, result.targetFiberId);
            std::cout << "  [" << i << "]"
                      << " distance_windings=" << result.windingDistance
                      << " source=" << fiberLabel(source, result.sourceFiberId)
                      << " target=" << fiberLabel(target, result.targetFiberId)
                      << " src_idx=" << (source ? arclengthFraction(source->fiber,
                                                                    result.sourceArclength)
                                                : 0.0)
                      << " tgt_idx=" << (target ? arclengthFraction(target->fiber,
                                                                    result.targetArclength)
                                                : 0.0)
                      << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
