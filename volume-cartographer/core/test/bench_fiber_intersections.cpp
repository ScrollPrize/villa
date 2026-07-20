#include "vc/atlas/Atlas.hpp"
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
    fs::path atlasDir;
    fs::path lasagnaManifest;
    std::optional<std::pair<uint64_t, uint64_t>> pairIds;
    std::optional<vc::atlas::FiberIntersectionOptimizationMode> optimizationMode;
    vc::atlas::FiberIntersectionBroadPhaseOptions broad;
    vc::atlas::FiberIntersectionCeresOptions ceres;

    Options()
    {
        ceres.optimizationMode =
            vc::atlas::FiberIntersectionOptimizationMode::WindingCeres;
        ceres.requireWindingScore = true;
    }
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
        << "  --atlas-dir <path>         Atlas directory; search atlas fibers -> non-atlas fibers\n"
        << "  --lasagna-manifest <path>  Required .lasagna.json for grad_mag winding distance\n"
        << "  --pair <source/target>     Profile one non-atlas runtime-id pair, e.g. 28/50\n"
        << "  --optimization <mode>      geometric-direct, geometric-ceres, or winding-ceres\n"
        << "  --max-distance <vx>        Broad-phase radius, default 100\n"
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

uint64_t parseUint64(const std::string& value, const char* name)
{
    size_t pos = 0;
    const unsigned long long parsed = std::stoull(value, &pos);
    if (pos != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<uint64_t>(parsed);
}

std::pair<uint64_t, uint64_t> parsePairIds(const std::string& value)
{
    const size_t separator = value.find_first_of("/,:");
    if (separator == std::string::npos) {
        throw std::runtime_error("invalid --pair, expected source/target runtime ids");
    }
    const uint64_t source = parseUint64(value.substr(0, separator), "--pair source");
    const uint64_t target = parseUint64(value.substr(separator + 1), "--pair target");
    if (source == 0 || target == 0 || source == target) {
        throw std::runtime_error("invalid --pair, ids must be non-zero and distinct");
    }
    return {source, target};
}

vc::atlas::FiberIntersectionOptimizationMode parseOptimizationMode(
    const std::string& value)
{
    if (value == "geometric-direct") {
        return vc::atlas::FiberIntersectionOptimizationMode::GeometricDirectWalk;
    }
    if (value == "geometric-ceres") {
        return vc::atlas::FiberIntersectionOptimizationMode::GeometricCeres;
    }
    if (value == "winding-ceres") {
        return vc::atlas::FiberIntersectionOptimizationMode::WindingCeres;
    }
    throw std::runtime_error("unknown --optimization mode: " + value);
}

std::string optimizationModeName(
    vc::atlas::FiberIntersectionOptimizationMode mode)
{
    switch (mode) {
    case vc::atlas::FiberIntersectionOptimizationMode::GeometricDirectWalk:
        return "geometric-direct";
    case vc::atlas::FiberIntersectionOptimizationMode::GeometricCeres:
        return "geometric-ceres";
    case vc::atlas::FiberIntersectionOptimizationMode::WindingCeres:
        return "winding-ceres";
    }
    return "unknown";
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
        } else if (arg == "--atlas-dir") {
            options.atlasDir = requireValue("--atlas-dir");
        } else if (arg == "--lasagna-manifest") {
            options.lasagnaManifest = requireValue("--lasagna-manifest");
        } else if (arg == "--pair") {
            options.pairIds = parsePairIds(requireValue("--pair"));
        } else if (arg == "--optimization") {
            options.optimizationMode =
                parseOptimizationMode(requireValue("--optimization"));
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
    (void)fallbackId;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    const nlohmann::json root = nlohmann::json::parse(in);

    vc::atlas::FiberPolyline fiber;
    fiber.id = 0;
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

fs::path packageRelativeFiberPath(const TimedFiber& fiber)
{
    return fs::path("fibers") / fiber.path.filename();
}

const TimedFiber* findFiberByPath(const std::vector<TimedFiber>& fibers, const fs::path& path)
{
    const std::string key = vc::atlas::atlasFiberPathKey(path);
    const auto it = std::find_if(fibers.begin(), fibers.end(), [&key](const TimedFiber& item) {
        return vc::atlas::atlasFiberPathKey(packageRelativeFiberPath(item)) == key;
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
    return packageRelativeFiberPath(*fiber).generic_string();
}

const TimedFiber* findFiberByRuntimeId(const std::vector<TimedFiber>& fibers,
                                       uint64_t runtimeId)
{
    const auto it = std::find_if(fibers.begin(), fibers.end(), [runtimeId](const auto& item) {
        return item.fiber.id == runtimeId;
    });
    return it == fibers.end() ? nullptr : &*it;
}

std::vector<vc::atlas::FiberPolyline> fiberPolylinesFor(
    const std::vector<TimedFiber>& fibers)
{
    std::vector<vc::atlas::FiberPolyline> polylines;
    polylines.reserve(fibers.size());
    for (const auto& item : fibers) {
        polylines.push_back(item.fiber);
    }
    return polylines;
}

struct PairProfileStats {
    size_t candidateFiberCount = 0;
    size_t candidateFiberCandidates = 0;
    int64_t candidateFiberMs = 0;
    size_t rawPairCandidates = 0;
    int64_t pairCandidatesReadyMs = 0;
    size_t refineStarted = 0;
    size_t refineFinished = 0;
    int64_t refineMs = 0;
    int64_t maxRefineMs = 0;
    size_t scoringStarts = 0;
    size_t scoringFinishes = 0;
    size_t scoringResults = 0;
    int64_t scoringMs = 0;
    int64_t pairMs = 0;
    size_t finalPairResults = 0;
};

void updatePairProfileStats(PairProfileStats& stats,
                            const vc::atlas::FiberIntersectionDiagnostic& diagnostic)
{
    switch (diagnostic.event) {
    case vc::atlas::FiberIntersectionDiagnosticEvent::CandidateFiberFinished:
        ++stats.candidateFiberCount;
        stats.candidateFiberCandidates += diagnostic.candidateCount;
        stats.candidateFiberMs += diagnostic.elapsedMs;
        break;
    case vc::atlas::FiberIntersectionDiagnosticEvent::PairCandidatesReady:
        stats.rawPairCandidates = diagnostic.candidateCount;
        stats.pairCandidatesReadyMs = diagnostic.elapsedMs;
        break;
    case vc::atlas::FiberIntersectionDiagnosticEvent::CandidateRefineStarted:
        ++stats.refineStarted;
        break;
    case vc::atlas::FiberIntersectionDiagnosticEvent::CandidateRefineFinished:
        ++stats.refineFinished;
        stats.refineMs += diagnostic.elapsedMs;
        stats.maxRefineMs = std::max(stats.maxRefineMs, diagnostic.elapsedMs);
        break;
    case vc::atlas::FiberIntersectionDiagnosticEvent::PairScoringStarted:
        ++stats.scoringStarts;
        stats.scoringResults = diagnostic.resultCount;
        break;
    case vc::atlas::FiberIntersectionDiagnosticEvent::PairScoringFinished:
        ++stats.scoringFinishes;
        stats.scoringMs += diagnostic.elapsedMs;
        stats.scoringResults = diagnostic.resultCount;
        break;
    case vc::atlas::FiberIntersectionDiagnosticEvent::PairFinished:
        stats.pairMs = diagnostic.elapsedMs;
        stats.finalPairResults = diagnostic.resultCount;
        break;
    default:
        break;
    }
}

void printPairProfile(const Options& options,
                      const std::vector<TimedFiber>& fibers,
                      const vc::lasagna::LasagnaNormalSampler& windingSampler)
{
    if (!options.pairIds) {
        throw std::runtime_error("pair profile requested without --pair");
    }

    vc::atlas::FiberIntersectionCeresOptions ceres = options.ceres;
    ceres.optimizationMode = options.optimizationMode.value_or(
        vc::atlas::FiberIntersectionOptimizationMode::GeometricDirectWalk);
    ceres.requireWindingScore = true;

    const uint64_t sourceId = options.pairIds->first;
    const uint64_t targetId = options.pairIds->second;
    const TimedFiber* source = findFiberByRuntimeId(fibers, sourceId);
    const TimedFiber* target = findFiberByRuntimeId(fibers, targetId);
    if (!source || !target) {
        std::ostringstream out;
        out << "runtime pair " << sourceId << "/" << targetId
            << " not found in " << options.fiberDir;
        throw std::runtime_error(out.str());
    }

    PairProfileStats stats;
    const auto start = Clock::now();
    auto results = vc::atlas::searchFiberIntersections(
        fiberPolylinesFor(fibers),
        {sourceId},
        {targetId},
        nullptr,
        options.broad,
        ceres,
        &windingSampler,
        vc::atlas::FiberIntersectionProgressCallback{},
        vc::atlas::FiberIntersectionCancelCallback{},
        [&](const vc::atlas::FiberIntersectionDiagnostic& diagnostic) {
            updatePairProfileStats(stats, diagnostic);
        });
    const double totalMs = elapsedMs(start, Clock::now());

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        if (a.windingDistance != b.windingDistance) return a.windingDistance < b.windingDistance;
        if (a.sourceArclength != b.sourceArclength) return a.sourceArclength < b.sourceArclength;
        return a.targetArclength < b.targetArclength;
    });

    std::cout << "\npair_profile:\n";
    std::cout << "  pair=" << sourceId << "/" << targetId << "\n";
    std::cout << "  source=" << fiberLabel(source, sourceId) << "\n";
    std::cout << "  target=" << fiberLabel(target, targetId) << "\n";
    std::cout << "  optimization=" << optimizationModeName(ceres.optimizationMode) << "\n";
    std::cout << "  total_ms=" << totalMs << "\n";
    std::cout << "  candidate_fibers=" << stats.candidateFiberCount
              << " candidate_fiber_candidates=" << stats.candidateFiberCandidates
              << " candidate_fiber_ms=" << stats.candidateFiberMs << "\n";
    std::cout << "  raw_pair_candidates=" << stats.rawPairCandidates
              << " pair_candidates_ready_ms=" << stats.pairCandidatesReadyMs << "\n";
    std::cout << "  refine_started=" << stats.refineStarted
              << " refine_finished=" << stats.refineFinished
              << " refine_total_ms=" << stats.refineMs
              << " refine_max_ms=" << stats.maxRefineMs << "\n";
    std::cout << "  final_scoring_starts=" << stats.scoringStarts
              << " final_scoring_finishes=" << stats.scoringFinishes
              << " final_scoring_results=" << stats.scoringResults
              << " final_scoring_ms=" << stats.scoringMs << "\n";
    std::cout << "  pair_finished_ms=" << stats.pairMs
              << " final_pair_results=" << stats.finalPairResults
              << " returned_results=" << results.size() << "\n";

    std::cout << "\npair_results:\n";
    const size_t resultLimit = std::min<size_t>(results.size(), 20);
    for (size_t i = 0; i < resultLimit; ++i) {
        const auto& result = results[i];
        std::cout << "  [" << i << "]"
                  << " distance_windings=" << result.windingDistance
                  << " candidate_vx=" << result.candidateDistance
                  << " ceres_solves=" << result.ceresSolves
                  << " ceres_iterations=" << result.ceresIterations
                  << " src_idx=" << arclengthFraction(source->fiber,
                                                      result.sourceArclength)
                  << " tgt_idx=" << arclengthFraction(target->fiber,
                                                      result.targetArclength)
                  << "\n";
    }
    if (results.size() > resultLimit) {
        std::cout << "  ... " << (results.size() - resultLimit)
                  << " more results omitted\n";
    }
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
        if (!options.atlasDir.empty()) {
            std::cout << "atlas_dir: " << options.atlasDir << "\n";
        }
        std::cout << "lasagna_manifest: " << options.lasagnaManifest << "\n";
        if (options.pairIds) {
            std::cout << "pair: " << options.pairIds->first
                      << "/" << options.pairIds->second << "\n";
        }
        std::cout << "broad_phase: maxDistance=" << options.broad.maxDistance
                  << " maxSampleSpacing=" << options.broad.maxSampleSpacing
                  << " seedStride=" << options.broad.seedStride
                  << " clusterArclength=" << options.broad.clusterArclength << "\n";
        std::cout << "ceres: maxIterations=" << options.ceres.maxIterations
                  << " deduplicateArclength=" << options.ceres.deduplicateArclength << "\n";

        std::vector<fs::path> canonicalPaths;
        canonicalPaths.reserve(fibers.size());
        for (const auto& item : fibers) {
            canonicalPaths.push_back(packageRelativeFiberPath(item));
        }
        const auto runtimeIds = vc::atlas::makeFiberRuntimeIdentityMap(canonicalPaths);
        for (auto& item : fibers) {
            item.fiber.id = runtimeIds.idForPath(packageRelativeFiberPath(item));
        }
        if (options.pairIds) {
            std::cout << "\nfibers:\n";
            std::cout << "  count=" << fibers.size() << "\n";
            printPairProfile(options, fibers, windingSampler);
            return 0;
        }

        std::cout << "\nfibers:\n";
        for (const auto& item : fibers) {
            std::cout << "  path=" << packageRelativeFiberPath(item).generic_string()
                      << " runtime_id=" << item.fiber.id
                      << " generation=" << item.fiber.generation
                      << " points=" << item.fiber.points.size()
                      << "\n";
        }

        if (options.atlasDir.empty()) {
            throw std::runtime_error("--atlas-dir is required unless --pair is set");
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
            std::cout << "  fiber_path=" << packageRelativeFiberPath(item).generic_string()
                      << " runtime_id=" << item.fiber.id
                      << " ms=" << item.indexMs << "\n";
        }

        std::cout << "\nsource_search:\n";
        for (auto& item : fibers) {
            const auto start = Clock::now();
            auto candidates = index.candidatesForFiber(item.fiber, options.broad);
            const auto end = Clock::now();
            item.searchMs = elapsedMs(start, end);
            item.candidateCount = candidates.size();
            std::cout << "  source_path=" << packageRelativeFiberPath(item).generic_string()
                      << " runtime_id=" << item.fiber.id
                      << " ms=" << item.searchMs
                      << " candidates=" << item.candidateCount << "\n";
        }

        std::vector<vc::atlas::FiberPolyline> fiberPolylines;
        fiberPolylines.reserve(fibers.size());
        const vc::atlas::Atlas atlas = vc::atlas::Atlas::load(options.atlasDir);
        const std::vector<std::string> atlasKeys =
            vc::atlas::atlasMappedFiberPathKeys(atlas);
        if (atlasKeys.empty()) {
            throw std::runtime_error("atlas has no fiber mappings: " +
                                     options.atlasDir.string());
        }
        std::cout << "\natlas_mappings:\n";
        for (const auto& mapping : atlas.fibers) {
            std::cout << "  fiber_path=" << mapping.fiberPath.generic_string()
                      << " key=" << vc::atlas::atlasFiberPathKey(mapping.fiberPath)
                      << "\n";
        }
        for (const auto& item : fibers) {
            fiberPolylines.push_back(item.fiber);
        }
        const auto searchSets = vc::atlas::atlasFiberSearchSets(atlas, runtimeIds);
        const std::vector<uint64_t>& sourceFiberIds = searchSets.sourceFiberIds;
        const std::vector<uint64_t>& targetFiberIds = searchSets.targetFiberIds;
        if (sourceFiberIds.empty()) {
            throw std::runtime_error("none of the atlas fibers are present in " +
                                     options.fiberDir.string());
        }
        if (targetFiberIds.empty()) {
            throw std::runtime_error("no non-atlas fibers are present in " +
                                     options.fiberDir.string());
        }

        std::cout << "\nshared_search_mode: atlas-fibers-to-non-atlas-fibers\n";
        std::cout << "  source_paths=";
        for (size_t i = 0; i < searchSets.sourceFiberPaths.size(); ++i) {
            if (i > 0) std::cout << ',';
            std::cout << searchSets.sourceFiberPaths[i].generic_string();
        }
        std::cout << "\n  target_paths=";
        for (size_t i = 0; i < searchSets.targetFiberPaths.size(); ++i) {
            if (i > 0) std::cout << ',';
            std::cout << searchSets.targetFiberPaths[i].generic_string();
        }
        std::cout << "\n";

        const auto finalStart = Clock::now();
        std::vector<vc::atlas::FiberIntersectionResult> allResults =
            vc::atlas::searchFiberIntersections(fiberPolylines,
                                                sourceFiberIds,
                                                targetFiberIds,
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
            const fs::path sourcePath = runtimeIds.pathForId(result.sourceFiberId);
            const fs::path targetPath = runtimeIds.pathForId(result.targetFiberId);
            const auto* source = findFiberByPath(fibers, sourcePath);
            const auto* target = findFiberByPath(fibers, targetPath);
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
