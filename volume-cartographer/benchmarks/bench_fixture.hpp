#pragma once

// The benchmarks' shared fixture, without Qt or VC3D: command-line arguments, the
// data manifest (benchmarks.json: volumes, segments, fibers, the cache that holds
// them, and named cases per benchmark), the chosen case's parameters, the data
// loaded from the cache, statistics and the report.
//
// Builds in vc_reader trees and in origin/main's tree (VC_BENCH_VC_READER unset).
// Only vc_reader trees fetch remote data (--prefetch, cache states cold and partial);
// every tree reads what the prefetch recorded in the cache's index.
//
// The timed volume's cache state (parameter cache_state):
//   cached   every chunk read is on disk (the prefetched copy, opened as a local folder);
//   cold     nothing on disk: the volume is streamed from its URL into an empty scratch cache;
//   partial  the scratch cache starts with the cached copy's partial_levels (default: all
//            but level 0); the rest is fetched from the URL during the run.
// Scratch caches live under "scratch" (default: <cache>-scratch) and are removed at exit.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

class QuadSurface;

namespace vcbench {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start);

// --key value / --key=value / --flag arguments.
class Args {
public:
    Args(int argc, char** argv);
    std::optional<std::string> get(const std::string& key) const;  // marks the key used
    std::string str(const std::string& key, const std::string& fallback = {}) const;
    bool flag(const std::string& key) const;
    // Fails when an argument was given that nothing asked for.
    void rejectUnused() const;

private:
    std::map<std::string, std::string> values_;
    mutable std::set<std::string> used_;
};

struct Summary {
    std::size_t n = 0;
    double mean = 0, min = 0, p50 = 0, p95 = 0, max = 0, total = 0;
};
Summary summarize(std::vector<double> values);

// Where and as what a run's results are written.
struct RunInfo {
    std::string timestamp;  // UTC, ISO 8601, when the harness started
    std::string runId;      // --run-id (run.sh: one per invocation), else the timestamp
    std::string commit;     // --commit (run.sh: the build tree's source commit), else "unknown"
    std::string jsonPath;   // --json: append one JSON line per metric
    std::string csvPath;    // --csv, else the manifest's "csv", else --json's path with .csv
};

// One benchmark run's results: steps with their samples (ms and counts).
class Report {
public:
    Report(std::string benchmark, std::string pipeline, std::string caseName, std::string cacheState,
           std::map<std::string, std::string> config, RunInfo run = {});
    void add(const std::string& group, const std::string& metric, double value);
    void setConfig(std::map<std::string, std::string> config) { config_ = std::move(config); }
    // Prints a table and appends the results to the run's JSON and CSV files, if set.
    void finish() const;

private:
    std::string benchmark_, pipeline_, case_, cacheState_;
    std::map<std::string, std::string> config_;
    RunInfo run_;
    std::vector<std::string> order_;
    std::map<std::string, std::vector<double>> samples_;  // "group|metric"
};

// A volume, segment or fiber folder of the manifest.
struct DataItem {
    std::string kind, name;
    std::string url;              // remote source, copied into the cache by the prefetch
    std::filesystem::path path;   // where it is read from: the manifest's path or the cached copy
    std::string volume;           // segments and fibers: the volume their coordinates belong to
    double scale = 1;             // segments and fibers: factor to that volume's level-0 voxels
};

struct Fiber {
    std::string name;
    std::vector<cv::Vec3d> points;  // level-0 voxels
};

// Where a run reads the timed volume from.
struct VolumeSource {
    std::string location;              // a local folder, or the URL when remote
    bool remote = false;
    std::filesystem::path cacheRoot;   // remote: VC3D's cache root for the run
};

class Fixture {
public:
    // Reads --config (default: benchmarks.json beside the sources), picks the case of
    // `benchmark` named by --case (or its only case). --list-cases prints the names.
    Fixture(int argc, char** argv, std::string benchmark);
    ~Fixture();
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    // --list-cases was given: the names are printed and the harness should exit 0.
    bool listOnly() const { return listOnly_; }
    const std::string& benchmark() const { return benchmark_; }
    const std::string& caseName() const { return caseName_; }
    const std::filesystem::path& cacheRoot() const { return cacheRoot_; }
    bool prefetching() const { return prefetch_; }
    // "cached", "cold" or "partial" (see above); "prefetch" with --prefetch.
    const std::string& cacheState() const { return cacheState_; }
    const std::string& backend() const { return backend_; }
    // "vc_reader-cpu" / "vc_reader-cuda" with vc_reader, "chunkcache" on origin/main.
    const std::string& pipelineName() const { return pipeline_; }

    // A parameter: --kebab-case on the command line, else the case's snake_case key,
    // else the manifest's "defaults", else `fallback`.
    std::string str(const std::string& key, const std::string& fallback = {});
    double num(const std::string& key, double fallback);
    int integer(const std::string& key, int fallback) { return int(num(key, fallback)); }
    bool flag(const std::string& key, bool fallback = false);
    // A list of numbers: a JSON array, or "a,b,c" on the command line. Empty if unset.
    std::vector<double> numbers(const std::string& key);
    // Fails on case keys or arguments that no parameter call read (typos).
    void rejectUnused() const;

    // The case's data: "volume" (else its segment's or fibers' volume), "segment", "fibers".
    // With --prefetch, remote segments and fibers are downloaded into the cache first.
    const DataItem& volume();
    const DataItem& segment();
    const DataItem& fibers();
    // The case's segment in its volume's level-0 voxels (loaded once).
    std::shared_ptr<QuadSurface> loadSegment();
    // The `count` fibers with the most line points (ties by file name), in level-0 voxels.
    std::vector<Fiber> loadFibers(int count);

    // Prepares the volume for the run's cache state. Remote sources write VC3D's
    // cache.json into `configDir` (the process's VC3D_CONFIG_DIR) with the run's cache
    // root: the fixture's cache with --prefetch, a fresh scratch cache for cold and
    // partial (seeded for partial). Call once.
    VolumeSource volumeSource(const std::filesystem::path& configDir);

    // Fixture facts for the report (case, cache, data locations).
    std::map<std::string, std::string> config() const { return config_; }
    void note(const std::string& key, const std::string& value) { config_[key] = value; }
    Report makeReport() const;
    const RunInfo& run() const { return run_; }
    const std::string& dumpDir() const { return dumpDir_; }

private:
    const nlohmann::json* lookup(const std::string& key);
    const DataItem& item(const std::string& kind, const std::string& name);
    void recordInIndex(const DataItem& item);

    Args args_;
    std::string benchmark_, caseName_, backend_, pipeline_, dumpDir_;
    RunInfo run_;
    std::filesystem::path configPath_, cacheRoot_;
    nlohmann::json manifest_, case_;
    std::set<std::string> usedCaseKeys_;
    std::map<std::string, DataItem> items_;  // "kind/name"
    std::map<std::string, std::string> config_;
    std::shared_ptr<QuadSurface> segment_;
    std::string cacheState_;
    std::filesystem::path scratch_;  // this run's scratch cache, removed by the destructor
    bool listOnly_ = false, prefetch_ = false, keepScratch_ = false;
};

// Loads a tifxyz and scales its vertices into the volume's level-0 voxels.
std::shared_ptr<QuadSurface> loadSurface(const std::filesystem::path& tifxyz, float coordinateScale);
// A valid vertex near the grid centre (volume coordinates).
cv::Vec3f surfaceCenter(QuadSurface& surface);
// Valid vertices sampled evenly over the grid, in a fixed order.
std::vector<cv::Vec3f> surfaceSamplePoints(QuadSurface& surface, int count);

} // namespace vcbench
