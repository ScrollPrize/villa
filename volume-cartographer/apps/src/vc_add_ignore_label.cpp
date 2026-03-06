#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/BinaryPyramid.hpp"
#include "vc/core/util/Zarr.hpp"

#include <boost/program_options.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

constexpr int kDefaultAngleBins = 72;
constexpr int kDefaultComputeLevel = 2;
constexpr int kDefaultOutputLevel = 0;
constexpr double kDefaultAlpha = 0.005;
constexpr double kDefaultShrink = 0.95;
constexpr int kDefaultInnerSmoothing = 5;
constexpr int kDefaultOuterSmoothing = 5;
constexpr int kDefaultIgnoreValue = 127;
constexpr std::size_t kLockPoolSize = 4096;
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kRadPerDeg = kPi / 180.0;
constexpr double kDefaultRamBudgetGb = 0.0;   // 0 => auto
constexpr double kDefaultAutoRamFraction = 0.60;

using Shape3 = std::array<std::size_t, 3>;
using AlphaWrapKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using AlphaWrapPoint = AlphaWrapKernel::Point_3;
using AlphaWrapMesh = CGAL::Surface_mesh<AlphaWrapPoint>;

enum class MapMode {
    legacy,
    exact,
    fast,
};

enum class ProcessingMode {
    legacySlice,
    chunkAlphaWrap,
};

enum class AlgoMode {
    polar,
    contour,
    hull,
    ray,
};

enum class ResultMode {
    same,
    fast,
};

struct ChunkIndex {
    std::size_t z = 0;
    std::size_t y = 0;
    std::size_t x = 0;

    bool operator==(const ChunkIndex& other) const noexcept
    {
        return z == other.z && y == other.y && x == other.x;
    }
};

struct Box3 {
    Shape3 origin = {0, 0, 0};
    Shape3 shape = {0, 0, 0};
};

struct ChunkIndexHash {
    std::size_t operator()(const ChunkIndex& c) const noexcept
    {
        std::size_t h = 1469598103934665603ull;
        h ^= c.z;
        h *= 1099511628211ull;
        h ^= c.y;
        h *= 1099511628211ull;
        h ^= c.x;
        h *= 1099511628211ull;
        return h;
    }
};

static bool chunkIndexLess(const ChunkIndex& a, const ChunkIndex& b)
{
    if (a.z != b.z) {
        return a.z < b.z;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.x < b.x;
}

struct Config {
    fs::path inputRoot;
    fs::path outputRoot;
    ProcessingMode mode = ProcessingMode::chunkAlphaWrap;
    int ignoreValue = kDefaultIgnoreValue;
    double alpha = kDefaultAlpha;
    double chunkAlpha = 0.0;
    int nAngleBins = kDefaultAngleBins;
    double shrinkFactor = kDefaultShrink;
    int computeLevel = kDefaultComputeLevel;
    int outputLevel = kDefaultOutputLevel;
    int workers = 0;
    int zMin = 0;
    int zMax = -1;  // exclusive
    int innerSmoothingWindow = kDefaultInnerSmoothing;
    int outerSmoothingWindow = kDefaultOuterSmoothing;
    bool skipOuter = false;
    bool skipInner = false;
    bool existingChunksOnly = true;
    std::string cacheMode = "auto";
    double ramBudgetGb = kDefaultRamBudgetGb;
    bool overwrite = false;
    bool reuseOutputTree = false;
    MapMode mapMode = MapMode::exact;
    bool fastAtan2 = false;
    bool useSquaredDist = false;
    bool chunkZSlab = false;
    bool rebuildPyramid = true;
    AlgoMode algoMode = AlgoMode::polar;
    ResultMode resultMode = ResultMode::fast;
    bool profileEnabled = false;
    fs::path profileJsonPath;
    std::optional<int> visualizeSlice;
    fs::path visualizeDir = "ignore_label_vis";
    bool selfTest = false;
    bool verbose = false;
};

enum class ComputeCacheMode {
    stream,
    preload
};

struct CachePlan {
    ComputeCacheMode mode = ComputeCacheMode::stream;
    std::size_t budgetBytes = 0;
    std::size_t requiredBytes = 0;
    std::size_t availableBytes = 0;
    double autoFraction = kDefaultAutoRamFraction;
};

struct ProcessingStats {
    std::size_t skippedMissingLevel0 = 0;
    std::size_t skippedMissingPyramid = 0;
    std::size_t preloadedComputeBytes = 0;
    std::size_t preloadedComputeSlices = 0;
    std::size_t cacheBudgetBytes = 0;
    std::size_t availableRamBytes = 0;
    std::string cacheMode;
    std::size_t chunksTouchedInMasks = 0;
    std::size_t touchedLevel0Writes = 0;
    std::size_t readChunkCount = 0;
    std::size_t writeChunkCount = 0;
    std::size_t pyramidChunkWrites = 0;
    std::string mapMode = "exact";
    bool fastAtan2 = false;
    bool useSquaredDist = false;
    bool chunkZSlab = false;
    bool rebuildPyramid = true;
    bool reuseOutputTree = false;
    std::string resultMode = "same";
};

struct TimingAccumulator {
    std::size_t nanos = 0;

    void add(double seconds)
    {
        nanos += static_cast<std::size_t>(seconds * 1e9);
    }

    double seconds() const { return static_cast<double>(nanos) / 1e9; }
};

struct ProfileStats {
    TimingAccumulator tSliceLoad;
    TimingAccumulator tMaskBuild;
    TimingAccumulator tMaskMerge;
    TimingAccumulator tResizeLegacy;
    TimingAccumulator tTouchDiscovery;
    TimingAccumulator tChunkApplyCpu;
    TimingAccumulator tChunkReadIo;
    TimingAccumulator tChunkWriteIo;
    TimingAccumulator tPyramidRead;
    TimingAccumulator tPyramidWrite;

    std::size_t slicesTotal = 0;
    std::size_t slicesSkippedEmpty = 0;
    std::size_t nonzeroFgSlices = 0;
    std::size_t computeChunksTotal = 0;
    std::size_t computeChunksSkippedEmpty = 0;
    std::size_t computeChunksWrapped = 0;
    std::size_t fgPixels = 0;
    std::size_t maskPixels = 0;
    std::size_t touchedChunksLevel0 = 0;
    std::size_t skippedMissingLevel0 = 0;
    std::size_t skippedMissingPyramid = 0;
    std::size_t chunkIoReads = 0;
    std::size_t chunkIoWrites = 0;
    std::size_t chunkCacheHits = 0;
    std::size_t chunkCacheMisses = 0;
    std::size_t bytesRead = 0;
    std::size_t bytesWritten = 0;
    std::size_t inputVoxels = 0;
    std::size_t pyramidReadCalls = 0;
    std::size_t pyramidWriteCalls = 0;
    std::size_t totalChunksInMasks = 0;

    void accumulate(const ProfileStats& other)
    {
        tSliceLoad.nanos += other.tSliceLoad.nanos;
        tMaskBuild.nanos += other.tMaskBuild.nanos;
        tMaskMerge.nanos += other.tMaskMerge.nanos;
        tResizeLegacy.nanos += other.tResizeLegacy.nanos;
        tTouchDiscovery.nanos += other.tTouchDiscovery.nanos;
        tChunkApplyCpu.nanos += other.tChunkApplyCpu.nanos;
        tChunkReadIo.nanos += other.tChunkReadIo.nanos;
        tChunkWriteIo.nanos += other.tChunkWriteIo.nanos;
        tPyramidRead.nanos += other.tPyramidRead.nanos;
        tPyramidWrite.nanos += other.tPyramidWrite.nanos;

        slicesTotal += other.slicesTotal;
        slicesSkippedEmpty += other.slicesSkippedEmpty;
        nonzeroFgSlices += other.nonzeroFgSlices;
        computeChunksTotal += other.computeChunksTotal;
        computeChunksSkippedEmpty += other.computeChunksSkippedEmpty;
        computeChunksWrapped += other.computeChunksWrapped;
        fgPixels += other.fgPixels;
        maskPixels += other.maskPixels;
        touchedChunksLevel0 += other.touchedChunksLevel0;
        skippedMissingLevel0 += other.skippedMissingLevel0;
        skippedMissingPyramid += other.skippedMissingPyramid;
        chunkIoReads += other.chunkIoReads;
        chunkIoWrites += other.chunkIoWrites;
        chunkCacheHits += other.chunkCacheHits;
        chunkCacheMisses += other.chunkCacheMisses;
        bytesRead += other.bytesRead;
        bytesWritten += other.bytesWritten;
        inputVoxels += other.inputVoxels;
        pyramidReadCalls += other.pyramidReadCalls;
        pyramidWriteCalls += other.pyramidWriteCalls;
        totalChunksInMasks += other.totalChunksInMasks;
    }

    double totalSeconds() const
    {
        return tSliceLoad.seconds() + tMaskBuild.seconds() + tMaskMerge.seconds() +
               tResizeLegacy.seconds() + tTouchDiscovery.seconds() + tChunkApplyCpu.seconds() +
               tChunkReadIo.seconds() + tChunkWriteIo.seconds() + tPyramidRead.seconds() +
               tPyramidWrite.seconds();
    }
};

struct ScopedTimer {
    std::chrono::steady_clock::time_point start;
    TimingAccumulator* acc;

    explicit ScopedTimer(TimingAccumulator& target)
        : start(std::chrono::steady_clock::now())
        , acc(&target)
    {
    }

    ~ScopedTimer()
    {
        if (acc) {
            auto end = std::chrono::steady_clock::now();
            acc->add(std::chrono::duration<double>(end - start).count());
        }
    }
};

using StageProgressCallback = std::function<void(std::size_t)>;

static std::size_t progressCadence(const std::size_t total)
{
    return std::max<std::size_t>(1, total / 200);
}

static bool shouldEmitStructuredProgress(const std::size_t current,
                                         const std::size_t total,
                                         const std::size_t cadence)
{
    return current == 1 || current == total || (current % cadence) == 0;
}

static void emitStructuredStage(const bool enabled,
                                const char* name,
                                const std::size_t total,
                                const std::optional<int>& level = std::nullopt)
{
    if (!enabled) {
        return;
    }
    std::cerr << "VC_STAGE name=" << name;
    if (level.has_value()) {
        std::cerr << " level=" << *level;
    }
    std::cerr << " total=" << total << '\n';
}

static void emitStructuredProgress(const bool enabled,
                                   const char* name,
                                   const std::size_t current,
                                   const std::size_t total,
                                   const std::optional<int>& level = std::nullopt)
{
    if (!enabled) {
        return;
    }
    std::cerr << "VC_PROGRESS name=" << name;
    if (level.has_value()) {
        std::cerr << " level=" << *level;
    }
    std::cerr << " current=" << current
              << " total=" << total << '\n';
}

struct NearestNeighborMap {
    std::vector<std::uint32_t> dstToSrcY;
    std::vector<std::uint32_t> dstToSrcX;
    std::vector<std::uint32_t> srcToDstY0;
    std::vector<std::uint32_t> srcToDstY1;
    std::vector<std::uint32_t> srcToDstX0;
    std::vector<std::uint32_t> srcToDstX1;
};

static bool isNumericDirName(const std::string& name)
{
    if (name.empty()) {
        return false;
    }
    for (const char c : name) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

static std::vector<int> discoverNumericLevels(const fs::path& root)
{
    std::vector<int> levels;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!isNumericDirName(name)) {
            continue;
        }
        if (!fs::exists(entry.path() / ".zarray")) {
            continue;
        }
        levels.push_back(std::stoi(name));
    }
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
}

static std::string mapModeToString(MapMode mode)
{
    switch (mode) {
    case MapMode::legacy:
        return "legacy";
    case MapMode::exact:
        return "exact";
    case MapMode::fast:
        return "fast";
    default:
        return "unknown";
    }
}

static std::string processingModeToString(ProcessingMode mode)
{
    switch (mode) {
    case ProcessingMode::legacySlice:
        return "legacy-slice";
    case ProcessingMode::chunkAlphaWrap:
        return "chunk-alpha-wrap";
    default:
        return "unknown";
    }
}

static std::string algoModeToString(AlgoMode mode)
{
    switch (mode) {
    case AlgoMode::polar:
        return "polar";
    case AlgoMode::contour:
        return "contour";
    case AlgoMode::hull:
        return "hull";
    case AlgoMode::ray:
        return "ray";
    default:
        return "unknown";
    }
}

static std::string resultModeToString(ResultMode mode)
{
    switch (mode) {
    case ResultMode::same:
        return "same";
    case ResultMode::fast:
        return "fast";
    default:
        return "unknown";
    }
}

static MapMode parseMapMode(const std::string& value)
{
    if (value == "legacy") {
        return MapMode::legacy;
    }
    if (value == "exact") {
        return MapMode::exact;
    }
    if (value == "fast") {
        return MapMode::fast;
    }
    throw std::runtime_error("invalid --map-mode: " + value + " (expected legacy|exact|fast)");
}

static AlgoMode parseAlgoMode(const std::string& value)
{
    if (value == "polar") {
        return AlgoMode::polar;
    }
    if (value == "contour") {
        return AlgoMode::contour;
    }
    if (value == "hull") {
        return AlgoMode::hull;
    }
    if (value == "ray") {
        return AlgoMode::ray;
    }
    throw std::runtime_error("invalid --algo-mode: " + value + " (expected polar|contour|hull|ray)");
}

static ResultMode parseResultMode(const std::string& value)
{
    if (value == "same") {
        return ResultMode::same;
    }
    if (value == "fast") {
        return ResultMode::fast;
    }
    throw std::runtime_error("invalid --result-mode: " + value + " (expected same|fast)");
}

static ProcessingMode parseProcessingMode(const std::string& value)
{
    if (value == "legacy-slice") {
        return ProcessingMode::legacySlice;
    }
    if (value == "chunk-alpha-wrap") {
        return ProcessingMode::chunkAlphaWrap;
    }
    throw std::runtime_error(
        "invalid --mode: " + value + " (expected legacy-slice|chunk-alpha-wrap)");
}

static bool parseUIntToken(std::string_view token, std::size_t& value)
{
    if (token.empty()) {
        return false;
    }
    std::size_t parsed = 0;
    const char* begin = token.data();
    const char* end = token.data() + token.size();
    const auto res = std::from_chars(begin, end, parsed);
    if (res.ec != std::errc() || res.ptr != end) {
        return false;
    }
    value = parsed;
    return true;
}

static bool parseChunkFilename(const std::string& name, ChunkIndex& out)
{
    const auto p1 = name.find('.');
    if (p1 == std::string::npos) {
        return false;
    }
    const auto p2 = name.find('.', p1 + 1);
    if (p2 == std::string::npos || name.find('.', p2 + 1) != std::string::npos) {
        return false;
    }

    std::size_t z = 0;
    std::size_t y = 0;
    std::size_t x = 0;
    if (!parseUIntToken(std::string_view{name}.substr(0, p1), z)) {
        return false;
    }
    if (!parseUIntToken(std::string_view{name}.substr(p1 + 1, p2 - p1 - 1), y)) {
        return false;
    }
    if (!parseUIntToken(std::string_view{name}.substr(p2 + 1), x)) {
        return false;
    }
    out = ChunkIndex{z, y, x};
    return true;
}

static std::string chunkFilename(const ChunkIndex& c)
{
    return std::to_string(c.z) + "." + std::to_string(c.y) + "." + std::to_string(c.x);
}

static std::unordered_set<ChunkIndex, ChunkIndexHash> scanLevelExistingChunks(const fs::path& levelPath)
{
    std::unordered_set<ChunkIndex, ChunkIndexHash> out;
    for (const auto& entry : fs::directory_iterator(levelPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        ChunkIndex c;
        if (!parseChunkFilename(entry.path().filename().string(), c)) {
            continue;
        }
        out.insert(c);
    }
    return out;
}

static std::unordered_map<int, std::unordered_set<ChunkIndex, ChunkIndexHash>> scanExistingChunksByLevel(
    const fs::path& inputRoot,
    const std::vector<int>& levels)
{
    std::unordered_map<int, std::unordered_set<ChunkIndex, ChunkIndexHash>> out;
    out.reserve(levels.size());
    for (int level : levels) {
        out.emplace(level, scanLevelExistingChunks(inputRoot / std::to_string(level)));
    }
    return out;
}

static std::optional<std::size_t> memAvailableBytes()
{
#if defined(__linux__)
    std::ifstream in("/proc/meminfo");
    if (!in) {
        return std::nullopt;
    }
    std::string key;
    std::size_t valueKb = 0;
    std::string unit;
    while (in >> key >> valueKb >> unit) {
        if (key == "MemAvailable:") {
            return valueKb * std::size_t{1024};
        }
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

static nlohmann::json readJsonFile(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open json file: " + path.string());
    }
    return nlohmann::json::parse(in);
}

static void writeJsonFile(const fs::path& path, const nlohmann::json& j)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to create json file: " + path.string());
    }
    out << j.dump(2) << '\n';
    if (!out) {
        throw std::runtime_error("failed to write json file: " + path.string());
    }
}

static std::string nowIsoUtc()
{
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tmUtc{};
#if defined(_WIN32)
    gmtime_s(&tmUtc, &t);
#else
    gmtime_r(&t, &tmUtc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tmUtc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

static Shape3 toShape3(const std::vector<std::size_t>& shape)
{
    if (shape.size() != 3) {
        throw std::runtime_error("expected 3D dataset");
    }
    return {shape[0], shape[1], shape[2]};
}

static bool hasAnyNonZero(const std::vector<uint8_t>& data)
{
    return std::any_of(data.begin(), data.end(), [](uint8_t v) { return v != 0; });
}

static bool hasAnyNonZero(const uint8_t* data, std::size_t n)
{
    return std::any_of(data, data + n, [](uint8_t v) { return v != 0; });
}

static std::size_t countNonZero(const uint8_t* data, std::size_t n)
{
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] != 0) {
            ++c;
        }
    }
    return c;
}

static std::size_t countNonZero(const cv::Mat1b& m)
{
    if (m.empty()) {
        return 0;
    }
    if (m.isContinuous()) {
        return countNonZero(m.ptr<uint8_t>(0), m.total());
    }

    std::size_t c = 0;
    for (int y = 0; y < m.rows; ++y) {
        c += countNonZero(m.ptr<uint8_t>(y), static_cast<std::size_t>(m.cols));
    }
    return c;
}

static std::size_t volumeElements(const Shape3& shape)
{
    return shape[0] * shape[1] * shape[2];
}

static std::size_t linearIndex(const Shape3& shape,
                               std::size_t z,
                               std::size_t y,
                               std::size_t x)
{
    return (z * shape[1] + y) * shape[2] + x;
}

static bool boxIntersectsZRange(const Box3& box, int zStart, int zStop)
{
    const std::size_t boxZ0 = box.origin[0];
    const std::size_t boxZ1 = box.origin[0] + box.shape[0];
    return boxZ0 < static_cast<std::size_t>(zStop) && static_cast<std::size_t>(zStart) < boxZ1;
}

static Box3 makeChunkBox(const ChunkIndex& chunk, const Shape3& chunkShape, const Shape3& volumeShape)
{
    const Shape3 origin = {
        chunk.z * chunkShape[0],
        chunk.y * chunkShape[1],
        chunk.x * chunkShape[2],
    };

    return {
        origin,
        {
            origin[0] < volumeShape[0] ? std::min(chunkShape[0], volumeShape[0] - origin[0]) : 0,
            origin[1] < volumeShape[1] ? std::min(chunkShape[1], volumeShape[1] - origin[1]) : 0,
            origin[2] < volumeShape[2] ? std::min(chunkShape[2], volumeShape[2] - origin[2]) : 0,
        },
    };
}

static Box3 expandAndClampBox(const Box3& box, std::size_t halo, const Shape3& volumeShape)
{
    const Shape3 origin = {
        box.origin[0] > halo ? box.origin[0] - halo : 0,
        box.origin[1] > halo ? box.origin[1] - halo : 0,
        box.origin[2] > halo ? box.origin[2] - halo : 0,
    };
    const Shape3 end = {
        std::min(volumeShape[0], box.origin[0] + box.shape[0] + halo),
        std::min(volumeShape[1], box.origin[1] + box.shape[1] + halo),
        std::min(volumeShape[2], box.origin[2] + box.shape[2] + halo),
    };
    return {
        origin,
        {
            end[0] - origin[0],
            end[1] - origin[1],
            end[2] - origin[2],
        },
    };
}

static Shape3 relativeOrigin(const Box3& inner, const Box3& outer)
{
    return {
        inner.origin[0] - outer.origin[0],
        inner.origin[1] - outer.origin[1],
        inner.origin[2] - outer.origin[2],
    };
}

static int wrapIndex(int idx, int n)
{
    int v = idx % n;
    if (v < 0) {
        v += n;
    }
    return v;
}

static double outerQuantileFromAlpha(double alpha)
{
    const double clamped = std::clamp(alpha, 0.0, 0.1);
    // Smaller alpha -> tighter hull, larger alpha -> looser hull.
    return std::clamp(0.55 + clamped * 35.0, 0.55, 0.98);
}

static std::vector<double> circularMedianSmooth(const std::vector<double>& values, int window)
{
    if (window <= 1 || values.empty()) {
        return values;
    }

    const int n = static_cast<int>(values.size());
    const int pad = window / 2;
    std::vector<double> out(n, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> local;
    local.reserve(window);

    for (int i = 0; i < n; ++i) {
        local.clear();
        for (int d = -pad; d <= pad; ++d) {
            const double v = values[wrapIndex(i + d, n)];
            if (std::isfinite(v)) {
                local.push_back(v);
            }
        }
        if (local.empty()) {
            continue;
        }
        std::sort(local.begin(), local.end());
        out[i] = local[local.size() / 2];
    }
    return out;
}

static bool buildDistanceBinsFromPoints(const std::vector<cv::Point>& points,
                                       int nAngleBins,
                                       double& centroidX,
                                       double& centroidY,
                                       std::vector<std::vector<double>>& distanceBins,
                                       std::size_t& pointCount,
                                       bool useFastAtan2,
                                       bool useSquaredDist)
{
    if (points.empty()) {
        return false;
    }

    double sumX = 0.0;
    double sumY = 0.0;
    for (const auto& p : points) {
        sumX += static_cast<double>(p.x);
        sumY += static_cast<double>(p.y);
    }

    pointCount = points.size();

    if (pointCount < 3) {
        return false;
    }

    centroidX = sumX / static_cast<double>(pointCount);
    centroidY = sumY / static_cast<double>(pointCount);

    distanceBins.assign(nAngleBins, {});
    for (auto& bin : distanceBins) {
        bin.reserve(128);
    }

    constexpr double twoPi = 2.0 * kPi;
    for (const auto& p : points) {
        const double dx = static_cast<double>(p.x) - centroidX;
        const double dy = static_cast<double>(p.y) - centroidY;
        const double angle = useFastAtan2 ? static_cast<double>(cv::fastAtan2(dy, dx))
                                          : std::atan2(dy, dx); // [-pi, pi] for atan2
        double norm = useFastAtan2 ? angle / 360.0 : (angle + kPi) / twoPi;
        if (useFastAtan2) {
            if (norm >= 1.0) {
                norm -= 1.0;
            } else if (norm < 0.0) {
                norm += 1.0;
            }
        }
        int bin = static_cast<int>(norm * static_cast<double>(nAngleBins));
        if (bin < 0) {
            bin = 0;
        } else if (bin >= nAngleBins) {
            bin = nAngleBins - 1;
        }
        const double dist = dx * dx + dy * dy;
        distanceBins[bin].push_back(useSquaredDist ? dist : std::sqrt(dist));
    }
    return true;
}

static bool buildDistanceBins(const cv::Mat1b& labelSlice,
                             int nAngleBins,
                             double& centroidX,
                             double& centroidY,
                             std::vector<std::vector<double>>& distanceBins,
                             std::size_t& pointCount,
                             bool useFastAtan2,
                             bool useSquaredDist)
{
    std::vector<cv::Point> points;
    points.reserve(static_cast<std::size_t>(labelSlice.rows * labelSlice.cols / 32));
    for (int y = 0; y < labelSlice.rows; ++y) {
        const uint8_t* row = labelSlice.ptr<uint8_t>(y);
        for (int x = 0; x < labelSlice.cols; ++x) {
            if (row[x] != 0) {
                points.emplace_back(x, y);
            }
        }
    }
    return buildDistanceBinsFromPoints(points, nAngleBins, centroidX, centroidY, distanceBins, pointCount,
                                      useFastAtan2, useSquaredDist);
}

static bool buildDistanceBinsFromContours(const cv::Mat1b& labelSlice,
                                         bool useHull,
                                         int nAngleBins,
                                         double& centroidX,
                                         double& centroidY,
                                         std::vector<std::vector<double>>& distanceBins,
                                         std::size_t& pointCount,
                                         bool useFastAtan2,
                                         bool useSquaredDist)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(labelSlice, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        return false;
    }

    std::vector<cv::Point> points;
    for (const auto& contour : contours) {
        if (contour.empty()) {
            continue;
        }
        if (useHull && contour.size() >= 3) {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            points.insert(points.end(), hull.begin(), hull.end());
        } else {
            points.insert(points.end(), contour.begin(), contour.end());
        }
    }
    return buildDistanceBinsFromPoints(points, nAngleBins, centroidX, centroidY, distanceBins, pointCount,
                                      useFastAtan2, useSquaredDist);
}

static bool buildDistanceBinsByRay(const cv::Mat1b& labelSlice,
                                  int nAngleBins,
                                  double& centroidX,
                                  double& centroidY,
                                  std::vector<std::vector<double>>& distanceBins,
                                  std::size_t& pointCount,
                                  bool useSquaredDist)
{
    std::vector<cv::Point> fg;
    cv::findNonZero(labelSlice, fg);
    if (fg.empty()) {
        return false;
    }

    centroidX = 0.0;
    centroidY = 0.0;
    for (const auto& p : fg) {
        centroidX += static_cast<double>(p.x);
        centroidY += static_cast<double>(p.y);
    }
    centroidX /= static_cast<double>(fg.size());
    centroidY /= static_cast<double>(fg.size());

    const int maxR = static_cast<int>(
        std::ceil(std::hypot(static_cast<double>(labelSlice.cols), static_cast<double>(labelSlice.rows))));
    pointCount = 0;
    distanceBins.assign(nAngleBins, {});
    for (auto& bin : distanceBins) {
        bin.reserve(4);
    }

    constexpr double twoPi = 2.0 * kPi;
    for (int i = 0; i < nAngleBins; ++i) {
        const double angle = -kPi + (static_cast<double>(i) + 0.5) * twoPi / static_cast<double>(nAngleBins);
        const double vx = std::cos(angle);
        const double vy = std::sin(angle);
        bool wasInside = false;
        int lastInside = -1;

        for (int t = 0; t <= maxR; ++t) {
            const int x = static_cast<int>(std::llround(centroidX + vx * static_cast<double>(t)));
            const int y = static_cast<int>(std::llround(centroidY + vy * static_cast<double>(t)));
            if (x < 0 || x >= labelSlice.cols || y < 0 || y >= labelSlice.rows) {
                break;
            }

            const bool inside = labelSlice.ptr<uint8_t>(y)[x] != 0;
            if (inside && !wasInside) {
                const double rr = static_cast<double>(t);
                distanceBins[i].push_back(useSquaredDist ? rr * rr : rr);
                ++pointCount;
                wasInside = true;
                lastInside = t;
            } else if (inside) {
                lastInside = t;
            } else if (!inside && wasInside) {
                const double rr = static_cast<double>(lastInside);
                distanceBins[i].push_back(useSquaredDist ? rr * rr : rr);
                ++pointCount;
                wasInside = false;
                lastInside = -1;
            }
        }
        if (wasInside && lastInside >= 0) {
            const double rr = static_cast<double>(lastInside);
            distanceBins[i].push_back(useSquaredDist ? rr * rr : rr);
            ++pointCount;
        }
    }
    return pointCount >= nAngleBins;
}

static cv::Mat1b polygonMaskFromPolar(const std::vector<double>& radii,
                                      double centroidX,
                                      double centroidY,
                                      int width,
                                      int height)
{
    std::vector<cv::Point> polygon;
    polygon.reserve(radii.size());
    constexpr double twoPi = 2.0 * kPi;

    for (int i = 0; i < static_cast<int>(radii.size()); ++i) {
        const double r = radii[i];
        if (!std::isfinite(r) || r <= 0.0) {
            continue;
        }
        const double angle = -kPi + (static_cast<double>(i) + 0.5) * twoPi /
                                        static_cast<double>(radii.size());
        const int px = static_cast<int>(std::llround(centroidX + r * std::cos(angle)));
        const int py = static_cast<int>(std::llround(centroidY + r * std::sin(angle)));
        polygon.emplace_back(px, py);
    }

    cv::Mat1b inside(height, width, uint8_t(0));
    if (polygon.size() < 3) {
        return inside;
    }
    cv::fillPoly(inside, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255));
    return inside;
}

struct SliceMaskPair {
    cv::Mat1b inner;
    cv::Mat1b outer;
};

static SliceMaskPair detectIgnoreMasks(const cv::Mat1b& labelSlice,
                                       int nAngleBins,
                                       double alpha,
                                       int outerSmoothingWindow,
                                       double shrinkFactor,
                                       int innerSmoothingWindow,
                                       bool needOuter,
                                       bool needInner,
                                       bool useFastAtan2,
                                       bool useSquaredDist,
                                       AlgoMode algoMode)
{
    SliceMaskPair result{
        cv::Mat1b(labelSlice.size(), uint8_t(0)),
        cv::Mat1b(labelSlice.size(), uint8_t(0)),
    };
    if (!needOuter && !needInner) {
        return result;
    }

    double centroidX = 0.0;
    double centroidY = 0.0;
    std::vector<std::vector<double>> distanceBins;
    std::size_t pointCount = 0;
    bool ok = false;
    if (algoMode == AlgoMode::polar) {
        ok = buildDistanceBins(labelSlice,
                               nAngleBins,
                               centroidX,
                               centroidY,
                               distanceBins,
                               pointCount,
                               useFastAtan2,
                               useSquaredDist);
    } else if (algoMode == AlgoMode::contour || algoMode == AlgoMode::hull) {
        ok = buildDistanceBinsFromContours(labelSlice,
                                          algoMode == AlgoMode::hull,
                                          nAngleBins,
                                          centroidX,
                                          centroidY,
                                          distanceBins,
                                          pointCount,
                                          useFastAtan2,
                                          useSquaredDist);
        if (!ok) {
            ok = buildDistanceBins(labelSlice,
                                   nAngleBins,
                                   centroidX,
                                   centroidY,
                                   distanceBins,
                                   pointCount,
                                   useFastAtan2,
                                   useSquaredDist);
        }
    } else if (algoMode == AlgoMode::ray) {
        ok = buildDistanceBinsByRay(labelSlice,
                                    nAngleBins,
                                    centroidX,
                                    centroidY,
                                    distanceBins,
                                    pointCount,
                                    useSquaredDist);
    }

    if (!ok) {
        return result;
    }

    if (needOuter) {
        const bool convertBack = useSquaredDist;
        const double quantile = outerQuantileFromAlpha(alpha);
        std::vector<double> outerRadii(nAngleBins, std::numeric_limits<double>::quiet_NaN());
        for (int i = 0; i < nAngleBins; ++i) {
            auto& bin = distanceBins[i];
            if (bin.empty()) {
                continue;
            }
            const double pos = quantile * static_cast<double>(bin.size() - 1);
            const std::size_t idx =
                std::min(static_cast<std::size_t>(std::llround(pos)), bin.size() - 1);
            std::nth_element(bin.begin(), bin.begin() + static_cast<std::ptrdiff_t>(idx), bin.end());
            outerRadii[i] = bin[idx];
        }
        if (convertBack) {
            for (double& r : outerRadii) {
                if (std::isfinite(r)) {
                    r = std::sqrt(r);
                }
            }
        }
        outerRadii = circularMedianSmooth(outerRadii, outerSmoothingWindow);

        cv::Mat1b inside = polygonMaskFromPolar(
            outerRadii, centroidX, centroidY, labelSlice.cols, labelSlice.rows);
        for (int y = 0; y < labelSlice.rows; ++y) {
            const uint8_t* lrow = labelSlice.ptr<uint8_t>(y);
            const uint8_t* irow = inside.ptr<uint8_t>(y);
            uint8_t* orow = result.outer.ptr<uint8_t>(y);
            for (int x = 0; x < labelSlice.cols; ++x) {
                if (lrow[x] == 0 && irow[x] == 0) {
                    orow[x] = 255;
                }
            }
        }
    }

    if (needInner) {
        const bool convertBack = useSquaredDist;
        std::vector<double> minRadii(nAngleBins, std::numeric_limits<double>::quiet_NaN());
        for (int i = 0; i < nAngleBins; ++i) {
            if (distanceBins[i].empty()) {
                continue;
            }
            minRadii[i] = *std::min_element(distanceBins[i].begin(), distanceBins[i].end());
        }
        if (convertBack) {
            for (double& r : minRadii) {
                if (std::isfinite(r)) {
                    r = std::sqrt(r);
                }
            }
        }
        minRadii = circularMedianSmooth(minRadii, innerSmoothingWindow);

        for (double& r : minRadii) {
            if (std::isfinite(r)) {
                r *= shrinkFactor;
            }
        }

        cv::Mat1b inside = polygonMaskFromPolar(
            minRadii, centroidX, centroidY, labelSlice.cols, labelSlice.rows);
        for (int y = 0; y < labelSlice.rows; ++y) {
            const uint8_t* lrow = labelSlice.ptr<uint8_t>(y);
            const uint8_t* irow = inside.ptr<uint8_t>(y);
            uint8_t* orow = result.inner.ptr<uint8_t>(y);
            for (int x = 0; x < labelSlice.cols; ++x) {
                if (irow[x] != 0 && lrow[x] == 0) {
                    orow[x] = 255;
                }
            }
        }
    }

    return result;
}

static cv::Mat1b detectInnerRegion(const cv::Mat1b& labelSlice,
                                   int nAngleBins,
                                   double shrinkFactor,
                                   int smoothingWindow,
                                   bool useFastAtan2,
                                   bool useSquaredDist,
                                   AlgoMode algoMode)
{
    return detectIgnoreMasks(labelSlice,
                             nAngleBins,
                             kDefaultAlpha,
                             kDefaultOuterSmoothing,
                             shrinkFactor,
                             smoothingWindow,
                             false,
                             true,
                             useFastAtan2,
                             useSquaredDist,
                             algoMode)
        .inner;
}

static cv::Mat1b detectOuterRegion(const cv::Mat1b& labelSlice,
                                   int nAngleBins,
                                   double alpha,
                                   int smoothingWindow,
                                   bool useFastAtan2,
                                   bool useSquaredDist,
                                   AlgoMode algoMode)
{
    return detectIgnoreMasks(labelSlice,
                             nAngleBins,
                             alpha,
                             smoothingWindow,
                             kDefaultShrink,
                             kDefaultInnerSmoothing,
                             true,
                             false,
                             useFastAtan2,
                             useSquaredDist,
                             algoMode)
        .outer;
}

static cv::Mat1b upscaleMaskNearest(const cv::Mat1b& mask, int outH, int outW)
{
    if (mask.rows == outH && mask.cols == outW) {
        return mask.clone();
    }
    cv::Mat1b upscaled;
    cv::resize(mask, upscaled, cv::Size(outW, outH), 0.0, 0.0, cv::INTER_NEAREST);
    return upscaled;
}

static NearestNeighborMap buildNearestNeighborMap(std::size_t srcH,
                                                 std::size_t srcW,
                                                 std::size_t dstH,
                                                 std::size_t dstW)
{
    NearestNeighborMap map;
    map.dstToSrcY.resize(dstH);
    map.dstToSrcX.resize(dstW);
    map.srcToDstY0.assign(srcH, static_cast<std::uint32_t>(dstH));
    map.srcToDstY1.assign(srcH, 0);
    map.srcToDstX0.assign(srcW, static_cast<std::uint32_t>(dstW));
    map.srcToDstX1.assign(srcW, 0);

    if (srcH == 0 || srcW == 0 || dstH == 0 || dstW == 0) {
        return map;
    }

    if (srcH == dstH && srcW == dstW) {
        for (std::size_t y = 0; y < srcH; ++y) {
            map.dstToSrcY[y] = static_cast<std::uint32_t>(y);
            map.srcToDstY0[y] = static_cast<std::uint32_t>(y);
            map.srcToDstY1[y] = static_cast<std::uint32_t>(y + 1);
        }
        for (std::size_t x = 0; x < srcW; ++x) {
            map.dstToSrcX[x] = static_cast<std::uint32_t>(x);
            map.srcToDstX0[x] = static_cast<std::uint32_t>(x);
            map.srcToDstX1[x] = static_cast<std::uint32_t>(x + 1);
        }
        return map;
    }

    cv::Mat yMapSrc(static_cast<int>(srcH), 1, CV_32S);
    for (std::size_t y = 0; y < srcH; ++y) {
        yMapSrc.ptr<int>()[y] = static_cast<int>(y);
    }
    cv::Mat yMapDst;
    cv::resize(yMapSrc, yMapDst, cv::Size(1, static_cast<int>(dstH)), 0.0, 0.0, cv::INTER_NEAREST);

    cv::Mat xMapSrc(1, static_cast<int>(srcW), CV_32S);
    for (std::size_t x = 0; x < srcW; ++x) {
        xMapSrc.ptr<int>()[x] = static_cast<int>(x);
    }
    cv::Mat xMapDst;
    cv::resize(xMapSrc, xMapDst, cv::Size(static_cast<int>(dstW), 1), 0.0, 0.0, cv::INTER_NEAREST);

    const int* yMapPtr = yMapDst.ptr<int>();
    const int* xMapPtr = xMapDst.ptr<int>();

    for (std::size_t y = 0; y < dstH; ++y) {
        const std::size_t srcY = static_cast<std::size_t>(yMapPtr[y]);
        map.dstToSrcY[y] = static_cast<std::uint32_t>(srcY);
        map.srcToDstY0[srcY] = std::min(map.srcToDstY0[srcY], static_cast<std::uint32_t>(y));
        map.srcToDstY1[srcY] = std::max(map.srcToDstY1[srcY], static_cast<std::uint32_t>(y));
    }
    for (std::size_t y = 0; y < srcH; ++y) {
        if (map.srcToDstY1[y] >= map.srcToDstY0[y]) {
            map.srcToDstY1[y] += 1;
        }
    }

    for (std::size_t x = 0; x < dstW; ++x) {
        const std::size_t srcX = static_cast<std::size_t>(xMapPtr[x]);
        map.dstToSrcX[x] = static_cast<std::uint32_t>(srcX);
        map.srcToDstX0[srcX] = std::min(map.srcToDstX0[srcX], static_cast<std::uint32_t>(x));
        map.srcToDstX1[srcX] = std::max(map.srcToDstX1[srcX], static_cast<std::uint32_t>(x));
    }
    for (std::size_t x = 0; x < srcW; ++x) {
        if (map.srcToDstX1[x] >= map.srcToDstX0[x]) {
            map.srcToDstX1[x] += 1;
        }
    }

    return map;
}

static std::size_t hashChunk(std::size_t z, std::size_t y, std::size_t x)
{
    std::size_t h = 1469598103934665603ull;
    h ^= z + 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    h ^= y + 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    h ^= x + 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    return h;
}

class ChunkLockPool {
public:
    explicit ChunkLockPool(std::size_t count) : locks_(count) {}

    std::mutex& lockFor(std::size_t z, std::size_t y, std::size_t x)
    {
        return locks_[hashChunk(z, y, x) % locks_.size()];
    }

private:
    std::vector<std::mutex> locks_;
};

static void applyMaskToOutputZRangeLegacy(
    vc::VcDataset& output,
    const Shape3& outShape,
    const Shape3& outChunk,
    std::size_t outZ0,
    std::size_t outZ1,
    const cv::Mat1b& mask,
    const std::vector<std::pair<std::size_t, std::size_t>>& touchedChunkXY,
    const std::unordered_set<ChunkIndex, ChunkIndexHash>* existingLevel0Chunks,
    bool existingChunksOnly,
    uint8_t ignoreValue,
    ChunkLockPool& lockPool,
    std::unordered_set<ChunkIndex, ChunkIndexHash>& touchedLevel0,
    std::size_t& skippedMissingChunks,
    ProfileStats* profile = nullptr)
{
    const std::size_t chunkZ = outChunk[0];
    const std::size_t chunkY = outChunk[1];
    const std::size_t chunkX = outChunk[2];
    const std::size_t chunkElems = chunkZ * chunkY * chunkX;
    const std::size_t chunkBytes = chunkElems * sizeof(uint8_t);

    std::vector<std::pair<std::size_t, std::vector<std::size_t>>> czToLocalZ;
    czToLocalZ.reserve(2);
    for (std::size_t outZ = outZ0; outZ < outZ1; ++outZ) {
        const std::size_t cz = outZ / chunkZ;
        const std::size_t lz = outZ % chunkZ;
        if (czToLocalZ.empty() || czToLocalZ.back().first != cz) {
            czToLocalZ.push_back({cz, {}});
        }
        czToLocalZ.back().second.push_back(lz);
    }

    std::vector<uint8_t> chunkBuf(chunkElems, 0);
    for (const auto& ycxc : touchedChunkXY) {
        const std::size_t cy = ycxc.first;
        const std::size_t cx = ycxc.second;
        const std::size_t y0 = cy * chunkY;
        const std::size_t x0 = cx * chunkX;
        const std::size_t h = std::min(chunkY, outShape[1] - y0);
        const std::size_t w = std::min(chunkX, outShape[2] - x0);

        for (const auto& group : czToLocalZ) {
            const std::size_t cz = group.first;
            const ChunkIndex chunk{cz, cy, cx};

            if (existingChunksOnly &&
                existingLevel0Chunks != nullptr &&
                existingLevel0Chunks->find(chunk) == existingLevel0Chunks->end()) {
                ++skippedMissingChunks;
                continue;
            }

            bool changed = false;
            {
                std::lock_guard<std::mutex> guard(lockPool.lockFor(cz, cy, cx));
                auto ioStart = std::chrono::steady_clock::now();
                chunkBuf.assign(chunkElems, 0);
                output.readChunk(cz, cy, cx, chunkBuf.data());
                if (profile) {
                    auto ioEnd = std::chrono::steady_clock::now();
                    profile->tChunkReadIo.add(
                        std::chrono::duration<double>(ioEnd - ioStart).count());
                    ++profile->chunkIoReads;
                    profile->bytesRead += chunkBytes;
                }

                auto cpuStart = std::chrono::steady_clock::now();
                for (const std::size_t lz : group.second) {
                    for (std::size_t yy = 0; yy < h; ++yy) {
                        const uint8_t* mrow = mask.ptr<uint8_t>(static_cast<int>(y0 + yy)) +
                                              static_cast<int>(x0);
                        for (std::size_t xx = 0; xx < w; ++xx) {
                            if (mrow[xx] == 0) {
                                continue;
                            }
                            const std::size_t idx = lz * chunkY * chunkX + yy * chunkX + xx;
                            if (chunkBuf[idx] == 0) {
                                chunkBuf[idx] = ignoreValue;
                                changed = true;
                            }
                        }
                    }
                }
                if (profile) {
                    auto cpuEnd = std::chrono::steady_clock::now();
                    profile->tChunkApplyCpu.add(
                        std::chrono::duration<double>(cpuEnd - cpuStart).count());
                }

                if (changed) {
                    auto ioWriteStart = std::chrono::steady_clock::now();
                    output.writeChunk(cz, cy, cx, chunkBuf.data(), chunkBytes);
                    if (profile) {
                        auto ioWriteEnd = std::chrono::steady_clock::now();
                        profile->tChunkWriteIo.add(
                            std::chrono::duration<double>(ioWriteEnd - ioWriteStart).count());
                        ++profile->chunkIoWrites;
                        profile->bytesWritten += chunkBytes;
                    }
                    touchedLevel0.insert(chunk);
                }
            }
        }
    }
}

static void applyMaskToOutputZRangeMapped(
    vc::VcDataset& output,
    const Shape3& outShape,
    const Shape3& outChunk,
    const NearestNeighborMap& nn,
    std::size_t outZ0,
    std::size_t outZ1,
    const cv::Mat1b& mask,
    const std::vector<std::pair<std::size_t, std::size_t>>& touchedChunkXY,
    const std::unordered_set<ChunkIndex, ChunkIndexHash>* existingLevel0Chunks,
    bool existingChunksOnly,
    uint8_t ignoreValue,
    ChunkLockPool& lockPool,
    std::unordered_set<ChunkIndex, ChunkIndexHash>& touchedLevel0,
    std::size_t& skippedMissingChunks,
    ProfileStats* profile = nullptr)
{
    const std::size_t chunkY = outChunk[1];
    const std::size_t chunkX = outChunk[2];
    const std::size_t chunkElems = outChunk[0] * chunkY * chunkX;
    const std::size_t chunkBytes = chunkElems * sizeof(uint8_t);

    std::vector<std::pair<std::size_t, std::vector<std::size_t>>> czToLocalZ;
    czToLocalZ.reserve(2);
    for (std::size_t outZ = outZ0; outZ < outZ1; ++outZ) {
        const std::size_t cz = outZ / outChunk[0];
        const std::size_t lz = outZ % outChunk[0];
        if (czToLocalZ.empty() || czToLocalZ.back().first != cz) {
            czToLocalZ.push_back({cz, {}});
        }
        czToLocalZ.back().second.push_back(lz);
    }
    if (czToLocalZ.empty()) {
        return;
    }

    std::vector<uint8_t> chunkBuf(chunkElems, 0);
    for (const auto& ycxc : touchedChunkXY) {
        const std::size_t cy = ycxc.first;
        const std::size_t cx = ycxc.second;
        const std::size_t y0 = cy * chunkY;
        const std::size_t x0 = cx * chunkX;
        const std::size_t h = std::min(chunkY, outShape[1] - y0);
        const std::size_t w = std::min(chunkX, outShape[2] - x0);

        std::vector<std::uint32_t> outToSrcY(h);
        for (std::size_t yy = 0; yy < h; ++yy) {
            outToSrcY[yy] = nn.dstToSrcY[y0 + yy];
        }
        std::vector<std::uint32_t> outToSrcX(w);
        for (std::size_t xx = 0; xx < w; ++xx) {
            outToSrcX[xx] = nn.dstToSrcX[x0 + xx];
        }

        for (const auto& group : czToLocalZ) {
            const std::size_t cz = group.first;
            const ChunkIndex chunk{cz, cy, cx};
            if (existingChunksOnly &&
                existingLevel0Chunks != nullptr &&
                existingLevel0Chunks->find(chunk) == existingLevel0Chunks->end()) {
                ++skippedMissingChunks;
                continue;
            }

            bool changed = false;
            {
                std::lock_guard<std::mutex> guard(lockPool.lockFor(cz, cy, cx));
                auto ioStart = std::chrono::steady_clock::now();
                chunkBuf.assign(chunkElems, 0);
                output.readChunk(cz, cy, cx, chunkBuf.data());
                if (profile) {
                    auto ioEnd = std::chrono::steady_clock::now();
                    profile->tChunkReadIo.add(std::chrono::duration<double>(ioEnd - ioStart).count());
                    ++profile->chunkIoReads;
                    profile->bytesRead += chunkBytes;
                }

                auto cpuStart = std::chrono::steady_clock::now();
                const std::size_t sliceElems = chunkY * chunkX;
                for (const std::size_t lz : group.second) {
                    const std::size_t base = lz * sliceElems;
                    for (std::size_t yy = 0; yy < h; ++yy) {
                        const std::size_t srcYY = outToSrcY[yy];
                        const uint8_t* srcRow = mask.ptr<uint8_t>(static_cast<int>(srcYY));
                        const std::size_t dstBase = base + yy * chunkX;
                        for (std::size_t xx = 0; xx < w; ++xx) {
                            const std::size_t srcX = outToSrcX[xx];
                            if (srcRow[srcX] == 0) {
                                continue;
                            }
                            const std::size_t dstIdx = dstBase + xx;
                            if (chunkBuf[dstIdx] == 0) {
                                chunkBuf[dstIdx] = ignoreValue;
                                changed = true;
                            }
                        }
                    }
                }
                if (profile) {
                    auto cpuEnd = std::chrono::steady_clock::now();
                    profile->tChunkApplyCpu.add(std::chrono::duration<double>(cpuEnd - cpuStart).count());
                }

                if (changed) {
                    auto ioWriteStart = std::chrono::steady_clock::now();
                    output.writeChunk(cz, cy, cx, chunkBuf.data(), chunkBytes);
                    if (profile) {
                        auto ioWriteEnd = std::chrono::steady_clock::now();
                        profile->tChunkWriteIo.add(
                            std::chrono::duration<double>(ioWriteEnd - ioWriteStart).count());
                        ++profile->chunkIoWrites;
                        profile->bytesWritten += chunkBytes;
                    }
                    touchedLevel0.insert(chunk);
                }
            }
        }
    }
}

static bool writeChunkBufferMapped(
    const cv::Mat1b& mask,
    const Shape3& outChunk,
    std::size_t slabOutZ0,
    std::size_t slabOutZ1,
    const std::vector<std::uint32_t>& outToSrcY,
    const std::vector<std::uint32_t>& outToSrcX,
    const std::size_t outZ0,
    const std::size_t outZ1,
    std::vector<std::uint8_t>& chunkBuf,
    uint8_t ignoreValue,
    ProfileStats* profile = nullptr)
{
    const std::size_t h = outToSrcY.size();
    const std::size_t w = outToSrcX.size();
    if (h == 0 || w == 0) {
        return false;
    }

    const std::size_t localZ0 = std::max(outZ0, slabOutZ0);
    const std::size_t localZ1 = std::min(outZ1, slabOutZ1);
    if (localZ0 >= localZ1) {
        return false;
    }

    auto cpuStart = std::chrono::steady_clock::now();
    const std::size_t sliceElems = outChunk[1] * outChunk[2];
    bool changed = false;
    for (std::size_t outZ = localZ0; outZ < localZ1; ++outZ) {
        const std::size_t localZ = outZ - slabOutZ0;
        const std::size_t base = localZ * sliceElems;
        for (std::size_t yy = 0; yy < h; ++yy) {
            const std::size_t srcYY = outToSrcY[yy];
            const uint8_t* srcRow = mask.ptr<uint8_t>(static_cast<int>(srcYY));
            const std::size_t dstRowBase = base + yy * outChunk[2];
            for (std::size_t xx = 0; xx < w; ++xx) {
                const std::size_t srcX = outToSrcX[xx];
                if (srcRow[srcX] == 0) {
                    continue;
                }
                const std::size_t dstIdx = dstRowBase + xx;
                if (chunkBuf[dstIdx] == 0) {
                    chunkBuf[dstIdx] = ignoreValue;
                    changed = true;
                }
            }
        }
    }
    if (profile) {
        auto cpuEnd = std::chrono::steady_clock::now();
        profile->tChunkApplyCpu.add(std::chrono::duration<double>(cpuEnd - cpuStart).count());
    }
    return changed;
}

static std::vector<std::pair<std::size_t, std::size_t>> collectTouchedChunkXY(
    const cv::Mat1b& mask,
    const Shape3& outShape,
    const Shape3& outChunk)
{
    const std::size_t chunkY = outChunk[1];
    const std::size_t chunkX = outChunk[2];
    const std::size_t nCY = (outShape[1] + chunkY - 1) / chunkY;
    const std::size_t nCX = (outShape[2] + chunkX - 1) / chunkX;

    std::vector<uint8_t> touched(nCY * nCX, uint8_t(0));
    for (int y = 0; y < mask.rows; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        const std::size_t cy = static_cast<std::size_t>(y) / chunkY;
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x] == 0) {
                continue;
            }
            const std::size_t cx = static_cast<std::size_t>(x) / chunkX;
            touched[cy * nCX + cx] = 1;
        }
    }

    std::vector<std::pair<std::size_t, std::size_t>> out;
    out.reserve(64);
    for (std::size_t cy = 0; cy < nCY; ++cy) {
        for (std::size_t cx = 0; cx < nCX; ++cx) {
            if (touched[cy * nCX + cx] != 0) {
                out.emplace_back(cy, cx);
            }
        }
    }
    return out;
}

static std::vector<std::pair<std::size_t, std::size_t>> collectTouchedChunkXYMapped(
    const cv::Mat1b& mask,
    const NearestNeighborMap& nnMap,
    const Shape3& outShape,
    const Shape3& outChunk)
{
    const std::size_t chunkY = outChunk[1];
    const std::size_t chunkX = outChunk[2];
    const std::size_t nCY = (outShape[1] + chunkY - 1) / chunkY;
    const std::size_t nCX = (outShape[2] + chunkX - 1) / chunkX;
    if (mask.empty() || nCY == 0 || nCX == 0) {
        return {};
    }

    std::vector<std::uint32_t> seen(nCY * nCX, 0);
    const std::uint32_t token = 1;

    std::vector<std::pair<std::size_t, std::size_t>> out;
    out.reserve(128);

    for (int y = 0; y < mask.rows; ++y) {
        const std::size_t sy = static_cast<std::size_t>(y);
        if (sy >= nnMap.srcToDstY0.size()) {
            break;
        }

        const std::size_t dstY0 = nnMap.srcToDstY0[sy];
        const std::size_t dstY1 = nnMap.srcToDstY1[sy];
        if (dstY1 <= dstY0) {
            continue;
        }

        const std::size_t cy0 = dstY0 / chunkY;
        const std::size_t cy1 = std::min((dstY1 - 1) / chunkY, nCY - 1);
        const uint8_t* row = mask.ptr<uint8_t>(y);

        int x = 0;
        while (x < mask.cols) {
            while (x < mask.cols && row[x] == 0) {
                ++x;
            }
            if (x >= mask.cols) {
                break;
            }
            const int x0 = x;
            while (x < mask.cols && row[x] != 0) {
                ++x;
            }
            const int x1 = x;

            const std::size_t sx0 = static_cast<std::size_t>(x0);
            const std::size_t sx1 = static_cast<std::size_t>(x1 - 1);
            if (sx1 >= nnMap.srcToDstX0.size()) {
                continue;
            }

            const std::size_t dstX0 = nnMap.srcToDstX0[sx0];
            const std::size_t dstX1 = nnMap.srcToDstX1[sx1];
            if (dstX1 <= dstX0) {
                continue;
            }

            const std::size_t cx0 = dstX0 / chunkX;
            const std::size_t cx1 = std::min((dstX1 - 1) / chunkX, nCX - 1);
            for (std::size_t cy = cy0; cy <= cy1; ++cy) {
                const std::size_t rowBase = cy * nCX;
                for (std::size_t cx = cx0; cx <= cx1; ++cx) {
                    const std::size_t idx = rowBase + cx;
                    if (seen[idx] != token) {
                        seen[idx] = token;
                        out.push_back({cy, cx});
                    }
                }
            }
        }
    }
    return out;
}

static std::vector<std::pair<std::size_t, std::size_t>> collectTouchedChunkXYMappedFast(
    const cv::Mat1b& mask,
    const NearestNeighborMap& nnMap,
    const Shape3& outShape,
    const Shape3& outChunk)
{
    const std::size_t chunkY = outChunk[1];
    const std::size_t chunkX = outChunk[2];
    const std::size_t nCY = (outShape[1] + chunkY - 1) / chunkY;
    const std::size_t nCX = (outShape[2] + chunkX - 1) / chunkX;
    if (mask.empty() || outShape[1] == 0 || outShape[2] == 0) {
        return {};
    }

    int minX = mask.cols;
    int maxX = -1;
    int minY = mask.rows;
    int maxY = -1;
    for (int y = 0; y < mask.rows; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        int rowMin = mask.cols;
        int rowMax = -1;
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x] == 0) {
                continue;
            }
            rowMin = std::min(rowMin, x);
            rowMax = x;
        }
        if (rowMax >= rowMin) {
            minY = std::min(minY, y);
            maxY = y;
            minX = std::min(minX, rowMin);
            maxX = std::max(maxX, rowMax);
        }
    }

    if (maxX < minX || maxY < minY) {
        return {};
    }

    if (minX < 0 || minY < 0 || maxX < minX || maxY < minY ||
        static_cast<std::size_t>(maxX) >= nnMap.srcToDstX0.size() ||
        static_cast<std::size_t>(maxY) >= nnMap.srcToDstY0.size()) {
        return {};
    }

    const std::size_t dstY0 = nnMap.srcToDstY0[static_cast<std::size_t>(minY)];
    const std::size_t dstY1 = nnMap.srcToDstY1[static_cast<std::size_t>(maxY)];
    const std::size_t dstX0 = nnMap.srcToDstX0[static_cast<std::size_t>(minX)];
    const std::size_t dstX1 = nnMap.srcToDstX1[static_cast<std::size_t>(maxX)];
    if (dstY1 <= dstY0 || dstX1 <= dstX0) {
        return {};
    }

    const std::size_t cy0 = dstY0 / chunkY;
    const std::size_t cy1 = (dstY1 - 1) / chunkY;
    const std::size_t cx0 = dstX0 / chunkX;
    const std::size_t cx1 = (dstX1 - 1) / chunkX;

    std::vector<std::pair<std::size_t, std::size_t>> out;
    out.reserve((cy1 - cy0 + 1) * (cx1 - cx0 + 1));
    for (std::size_t cy = cy0; cy <= cy1 && cy < nCY; ++cy) {
        for (std::size_t cx = cx0; cx <= cx1 && cx < nCX; ++cx) {
            out.emplace_back(cy, cx);
        }
    }
    return out;
}

struct SlabChunkState {
    bool loaded = false;
    bool dirty = false;
    std::size_t h = 0;
    std::size_t w = 0;
    std::vector<std::uint8_t> buf;
    std::vector<std::uint32_t> outToSrcY;
    std::vector<std::uint32_t> outToSrcX;
};

static std::uint64_t slabChunkKey(std::size_t cy, std::size_t cx)
{
    return (static_cast<std::uint64_t>(cy) << 32) | static_cast<std::uint64_t>(cx);
}

static CachePlan chooseCachePlan(const Config& cfg,
                                 std::size_t zCount,
                                 const Shape3& computeShape)
{
    CachePlan plan;
    const std::size_t sliceElems = computeShape[1] * computeShape[2];
    plan.requiredBytes = zCount * sliceElems;

    const auto available = memAvailableBytes();
    plan.availableBytes = available.value_or(0);

    if (cfg.ramBudgetGb > 0.0) {
        plan.budgetBytes = static_cast<std::size_t>(cfg.ramBudgetGb * 1024.0 * 1024.0 * 1024.0);
    } else if (available.has_value()) {
        plan.budgetBytes = static_cast<std::size_t>(
            static_cast<double>(*available) * kDefaultAutoRamFraction);
    } else {
        plan.budgetBytes = std::size_t{8} * 1024 * 1024 * 1024;
    }

    if (cfg.cacheMode == "stream") {
        plan.mode = ComputeCacheMode::stream;
    } else if (cfg.cacheMode == "preload") {
        if (plan.requiredBytes > plan.budgetBytes) {
            throw std::runtime_error(
                "requested --cache-mode preload but required bytes exceed RAM budget");
        }
        plan.mode = ComputeCacheMode::preload;
    } else if (cfg.cacheMode == "auto") {
        plan.mode = (plan.requiredBytes <= plan.budgetBytes)
            ? ComputeCacheMode::preload
            : ComputeCacheMode::stream;
    } else {
        throw std::runtime_error("--cache-mode must be one of: auto, stream, preload");
    }
    return plan;
}

static void copyRootMetadata(const fs::path& inputRoot,
                             const fs::path& outputRoot,
                             const std::unordered_set<int>& levelSet)
{
    for (const auto& entry : fs::directory_iterator(inputRoot)) {
        const auto name = entry.path().filename().string();
        if (entry.is_directory() && isNumericDirName(name)) {
            const int level = std::stoi(name);
            if (levelSet.find(level) != levelSet.end()) {
                continue;
            }
        }
        const fs::path dst = outputRoot / entry.path().filename();
        fs::copy(entry.path(),
                 dst,
                 fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    }
}

static void copyLevelFromExisting(const fs::path& inputLevelPath,
                                  const fs::path& outputLevelPath,
                                  const std::unordered_set<ChunkIndex, ChunkIndexHash>& existingChunks,
                                  std::size_t workers,
                                  std::atomic<std::size_t>* copiedChunks = nullptr,
                                  const StageProgressCallback& progressCallback = {})
{
    fs::create_directories(outputLevelPath);

    for (const auto& entry : fs::directory_iterator(inputLevelPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        ChunkIndex c;
        if (!parseChunkFilename(entry.path().filename().string(), c)) {
            // Copy level metadata files (e.g., .zarray, .zattrs) unconditionally.
            fs::copy_file(entry.path(),
                          outputLevelPath / entry.path().filename(),
                          fs::copy_options::overwrite_existing);
        }
    }

    std::vector<ChunkIndex> chunks;
    chunks.reserve(existingChunks.size());
    for (const auto& c : existingChunks) {
        chunks.push_back(c);
    }
    std::sort(chunks.begin(), chunks.end(), chunkIndexLess);

    const std::size_t nWorkers = std::max<std::size_t>(1, workers);
    std::atomic<std::size_t> next{0};
    std::atomic<bool> hadError{false};
    std::mutex errMutex;
    std::string firstError;
    std::vector<std::thread> threads;
    threads.reserve(nWorkers);

    for (std::size_t tid = 0; tid < nWorkers; ++tid) {
        threads.emplace_back([&, tid]() {
            (void)tid;
            while (true) {
                if (hadError.load()) {
                    break;
                }
                const std::size_t idx = next.fetch_add(1);
                if (idx >= chunks.size()) {
                    break;
                }
                try {
                    const fs::path src = inputLevelPath / chunkFilename(chunks[idx]);
                    const fs::path dst = outputLevelPath / src.filename();
                    fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
                    if (copiedChunks != nullptr) {
                        const std::size_t done = copiedChunks->fetch_add(1) + 1;
                        if (progressCallback) {
                            progressCallback(done);
                        }
                    }
                } catch (const std::exception& e) {
                    hadError.store(true);
                    std::lock_guard<std::mutex> lk(errMutex);
                    if (firstError.empty()) {
                        firstError = e.what();
                    }
                    break;
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (hadError.load()) {
        throw std::runtime_error("failed copying level chunks: " + firstError);
    }
}

static void copyInputTreeChunkwise(
    const fs::path& inputRoot,
    const fs::path& outputRoot,
    const std::vector<int>& levels,
    const std::unordered_map<int, std::unordered_set<ChunkIndex, ChunkIndexHash>>& existingByLevel,
    std::size_t workers,
    bool structuredProgress = false)
{
    std::unordered_set<int> levelSet(levels.begin(), levels.end());
    std::size_t totalChunkCopies = 0;
    for (int level : levels) {
        const auto it = existingByLevel.find(level);
        if (it == existingByLevel.end()) {
            throw std::runtime_error("missing existing chunk index for level " + std::to_string(level));
        }
        totalChunkCopies += it->second.size();
    }

    const std::size_t progressTotal = std::max<std::size_t>(1, totalChunkCopies);
    const std::size_t cadence = progressCadence(progressTotal);
    std::atomic<std::size_t> copiedChunks{0};
    if (structuredProgress) {
        emitStructuredStage(true, "copy", progressTotal);
    }

    copyRootMetadata(inputRoot, outputRoot, levelSet);

    for (int level : levels) {
        const auto it = existingByLevel.find(level);
        copyLevelFromExisting(inputRoot / std::to_string(level),
                              outputRoot / std::to_string(level),
                              it->second,
                              workers,
                              &copiedChunks,
                              [&](const std::size_t done) {
            if (structuredProgress &&
                shouldEmitStructuredProgress(done, progressTotal, cadence)) {
                emitStructuredProgress(true, "copy", done, progressTotal);
            }
        });
    }

    if (structuredProgress && totalChunkCopies == 0) {
        emitStructuredProgress(true, "copy", 1, progressTotal);
    }
}

static void validateReusableOutputTree(const fs::path& inputRoot,
                                     const fs::path& outputRoot,
                                     const std::vector<int>& levels)
{
    if (!fs::exists(outputRoot) || !fs::is_directory(outputRoot)) {
        throw std::runtime_error("--reuse-output-tree requires an existing output directory");
    }

    for (int level : levels) {
        const fs::path inputLevel = inputRoot / std::to_string(level);
        const fs::path outputLevel = outputRoot / std::to_string(level);
        if (!fs::exists(outputLevel / ".zarray")) {
            throw std::runtime_error("output level is missing .zarray: " + outputLevel.string());
        }

        vc::VcDataset inLevel(inputLevel);
        vc::VcDataset outLevel(outputLevel);
        if (inLevel.getDtype() != outLevel.getDtype()) {
            throw std::runtime_error("dtype mismatch for reused output level " +
                                     std::to_string(level));
        }
        if (toShape3(inLevel.shape()) != toShape3(outLevel.shape())) {
            throw std::runtime_error("shape mismatch for reused output level " +
                                     std::to_string(level));
        }
        if (toShape3(inLevel.defaultChunkShape()) != toShape3(outLevel.defaultChunkShape())) {
            throw std::runtime_error("chunk-shape mismatch for reused output level " +
                                     std::to_string(level));
        }
    }
}

static bool readAlignedSourceRegionByChunk(vc::VcDataset& src,
                                          const Shape3& srcShape,
                                          const Shape3& srcChunk,
                                          std::size_t srcZ0,
                                          std::size_t srcY0,
                                          std::size_t srcX0,
                                          std::size_t srcActualZ,
                                          std::size_t srcActualY,
                                          std::size_t srcActualX,
                                          std::vector<uint8_t>& dstBuf)
{
    if (srcActualZ == 0 || srcActualY == 0 || srcActualX == 0 || srcShape[0] == 0 ||
        srcShape[1] == 0 || srcShape[2] == 0 || srcChunk[0] == 0 || srcChunk[1] == 0 || srcChunk[2] == 0) {
        return false;
    }

    if (srcZ0 + srcActualZ > srcShape[0] || srcY0 + srcActualY > srcShape[1] ||
        srcX0 + srcActualX > srcShape[2]) {
        return false;
    }

    if ((srcZ0 % srcChunk[0] != 0) || (srcY0 % srcChunk[1] != 0) ||
        (srcX0 % srcChunk[2] != 0) || (srcActualZ % srcChunk[0] != 0) ||
        (srcActualY % srcChunk[1] != 0) || (srcActualX % srcChunk[2] != 0)) {
        return false;
    }

    const std::size_t zChunks = srcActualZ / srcChunk[0];
    const std::size_t yChunks = srcActualY / srcChunk[1];
    const std::size_t xChunks = srcActualX / srcChunk[2];
    if (zChunks == 0 || yChunks == 0 || xChunks == 0) {
        return false;
    }

    const std::size_t dstElems = srcActualZ * srcActualY * srcActualX;
    dstBuf.assign(dstElems, 0);

    std::vector<uint8_t> chunkBuf(srcChunk[0] * srcChunk[1] * srcChunk[2], 0);
    const std::size_t zChunkStart = srcZ0 / srcChunk[0];
    const std::size_t yChunkStart = srcY0 / srcChunk[1];
    const std::size_t xChunkStart = srcX0 / srcChunk[2];

    for (std::size_t zc = zChunkStart; zc < zChunkStart + zChunks; ++zc) {
                const std::size_t dstZ = (zc - zChunkStart) * srcChunk[0];
                for (std::size_t yc = yChunkStart; yc < yChunkStart + yChunks; ++yc) {
                    const std::size_t dstY = (yc - yChunkStart) * srcChunk[1];
                    for (std::size_t xc = xChunkStart; xc < xChunkStart + xChunks; ++xc) {
                        const std::size_t dstX = (xc - xChunkStart) * srcChunk[2];
                        try {
                            if (!src.readChunk(zc, yc, xc, chunkBuf.data())) {
                                std::fill(chunkBuf.begin(), chunkBuf.end(), uint8_t(0));
                            }
                        } catch (...) {
                            return false;
                        }
                        for (std::size_t z = 0; z < srcChunk[0]; ++z) {
                    const std::size_t dstRowBase = (dstZ + z) * srcActualY * srcActualX + dstY * srcActualX + dstX;
                    const std::size_t chunkRowBase = z * srcChunk[1] * srcChunk[2];
                    for (std::size_t y = 0; y < srcChunk[1]; ++y) {
                        const std::size_t dstBase = dstRowBase + y * srcActualX;
                        const std::size_t chunkBase = chunkRowBase + y * srcChunk[2];
                        std::copy_n(chunkBuf.data() + chunkBase, srcChunk[2], dstBuf.data() + dstBase);
                    }
                }
            }
        }
    }

    return true;
}

static std::vector<ChunkIndex> buildTouchedParents(const std::vector<ChunkIndex>& sourceTouched,
                                                   const Shape3& sourceShape,
                                                   const Shape3& sourceChunk,
                                                   const Shape3& targetShape,
                                                   const Shape3& targetChunk)
{
    if (sourceTouched.empty()) {
        return {};
    }

    const std::size_t targetChunksZ = (targetShape[0] + targetChunk[0] - 1) / targetChunk[0];
    const std::size_t targetChunksY = (targetShape[1] + targetChunk[1] - 1) / targetChunk[1];
    const std::size_t targetChunksX = (targetShape[2] + targetChunk[2] - 1) / targetChunk[2];

    std::unordered_set<ChunkIndex, ChunkIndexHash> touched;
    touched.reserve(sourceTouched.size());

    for (const auto& src : sourceTouched) {
        const std::size_t srcZ0 = src.z * sourceChunk[0];
        const std::size_t srcY0 = src.y * sourceChunk[1];
        const std::size_t srcX0 = src.x * sourceChunk[2];

        if (srcZ0 >= sourceShape[0] || srcY0 >= sourceShape[1] || srcX0 >= sourceShape[2]) {
            continue;
        }

        const std::size_t srcZ1 = std::min(sourceShape[0] - 1, srcZ0 + sourceChunk[0] - 1);
        const std::size_t srcY1 = std::min(sourceShape[1] - 1, srcY0 + sourceChunk[1] - 1);
        const std::size_t srcX1 = std::min(sourceShape[2] - 1, srcX0 + sourceChunk[2] - 1);

        const std::size_t tgtZMin = (srcZ0 / 2) / targetChunk[0];
        const std::size_t tgtZMax = (srcZ1 / 2) / targetChunk[0];
        const std::size_t tgtYMin = (srcY0 / 2) / targetChunk[1];
        const std::size_t tgtYMax = (srcY1 / 2) / targetChunk[1];
        const std::size_t tgtXMin = (srcX0 / 2) / targetChunk[2];
        const std::size_t tgtXMax = (srcX1 / 2) / targetChunk[2];

        for (std::size_t z = tgtZMin; z <= tgtZMax && z < targetChunksZ; ++z) {
            for (std::size_t y = tgtYMin; y <= tgtYMax && y < targetChunksY; ++y) {
                for (std::size_t x = tgtXMin; x <= tgtXMax && x < targetChunksX; ++x) {
                    touched.insert({z, y, x});
                }
            }
        }
    }

    std::vector<ChunkIndex> result;
    result.reserve(touched.size());
    for (const auto& c : touched) {
        result.push_back(c);
    }
    std::sort(result.begin(), result.end(), chunkIndexLess);
    return result;
}

static std::vector<ChunkIndex> buildLabelPriorityPyramidLevelTouched(
    const fs::path& outputRoot,
    int level,
    const std::vector<ChunkIndex>& sourceTouched,
    std::size_t workers,
    uint8_t ignoreValue,
    const std::unordered_set<ChunkIndex, ChunkIndexHash>* existingDstChunks,
    bool existingChunksOnly,
    std::atomic<std::size_t>& skippedMissingChunks,
    bool structuredProgress = false,
    ProfileStats* profile = nullptr)
{
    const fs::path srcPath = outputRoot / std::to_string(level - 1);
    const fs::path dstPath = outputRoot / std::to_string(level);

    vc::VcDataset src(srcPath);
    vc::VcDataset dst(dstPath);

    const Shape3 srcShape = toShape3(src.shape());
    const Shape3 srcChunk = toShape3(src.defaultChunkShape());
    const Shape3 dstShape = toShape3(dst.shape());
    const Shape3 dstChunk = toShape3(dst.defaultChunkShape());
    const std::size_t dstChunkElems = dstChunk[0] * dstChunk[1] * dstChunk[2];

    std::vector<ChunkIndex> touchedChunks = buildTouchedParents(sourceTouched,
                                                                srcShape,
                                                                srcChunk,
                                                                dstShape,
                                                                dstChunk);
    if (touchedChunks.empty()) {
        return {};
    }

    if (existingChunksOnly && existingDstChunks != nullptr) {
        std::vector<ChunkIndex> filtered;
        filtered.reserve(touchedChunks.size());
        for (const auto& c : touchedChunks) {
            if (existingDstChunks->find(c) != existingDstChunks->end()) {
                filtered.push_back(c);
            } else {
                skippedMissingChunks.fetch_add(1);
            }
        }
        touchedChunks.swap(filtered);
    }
    const std::size_t progressTotal = std::max<std::size_t>(1, touchedChunks.size());
    const std::size_t cadence = progressCadence(progressTotal);
    if (structuredProgress) {
        emitStructuredStage(true, "pyramid", progressTotal, level);
    }
    if (touchedChunks.empty()) {
        if (structuredProgress) {
            emitStructuredProgress(true, "pyramid", 1, progressTotal, level);
        }
        return {};
    }

    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::atomic<bool> hadError{false};
    std::vector<std::thread> threads;
    threads.reserve(workers);
    std::mutex outTouchedMutex;
    std::mutex profileMutex;
    std::mutex errMutex;
    std::string firstError;
    std::vector<ChunkIndex> outputTouched;
    outputTouched.reserve(touchedChunks.size());
    ProfileStats profileAccum;

    for (std::size_t tid = 0; tid < workers; ++tid) {
        threads.emplace_back([&, tid]() {
            (void)tid;
            vc::VcDataset srcLocal(srcPath);
            vc::VcDataset dstLocal(dstPath);
            std::vector<uint8_t> dstBuf(dstChunkElems, 0);
            std::vector<uint8_t> srcBuf;
            ProfileStats localProfile;

            while (true) {
                if (hadError.load()) {
                    break;
                }
                const std::size_t idx = next.fetch_add(1);
                if (idx >= touchedChunks.size()) {
                    break;
                }
                try {
                    const ChunkIndex chunk = touchedChunks[idx];
                    const std::size_t cz = chunk.z;
                    const std::size_t cy = chunk.y;
                    const std::size_t cx = chunk.x;

                    const std::size_t srcZ0 = cz * dstChunk[0] * 2;
                    const std::size_t srcY0 = cy * dstChunk[1] * 2;
                    const std::size_t srcX0 = cx * dstChunk[2] * 2;

                    if (srcZ0 >= srcShape[0] || srcY0 >= srcShape[1] || srcX0 >= srcShape[2]) {
                        const std::size_t progressDone = done.fetch_add(1) + 1;
                        if (structuredProgress &&
                            shouldEmitStructuredProgress(progressDone, progressTotal, cadence)) {
                            emitStructuredProgress(true, "pyramid", progressDone, progressTotal, level);
                        }
                        continue;
                    }

                    const std::size_t srcActualZ = std::min(srcShape[0] - srcZ0, dstChunk[0] * 2);
                    const std::size_t srcActualY = std::min(srcShape[1] - srcY0, dstChunk[1] * 2);
                    const std::size_t srcActualX = std::min(srcShape[2] - srcX0, dstChunk[2] * 2);

                    const std::size_t srcElems = srcActualZ * srcActualY * srcActualX;
                    auto tReadStart = std::chrono::steady_clock::now();
                    if (!readAlignedSourceRegionByChunk(srcLocal,
                                                       srcShape,
                                                       srcChunk,
                                                       srcZ0,
                                                       srcY0,
                                                       srcX0,
                                                       srcActualZ,
                                                       srcActualY,
                                                       srcActualX,
                                                       srcBuf)) {
                        srcBuf.assign(srcElems, 0);
                        srcLocal.readRegion({srcZ0, srcY0, srcX0},
                                            {srcActualZ, srcActualY, srcActualX},
                                            srcBuf.data());
                    }
                    auto tReadEnd = std::chrono::steady_clock::now();
                    localProfile.tPyramidRead.add(std::chrono::duration<double>(tReadEnd - tReadStart).count());
                    ++localProfile.pyramidReadCalls;
                    localProfile.bytesRead += srcElems * sizeof(uint8_t);

                    std::fill(dstBuf.begin(), dstBuf.end(), 0);
                    vc::core::util::downsampleLabelPriority(
                        srcBuf.data(),
                        vc::core::util::Shape3{srcActualZ, srcActualY, srcActualX},
                        dstBuf.data(),
                        vc::core::util::Shape3{dstChunk[0], dstChunk[1], dstChunk[2]},
                        ignoreValue);

                    if (hasAnyNonZero(dstBuf)) {
                        auto tWriteStart = std::chrono::steady_clock::now();
                        dstLocal.writeChunk(cz, cy, cx, dstBuf.data(), dstBuf.size() * sizeof(uint8_t));
                        auto tWriteEnd = std::chrono::steady_clock::now();
                        localProfile.tPyramidWrite.add(std::chrono::duration<double>(tWriteEnd - tWriteStart).count());
                        ++localProfile.pyramidWriteCalls;
                        localProfile.bytesWritten += dstBuf.size() * sizeof(uint8_t);
                        std::lock_guard<std::mutex> lk(outTouchedMutex);
                        outputTouched.push_back(chunk);
                    }
                    const std::size_t progressDone = done.fetch_add(1) + 1;
                    if (structuredProgress &&
                        shouldEmitStructuredProgress(progressDone, progressTotal, cadence)) {
                        emitStructuredProgress(true, "pyramid", progressDone, progressTotal, level);
                    }
                } catch (const std::exception& e) {
                    hadError.store(true);
                    std::lock_guard<std::mutex> lk(errMutex);
                    if (firstError.empty()) {
                        firstError = e.what();
                    }
                    break;
                }
            }

            if (profile) {
                std::lock_guard<std::mutex> lk(profileMutex);
                profileAccum.accumulate(localProfile);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    if (profile) {
        profile->accumulate(profileAccum);
    }
    if (hadError.load()) {
        throw std::runtime_error("failed building pyramid level " + std::to_string(level) +
                                 ": " + firstError);
    }
    std::sort(outputTouched.begin(), outputTouched.end(), chunkIndexLess);
    outputTouched.erase(std::unique(outputTouched.begin(), outputTouched.end()),
                        outputTouched.end());
    return outputTouched;
}

static bool hasForegroundInRelativeBox(const std::vector<uint8_t>& volume,
                                       const Shape3& volumeShape,
                                       const Shape3& relOrigin,
                                       const Shape3& boxShape)
{
    if (boxShape[0] == 0 || boxShape[1] == 0 || boxShape[2] == 0) {
        return false;
    }
    for (std::size_t z = 0; z < boxShape[0]; ++z) {
        for (std::size_t y = 0; y < boxShape[1]; ++y) {
            const std::size_t base = linearIndex(volumeShape,
                                                 relOrigin[0] + z,
                                                 relOrigin[1] + y,
                                                 relOrigin[2]);
            for (std::size_t x = 0; x < boxShape[2]; ++x) {
                if (volume[base + x] != 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

static std::size_t countNonZeroInRelativeBox(const std::vector<uint8_t>& volume,
                                             const Shape3& volumeShape,
                                             const Shape3& relOrigin,
                                             const Shape3& boxShape)
{
    std::size_t count = 0;
    for (std::size_t z = 0; z < boxShape[0]; ++z) {
        for (std::size_t y = 0; y < boxShape[1]; ++y) {
            const std::size_t base = linearIndex(volumeShape,
                                                 relOrigin[0] + z,
                                                 relOrigin[1] + y,
                                                 relOrigin[2]);
            count += countNonZero(volume.data() + base, boxShape[2]);
        }
    }
    return count;
}

static AlphaWrapPoint voxelCenterPoint(std::size_t z, std::size_t y, std::size_t x)
{
    return AlphaWrapPoint(static_cast<double>(z) + 0.5,
                          static_cast<double>(y) + 0.5,
                          static_cast<double>(x) + 0.5);
}

static bool bboxContains(const CGAL::Bbox_3& bbox, const AlphaWrapPoint& p)
{
    return p.x() >= bbox.xmin() && p.x() <= bbox.xmax() &&
           p.y() >= bbox.ymin() && p.y() <= bbox.ymax() &&
           p.z() >= bbox.zmin() && p.z() <= bbox.zmax();
}

static std::vector<uint8_t> classifyOuterAlphaWrapCore(const std::vector<uint8_t>& halo,
                                                       const Box3& haloBox,
                                                       const Box3& coreBox,
                                                       double alpha,
                                                       double offset)
{
    std::vector<AlphaWrapPoint> points;
    points.reserve(countNonZero(halo.data(), halo.size()));
    for (std::size_t z = 0; z < haloBox.shape[0]; ++z) {
        for (std::size_t y = 0; y < haloBox.shape[1]; ++y) {
            const std::size_t base = linearIndex(haloBox.shape, z, y, 0);
            for (std::size_t x = 0; x < haloBox.shape[2]; ++x) {
                if (halo[base + x] == 0) {
                    continue;
                }
                points.push_back(voxelCenterPoint(haloBox.origin[0] + z,
                                                  haloBox.origin[1] + y,
                                                  haloBox.origin[2] + x));
            }
        }
    }

    const Shape3 coreRel = relativeOrigin(coreBox, haloBox);
    std::vector<uint8_t> ignore(volumeElements(coreBox.shape), 0);
    if (points.size() < 4) {
        return ignore;
    }

    AlphaWrapMesh wrap;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    if (num_faces(wrap) == 0) {
        return ignore;
    }

    const CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(wrap);
    CGAL::Side_of_triangle_mesh<AlphaWrapMesh, AlphaWrapKernel> sideOfMesh(wrap);

    for (std::size_t z = 0; z < coreBox.shape[0]; ++z) {
        for (std::size_t y = 0; y < coreBox.shape[1]; ++y) {
            for (std::size_t x = 0; x < coreBox.shape[2]; ++x) {
                const std::size_t haloIdx = linearIndex(haloBox.shape,
                                                        coreRel[0] + z,
                                                        coreRel[1] + y,
                                                        coreRel[2] + x);
                if (halo[haloIdx] != 0) {
                    continue;
                }

                const AlphaWrapPoint p = voxelCenterPoint(coreBox.origin[0] + z,
                                                          coreBox.origin[1] + y,
                                                          coreBox.origin[2] + x);
                if (!bboxContains(bbox, p) || sideOfMesh(p) == CGAL::ON_UNBOUNDED_SIDE) {
                    ignore[linearIndex(coreBox.shape, z, y, x)] = 255;
                }
            }
        }
    }

    return ignore;
}

static void writeReplicatedCoreMaskToLevel0(
    vc::VcDataset& output,
    const Shape3& outShape,
    const Shape3& outChunk,
    const Box3& coreBox,
    int zStart,
    int zStop,
    std::size_t scale,
    const std::vector<uint8_t>& ignoreCore,
    const std::unordered_set<ChunkIndex, ChunkIndexHash>* existingLevel0Chunks,
    bool existingChunksOnly,
    uint8_t ignoreValue,
    std::unordered_set<ChunkIndex, ChunkIndexHash>& touchedLevel0,
    std::size_t& skippedMissingChunks,
    ProfileStats* profile = nullptr)
{
    const std::size_t clipLocalZ0 = coreBox.origin[0] < static_cast<std::size_t>(zStart)
        ? static_cast<std::size_t>(zStart) - coreBox.origin[0]
        : 0;
    const std::size_t clipLocalZ1 = coreBox.origin[0] + coreBox.shape[0] > static_cast<std::size_t>(zStop)
        ? static_cast<std::size_t>(zStop) - coreBox.origin[0]
        : coreBox.shape[0];
    if (clipLocalZ0 >= clipLocalZ1) {
        return;
    }

    const std::size_t outBaseZ = coreBox.origin[0] * scale;
    const std::size_t outBaseY = coreBox.origin[1] * scale;
    const std::size_t outBaseX = coreBox.origin[2] * scale;
    const std::size_t regionZ0 = outBaseZ + clipLocalZ0 * scale;
    const std::size_t regionZ1 = outBaseZ + clipLocalZ1 * scale;
    const std::size_t regionY0 = outBaseY;
    const std::size_t regionY1 = outBaseY + coreBox.shape[1] * scale;
    const std::size_t regionX0 = outBaseX;
    const std::size_t regionX1 = outBaseX + coreBox.shape[2] * scale;

    if (regionZ0 >= regionZ1 || regionY0 >= regionY1 || regionX0 >= regionX1) {
        return;
    }

    const std::size_t chunkElems = volumeElements(outChunk);
    const std::size_t chunkBytes = chunkElems * sizeof(uint8_t);
    const std::size_t cz0 = regionZ0 / outChunk[0];
    const std::size_t cz1 = (regionZ1 - 1) / outChunk[0];
    const std::size_t cy0 = regionY0 / outChunk[1];
    const std::size_t cy1 = (regionY1 - 1) / outChunk[1];
    const std::size_t cx0 = regionX0 / outChunk[2];
    const std::size_t cx1 = (regionX1 - 1) / outChunk[2];

    std::vector<uint8_t> chunkBuf(chunkElems, 0);
    for (std::size_t cz = cz0; cz <= cz1; ++cz) {
        for (std::size_t cy = cy0; cy <= cy1; ++cy) {
            for (std::size_t cx = cx0; cx <= cx1; ++cx) {
                const ChunkIndex chunk{cz, cy, cx};
                if (existingChunksOnly &&
                    existingLevel0Chunks != nullptr &&
                    existingLevel0Chunks->find(chunk) == existingLevel0Chunks->end()) {
                    ++skippedMissingChunks;
                    continue;
                }

                const std::size_t chunkGlobalZ0 = cz * outChunk[0];
                const std::size_t chunkGlobalY0 = cy * outChunk[1];
                const std::size_t chunkGlobalX0 = cx * outChunk[2];
                const std::size_t overlapZ0 = std::max(chunkGlobalZ0, regionZ0);
                const std::size_t overlapZ1 = std::min(chunkGlobalZ0 + outChunk[0], regionZ1);
                const std::size_t overlapY0 = std::max(chunkGlobalY0, regionY0);
                const std::size_t overlapY1 = std::min(chunkGlobalY0 + outChunk[1], regionY1);
                const std::size_t overlapX0 = std::max(chunkGlobalX0, regionX0);
                const std::size_t overlapX1 = std::min(chunkGlobalX0 + outChunk[2], regionX1);
                if (overlapZ0 >= overlapZ1 || overlapY0 >= overlapY1 || overlapX0 >= overlapX1) {
                    continue;
                }

                if (profile) {
                    ++profile->totalChunksInMasks;
                }

                auto readStart = std::chrono::steady_clock::now();
                chunkBuf.assign(chunkElems, 0);
                output.readChunk(cz, cy, cx, chunkBuf.data());
                if (profile) {
                    auto readEnd = std::chrono::steady_clock::now();
                    profile->tChunkReadIo.add(std::chrono::duration<double>(readEnd - readStart).count());
                    ++profile->chunkIoReads;
                    profile->bytesRead += chunkBytes;
                }

                bool changed = false;
                auto cpuStart = std::chrono::steady_clock::now();
                for (std::size_t outZ = overlapZ0; outZ < overlapZ1; ++outZ) {
                    const std::size_t srcLocalZ = (outZ - outBaseZ) / scale;
                    const std::size_t dstLocalZ = outZ - chunkGlobalZ0;
                    for (std::size_t outY = overlapY0; outY < overlapY1; ++outY) {
                        const std::size_t srcLocalY = (outY - outBaseY) / scale;
                        const std::size_t dstLocalY = outY - chunkGlobalY0;
                        for (std::size_t outX = overlapX0; outX < overlapX1; ++outX) {
                            const std::size_t srcLocalX = (outX - outBaseX) / scale;
                            if (ignoreCore[linearIndex(coreBox.shape,
                                                       srcLocalZ,
                                                       srcLocalY,
                                                       srcLocalX)] == 0) {
                                continue;
                            }
                            const std::size_t dstIdx = linearIndex(outChunk,
                                                                   dstLocalZ,
                                                                   dstLocalY,
                                                                   outX - chunkGlobalX0);
                            if (chunkBuf[dstIdx] == 0) {
                                chunkBuf[dstIdx] = ignoreValue;
                                changed = true;
                            }
                        }
                    }
                }
                if (profile) {
                    auto cpuEnd = std::chrono::steady_clock::now();
                    profile->tChunkApplyCpu.add(std::chrono::duration<double>(cpuEnd - cpuStart).count());
                }

                if (!changed) {
                    continue;
                }

                auto writeStart = std::chrono::steady_clock::now();
                output.writeChunk(cz, cy, cx, chunkBuf.data(), chunkBytes);
                if (profile) {
                    auto writeEnd = std::chrono::steady_clock::now();
                    profile->tChunkWriteIo.add(std::chrono::duration<double>(writeEnd - writeStart).count());
                    ++profile->chunkIoWrites;
                    profile->bytesWritten += chunkBytes;
                }
                touchedLevel0.insert(chunk);
            }
        }
    }
}

static std::vector<ChunkIndex> collectActiveComputeChunks(
    const std::unordered_set<ChunkIndex, ChunkIndexHash>& existingComputeChunks,
    const Shape3& computeShape,
    const Shape3& computeChunk,
    int zStart,
    int zStop)
{
    std::vector<ChunkIndex> out;
    out.reserve(existingComputeChunks.size());
    for (const auto& chunk : existingComputeChunks) {
        const Box3 box = makeChunkBox(chunk, computeChunk, computeShape);
        if (box.shape[0] == 0 || box.shape[1] == 0 || box.shape[2] == 0) {
            continue;
        }
        if (!boxIntersectsZRange(box, zStart, zStop)) {
            continue;
        }
        out.push_back(chunk);
    }
    std::sort(out.begin(), out.end(), chunkIndexLess);
    return out;
}

static bool valueIsRasterized(const nlohmann::json& meta)
{
    if (!meta.contains("label_volume")) {
        return false;
    }
    if (!meta["label_volume"].is_string()) {
        return false;
    }
    return meta["label_volume"].get<std::string>() == "rasterized";
}

static void validateInputMetadata(const fs::path& inputRoot)
{
    const fs::path metaPath = inputRoot / "meta.json";
    if (!fs::exists(metaPath)) {
        throw std::runtime_error("input volume is missing meta.json");
    }
    const auto meta = readJsonFile(metaPath);
    if (!valueIsRasterized(meta)) {
        throw std::runtime_error(
            "input volume is not rasterized (meta.json must contain \"label_volume\": \"rasterized\")");
    }
}

static void writeOutputMetadata(const Config& cfg,
                                const Shape3& shape0,
                                const nlohmann::json& inputMeta,
                                const ProcessingStats& stats)
{
    nlohmann::json meta = inputMeta;

    const std::string uuid = cfg.outputRoot.filename().string();
    meta["type"] = "vol";
    meta["uuid"] = uuid;
    meta["name"] = uuid;
    meta["width"] = static_cast<long long>(shape0[2]);
    meta["height"] = static_cast<long long>(shape0[1]);
    meta["slices"] = static_cast<long long>(shape0[0]);
    meta["format"] = "zarr";
    meta["min"] = 0.0;
    meta["max"] = 255.0;
    meta["label_volume"] = "rasterized";
    meta["ignore_label_value"] = cfg.ignoreValue;
    meta["ignore_label_added_at"] = nowIsoUtc();
    meta["ignore_label_tool"] = "vc_add_ignore_label";
    meta["source_zarr"] = fs::weakly_canonical(cfg.inputRoot).string();
    meta["ignore_label_params"] = {
        {"mode", processingModeToString(cfg.mode)},
        {"alpha", cfg.alpha},
        {"chunk_alpha", cfg.chunkAlpha},
        {"n_angle_bins", cfg.nAngleBins},
        {"shrink_factor", cfg.shrinkFactor},
        {"compute_level", cfg.computeLevel},
        {"output_level", cfg.outputLevel},
        {"existing_chunks_only", cfg.existingChunksOnly},
        {"algo_mode", algoModeToString(cfg.algoMode)},
        {"map_mode", mapModeToString(cfg.mapMode)},
        {"fast_atan2", cfg.fastAtan2},
        {"use_squared_dist", cfg.useSquaredDist},
        {"chunk_z_slab", stats.chunkZSlab},
        {"rebuild_pyramid", stats.rebuildPyramid},
        {"reuse_output_tree", stats.reuseOutputTree},
        {"result_mode", stats.resultMode},
        {"cache_mode", stats.cacheMode},
        {"cache_budget_bytes", static_cast<long long>(stats.cacheBudgetBytes)},
        {"available_ram_bytes", static_cast<long long>(stats.availableRamBytes)},
        {"preloaded_compute_bytes", static_cast<long long>(stats.preloadedComputeBytes)},
        {"preloaded_compute_slices", static_cast<long long>(stats.preloadedComputeSlices)},
        {"skipped_missing_chunks_level0", static_cast<long long>(stats.skippedMissingLevel0)},
        {"skipped_missing_chunks_pyramid", static_cast<long long>(stats.skippedMissingPyramid)},
        {"skip_outer", cfg.skipOuter},
        {"skip_inner", cfg.skipInner},
        {"z_min", cfg.zMin},
        {"z_max", cfg.zMax}
    };

    writeJsonFile(cfg.outputRoot / "meta.json", meta);
}

static void writeProfileJson(const fs::path& path,
                            const Config& cfg,
                            bool reuseOutputTree,
                            const ProfileStats& profile,
                            const std::size_t totalWorkItems)
{
    nlohmann::json j;
    j["input"] = cfg.inputRoot.string();
    j["output"] = cfg.outputRoot.string();
    j["workers"] = static_cast<long long>(cfg.workers);
    j["mode"] = processingModeToString(cfg.mode);
    j["compute_level"] = cfg.computeLevel;
    j["output_level"] = cfg.outputLevel;
    j["chunk_alpha"] = cfg.chunkAlpha;
    j["map_mode"] = mapModeToString(cfg.mapMode);
    j["fast_atan2"] = cfg.fastAtan2;
    j["use_squared_dist"] = cfg.useSquaredDist;
    j["chunk_z_slab"] = cfg.chunkZSlab;
    j["rebuild_pyramid"] = cfg.rebuildPyramid;
    j["reuse_output_tree"] = reuseOutputTree;
    j["result_mode"] = resultModeToString(cfg.resultMode);
    j["cache_mode"] = cfg.cacheMode;
    j["algo_mode"] = algoModeToString(cfg.algoMode);
    j["z_min"] = cfg.zMin;
    j["z_max"] = cfg.zMax;
    if (cfg.mode == ProcessingMode::chunkAlphaWrap) {
        j["chunk_total"] = static_cast<long long>(totalWorkItems);
    } else {
        j["slice_total"] = static_cast<long long>(totalWorkItems);
    }
    j["slices_total"] = static_cast<long long>(profile.slicesTotal);
    j["slices_skipped_empty"] = static_cast<long long>(profile.slicesSkippedEmpty);
    j["slices_nonzero_fg"] = static_cast<long long>(profile.nonzeroFgSlices);
    j["compute_chunks_total"] = static_cast<long long>(profile.computeChunksTotal);
    j["compute_chunks_skipped_empty"] = static_cast<long long>(profile.computeChunksSkippedEmpty);
    j["compute_chunks_wrapped"] = static_cast<long long>(profile.computeChunksWrapped);
    j["fg_pixels"] = static_cast<long long>(profile.fgPixels);
    j["mask_pixels"] = static_cast<long long>(profile.maskPixels);
    j["touched_chunks_level0"] = static_cast<long long>(profile.touchedChunksLevel0);
    j["chunk_cache_hits"] = static_cast<long long>(profile.chunkCacheHits);
    j["chunk_cache_misses"] = static_cast<long long>(profile.chunkCacheMisses);
    j["chunk_io_reads"] = static_cast<long long>(profile.chunkIoReads);
    j["chunk_io_writes"] = static_cast<long long>(profile.chunkIoWrites);
    j["bytes_read"] = static_cast<long long>(profile.bytesRead);
    j["bytes_written"] = static_cast<long long>(profile.bytesWritten);
    j["pyramid_read_calls"] = static_cast<long long>(profile.pyramidReadCalls);
    j["pyramid_write_calls"] = static_cast<long long>(profile.pyramidWriteCalls);
    j["total_chunks_in_masks"] = static_cast<long long>(profile.totalChunksInMasks);
    j["skipped_missing_level0"] = static_cast<long long>(profile.skippedMissingLevel0);
    j["skipped_missing_pyramid"] = static_cast<long long>(profile.skippedMissingPyramid);
    j["input_voxels"] = static_cast<long long>(profile.inputVoxels);
    j["timing_sec"] = {
        {"slice_load", profile.tSliceLoad.seconds()},
        {"mask_build", profile.tMaskBuild.seconds()},
        {"mask_merge", profile.tMaskMerge.seconds()},
        {"resize_legacy", profile.tResizeLegacy.seconds()},
        {"touch_discovery", profile.tTouchDiscovery.seconds()},
        {"chunk_apply_cpu", profile.tChunkApplyCpu.seconds()},
        {"chunk_read_io", profile.tChunkReadIo.seconds()},
        {"chunk_write_io", profile.tChunkWriteIo.seconds()},
        {"pyramid_read", profile.tPyramidRead.seconds()},
        {"pyramid_write", profile.tPyramidWrite.seconds()},
        {"total", profile.totalSeconds()},
    };

    if (totalWorkItems > 0) {
        const double totalSec = std::max(1e-9, profile.totalSeconds());
        j["throughput"] = {
            {"input_voxels_per_sec", static_cast<double>(profile.inputVoxels) / totalSec},
            {"chunks_updated_per_sec",
             static_cast<double>(profile.chunkIoWrites) / totalSec},
            {"bytes_read_per_sec", static_cast<double>(profile.bytesRead) / totalSec},
            {"bytes_written_per_sec", static_cast<double>(profile.bytesWritten) / totalSec},
        };
    }
    writeJsonFile(path, j);
}

static int processChunkAlphaWrap(
    const Config& cfg,
    const std::vector<int>& levels,
    const nlohmann::json& inputMeta,
    const std::unordered_map<int, std::unordered_set<ChunkIndex, ChunkIndexHash>>& existingByLevel,
    bool reuseOutputTree,
    double copyStageSeconds,
    std::size_t workers,
    const Shape3& computeShape,
    const Shape3& outShape,
    const Shape3& outChunk,
    int zStart,
    int zStop)
{
    vc::VcDataset computeDs(cfg.inputRoot / std::to_string(cfg.computeLevel));
    vc::VcDataset out0(cfg.outputRoot / "0");
    const Shape3 computeChunk = toShape3(computeDs.defaultChunkShape());
    if (computeChunk != outChunk) {
        throw std::runtime_error("chunk-alpha-wrap currently requires matching compute/output chunk shapes");
    }

    const std::size_t scale =
        std::size_t{1} << static_cast<std::size_t>(cfg.computeLevel - cfg.outputLevel);
    if ((outChunk[0] % scale) != 0 || (outChunk[1] % scale) != 0 || (outChunk[2] % scale) != 0) {
        throw std::runtime_error("chunk-alpha-wrap requires output chunk sizes divisible by scale");
    }

    const auto itLevel0 = existingByLevel.find(0);
    if (itLevel0 == existingByLevel.end()) {
        throw std::runtime_error("missing existing chunk index for level 0");
    }
    const auto itComputeLevel = existingByLevel.find(cfg.computeLevel);
    if (itComputeLevel == existingByLevel.end()) {
        throw std::runtime_error("missing existing chunk index for compute level");
    }
    const auto& existingLevel0Chunks = itLevel0->second;
    const auto activeComputeChunks = collectActiveComputeChunks(itComputeLevel->second,
                                                               computeShape,
                                                               computeChunk,
                                                               zStart,
                                                               zStop);

    const std::size_t total = activeComputeChunks.size();
    const std::size_t offsetVoxels =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(cfg.chunkAlpha / 4.0)));
    const std::size_t haloVoxels =
        static_cast<std::size_t>(std::ceil(cfg.chunkAlpha + static_cast<double>(offsetVoxels)));
    const std::size_t wrapProgressTotal = std::max<std::size_t>(1, total);
    const std::size_t wrapCadence = progressCadence(wrapProgressTotal);

    if (cfg.verbose) {
        std::cerr << "[config] mode=" << processingModeToString(cfg.mode)
                  << " compute_level=" << cfg.computeLevel
                  << " output_level=" << cfg.outputLevel
                  << " chunk_alpha=" << cfg.chunkAlpha
                  << " workers=" << workers
                  << " output_tree=" << (reuseOutputTree ? "reuse" : "copy")
                  << " z_range=[" << zStart << "," << zStop << ")\n";
        std::cerr << "[chunk-alpha-wrap] chunks=" << total
                  << " alpha=" << cfg.chunkAlpha
                  << " offset=" << offsetVoxels
                  << " halo=" << haloVoxels
                  << " scale=" << scale << '\n';
        emitStructuredStage(true, "wrap", wrapProgressTotal);
        if (total == 0) {
            emitStructuredProgress(true, "wrap", 1, wrapProgressTotal);
        }
    }

    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::atomic<std::size_t> skippedMissingLevel0{0};
    std::atomic<std::size_t> skippedMissingPyramid{0};
    std::atomic<bool> hadError{false};
    std::mutex ioMutex;
    std::mutex touchedMutex;
    std::mutex profileMutex;
    std::mutex errMutex;
    std::string firstError;
    std::unordered_set<ChunkIndex, ChunkIndexHash> touchedLevel0Global;
    ProfileStats profile;
    std::vector<std::thread> threads;
    threads.reserve(workers);

    const auto progress = [&](std::size_t d) {
        if (cfg.verbose && shouldEmitStructuredProgress(d, wrapProgressTotal, wrapCadence)) {
            emitStructuredProgress(true, "wrap", d, wrapProgressTotal);
        }
        if (cfg.verbose && (d == 1 || (d % 16) == 0 || d == total)) {
            std::lock_guard<std::mutex> lock(ioMutex);
            std::cerr << "[ignore] " << d << "/" << total << " compute chunks\n";
        }
    };

    const auto wrapStageStart = std::chrono::steady_clock::now();

    for (std::size_t tid = 0; tid < workers; ++tid) {
        threads.emplace_back([&, tid]() {
            (void)tid;
            vc::VcDataset localCompute(cfg.inputRoot / std::to_string(cfg.computeLevel));
            vc::VcDataset localOut0(cfg.outputRoot / "0");
            std::vector<uint8_t> haloBuf;
            std::unordered_set<ChunkIndex, ChunkIndexHash> localTouchedLevel0;
            localTouchedLevel0.reserve(128);
            std::size_t localSkippedLevel0 = 0;
            ProfileStats localProfile;

            while (true) {
                if (hadError.load()) {
                    break;
                }
                const std::size_t idx = next.fetch_add(1);
                if (idx >= total) {
                    break;
                }

                try {
                    const ChunkIndex chunk = activeComputeChunks[idx];
                    const Box3 coreBox = makeChunkBox(chunk, computeChunk, computeShape);
                    const std::size_t clipLocalZ0 = coreBox.origin[0] < static_cast<std::size_t>(zStart)
                        ? static_cast<std::size_t>(zStart) - coreBox.origin[0]
                        : 0;
                    const std::size_t clipLocalZ1 =
                        coreBox.origin[0] + coreBox.shape[0] > static_cast<std::size_t>(zStop)
                            ? static_cast<std::size_t>(zStop) - coreBox.origin[0]
                            : coreBox.shape[0];
                    const Shape3 clippedShape = {
                        clipLocalZ1 > clipLocalZ0 ? clipLocalZ1 - clipLocalZ0 : 0,
                        coreBox.shape[1],
                        coreBox.shape[2],
                    };

                    ++localProfile.computeChunksTotal;
                    localProfile.inputVoxels += volumeElements(clippedShape);
                    if (clippedShape[0] == 0 || clippedShape[1] == 0 || clippedShape[2] == 0) {
                        const std::size_t d = done.fetch_add(1) + 1;
                        progress(d);
                        continue;
                    }

                    const Box3 haloBox = expandAndClampBox(coreBox, haloVoxels, computeShape);
                    haloBuf.assign(volumeElements(haloBox.shape), 0);
                    {
                        ScopedTimer readTimer(localProfile.tSliceLoad);
                        localCompute.readRegion({haloBox.origin[0], haloBox.origin[1], haloBox.origin[2]},
                                                {haloBox.shape[0], haloBox.shape[1], haloBox.shape[2]},
                                                haloBuf.data());
                    }
                    localProfile.bytesRead += haloBuf.size() * sizeof(uint8_t);

                    const Shape3 coreRel = relativeOrigin(coreBox, haloBox);
                    const Shape3 clippedRel = {coreRel[0] + clipLocalZ0, coreRel[1], coreRel[2]};
                    if (!hasForegroundInRelativeBox(haloBuf, haloBox.shape, clippedRel, clippedShape)) {
                        ++localProfile.computeChunksSkippedEmpty;
                        const std::size_t d = done.fetch_add(1) + 1;
                        progress(d);
                        continue;
                    }

                    if (cfg.profileEnabled) {
                        localProfile.fgPixels += countNonZeroInRelativeBox(haloBuf,
                                                                           haloBox.shape,
                                                                           clippedRel,
                                                                           clippedShape);
                    }

                    std::vector<uint8_t> ignoreCore;
                    {
                        ScopedTimer wrapTimer(localProfile.tMaskBuild);
                        ignoreCore = classifyOuterAlphaWrapCore(haloBuf,
                                                                haloBox,
                                                                coreBox,
                                                                cfg.chunkAlpha,
                                                                static_cast<double>(offsetVoxels));
                    }
                    ++localProfile.computeChunksWrapped;

                    if (cfg.profileEnabled) {
                        localProfile.maskPixels += countNonZeroInRelativeBox(ignoreCore,
                                                                             coreBox.shape,
                                                                             {clipLocalZ0, 0, 0},
                                                                             clippedShape);
                    }

                    writeReplicatedCoreMaskToLevel0(localOut0,
                                                    outShape,
                                                    outChunk,
                                                    coreBox,
                                                    zStart,
                                                    zStop,
                                                    scale,
                                                    ignoreCore,
                                                    &existingLevel0Chunks,
                                                    cfg.existingChunksOnly,
                                                    static_cast<uint8_t>(cfg.ignoreValue),
                                                    localTouchedLevel0,
                                                    localSkippedLevel0,
                                                    &localProfile);

                    const std::size_t d = done.fetch_add(1) + 1;
                    progress(d);
                } catch (const std::exception& e) {
                    hadError.store(true);
                    std::lock_guard<std::mutex> lk(errMutex);
                    if (firstError.empty()) {
                        firstError = e.what();
                    }
                    break;
                }
            }

            skippedMissingLevel0.fetch_add(localSkippedLevel0);
            {
                std::lock_guard<std::mutex> lock(touchedMutex);
                touchedLevel0Global.insert(localTouchedLevel0.begin(), localTouchedLevel0.end());
            }
            {
                std::lock_guard<std::mutex> lock(profileMutex);
                profile.accumulate(localProfile);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    if (hadError.load()) {
        throw std::runtime_error("failed while processing compute chunks: " + firstError);
    }
    const double wrapStageSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wrapStageStart).count();

    std::vector<ChunkIndex> activeTouched;
    activeTouched.reserve(touchedLevel0Global.size());
    for (const auto& c : touchedLevel0Global) {
        activeTouched.push_back(c);
    }
    std::sort(activeTouched.begin(), activeTouched.end(), chunkIndexLess);
    profile.touchedChunksLevel0 = activeTouched.size();

    double pyramidStageSeconds = 0.0;
    if (cfg.rebuildPyramid) {
        for (int level : levels) {
            if (level == 0) {
                continue;
            }
            if (activeTouched.empty()) {
                break;
            }
            if (cfg.verbose) {
                std::cerr << "[pyramid] building level " << level
                          << " touched=" << activeTouched.size() << '\n';
            }
            const std::unordered_set<ChunkIndex, ChunkIndexHash>* existingDst = nullptr;
            if (cfg.existingChunksOnly) {
                auto it = existingByLevel.find(level);
                if (it == existingByLevel.end()) {
                    throw std::runtime_error("missing existing chunk index for pyramid level " +
                                             std::to_string(level));
                }
                existingDst = &it->second;
            }
            const auto pyramidLevelStart = std::chrono::steady_clock::now();
            activeTouched = buildLabelPriorityPyramidLevelTouched(cfg.outputRoot,
                                                                  level,
                                                                  activeTouched,
                                                                  workers,
                                                                  static_cast<uint8_t>(cfg.ignoreValue),
                                                                  existingDst,
                                                                  cfg.existingChunksOnly,
                                                                  skippedMissingPyramid,
                                                                  cfg.verbose,
                                                                  &profile);
            pyramidStageSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pyramidLevelStart).count();
        }
    } else if (cfg.verbose) {
        std::cerr << "[pyramid] skipped (no-rebuild-pyramid)" << '\n';
    }

    ProcessingStats stats;
    stats.skippedMissingLevel0 = skippedMissingLevel0.load();
    stats.skippedMissingPyramid = skippedMissingPyramid.load();
    stats.preloadedComputeBytes = 0;
    stats.preloadedComputeSlices = 0;
    stats.cacheBudgetBytes = 0;
    stats.availableRamBytes = 0;
    stats.cacheMode = "chunk-region";
    stats.mapMode = "chunk-alpha-wrap";
    stats.fastAtan2 = false;
    stats.useSquaredDist = false;
    stats.chunkZSlab = false;
    stats.rebuildPyramid = cfg.rebuildPyramid;
    stats.reuseOutputTree = reuseOutputTree;
    stats.resultMode = resultModeToString(cfg.resultMode);

    writeOutputMetadata(cfg, outShape, inputMeta, stats);
    if (cfg.profileEnabled && !cfg.profileJsonPath.empty()) {
        writeProfileJson(cfg.profileJsonPath, cfg, reuseOutputTree, profile, total);
    }
    if (cfg.verbose) {
        const double totalStageSeconds = copyStageSeconds + wrapStageSeconds + pyramidStageSeconds;
        std::cerr << std::fixed << std::setprecision(3)
                  << "[summary] mode=" << processingModeToString(cfg.mode)
                  << " copy_seconds=" << copyStageSeconds
                  << " wrap_seconds=" << wrapStageSeconds
                  << " pyramid_seconds=" << pyramidStageSeconds
                  << " total_seconds=" << totalStageSeconds
                  << " wrapped_chunks=" << profile.computeChunksWrapped
                  << " touched_level0=" << profile.touchedChunksLevel0
                  << '\n';
        std::cerr << "VC_SUMMARY mode=" << processingModeToString(cfg.mode)
                  << " reuse_output_tree=" << (reuseOutputTree ? 1 : 0)
                  << " copy_seconds=" << copyStageSeconds
                  << " wrap_seconds=" << wrapStageSeconds
                  << " pyramid_seconds=" << pyramidStageSeconds
                  << " total_seconds=" << totalStageSeconds
                  << " compute_chunks=" << total
                  << " wrapped_chunks=" << profile.computeChunksWrapped
                  << " touched_level0=" << profile.touchedChunksLevel0
                  << '\n';
        std::cerr.unsetf(std::ios::floatfield);
    }
    std::cout << "vc_add_ignore_label completed\n";
    return EXIT_SUCCESS;
}

static void visualizeSlice(const Config& cfg)
{
    if (!cfg.visualizeSlice.has_value()) {
        return;
    }

    vc::VcDataset computeDs(cfg.inputRoot / std::to_string(cfg.computeLevel));
    const Shape3 computeShape = toShape3(computeDs.shape());

    const int z = *cfg.visualizeSlice;
    if (z < 0 || z >= static_cast<int>(computeShape[0])) {
        throw std::runtime_error("visualize slice out of range for compute level");
    }

    std::vector<uint8_t> slice(computeShape[1] * computeShape[2], 0);
    computeDs.readRegion({static_cast<std::size_t>(z), 0, 0},
                         {1, computeShape[1], computeShape[2]},
                         slice.data());
    cv::Mat1b label(static_cast<int>(computeShape[1]),
                    static_cast<int>(computeShape[2]),
                    slice.data());

    fs::create_directories(cfg.visualizeDir);

    const fs::path outputLevel = cfg.outputRoot / std::to_string(cfg.computeLevel);
    const bool hasOutputLevel = !cfg.outputRoot.empty() && fs::exists(outputLevel / ".zarray");
    if (hasOutputLevel) {
        vc::VcDataset outputDs(outputLevel);
        if (toShape3(outputDs.shape()) != computeShape) {
            throw std::runtime_error("visualize output level shape mismatch");
        }

        std::vector<uint8_t> outSlice(computeShape[1] * computeShape[2], 0);
        outputDs.readRegion({static_cast<std::size_t>(z), 0, 0},
                            {1, computeShape[1], computeShape[2]},
                            outSlice.data());
        cv::Mat1b outputIgnore(static_cast<int>(computeShape[1]),
                               static_cast<int>(computeShape[2]),
                               outSlice.data());
        cv::Mat1b ignoreMask(outputIgnore.rows, outputIgnore.cols, uint8_t(0));
        for (int y = 0; y < outputIgnore.rows; ++y) {
            const uint8_t* outRow = outputIgnore.ptr<uint8_t>(y);
            uint8_t* maskRow = ignoreMask.ptr<uint8_t>(y);
            for (int x = 0; x < outputIgnore.cols; ++x) {
                if (outRow[x] == static_cast<uint8_t>(cfg.ignoreValue)) {
                    maskRow[x] = 255;
                }
            }
        }

        cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_input.png")).string(), label);
        cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_ignore_mask.png")).string(),
                    ignoreMask);
        cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_output_ignore.png")).string(),
                    outputIgnore);

        cv::Mat3b vis(label.rows, label.cols, cv::Vec3b(0, 0, 0));
        for (int y = 0; y < label.rows; ++y) {
            const uint8_t* lrow = label.ptr<uint8_t>(y);
            const uint8_t* maskRow = ignoreMask.ptr<uint8_t>(y);
            cv::Vec3b* vrow = vis.ptr<cv::Vec3b>(y);
            for (int x = 0; x < label.cols; ++x) {
                if (lrow[x] != 0) {
                    vrow[x] = cv::Vec3b(255, 255, 255);
                }
                if (maskRow[x] != 0) {
                    vrow[x] = cv::Vec3b(255, 0, 0);
                }
            }
        }
        cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_combined.png")).string(), vis);
        return;
    }

    if (cfg.mode == ProcessingMode::chunkAlphaWrap) {
        throw std::runtime_error(
            "chunk-alpha-wrap visualization without an existing output zarr is not supported");
    }

    const SliceMaskPair masks = detectIgnoreMasks(label,
                                                  cfg.nAngleBins,
                                                  cfg.alpha,
                                                  cfg.outerSmoothingWindow,
                                                  cfg.shrinkFactor,
                                                  cfg.innerSmoothingWindow,
                                                  !cfg.skipOuter,
                                                  !cfg.skipInner,
                                                  cfg.fastAtan2,
                                                  cfg.useSquaredDist,
                                                  cfg.algoMode);
    const cv::Mat1b& outer = masks.outer;
    const cv::Mat1b& inner = masks.inner;

    cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_original.png")).string(), label);
    cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_outer.png")).string(), outer);
    cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_inner.png")).string(), inner);

    cv::Mat3b vis(label.rows, label.cols, cv::Vec3b(0, 0, 0));
    for (int y = 0; y < label.rows; ++y) {
        const uint8_t* lrow = label.ptr<uint8_t>(y);
        const uint8_t* orow = outer.ptr<uint8_t>(y);
        const uint8_t* irow = inner.ptr<uint8_t>(y);
        cv::Vec3b* vrow = vis.ptr<cv::Vec3b>(y);
        for (int x = 0; x < label.cols; ++x) {
            if (lrow[x] != 0) {
                vrow[x] = cv::Vec3b(255, 255, 255);
            }
            if (orow[x] != 0) {
                vrow[x] = cv::Vec3b(255, 0, 0);
            }
            if (irow[x] != 0) {
                vrow[x] = cv::Vec3b(0, 0, 255);
            }
        }
    }
    cv::imwrite((cfg.visualizeDir / ("slice_" + std::to_string(z) + "_combined.png")).string(), vis);
}

static bool runSelfTest()
{
    auto makeNoisyContour = [](int h, int w, unsigned seed) {
        cv::Mat1b m(h, w, uint8_t(0));
        cv::RNG rng(static_cast<int64_t>(seed));
        const cv::Point c(w / 2, h / 2);
        const auto put = [&](int y, int x) {
            if (0 <= y && y < h && 0 <= x && x < w) {
                m.at<uint8_t>(y, x) = 255;
            }
        };
        for (int a = 0; a < 360; ++a) {
            const double t = a * kRadPerDeg;
            const int r = static_cast<int>(0.35 * std::min(h, w) + 6 * std::sin(3.0 * t));
            const int x = c.x + static_cast<int>(std::llround(std::cos(t) * r));
            const int y = c.y + static_cast<int>(std::llround(std::sin(t) * r));
            put(y, x);
            if (rng.uniform(0.0, 1.0) < 0.05) {
                const int ny = y + rng.uniform(-2, 3);
                const int nx = x + rng.uniform(-2, 3);
                put(ny, nx);
            }
        }
        for (int i = 0; i < (h * w) / 80; ++i) {
            const int x = rng.uniform(0, w);
            const int y = rng.uniform(0, h);
            put(y, x);
        }
        return m;
    };

    auto makeShape = [&](int h, int w, int type) {
        cv::Mat1b m(h, w, uint8_t(0));
        switch (type) {
        case 0: {
            const cv::Point c(w / 2, h / 2);
            cv::circle(m, c, 70, cv::Scalar(255), -1);
            break;
        }
        case 1: {
            const cv::Point c(w / 2, h / 2);
            cv::circle(m, c, 80, cv::Scalar(255), -1);
            cv::circle(m, c, 50, cv::Scalar(0), 3);
            break;
        }
        case 2: {
            m = makeNoisyContour(h, w, 2026);
            break;
        }
        case 3: {
            const cv::Point c(w / 3, h / 3 * 2);
            cv::ellipse(m, c, cv::Size(55, 85), -25.0, 0.0, 360.0, cv::Scalar(255), -1);
            break;
        }
        case 4: {
            cv::circle(m, cv::Point(w / 3, h / 2), 32, cv::Scalar(255), -1);
            cv::circle(m, cv::Point((w * 2) / 3, h / 2), 28, cv::Scalar(255), -1);
            break;
        }
        case 5: {
            for (int i = 0; i < std::max(h, w); ++i) {
                const int y = (i * 7) % h;
                const int x = (i * 11) % w;
                m.at<uint8_t>(y, x) = 255;
            }
            break;
        }
        default: {
            cv::RNG rng(1337);
            for (int i = 0; i < (h * w) / 30; ++i) {
                const int y = rng.uniform(0, h);
                const int x = rng.uniform(0, w);
                m.at<uint8_t>(y, x) = 255;
            }
            break;
        }
        }
        return m;
    };

    auto toTouchedSet = [&](const std::vector<std::pair<std::size_t, std::size_t>>& items,
                           std::size_t nCX) {
        std::vector<std::size_t> out;
        out.reserve(items.size());
        for (const auto& p : items) {
            out.push_back(p.first * nCX + p.second);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };

    auto parityCheck = [&](const cv::Mat1b& label,
                           const std::string& name) -> bool {
        const Shape3 outShape = {1,
                                 static_cast<std::size_t>(label.rows * 4),
                                 static_cast<std::size_t>(label.cols * 3)};
        const Shape3 outChunk = {1, 32, 32};
        const auto nn = buildNearestNeighborMap(label.rows, label.cols,
                                               outShape[1], outShape[2]);
        const cv::Mat1b mask = [] (const cv::Mat1b& in) {
            const SliceMaskPair m = detectIgnoreMasks(in,
                                                      kDefaultAngleBins,
                                                      kDefaultAlpha,
                                                      kDefaultOuterSmoothing,
                                                      kDefaultShrink,
                                                      kDefaultInnerSmoothing,
                                                      true,
                                                      true,
                                                      false,
                                                      false,
                                                      AlgoMode::polar);
            cv::Mat1b out;
            out = m.outer.clone();
            cv::bitwise_or(out, m.inner, out);
            return out;
        }(label);

        const cv::Mat1b legacy = upscaleMaskNearest(
            mask,
            static_cast<int>(outShape[1]),
            static_cast<int>(outShape[2]));

        cv::Mat1b exact(static_cast<int>(outShape[1]),
                       static_cast<int>(outShape[2]),
                       uint8_t(0));
        for (int y = 0; y < static_cast<int>(outShape[1]); ++y) {
            const std::size_t srcY = nn.dstToSrcY[static_cast<std::size_t>(y)];
            const uint8_t* src = mask.ptr<uint8_t>(static_cast<int>(srcY));
            uint8_t* dst = exact.ptr<uint8_t>(y);
            for (int x = 0; x < static_cast<int>(outShape[2]); ++x) {
                dst[x] = src[nn.dstToSrcX[static_cast<std::size_t>(x)]];
            }
        }

        cv::Mat diff;
        cv::absdiff(legacy, exact, diff);
        if (cv::countNonZero(diff) != 0) {
            std::cerr << "self-test parity failed: legacy/exact raster mismatch for " << name
                      << '\n';
            return false;
        }

        const auto tLegacy = collectTouchedChunkXY(legacy, outShape, outChunk);
        const auto tExact = collectTouchedChunkXYMapped(mask, nn, outShape, outChunk);
        const auto tFast = collectTouchedChunkXYMappedFast(mask, nn, outShape, outChunk);
        const auto legacySet = toTouchedSet(tLegacy, (outShape[2] + outChunk[2] - 1) / outChunk[2]);
        const auto exactSet = toTouchedSet(tExact, (outShape[2] + outChunk[2] - 1) / outChunk[2]);
        const auto fastSet = toTouchedSet(tFast, (outShape[2] + outChunk[2] - 1) / outChunk[2]);
        if (legacySet != exactSet) {
            std::cerr << "self-test parity failed: touched set mismatch for " << name
                      << " (" << legacySet.size() << " != " << exactSet.size() << ")\n";
            return false;
        }
        if (!std::includes(fastSet.begin(), fastSet.end(), legacySet.begin(), legacySet.end())) {
            std::cerr << "self-test failed: fast touched set misses legacy chunks for " << name << '\n';
            return false;
        }
        return true;
    };

    auto checkShape = [&](const cv::Mat1b& label, const std::string& name) {
        const SliceMaskPair masks = detectIgnoreMasks(label,
                                                     kDefaultAngleBins,
                                                     kDefaultAlpha,
                                                     kDefaultOuterSmoothing,
                                                     kDefaultShrink,
                                                     kDefaultInnerSmoothing,
                                                     true,
                                                     true,
                                                     false,
                                                     false,
                                                     AlgoMode::polar);
        const int fg = cv::countNonZero(label);
        const int inner = cv::countNonZero(masks.inner);
        const int outer = cv::countNonZero(masks.outer);
        if (fg <= 0) {
            std::cerr << "self-test failed on " << name << ": empty foreground\n";
            return false;
        }
        if (inner <= 0 && outer <= 0) {
            std::cerr << "self-test failed on " << name
                      << ": both ignore masks are empty (fg=" << fg
                      << ", inner=" << inner << ", outer=" << outer << ")\n";
            return false;
        }

        cv::Mat1b overlap;
        cv::bitwise_and(masks.inner, label, overlap);
        if (cv::countNonZero(overlap) != 0) {
            std::cerr << "self-test failed on " << name << ": inner overlaps foreground\n";
            return false;
        }
        cv::bitwise_and(masks.outer, label, overlap);
        if (cv::countNonZero(overlap) != 0) {
            std::cerr << "self-test failed on " << name << ": outer overlaps foreground\n";
            return false;
        }

        if (!parityCheck(label, name)) {
            return false;
        }

        std::cout << "self-test shape ok: " << name << " fg=" << fg << " inner=" << inner
                  << " outer=" << outer << '\n';
        return true;
    };

    constexpr int H = 256;
    constexpr int W = 256;

    std::array<std::string, 7> names = {"disk", "ring", "noisy_contour", "off_center", "two_components",
                                        "diagonal", "random_sparse"};
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (!checkShape(makeShape(static_cast<int>(H), static_cast<int>(W), static_cast<int>(i)), names[i])) {
            return false;
        }
    }

    auto checkChunkAlphaWrapConsistency = [&]() {
        const Shape3 volumeShape = {48, 48, 48};
        const Shape3 chunkShape = {16, 16, 16};
        const ChunkIndex centerChunk{1, 1, 1};
        const Box3 coreBox = makeChunkBox(centerChunk, chunkShape, volumeShape);

        std::vector<uint8_t> occ(volumeElements(volumeShape), 0);
        for (std::size_t z = 0; z < volumeShape[0]; ++z) {
            for (std::size_t y = 0; y < volumeShape[1]; ++y) {
                for (std::size_t x = 0; x < volumeShape[2]; ++x) {
                    const double dz = static_cast<double>(z) + 0.5 - 24.0;
                    const double dy = static_cast<double>(y) + 0.5 - 24.0;
                    const double dx = static_cast<double>(x) + 0.5 - 24.0;
                    const double r = std::sqrt(dz * dz + dy * dy + dx * dx);
                    if (std::abs(r - 12.0) <= 0.75) {
                        occ[linearIndex(volumeShape, z, y, x)] = 255;
                    }
                }
            }
        }

        auto extractBox = [&](const Box3& box) {
            std::vector<uint8_t> out(volumeElements(box.shape), 0);
            for (std::size_t z = 0; z < box.shape[0]; ++z) {
                for (std::size_t y = 0; y < box.shape[1]; ++y) {
                    for (std::size_t x = 0; x < box.shape[2]; ++x) {
                        out[linearIndex(box.shape, z, y, x)] =
                            occ[linearIndex(volumeShape,
                                            box.origin[0] + z,
                                            box.origin[1] + y,
                                            box.origin[2] + x)];
                    }
                }
            }
            return out;
        };

        constexpr double alpha = 6.0;
        constexpr double offset = 2.0;
        const Box3 localHalo = expandAndClampBox(coreBox, 8, volumeShape);
        const Box3 refHalo = expandAndClampBox(coreBox, 20, volumeShape);

        const auto localIgnore = classifyOuterAlphaWrapCore(extractBox(localHalo),
                                                            localHalo,
                                                            coreBox,
                                                            alpha,
                                                            offset);
        const auto refIgnore = classifyOuterAlphaWrapCore(extractBox(refHalo),
                                                          refHalo,
                                                          coreBox,
                                                          alpha,
                                                          offset);
        if (localIgnore != refIgnore) {
            std::cerr << "self-test failed: chunk-alpha-wrap local/reference mismatch\n";
            return false;
        }
        return true;
    };

    auto checkAlignedReadMissingChunk = [&]() {
        const fs::path tmpRoot = fs::temp_directory_path() / "vc_add_ignore_selftest_missing_chunk";
        std::error_code ec;
        fs::remove_all(tmpRoot, ec);
        fs::create_directories(tmpRoot, ec);
        if (ec) {
            std::cerr << "self-test failed: unable to create temp root for missing-chunk test\n";
            return false;
        }

        try {
            auto ds = vc::createZarrDataset(tmpRoot,
                                            "0",
                                            {128, 128, 256},
                                            {128, 128, 128},
                                            vc::VcDtype::uint8,
                                            "none");
            std::vector<uint8_t> chunk(128 * 128 * 128, 11);
            ds->writeChunk(0, 0, 0, chunk.data(), chunk.size() * sizeof(uint8_t));

            std::vector<uint8_t> region;
            const bool ok = readAlignedSourceRegionByChunk(*ds,
                                                           {128, 128, 256},
                                                           {128, 128, 128},
                                                           0,
                                                           0,
                                                           0,
                                                           128,
                                                           128,
                                                           256,
                                                           region);
            if (!ok) {
                std::cerr << "self-test failed: aligned missing-chunk read returned false\n";
                fs::remove_all(tmpRoot, ec);
                return false;
            }

            for (std::size_t z = 0; z < 128; ++z) {
                for (std::size_t y = 0; y < 128; ++y) {
                    const std::size_t left = linearIndex({128, 128, 256}, z, y, 0);
                    const std::size_t right = linearIndex({128, 128, 256}, z, y, 128);
                    if (!std::all_of(region.begin() + static_cast<std::ptrdiff_t>(left),
                                     region.begin() + static_cast<std::ptrdiff_t>(left + 128),
                                     [](uint8_t v) { return v == 11; })) {
                        std::cerr << "self-test failed: aligned read corrupted present chunk\n";
                        fs::remove_all(tmpRoot, ec);
                        return false;
                    }
                    if (!std::all_of(region.begin() + static_cast<std::ptrdiff_t>(right),
                                     region.begin() + static_cast<std::ptrdiff_t>(right + 128),
                                     [](uint8_t v) { return v == 0; })) {
                        std::cerr << "self-test failed: aligned read duplicated missing chunk data\n";
                        fs::remove_all(tmpRoot, ec);
                        return false;
                    }
                }
            }

            fs::remove_all(tmpRoot, ec);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "self-test failed in missing-chunk test: " << e.what() << '\n';
            fs::remove_all(tmpRoot, ec);
            return false;
        }
    };

    if (!checkChunkAlphaWrapConsistency()) {
        return false;
    }
    if (!checkAlignedReadMissingChunk()) {
        return false;
    }

    std::cout << "self-test passed\n";
    return true;
}

static int process(const Config& cfg)
{
    if (cfg.selfTest) {
        return runSelfTest() ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (cfg.skipOuter && cfg.skipInner) {
        throw std::runtime_error("both --skip-outer and --skip-inner are set; nothing to do");
    }

    validateInputMetadata(cfg.inputRoot);
    const auto levels = discoverNumericLevels(cfg.inputRoot);
    if (levels.empty()) {
        throw std::runtime_error("input zarr has no numeric levels");
    }
    if (std::find(levels.begin(), levels.end(), cfg.computeLevel) == levels.end()) {
        throw std::runtime_error("compute level not found in input");
    }
    if (std::find(levels.begin(), levels.end(), cfg.outputLevel) == levels.end()) {
        throw std::runtime_error("output level not found in input");
    }
    if (cfg.computeLevel < cfg.outputLevel) {
        throw std::runtime_error("--compute-level must be >= --output-level");
    }
    if (cfg.outputLevel != 0) {
        throw std::runtime_error("this tool currently supports only --output-level 0");
    }

    if (cfg.visualizeSlice.has_value()) {
        visualizeSlice(cfg);
        std::cout << "visualizations written to " << cfg.visualizeDir << '\n';
        return EXIT_SUCCESS;
    }

    const bool outputExisted = fs::exists(cfg.outputRoot);
    if (cfg.reuseOutputTree && cfg.overwrite) {
        throw std::runtime_error("--reuse-output-tree and --overwrite are mutually exclusive");
    }

    bool reuseOutputTree = false;
    if (outputExisted) {
        if (!fs::is_directory(cfg.outputRoot)) {
            throw std::runtime_error("output path exists but is not a directory");
        }
        // Auto-enable reuse when output exists and overwrite is not requested.
        reuseOutputTree = cfg.reuseOutputTree || !cfg.overwrite;
        if (!reuseOutputTree) {
            fs::remove_all(cfg.outputRoot);
            fs::create_directories(cfg.outputRoot);
        } else if (cfg.verbose && !cfg.reuseOutputTree) {
            std::cerr << "[output] existing tree detected; auto-enabling reuse-output-tree" << '\n';
        }
    } else {
        fs::create_directories(cfg.outputRoot);
    }

    const std::size_t workers = cfg.workers > 0
        ? static_cast<std::size_t>(cfg.workers)
        : static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));

    const nlohmann::json inputMeta = readJsonFile(cfg.inputRoot / "meta.json");
    for (int level : levels) {
        vc::VcDataset inLevel(cfg.inputRoot / std::to_string(level));
        if (inLevel.getDtype() != vc::VcDtype::uint8) {
            throw std::runtime_error("only uint8 zarr inputs are supported");
        }
    }

    const auto existingByLevel = scanExistingChunksByLevel(cfg.inputRoot, levels);
    const auto copyStageStart = std::chrono::steady_clock::now();
    if (reuseOutputTree && outputExisted) {
        if (cfg.verbose) {
            std::cerr << "[copy] mode=reuse validating existing output tree\n";
            emitStructuredStage(true, "reuse", 1);
        }
        validateReusableOutputTree(cfg.inputRoot, cfg.outputRoot, levels);
        if (cfg.verbose) {
            emitStructuredProgress(true, "reuse", 1, 1);
        }
    } else {
        if (cfg.verbose) {
            std::size_t totalCopyChunks = 0;
            for (int level : levels) {
                const auto it = existingByLevel.find(level);
                if (it != existingByLevel.end()) {
                    totalCopyChunks += it->second.size();
                }
            }
            std::cerr << "[copy] mode=copy chunk_files=" << totalCopyChunks
                      << " levels=" << levels.size() << '\n';
        }
        copyInputTreeChunkwise(cfg.inputRoot,
                               cfg.outputRoot,
                               levels,
                               existingByLevel,
                               workers,
                               cfg.verbose);
    }
    const double copyStageSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - copyStageStart).count();

    vc::VcDataset computeDs(cfg.inputRoot / std::to_string(cfg.computeLevel));
    vc::VcDataset out0(cfg.outputRoot / "0");
    const Shape3 computeShape = toShape3(computeDs.shape());
    const Shape3 outShape = toShape3(out0.shape());
    const Shape3 outChunk = toShape3(out0.defaultChunkShape());

    int zStart = std::max(0, cfg.zMin);
    int zStop = cfg.zMax >= 0 ? std::min(cfg.zMax, static_cast<int>(computeShape[0]))
                              : static_cast<int>(computeShape[0]);
    if (zStart >= zStop) {
        throw std::runtime_error("empty z range after applying --z-min/--z-max");
    }

    if (cfg.mode == ProcessingMode::chunkAlphaWrap) {
        return processChunkAlphaWrap(cfg,
                                     levels,
                                     inputMeta,
                                     existingByLevel,
                                     reuseOutputTree,
                                     copyStageSeconds,
                                     workers,
                                     computeShape,
                                     outShape,
                                     outChunk,
                                     zStart,
                                     zStop);
    }

    const std::size_t zCount = static_cast<std::size_t>(zStop - zStart);
    const CachePlan cachePlan = chooseCachePlan(cfg, zCount, computeShape);
    const std::size_t sliceElems = computeShape[1] * computeShape[2];
    const std::size_t zScale =
        std::size_t{1} << static_cast<std::size_t>(cfg.computeLevel - cfg.outputLevel);

    ProfileStats profile;

    std::vector<uint8_t> preloadedCompute;
    if (cachePlan.mode == ComputeCacheMode::preload) {
        auto ioStart = std::chrono::steady_clock::now();
        preloadedCompute.resize(cachePlan.requiredBytes);
        computeDs.readRegion({static_cast<std::size_t>(zStart), 0, 0},
                             {zCount, computeShape[1], computeShape[2]},
                             preloadedCompute.data());
        auto ioEnd = std::chrono::steady_clock::now();
        profile.tSliceLoad.add(std::chrono::duration<double>(ioEnd - ioStart).count());
        profile.bytesRead += cachePlan.requiredBytes;
    }
    if (cfg.verbose) {
        std::cerr << "[config] mode=" << processingModeToString(cfg.mode)
                  << " compute_level=" << cfg.computeLevel
                  << " output_level=" << cfg.outputLevel
                  << " alpha=" << cfg.alpha
                  << " workers=" << workers
                  << " output_tree=" << (reuseOutputTree ? "reuse" : "copy")
                  << " z_range=[" << zStart << "," << zStop << ")\n";
        std::cerr << "[cache] mode="
                  << (cachePlan.mode == ComputeCacheMode::preload ? "preload" : "stream")
                  << " required=" << cachePlan.requiredBytes
                  << " budget=" << cachePlan.budgetBytes
                  << " available=" << cachePlan.availableBytes << '\n';
    }

    const bool useLegacyMap = cfg.mapMode == MapMode::legacy;
    const bool useFastMap = cfg.mapMode == MapMode::fast;
    NearestNeighborMap nnMap;
    if (!useLegacyMap) {
        nnMap = buildNearestNeighborMap(computeShape[1], computeShape[2], outShape[1], outShape[2]);
    }

    auto itLevel0 = existingByLevel.find(0);
    if (itLevel0 == existingByLevel.end()) {
        throw std::runtime_error("missing existing chunk index for level 0");
    }
    const auto& existingLevel0Chunks = itLevel0->second;

    ChunkLockPool lockPool(kLockPoolSize);

    std::atomic<int> nextZ{zStart};
    std::atomic<std::size_t> nextSlab{0};
    std::atomic<std::size_t> done{0};
    std::atomic<std::size_t> skippedMissingLevel0{0};
    std::atomic<std::size_t> skippedMissingPyramid{0};
    std::atomic<bool> hadError{false};
    const std::size_t total = static_cast<std::size_t>(zStop - zStart);
    std::mutex ioMutex;
    std::mutex touchedMutex;
    std::mutex profileMutex;
    std::mutex errMutex;
    std::string firstError;
    std::unordered_set<ChunkIndex, ChunkIndexHash> touchedLevel0Global;
    std::vector<std::thread> threads;
    threads.reserve(workers);

    const auto processSlice = [&](int z,
                                 vc::VcDataset& localCompute,
                                 vc::VcDataset& localOut0,
                                 std::vector<uint8_t>& sliceData,
                                 ProfileStats& localProfile,
                                 std::unordered_set<ChunkIndex, ChunkIndexHash>& localTouchedLevel0,
                                 std::size_t& localSkippedLevel0,
                                 cv::Mat1b& label) {
        localProfile.slicesTotal += 1;
        bool hasForeground = false;
        if (cachePlan.mode == ComputeCacheMode::preload) {
            const std::size_t relZ = static_cast<std::size_t>(z - zStart);
            uint8_t* ptr = preloadedCompute.data() + relZ * sliceElems;
            label = cv::Mat1b(static_cast<int>(computeShape[1]), static_cast<int>(computeShape[2]), ptr);
            hasForeground = hasAnyNonZero(ptr, sliceElems);
        } else {
            {
                ScopedTimer readTimer(localProfile.tSliceLoad);
                localCompute.readRegion({static_cast<std::size_t>(z), 0, 0},
                                       {1, computeShape[1], computeShape[2]},
                                       sliceData.data());
            }
            localProfile.bytesRead += sliceElems * sizeof(uint8_t);
            hasForeground = hasAnyNonZero(sliceData);
            label = cv::Mat1b(static_cast<int>(computeShape[1]),
                              static_cast<int>(computeShape[2]),
                              sliceData.data());
        }
        localProfile.inputVoxels += sliceElems;

        if (!hasForeground) {
            ++localProfile.slicesSkippedEmpty;
            return;
        }
        ++localProfile.nonzeroFgSlices;
        if (cfg.profileEnabled) {
            localProfile.fgPixels += countNonZero(label);
        }

        SliceMaskPair masks;
        {
            ScopedTimer maskTimer(localProfile.tMaskBuild);
            masks = detectIgnoreMasks(label,
                                     cfg.nAngleBins,
                                     cfg.alpha,
                                     cfg.outerSmoothingWindow,
                                     cfg.shrinkFactor,
                                     cfg.innerSmoothingWindow,
                                     !cfg.skipOuter,
                                     !cfg.skipInner,
                                     cfg.fastAtan2,
                                     cfg.useSquaredDist,
                                     cfg.algoMode);
        }

        cv::Mat1b combined;
        {
            ScopedTimer mergeTimer(localProfile.tMaskMerge);
            if (!cfg.skipOuter && !cfg.skipInner) {
                combined = masks.outer.clone();
                cv::bitwise_or(combined, masks.inner, combined);
            } else if (!cfg.skipOuter) {
                combined = masks.outer;
            } else {
                combined = masks.inner;
            }
        }

        if (cfg.profileEnabled) {
            localProfile.maskPixels += countNonZero(combined);
        }

        if (useLegacyMap) {
            cv::Mat1b upscaled;
            {
                ScopedTimer resizeTimer(localProfile.tResizeLegacy);
                upscaled = upscaleMaskNearest(
                    combined, static_cast<int>(outShape[1]), static_cast<int>(outShape[2]));
            }

            const auto touchedChunkXY = collectTouchedChunkXY(upscaled, outShape, outChunk);
            localProfile.totalChunksInMasks += touchedChunkXY.size();
            if (touchedChunkXY.empty()) {
                ++localProfile.slicesSkippedEmpty;
                return;
            }
            const std::size_t outZ0 = static_cast<std::size_t>(z) * zScale;
            const std::size_t outZ1 = std::min(outZ0 + zScale, outShape[0]);
            applyMaskToOutputZRangeLegacy(localOut0,
                                          outShape,
                                          outChunk,
                                          outZ0,
                                          outZ1,
                                          upscaled,
                                          touchedChunkXY,
                                          &existingLevel0Chunks,
                                          cfg.existingChunksOnly,
                                          static_cast<uint8_t>(cfg.ignoreValue),
                                          lockPool,
                                          localTouchedLevel0,
                                          localSkippedLevel0,
                                          &localProfile);
        } else {
            std::vector<std::pair<std::size_t, std::size_t>> touchedChunkXY;
            {
                ScopedTimer touchTimer(localProfile.tTouchDiscovery);
                touchedChunkXY = useFastMap ? collectTouchedChunkXYMappedFast(combined, nnMap, outShape, outChunk)
                                            : collectTouchedChunkXYMapped(combined, nnMap, outShape, outChunk);
            }
            localProfile.totalChunksInMasks += touchedChunkXY.size();
            if (touchedChunkXY.empty()) {
                ++localProfile.slicesSkippedEmpty;
                return;
            }
            const std::size_t outZ0 = static_cast<std::size_t>(z) * zScale;
            const std::size_t outZ1 = std::min(outZ0 + zScale, outShape[0]);
            applyMaskToOutputZRangeMapped(localOut0,
                                          outShape,
                                          outChunk,
                                          nnMap,
                                          outZ0,
                                          outZ1,
                                          combined,
                                          touchedChunkXY,
                                          &existingLevel0Chunks,
                                          cfg.existingChunksOnly,
                                          static_cast<uint8_t>(cfg.ignoreValue),
                                          lockPool,
                                          localTouchedLevel0,
                                          localSkippedLevel0,
                                          &localProfile);
        }
    };

    const bool useChunkZSlab = !useLegacyMap && cfg.chunkZSlab;
    const std::size_t sliceProgressTotal = std::max<std::size_t>(1, total);
    const std::size_t sliceCadence = progressCadence(sliceProgressTotal);

    const auto processProgress = [&](std::size_t d) {
        if (cfg.verbose && shouldEmitStructuredProgress(d, sliceProgressTotal, sliceCadence)) {
            emitStructuredProgress(true, "slice", d, sliceProgressTotal);
        }
        if (cfg.verbose && (d == 1 || (d % 32) == 0 || d == total)) {
            std::lock_guard<std::mutex> lock(ioMutex);
            std::cerr << "[ignore] " << d << "/" << total << " slices\n";
        }
    };

    if (cfg.verbose) {
        emitStructuredStage(true, "slice", sliceProgressTotal);
        if (total == 0) {
            emitStructuredProgress(true, "slice", 1, sliceProgressTotal);
        }
    }

    const auto sliceStageStart = std::chrono::steady_clock::now();

    if (useChunkZSlab) {
        const std::size_t slabCount = (outShape[0] + outChunk[0] - 1) / outChunk[0];
        const std::size_t outChunkElems = outChunk[0] * outChunk[1] * outChunk[2];
        const std::size_t outChunkBytes = outChunkElems * sizeof(uint8_t);
        for (std::size_t tid = 0; tid < workers; ++tid) {
            threads.emplace_back([&, tid]() {
                (void)tid;
                vc::VcDataset localCompute(cfg.inputRoot / std::to_string(cfg.computeLevel));
                vc::VcDataset localOut0(cfg.outputRoot / "0");
                std::vector<uint8_t> sliceData(computeShape[1] * computeShape[2], 0);
                std::unordered_set<ChunkIndex, ChunkIndexHash> localTouchedLevel0;
                localTouchedLevel0.reserve(512);
                std::size_t localSkipped = 0;
                ProfileStats localProfile;

                while (true) {
                    if (hadError.load()) {
                        break;
                    }
                    const std::size_t slab = nextSlab.fetch_add(1);
                    if (slab >= slabCount) {
                        break;
                    }

                    try {
                        const std::size_t outZ0Slab = slab * outChunk[0];
                        const std::size_t outZ1Slab = std::min(outZ0Slab + outChunk[0], outShape[0]);
                        const std::size_t z0Compute =
                            std::max<std::size_t>(zStart,
                                                  (outZ0Slab + zScale - 1) / zScale);
                        const std::size_t z1Compute =
                            std::min<std::size_t>(zStop,
                                                  (outZ1Slab + zScale - 1) / zScale);
                        if (z0Compute >= z1Compute) {
                            continue;
                        }

                        std::unordered_map<std::uint64_t, SlabChunkState> slabChunks;
                        std::unordered_set<std::uint64_t> missingChunks;
                        slabChunks.reserve(256);

                        for (std::size_t z = z0Compute; z < z1Compute; ++z) {
                            ++localProfile.slicesTotal;
                            cv::Mat1b label;
                            bool hasForeground = false;
                            if (cachePlan.mode == ComputeCacheMode::preload) {
                                const std::size_t relZ = z - static_cast<std::size_t>(zStart);
                                uint8_t* ptr = preloadedCompute.data() + relZ * sliceElems;
                                label = cv::Mat1b(static_cast<int>(computeShape[1]),
                                                  static_cast<int>(computeShape[2]),
                                                  ptr);
                                hasForeground = hasAnyNonZero(ptr, sliceElems);
                            } else {
                                {
                                    ScopedTimer readTimer(localProfile.tSliceLoad);
                                    localCompute.readRegion({z, 0, 0},
                                                           {1, computeShape[1], computeShape[2]},
                                                           sliceData.data());
                                }
                                localProfile.bytesRead += sliceElems * sizeof(uint8_t);
                                hasForeground = hasAnyNonZero(sliceData);
                                label = cv::Mat1b(static_cast<int>(computeShape[1]),
                                                  static_cast<int>(computeShape[2]),
                                                  sliceData.data());
                            }

                            localProfile.inputVoxels += sliceElems;
                            if (!hasForeground) {
                                ++localProfile.slicesSkippedEmpty;
                                continue;
                            }
                            ++localProfile.nonzeroFgSlices;
                            if (cfg.profileEnabled) {
                                localProfile.fgPixels += countNonZero(label);
                            }

                            SliceMaskPair masks;
                            {
                                ScopedTimer maskTimer(localProfile.tMaskBuild);
                                masks = detectIgnoreMasks(label,
                                                          cfg.nAngleBins,
                                                          cfg.alpha,
                                                          cfg.outerSmoothingWindow,
                                                          cfg.shrinkFactor,
                                                          cfg.innerSmoothingWindow,
                                                          !cfg.skipOuter,
                                                          !cfg.skipInner,
                                                          cfg.fastAtan2,
                                                          cfg.useSquaredDist,
                                                          cfg.algoMode);
                            }

                            cv::Mat1b combined;
                            {
                                ScopedTimer mergeTimer(localProfile.tMaskMerge);
                                if (!cfg.skipOuter && !cfg.skipInner) {
                                    combined = masks.outer.clone();
                                    cv::bitwise_or(combined, masks.inner, combined);
                                } else if (!cfg.skipOuter) {
                                    combined = masks.outer;
                                } else {
                                    combined = masks.inner;
                                }
                            }
                            if (cfg.profileEnabled) {
                                localProfile.maskPixels += countNonZero(combined);
                            }

                            std::vector<std::pair<std::size_t, std::size_t>> touchedChunkXY;
                            {
                                ScopedTimer touchTimer(localProfile.tTouchDiscovery);
                                touchedChunkXY = useFastMap ? collectTouchedChunkXYMappedFast(combined, nnMap, outShape, outChunk)
                                                            : collectTouchedChunkXYMapped(combined, nnMap, outShape, outChunk);
                            }
                            localProfile.totalChunksInMasks += touchedChunkXY.size();
                            if (touchedChunkXY.empty()) {
                                ++localProfile.slicesSkippedEmpty;
                                continue;
                            }

                            const std::size_t outZ0 = z * zScale;
                            const std::size_t outZ1 = std::min(outZ0 + zScale, outShape[0]);
                            for (const auto& ycxc : touchedChunkXY) {
                                const std::size_t cy = ycxc.first;
                                const std::size_t cx = ycxc.second;
                                const std::size_t y0 = cy * outChunk[1];
                                const std::size_t x0 = cx * outChunk[2];
                                const std::uint64_t key = slabChunkKey(cy, cx);

                                auto it = slabChunks.find(key);
                                if (it == slabChunks.end()) {
                                    ++localProfile.chunkCacheMisses;
                                    SlabChunkState state;
                                    state.buf.assign(outChunkElems, 0);
                                    state.h = std::min(outChunk[1], outShape[1] - y0);
                                    state.w = std::min(outChunk[2], outShape[2] - x0);
                                    state.outToSrcY.resize(state.h);
                                    for (std::size_t yy = 0; yy < state.h; ++yy) {
                                        state.outToSrcY[yy] = nnMap.dstToSrcY[y0 + yy];
                                    }
                                    state.outToSrcX.resize(state.w);
                                    for (std::size_t xx = 0; xx < state.w; ++xx) {
                                        state.outToSrcX[xx] = nnMap.dstToSrcX[x0 + xx];
                                    }

                                    const ChunkIndex chunk = {slab, cy, cx};
                                    if (cfg.existingChunksOnly &&
                                        existingLevel0Chunks.find(chunk) == existingLevel0Chunks.end()) {
                                        if (missingChunks.insert(key).second) {
                                            ++localSkipped;
                                        }
                                        continue;
                                    }

                                    auto readTimer = std::chrono::steady_clock::now();
                                    localOut0.readChunk(slab, cy, cx, state.buf.data());
                                    auto readEnd = std::chrono::steady_clock::now();
                                    localProfile.tChunkReadIo.add(std::chrono::duration<double>(readEnd - readTimer).count());
                                    ++localProfile.chunkIoReads;
                                    localProfile.bytesRead += outChunkBytes;
                                    state.loaded = true;
                                    slabChunks.emplace(key, std::move(state));
                                    it = slabChunks.find(key);
                                } else {
                                    ++localProfile.chunkCacheHits;
                                }

                                if (it == slabChunks.end()) {
                                    continue;
                                }

                                const bool changed = writeChunkBufferMapped(combined,
                                                                            outChunk,
                                                                            outZ0Slab,
                                                                            outZ1Slab,
                                                                            it->second.outToSrcY,
                                                                            it->second.outToSrcX,
                                                                            outZ0,
                                                                            outZ1,
                                                                            it->second.buf,
                                                                            static_cast<uint8_t>(cfg.ignoreValue),
                                                                            &localProfile);
                                if (changed) {
                                    it->second.dirty = true;
                                }
                            }
                            const std::size_t d = done.fetch_add(1) + 1;
                            processProgress(d);
                        }

                        std::vector<std::uint64_t> dirtyKeys;
                        dirtyKeys.reserve(slabChunks.size());
                        for (const auto& kv : slabChunks) {
                            if (kv.second.loaded && kv.second.dirty) {
                                dirtyKeys.push_back(kv.first);
                            }
                        }
                        std::sort(dirtyKeys.begin(), dirtyKeys.end());

                        for (const std::uint64_t packed : dirtyKeys) {
                            auto it = slabChunks.find(packed);
                            if (it == slabChunks.end()) {
                                continue;
                            }
                            const std::size_t cx = static_cast<std::size_t>(packed & 0xFFFFFFFFULL);
                            const std::size_t cy = static_cast<std::size_t>((packed >> 32) & 0xFFFFFFFFULL);
                            const ChunkIndex chunk = {slab, cy, cx};

                            auto writeTimer = std::chrono::steady_clock::now();
                            localOut0.writeChunk(slab, cy, cx, it->second.buf.data(), outChunkBytes);
                            auto writeEnd = std::chrono::steady_clock::now();
                            localProfile.tChunkWriteIo.add(std::chrono::duration<double>(writeEnd - writeTimer).count());
                            ++localProfile.chunkIoWrites;
                            localProfile.bytesWritten += outChunkBytes;
                            it->second.loaded = false;
                            localTouchedLevel0.insert(chunk);
                        }
                    } catch (const std::exception& e) {
                        hadError.store(true);
                        std::lock_guard<std::mutex> lk(errMutex);
                        if (firstError.empty()) {
                            firstError = e.what();
                        }
                        break;
                    }
                }

                skippedMissingLevel0.fetch_add(localSkipped);
                {
                    std::lock_guard<std::mutex> lock(touchedMutex);
                    touchedLevel0Global.insert(localTouchedLevel0.begin(), localTouchedLevel0.end());
                }
                {
                    std::lock_guard<std::mutex> lock(profileMutex);
                    profile.accumulate(localProfile);
                }
            });
        }
    } else {
        for (std::size_t tid = 0; tid < workers; ++tid) {
            threads.emplace_back([&, tid]() {
                (void)tid;
                vc::VcDataset localCompute(cfg.inputRoot / std::to_string(cfg.computeLevel));
                vc::VcDataset localOut0(cfg.outputRoot / "0");
                std::vector<uint8_t> sliceData(computeShape[1] * computeShape[2], 0);
                std::unordered_set<ChunkIndex, ChunkIndexHash> localTouchedLevel0;
                localTouchedLevel0.reserve(512);
                std::size_t localSkippedLevel0 = 0;
                ProfileStats localProfile;

                while (true) {
                    if (hadError.load()) {
                        break;
                    }
                    const int z = nextZ.fetch_add(1);
                    if (z >= zStop) {
                        break;
                    }

                    try {
                        cv::Mat1b label;
                        processSlice(z,
                                     localCompute,
                                     localOut0,
                                     sliceData,
                                     localProfile,
                                     localTouchedLevel0,
                                     localSkippedLevel0,
                                     label);
                        const std::size_t d = done.fetch_add(1) + 1;
                        processProgress(d);
                    } catch (const std::exception& e) {
                        hadError.store(true);
                        std::lock_guard<std::mutex> lk(errMutex);
                        if (firstError.empty()) {
                            firstError = e.what();
                        }
                        break;
                    }
                }
                skippedMissingLevel0.fetch_add(localSkippedLevel0);
                {
                    std::lock_guard<std::mutex> lock(touchedMutex);
                    touchedLevel0Global.insert(localTouchedLevel0.begin(), localTouchedLevel0.end());
                }
                {
                    std::lock_guard<std::mutex> lock(profileMutex);
                    profile.accumulate(localProfile);
                }
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }
    if (hadError.load()) {
        throw std::runtime_error("failed while processing slices: " + firstError);
    }
    const double sliceStageSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sliceStageStart).count();

    std::vector<ChunkIndex> activeTouched;
    activeTouched.reserve(touchedLevel0Global.size());
    for (const auto& c : touchedLevel0Global) {
        activeTouched.push_back(c);
    }
    std::sort(activeTouched.begin(), activeTouched.end(), chunkIndexLess);
    profile.touchedChunksLevel0 = activeTouched.size();

    double pyramidStageSeconds = 0.0;
    // Rebuild only touched pyramid regions from output level-0 upwards.
    if (cfg.rebuildPyramid) {
        for (int level : levels) {
            if (level == 0) {
                continue;
            }
            if (activeTouched.empty()) {
                break;
            }
            if (cfg.verbose) {
                std::cerr << "[pyramid] building level " << level
                          << " touched=" << activeTouched.size() << '\n';
            }
            const std::unordered_set<ChunkIndex, ChunkIndexHash>* existingDst = nullptr;
            if (cfg.existingChunksOnly) {
                auto it = existingByLevel.find(level);
                if (it == existingByLevel.end()) {
                    throw std::runtime_error("missing existing chunk index for pyramid level " +
                                             std::to_string(level));
                }
                existingDst = &it->second;
            }
            const auto pyramidLevelStart = std::chrono::steady_clock::now();
            activeTouched = buildLabelPriorityPyramidLevelTouched(cfg.outputRoot,
                                                                  level,
                                                                  activeTouched,
                                                                  workers,
                                                                  static_cast<uint8_t>(cfg.ignoreValue),
                                                                  existingDst,
                                                                  cfg.existingChunksOnly,
                                                                  skippedMissingPyramid,
                                                                  cfg.verbose,
                                                                  &profile);
            pyramidStageSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pyramidLevelStart).count();
        }
    } else if (cfg.verbose) {
        std::cerr << "[pyramid] skipped (no-rebuild-pyramid)" << '\n';
    }

    const Shape3 shape0 = toShape3(out0.shape());
    ProcessingStats stats;
    stats.skippedMissingLevel0 = skippedMissingLevel0.load();
    stats.skippedMissingPyramid = skippedMissingPyramid.load();
    stats.preloadedComputeBytes = cachePlan.mode == ComputeCacheMode::preload
        ? cachePlan.requiredBytes
        : 0;
    stats.preloadedComputeSlices = cachePlan.mode == ComputeCacheMode::preload ? zCount : 0;
    stats.cacheBudgetBytes = cachePlan.budgetBytes;
    stats.availableRamBytes = cachePlan.availableBytes;
    stats.cacheMode = cachePlan.mode == ComputeCacheMode::preload ? "preload" : "stream";
    stats.mapMode = mapModeToString(cfg.mapMode);
    stats.fastAtan2 = cfg.fastAtan2;
    stats.useSquaredDist = cfg.useSquaredDist;
    stats.chunkZSlab = useChunkZSlab;
    stats.rebuildPyramid = cfg.rebuildPyramid;
    stats.reuseOutputTree = reuseOutputTree;
    stats.resultMode = resultModeToString(cfg.resultMode);

    writeOutputMetadata(cfg, shape0, inputMeta, stats);
    if (cfg.profileEnabled && !cfg.profileJsonPath.empty()) {
        writeProfileJson(cfg.profileJsonPath, cfg, reuseOutputTree, profile, total);
    }
    if (cfg.verbose) {
        const double totalStageSeconds = copyStageSeconds + sliceStageSeconds + pyramidStageSeconds;
        std::cerr << std::fixed << std::setprecision(3)
                  << "[summary] mode=" << processingModeToString(cfg.mode)
                  << " copy_seconds=" << copyStageSeconds
                  << " slice_seconds=" << sliceStageSeconds
                  << " pyramid_seconds=" << pyramidStageSeconds
                  << " total_seconds=" << totalStageSeconds
                  << " touched_level0=" << profile.touchedChunksLevel0
                  << '\n';
        std::cerr << "VC_SUMMARY mode=" << processingModeToString(cfg.mode)
                  << " reuse_output_tree=" << (reuseOutputTree ? 1 : 0)
                  << " copy_seconds=" << copyStageSeconds
                  << " slice_seconds=" << sliceStageSeconds
                  << " pyramid_seconds=" << pyramidStageSeconds
                  << " total_seconds=" << totalStageSeconds
                  << " slices=" << total
                  << " touched_level0=" << profile.touchedChunksLevel0
                  << '\n';
        std::cerr.unsetf(std::ios::floatfield);
    }
    std::cout << "vc_add_ignore_label completed\n";
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        Config cfg;
        std::string modeArg = processingModeToString(Config{}.mode);
        std::string mapModeArg = mapModeToString(Config{}.mapMode);
        std::string algoModeArg = algoModeToString(Config{}.algoMode);
        std::string resultModeArg = resultModeToString(Config{}.resultMode);
        bool noChunkZSlab = false;
        bool noRebuildPyramid = false;
        bool fastAtan2LegacyAlias = false;

        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input", po::value<std::string>(), "Input rasterized OME-Zarr root")
            ("output", po::value<std::string>(), "Output OME-Zarr root")
            ("mode", po::value<std::string>(&modeArg)->default_value(processingModeToString(Config{}.mode)),
             "Processing mode: chunk-alpha-wrap | legacy-slice")
            ("ignore-value", po::value<int>(&cfg.ignoreValue)->default_value(kDefaultIgnoreValue),
             "Ignore label value [1..254] (default 127)")
            ("alpha", po::value<double>(&cfg.alpha)->default_value(kDefaultAlpha),
             "Legacy 2D alpha, or chunk alpha alias in chunk-alpha-wrap mode")
            ("chunk-alpha", po::value<double>(&cfg.chunkAlpha)->default_value(0.0),
             "Chunk alpha-wrap radius in compute-level voxels")
            ("n-angle-bins", po::value<int>(&cfg.nAngleBins)->default_value(kDefaultAngleBins),
             "Angular bins for polar boundary detection (default 72)")
            ("shrink-factor", po::value<double>(&cfg.shrinkFactor)->default_value(kDefaultShrink),
             "Inner boundary shrink factor (default 0.95)")
            ("compute-level", po::value<int>(&cfg.computeLevel)->default_value(kDefaultComputeLevel),
             "Pyramid level used for mask detection (default 2)")
            ("output-level", po::value<int>(&cfg.outputLevel)->default_value(kDefaultOutputLevel),
             "Output level receiving labels (currently only 0 supported)")
            ("workers", po::value<int>(&cfg.workers)->default_value(0),
             "Worker threads (0 = hardware concurrency)")
            ("z-min", po::value<int>(&cfg.zMin)->default_value(0),
             "Compute-level start slice (inclusive)")
            ("z-max", po::value<int>(&cfg.zMax)->default_value(-1),
             "Compute-level end slice (exclusive, -1 = end)")
            ("inner-smoothing-window", po::value<int>(&cfg.innerSmoothingWindow)->default_value(kDefaultInnerSmoothing),
             "Circular median smoothing window for inner boundary")
            ("outer-smoothing-window", po::value<int>(&cfg.outerSmoothingWindow)->default_value(kDefaultOuterSmoothing),
             "Circular median smoothing window for outer boundary")
            ("skip-outer", po::bool_switch(&cfg.skipOuter)->default_value(false),
             "Skip outer region detection")
            ("skip-inner", po::bool_switch(&cfg.skipInner)->default_value(false),
             "Skip inner region detection")
            ("existing-chunks-only", po::value<bool>(&cfg.existingChunksOnly)->default_value(true),
             "Only modify chunks that already exist in input zarr levels")
            ("map-mode", po::value<std::string>(&mapModeArg)->default_value(mapModeToString(Config{}.mapMode)),
             "Mapping mode for compute->output mask transfer: legacy | exact | fast")
            ("algo-mode", po::value<std::string>(&algoModeArg)->default_value(algoModeToString(Config{}.algoMode)),
             "Ignore mask geometry mode: polar | contour | hull | ray")
            ("result-mode", po::value<std::string>(&resultModeArg)->default_value(resultModeToString(Config{}.resultMode)),
             "Preset behavior mode: same | fast")
            ("fast-atan2", po::bool_switch(&cfg.fastAtan2)->default_value(false),
             "Use cv::fastAtan2 in polar binning")
            ("fast-atans2", po::bool_switch(&fastAtan2LegacyAlias)->default_value(false),
             "Deprecated alias for --fast-atan2")
            ("use-squared-dist", po::bool_switch(&cfg.useSquaredDist)->default_value(false),
             "Use squared-distance accumulation before final sqrt")
            ("chunk-z-slab", po::bool_switch(&cfg.chunkZSlab)->default_value(false),
             "Enable output chunk-z-slab aggregation to reduce repeated I/O")
            ("no-chunk-z-slab", po::bool_switch(&noChunkZSlab)->default_value(false),
             "Disable output chunk-z-slab aggregation")
            ("no-rebuild-pyramid", po::bool_switch(&noRebuildPyramid)->default_value(false),
             "Skip pyramid rebuild (levels > 0 remain copied from input)")
            ("reuse-output-tree", po::bool_switch(&cfg.reuseOutputTree)->default_value(false),
             "Reuse existing output tree in-place and skip full input-tree copy")
            ("cache-mode", po::value<std::string>(&cfg.cacheMode)->default_value("auto"),
             "Compute slice cache mode: auto | stream | preload")
            ("ram-budget-gb", po::value<double>(&cfg.ramBudgetGb)->default_value(kDefaultRamBudgetGb),
             "RAM budget in GiB for cache planner (0 = auto from MemAvailable)")
            ("visualize", po::value<int>(),
             "Visualize one compute-level slice (no output zarr is written)")
            ("visualize-dir", po::value<std::string>(),
             "Directory for visualization PNGs (default ignore_label_vis)")
            ("profile-json", po::value<std::string>(),
             "Write per-stage profile counters to JSON")
            ("overwrite", po::bool_switch(&cfg.overwrite)->default_value(false),
             "Overwrite output folder if it exists")
            ("self-test", po::bool_switch(&cfg.selfTest)->default_value(false),
             "Run built-in synthetic regression harness and exit")
            ("verbose,v", po::bool_switch(&cfg.verbose)->default_value(false),
             "Verbose progress output");

        po::positional_options_description pos;
        pos.add("input", 1);
        pos.add("output", 1);

        po::variables_map vm;
        const auto parsed = po::command_line_parser(argc, argv)
                                .options(desc)
                                .positional(pos)
                                .run();
        po::store(parsed, vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout
                << "vc_add_ignore_label: add ignore label to rasterized OME-Zarr\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " <input_omezarr> <output_omezarr> [options]\n\n"
                << desc << '\n';
            return EXIT_SUCCESS;
        }

        if (vm.count("visualize")) {
            cfg.visualizeSlice = vm["visualize"].as<int>();
        }
        if (vm.count("visualize-dir")) {
            cfg.visualizeDir = vm["visualize-dir"].as<std::string>();
        }
        if (vm.count("input")) {
            cfg.inputRoot = vm["input"].as<std::string>();
        }
        if (vm.count("output")) {
            cfg.outputRoot = vm["output"].as<std::string>();
        }
        if (vm.count("profile-json")) {
            cfg.profileEnabled = true;
            cfg.profileJsonPath = vm["profile-json"].as<std::string>();
        }

        cfg.mode = parseProcessingMode(modeArg);
        cfg.mapMode = parseMapMode(mapModeArg);
        cfg.algoMode = parseAlgoMode(algoModeArg);
        cfg.resultMode = parseResultMode(resultModeArg);

        if (fastAtan2LegacyAlias) {
            cfg.fastAtan2 = true;
        }
        if (noChunkZSlab && cfg.chunkZSlab) {
            throw std::runtime_error("--chunk-z-slab and --no-chunk-z-slab are mutually exclusive");
        }
        if (noChunkZSlab) {
            cfg.chunkZSlab = false;
        }
        if (noRebuildPyramid) {
            cfg.rebuildPyramid = false;
        }

        const bool mapModeExplicit = vm.count("map-mode") > 0 && !vm["map-mode"].defaulted();
        const bool algoModeExplicit = vm.count("algo-mode") > 0 && !vm["algo-mode"].defaulted();
        const bool resultModeExplicit =
            vm.count("result-mode") > 0 && !vm["result-mode"].defaulted();
        const bool alphaExplicit = vm.count("alpha") > 0 && !vm["alpha"].defaulted();
        const bool chunkAlphaExplicit =
            vm.count("chunk-alpha") > 0 && !vm["chunk-alpha"].defaulted();
        const bool nAngleBinsExplicit =
            vm.count("n-angle-bins") > 0 && !vm["n-angle-bins"].defaulted();
        const bool shrinkFactorExplicit =
            vm.count("shrink-factor") > 0 && !vm["shrink-factor"].defaulted();
        const bool innerSmoothingExplicit =
            vm.count("inner-smoothing-window") > 0 &&
            !vm["inner-smoothing-window"].defaulted();
        const bool outerSmoothingExplicit =
            vm.count("outer-smoothing-window") > 0 &&
            !vm["outer-smoothing-window"].defaulted();
        const bool fastAtan2Explicit =
            (vm.count("fast-atan2") > 0 && !vm["fast-atan2"].defaulted()) || fastAtan2LegacyAlias;
        const bool useSquaredDistExplicit =
            vm.count("use-squared-dist") > 0 && !vm["use-squared-dist"].defaulted();
        const bool chunkZSlabExplicit =
            (vm.count("chunk-z-slab") > 0 && !vm["chunk-z-slab"].defaulted()) || noChunkZSlab;

        if (cfg.mode == ProcessingMode::legacySlice && cfg.resultMode == ResultMode::fast) {
            if (!fastAtan2Explicit) {
                cfg.fastAtan2 = true;
            }
            if (!useSquaredDistExplicit) {
                cfg.useSquaredDist = true;
            }
            if (!mapModeExplicit) {
                cfg.mapMode = MapMode::fast;
            }
            if (!chunkZSlabExplicit) {
                cfg.chunkZSlab = true;
            }
            if (!algoModeExplicit) {
                cfg.algoMode = AlgoMode::polar;
            }
        } else if (cfg.mode == ProcessingMode::legacySlice) {
            if (!chunkZSlabExplicit) {
                cfg.chunkZSlab = true;
            }
        }

        if (!cfg.selfTest && cfg.mode == ProcessingMode::chunkAlphaWrap) {
            if (cfg.skipOuter) {
                throw std::runtime_error(
                    "chunk-alpha-wrap supports only outer labeling; --skip-outer is not supported");
            }
            if (!cfg.skipInner) {
                throw std::runtime_error(
                    "chunk-alpha-wrap currently supports only outer labeling; pass --skip-inner");
            }
            if (nAngleBinsExplicit || shrinkFactorExplicit || innerSmoothingExplicit ||
                outerSmoothingExplicit || algoModeExplicit || fastAtan2Explicit ||
                useSquaredDistExplicit || mapModeExplicit || chunkZSlabExplicit ||
                resultModeExplicit) {
                throw std::runtime_error(
                    "chunk-alpha-wrap does not support legacy 2D mask controls or mapping presets");
            }
            if (chunkAlphaExplicit && alphaExplicit &&
                std::abs(cfg.chunkAlpha - cfg.alpha) > 1e-12) {
                throw std::runtime_error(
                    "--alpha and --chunk-alpha disagree in chunk-alpha-wrap mode");
            }
            if (!chunkAlphaExplicit && !alphaExplicit) {
                throw std::runtime_error(
                    "chunk-alpha-wrap requires --chunk-alpha (or --alpha as an alias)");
            }
            cfg.chunkAlpha = chunkAlphaExplicit ? cfg.chunkAlpha : cfg.alpha;
        } else if (!cfg.selfTest && chunkAlphaExplicit) {
            throw std::runtime_error("--chunk-alpha is only valid in chunk-alpha-wrap mode");
        }

        if (!cfg.selfTest) {
            if (cfg.inputRoot.empty()) {
                throw std::runtime_error("missing input path");
            }
            if (!cfg.visualizeSlice.has_value() && cfg.outputRoot.empty()) {
                throw std::runtime_error("missing output path");
            }
            if (!fs::exists(cfg.inputRoot) || !fs::is_directory(cfg.inputRoot)) {
                throw std::runtime_error("input path is not a directory");
            }
        }

        if (cfg.ignoreValue <= 0 || cfg.ignoreValue >= 255) {
            throw std::runtime_error("--ignore-value must be in [1, 254]");
        }
        if (cfg.mode == ProcessingMode::legacySlice && cfg.nAngleBins < 8) {
            throw std::runtime_error("--n-angle-bins must be >= 8");
        }
        if (cfg.mode == ProcessingMode::legacySlice &&
            (cfg.shrinkFactor <= 0.0 || cfg.shrinkFactor > 1.5)) {
            throw std::runtime_error("--shrink-factor must be in (0, 1.5]");
        }
        if (cfg.computeLevel < 0 || cfg.outputLevel < 0) {
            throw std::runtime_error("compute/output levels must be >= 0");
        }
        if (cfg.mode == ProcessingMode::legacySlice &&
            (cfg.innerSmoothingWindow < 1 || cfg.outerSmoothingWindow < 1)) {
            throw std::runtime_error("smoothing windows must be >= 1");
        }
        if (!cfg.selfTest && cfg.mode == ProcessingMode::chunkAlphaWrap && cfg.chunkAlpha <= 0.0) {
            throw std::runtime_error("--chunk-alpha must be > 0");
        }
        if (cfg.ramBudgetGb < 0.0) {
            throw std::runtime_error("--ram-budget-gb must be >= 0");
        }
        if (cfg.cacheMode != "auto" && cfg.cacheMode != "stream" && cfg.cacheMode != "preload") {
            throw std::runtime_error("--cache-mode must be one of: auto, stream, preload");
        }
        if (!cfg.profileJsonPath.empty() && cfg.profileJsonPath == "-") {
            throw std::runtime_error("--profile-json requires a file path");
        }

        return process(cfg);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
