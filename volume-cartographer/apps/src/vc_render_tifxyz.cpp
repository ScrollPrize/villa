#include <iostream>
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StoreToolOptions.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/Zarr.hpp"
#include "vc/core/util/StreamOperators.hpp"

#include "vc/core/types/Volume.hpp"
#include <vc_reader/composite.hpp>
#include <vc_reader/zarr.hpp>
#include "vc/core/util/VoxelSizeMetadata.hpp"
#include "utils/Json.hpp"
#include <vc_reader/affine.hpp>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <boost/program_options.hpp>
#include <limits>
#include <mutex>
#include <cmath>
#include <set>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <exception>
#include <thread>
#include <optional>
#include <unordered_set>
#include <tiffio.h>
#include <omp.h>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace po = boost::program_options;
using Json = utils::Json;
using Affine = vcr::catalog::Affine;

// ============================================================
// Logging infrastructure
// ============================================================

static FILE* g_logFile = nullptr;       // non-null when --log-path active
static std::string g_logPrefix;         // e.g. "[part 2/8] " — prepended when logging to file
static bool g_flipNormals = false;      // negate surface normals (--flip-normals); reverses slice ordering along the normal
static bool g_logRunning = false;      // protected by g_logFlushMutex
static std::mutex g_logFlushMutex;
static std::condition_variable g_logFlushWake;
static std::thread g_logFlushThread;

// Log to file if active, otherwise to the given default stream.
// When logging to a shared file in multi-part mode, each line is prefixed with the part id.
static void logPrintf(FILE* defaultStream, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
static void logPrintf(FILE* defaultStream, const char* fmt, ...)
{
    FILE* out = g_logFile ? g_logFile : defaultStream;
    if (g_logFile && !g_logPrefix.empty())
        std::fputs(g_logPrefix.c_str(), out);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(out, fmt, ap);
    va_end(ap);
}

// Start background flush thread (every ~5 seconds)
static void startLogFlusher()
{
    if (!g_logFile) return;
    {
        std::lock_guard<std::mutex> lock(g_logFlushMutex);
        g_logRunning = true;
    }
    g_logFlushThread = std::thread([] {
        std::unique_lock<std::mutex> lock(g_logFlushMutex);
        while (!g_logFlushWake.wait_for(lock, std::chrono::seconds(5),
                                       [] { return !g_logRunning; })) {
            if (g_logFile) std::fflush(g_logFile);
        }
    });
}

static void stopLogFlusher()
{
    {
        // Change the predicate under the wait's mutex so a stop notification
        // cannot be lost between checking the flag and entering the wait.
        std::lock_guard<std::mutex> lock(g_logFlushMutex);
        g_logRunning = false;
    }
    g_logFlushWake.notify_one();
    if (g_logFlushThread.joinable()) g_logFlushThread.join();
    if (g_logFile) { std::fflush(g_logFile); std::fclose(g_logFile); g_logFile = nullptr; }
}

// ============================================================
// Affine transform utilities
// ============================================================

static void printMat4x4(const Affine& M, const char* header)
{
    if (header) logPrintf(stdout, "%s\n", header);
    for (int r = 0; r < 4; ++r) {
        logPrintf(stdout, "  [%12.6f, %12.6f, %12.6f, %12.6f]\n",
                  M(r,0), M(r,1), M(r,2), M(r,3));
    }
}

static std::pair<std::string, bool> parseAffineSpec(const std::string& spec)
{
    for (const auto& suffix : {":inv", ":invert", ":i"}) {
        std::string s(suffix);
        if (spec.size() > s.size() && spec.substr(spec.size() - s.size()) == s)
            return {spec.substr(0, spec.size() - s.size()), true};
    }
    return {spec, false};
}

// ============================================================
// Geometry helpers
// ============================================================

static void applyAffineTransform(cv::Mat_<cv::Vec3f>& points,
                                  cv::Mat_<cv::Vec3f>& normals,
                                  const Affine& T)
{
    const auto normalTransform = T.normalTransform();

    for (int y = 0; y < points.rows; y++)
        for (int x = 0; x < points.cols; x++) {
            cv::Vec3f& pt = points(y, x);
            if (std::isnan(pt[0])) continue;
            const auto q = T.apply({pt[0], pt[1], pt[2]});
            pt = cv::Vec3f(float(q[0]), float(q[1]), float(q[2]));
        }

    for (int y = 0; y < normals.rows; y++)
        for (int x = 0; x < normals.cols; x++) {
            cv::Vec3f& n = normals(y, x);
            if (std::isnan(n[0])) continue;
            const auto q = normalTransform.mapDirection({n[0], n[1], n[2]});
            n = cv::Vec3f(float(q[0]), float(q[1]), float(q[2]));
        }
}

static void normalizeNormals(cv::Mat_<cv::Vec3f>& nrm)
{
    for (int y = 0; y < nrm.rows; y++)
        for (int x = 0; x < nrm.cols; x++) {
            auto& v = nrm(y, x);
            if (std::isnan(v[0])) continue;
            const float L2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
            if (L2 > 0) {
                v /= std::sqrt(L2);
            }
        }
}

// ============================================================
// Rotation / flip helpers
// ============================================================

static int normalizeQuadrantRotation(double angleDeg, double tol = 0.5)
{
    double a = std::fmod(angleDeg, 360.0);
    if (a < 0) a += 360.0;
    for (int i = 0; i < 4; ++i)
        if (std::abs(a - i * 90.0) <= tol) return i;
    return -1;
}

static void rotateFlipIfNeeded(cv::Mat& m, int rotQuad, int flip_axis)
{
    if (rotQuad == 1)      cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (rotQuad == 2) cv::rotate(m, m, cv::ROTATE_180);
    else if (rotQuad == 3) cv::rotate(m, m, cv::ROTATE_90_CLOCKWISE);
    if (flip_axis == 0)      cv::flip(m, m, 0);
    else if (flip_axis == 1) cv::flip(m, m, 1);
    else if (flip_axis == 2) cv::flip(m, m, -1);
}

// ============================================================
// Surface generation helpers
// ============================================================

static void computeCanvasOrigin(const cv::Size& size, float& u0, float& v0)
{
    u0 = -0.5f * (size.width  - 1.0f);
    v0 = -0.5f * (size.height - 1.0f);
}

static void genTile(QuadSurface* surf, const cv::Size& size, float render_scale,
                    float u0, float v0, cv::Mat_<cv::Vec3f>& points, cv::Mat_<cv::Vec3f>& normals)
{
    surf->gen(&points, &normals, size, cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0, v0, 0));
    // NOTE: do NOT mutate `normals` here. gen() returns a view into a scratch
    // buffer it reuses across calls (thread_local per worker), so the next
    // gen() on this thread overwrites it. --flip-normals is applied later in
    // prepareBaseAndDirs() on the thread-private clone instead.
}

// Surface coordinates (working-level voxels, XYZ) to indices of the rendered
// pyramid level: index = scale * c + offset per axis, from the two levels' placements.
struct LevelMap {
    cv::Vec3f scale{1, 1, 1}, offset{0, 0, 0};
};

static LevelMap mapToLevel(const vcr::zarr::LevelPlacement& working, const vcr::zarr::LevelPlacement& rendered)
{
    LevelMap map;
    for (int a = 0; a < 3; ++a) {
        const int zyx = 2 - a;
        map.scale[a] = float(working.scale[zyx] / rendered.scale[zyx]);
        map.offset[a] = float((working.offset[zyx] - rendered.offset[zyx]) / rendered.scale[zyx]);
    }
    return map;
}

static void prepareBaseAndDirs(const cv::Mat_<cv::Vec3f>& pts, const cv::Mat_<cv::Vec3f>& nrm,
                                float scale_seg, const LevelMap& level,
                                bool hasAffine, const Affine& aff,
                                cv::Mat_<cv::Vec3f>& base, cv::Mat_<cv::Vec3f>& dirs)
{
    base = pts.clone(); base *= scale_seg;
    dirs = nrm.clone();
    // --flip-normals: negate on the private clone (safe under OpenMP), never on
    // gen()'s shared scratch view. Negation commutes with the affine + the
    // subsequent normalization, so applying it here is identical to flipping
    // the raw geometric normal. (-NaN stays NaN, so invalid pixels are intact.)
    if (g_flipNormals) dirs *= -1.0f;
    if (hasAffine) applyAffineTransform(base, dirs, aff);
    normalizeNormals(dirs);
    for (auto& p : base)
        for (int a = 0; a < 3; ++a)
            p[a] = p[a] * level.scale[a] + level.offset[a];
}

// The ZYX points of every pixel at every offset along its normal, layer-major
// (point k * n + p), computed in float as the renders always have.
static std::vector<double> samplePoints(const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                                        const std::vector<float>& offsets)
{
    const std::size_t n = base.total();
    std::vector<double> points(3 * n * offsets.size());
    double* out = points.data();
    for (const float off : offsets)
        for (std::size_t p = 0; p < n; ++p, out += 3) {
            const cv::Vec3f& b = base(int(p));
            const cv::Vec3f& d = dirs(int(p));
            out[0] = b[2] + d[2] * off;
            out[1] = b[1] + d[1] * off;
            out[2] = b[0] + d[0] * off;
        }
    return points;
}

// One trilinear image per offset. Pixels without a surface read 0; uint8 truncates
// and uint16 rounds, as the renders always have.
template <typename T>
static std::vector<cv::Mat_<T>> sampleLayers(vcr::Array& volume, const vcr::ElementType& type,
                                             const cv::Mat_<cv::Vec3f>& base, const cv::Mat_<cv::Vec3f>& dirs,
                                             const std::vector<float>& offsets)
{
    const std::size_t n = base.total();
    const auto points = samplePoints(base, dirs, offsets);
    std::vector<float> values(n * offsets.size());
    volume.sample_linear(points.data(), values.size(), type, false, values.data());
    std::vector<cv::Mat_<T>> out(offsets.size());
    constexpr float maxValue = float(std::numeric_limits<T>::max());
    constexpr float rounding = std::is_same_v<T, uint16_t> ? 0.5f : 0.0f;
    for (std::size_t k = 0; k < offsets.size(); ++k) {
        out[k].create(base.rows, base.cols);
        const float* v = values.data() + k * n;
        T* dst = out[k][0];
        for (std::size_t p = 0; p < n; ++p)
            dst[p] = std::isnan(v[p]) ? T(0) : T(std::clamp(v[p], 0.0f, maxValue) + rounding);
    }
    return out;
}

// The composite of every pixel's nearest-voxel layers of a uint8 volume; pixels
// without a surface or a remaining layer are 0.
static cv::Mat_<uint8_t> sampleComposite(vcr::Array& volume, const cv::Mat_<cv::Vec3f>& base,
                                         const cv::Mat_<cv::Vec3f>& dirs, const std::vector<float>& offsets,
                                         const vcr::composite::Options& method)
{
    const std::size_t n = base.total();
    const auto points = samplePoints(base, dirs, offsets);
    std::vector<uint8_t> voxels(n * offsets.size());
    volume.sample_nearest(points.data(), voxels.size(), voxels.data());
    std::vector<float> layers(voxels.size());
    for (std::size_t i = 0; i < voxels.size(); ++i) {
        const double* q = &points[3 * i];
        layers[i] = std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2])
            ? float(voxels[i]) : std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<float> values(n);
    vcr::composite::reduce(layers.data(), offsets.size(), n, method, values.data());
    cv::Mat_<uint8_t> out(base.rows, base.cols);
    uint8_t* dst = out[0];
    for (std::size_t p = 0; p < n; ++p)
        dst[p] = std::isnan(values[p]) ? uint8_t(0) : uint8_t(std::clamp(values[p], 0.0f, 255.0f));
    return out;
}

// ============================================================
// Accumulation (max / mean / median) — templated for u8 & u16
// ============================================================

enum class AccumType { Max, Mean, Median, Alpha, BeerLambert };

template <typename T>
static cv::Mat accumulate(const std::vector<cv::Mat>& samples, size_t start, size_t count, AccumType type, int cvType)
{
    if (count == 1) return samples[start];
    if (type == AccumType::Max) {
        cv::Mat out = samples[start].clone();
        for (size_t j = 1; j < count; ++j) cv::max(out, samples[start + j], out);
        return out;
    }
    if (type == AccumType::Mean) {
        cv::Mat sum; samples[start].convertTo(sum, CV_64F);
        for (size_t j = 1; j < count; ++j) { cv::Mat t; samples[start+j].convertTo(t, CV_64F); sum += t; }
        sum /= double(count);
        cv::Mat out; sum.convertTo(out, cvType);
        return out;
    }
    // Median
    int rows = samples[start].rows, cols = samples[start].cols;
    cv::Mat out(rows, cols, cvType);
    std::vector<T> vals(count);
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++) {
            for (size_t i = 0; i < count; i++) vals[i] = samples[start + i].at<T>(y, x);
            std::sort(vals.begin(), vals.end());
            out.at<T>(y, x) = (count % 2 == 1)
                ? vals[count / 2]
                : T((uint32_t(vals[count/2 - 1]) + uint32_t(vals[count/2]) + 1) / 2);
        }
    return out;
}

// ============================================================
// Sample offsets along the surface normal
// ============================================================

// Build the flat offset vector sampled by sampleLayers. Offsets are distances along the
// unit surface normal, in level-g voxels: base points are already scaled into level-g
// index space (prepareBaseAndDirs) and dirs are unit length, so a step of sliceStep
// advances exactly one level-g voxel. (Do NOT scale by ds_scale here — that would make
// the through-normal spacing the native/level-0 resolution while the in-plane spacing is
// level-g, yielding an anisotropic stack that mismatches the isotropic .zattrs scale.)
static std::vector<float> buildOffsetList(int numSlices, double sliceStep,
                                           const std::vector<float>& accumOffsets)
{
    double center = 0.5 * (std::max(1, numSlices) - 1.0);
    size_t sps = std::max<size_t>(1, accumOffsets.size());
    std::vector<float> out;
    out.reserve(numSlices * sps);
    for (int zi = 0; zi < numSlices; zi++) {
        float base = float((zi - center) * sliceStep);
        if (accumOffsets.empty())
            out.push_back(base);
        else
            for (float ao : accumOffsets)
                out.push_back(base + ao);
    }
    return out;
}

static std::vector<float> buildCompositeOffsetList(
    int compositeStart, int compositeEnd, double sliceStep)
{
    std::vector<float> out;
    out.reserve(std::max(0, compositeEnd - compositeStart + 1));
    for (int zi = compositeStart; zi <= compositeEnd; zi++)
        out.push_back(float(zi) * float(sliceStep));
    return out;
}

#ifdef __linux__
// Lowest memory limit from this process's cgroup up to the top of its mount: a limit on
// any ancestor (a systemd slice, a container) applies to every cgroup below it.
// /proc/self/cgroup gives the cgroup relative to its hierarchy's root, /proc/self/mountinfo
// where that hierarchy is mounted and which subtree the mount shows. 0 when none is set.
static unsigned long long cgroupMemoryLimit()
{
    const auto listed = [](const std::string& csv, const std::string& name) {
        return ("," + csv + ",").find("," + name + ",") != std::string::npos;
    };
    unsigned long long best = 0;
    std::ifstream mounts("/proc/self/mountinfo");
    for (std::string line; std::getline(mounts, line);) {
        // id parent major:minor root mountpoint options [optional...] - fstype source superoptions
        std::istringstream fields(line);
        std::string skip, root, mnt, fstype, superOpts;
        fields >> skip >> skip >> skip >> root >> mnt;
        while (fields >> skip && skip != "-") {}
        fields >> fstype >> skip >> superOpts;
        const bool v2 = fstype == "cgroup2";
        if (!v2 && !(fstype == "cgroup" && listed(superOpts, "memory"))) continue;

        // hierarchy:controllers:path, where cgroup v2 is the line "0::path"
        std::string path;
        std::ifstream cgroups("/proc/self/cgroup");
        for (std::string entry; std::getline(cgroups, entry);) {
            const auto a = entry.find(':'), b = entry.find(':', a + 1);
            if (a == std::string::npos || b == std::string::npos) continue;
            const std::string controllers = entry.substr(a + 1, b - a - 1);
            if (v2 ? entry.compare(0, a, "0") == 0 && controllers.empty() : listed(controllers, "memory"))
                path = entry.substr(b + 1);
        }
        if (path.empty() || path.find("/..") != std::string::npos) continue;
        // A mount that shows only a subtree (a container without its own cgroup namespace)
        // has the process's cgroup somewhere under that subtree, or not at all.
        if (root != "/") {
            if (path.compare(0, root.size(), root) != 0 || (path.size() > root.size() && path[root.size()] != '/'))
                continue;
            path.erase(0, root.size());
        }

        const std::string file = v2 ? "/memory.max" : "/memory.limit_in_bytes";
        std::string dir = mnt + path;
        while (dir.size() > mnt.size() && dir.back() == '/') dir.pop_back();
        for (;;) {
            std::ifstream in(dir + file); unsigned long long lim = 0;
            if (in >> lim && lim > 0 && (best == 0 || lim < best)) best = lim;
            if (dir.size() <= mnt.size()) break;
            dir.erase(dir.rfind('/'));
        }
    }
    return best;
}
#endif

// Memory this process can actually use: the lowest cgroup limit when one is set
// (a container, a systemd slice), otherwise the machine's RAM. 0 when unknown (Windows).
static size_t usableMemoryBytes()
{
    size_t bytes = 0;
#ifndef _WIN32
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) bytes = size_t(pages) * size_t(page);
#endif
#ifdef __linux__
    for (const char* f : {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        std::ifstream in(f); unsigned long long lim = 0;
        if (in >> lim && lim > 0 && (bytes == 0 || lim < bytes)) bytes = size_t(lim);
    }
    if (const unsigned long long lim = cgroupMemoryLimit(); lim > 0 && (bytes == 0 || lim < bytes))
        bytes = size_t(lim);
#endif
    return bytes;
}

static bool pathsEquivalent(const std::filesystem::path& a, const std::filesystem::path& b)
{
    std::error_code ecA, ecB;
    auto ca = std::filesystem::weakly_canonical(a, ecA);
    auto cb = std::filesystem::weakly_canonical(b, ecB);
    if (!ecA && !ecB) return ca == cb;
    return a.lexically_normal() == b.lexically_normal();
}

// Accumulates the sampled layers into one image per output slice.
template <typename T>
static std::vector<cv::Mat> processRawSlices(std::vector<cv::Mat_<T>>& raw, int numSlices,
                                              const std::vector<float>& accumOffsets,
                                              AccumType accumType, int cvType)
{
    size_t sps = std::max<size_t>(1, accumOffsets.size());
    std::vector<cv::Mat> result(numSlices);
    for (int zi = 0; zi < numSlices; zi++) {
        if (accumOffsets.empty()) {
            result[zi] = raw[zi];
        } else {
            // Gather samples into a generic vector
            std::vector<cv::Mat> samps(raw.begin() + zi * sps, raw.begin() + zi * sps + sps);
            result[zi] = accumulate<T>(samps, 0, sps, accumType, cvType);
        }
    }
    return result;
}

// ============================================================
// TIFF row writer
// ============================================================

// Writes full-width output rows starting at source row rowY0 into each slice's TIFF.
static void writeTifRows(std::vector<TiffWriter>& writers,
                          const std::vector<cv::Mat>& slices,
                          uint32_t rowY0, uint32_t tiffTileH,
                          uint32_t srcHeight, int rotQuad, int flipAxis)
{
    uint32_t numTiffTiles = (srcHeight + tiffTileH - 1) / tiffTileH;
    for (size_t zi = 0; zi < slices.size(); zi++) {
        for (uint32_t ty = 0; ty < uint32_t(slices[zi].rows); ty += tiffTileH) {
            uint32_t tdy = std::min(tiffTileH, uint32_t(slices[zi].rows) - ty);
            cv::Mat sub = slices[zi](cv::Rect(0, ty, slices[zi].cols, tdy));
            int dstBx, dstBy, rBX, rBY;
            uint32_t srcTileIdx = (rowY0 + ty) / tiffTileH;
            mapTileIndex(0, int(srcTileIdx), 1, int(numTiffTiles),
                         std::max(rotQuad, 0), flipAxis, dstBx, dstBy, rBX, rBY);
            writers[zi].writeTile(0, uint32_t(dstBy) * tiffTileH, sub);
        }
    }
}


// ============================================================
// Rendering: OMP-parallel over 128x128 output tiles, written to the zarr
// chunks and/or assembled into TIFF rows
// ============================================================

template <typename T>
static void renderTiles(
    QuadSurface* surf, vcr::Array& volume, const vcr::ElementType& type, const LevelMap& levelMap,
    const cv::Size& fullSize, const cv::Rect& crop, const cv::Size& tgtSize,
    float renderScale, float scaleSeg,
    bool hasAffine, const Affine& aff,
    int numSlices, double sliceStep,
    const std::vector<float>& accumOffsets, AccumType accumType,
    bool isComposite, int compositeStart, int compositeEnd,
    const vcr::composite::Options& compositeMethod,
    int rotQuad, int flipAxis,
    int numParts, int partId,
    int cvType,
    // Zarr output (null: TIFF only); chunks0 is the tile shape either way
    vcr::zarr::Writer* dsOut, const std::vector<size_t>& chunks0,
    size_t tilesXSrc, size_t tilesYSrc,
    // Pyramid datasets L1-L5 (empty = no inline pyramid)
    const std::vector<vcr::zarr::Writer*>& pyramidDs,
    // TIF output (optional)
    std::vector<TiffWriter>* tifWriters, uint32_t tiffTileH,
    bool quickTif,
    bool resume)
{
    const size_t CH = chunks0[1], CW = chunks0[2];
    const uint32_t numTileRows = static_cast<uint32_t>(tilesYSrc);
    const uint32_t numTileCols = static_cast<uint32_t>(tilesXSrc);

    // Pre-warm validMask cache (thread-safety: subsequent calls are read-only)
    surf->validMask();

    auto allOffsets = buildOffsetList(numSlices, sliceStep, accumOffsets);
    const auto compositeOffsets = isComposite
        ? buildCompositeOffsetList(compositeStart, compositeEnd, sliceStep)
        : std::vector<float>{};

    const bool wantTif = tifWriters && !tifWriters->empty();
    const bool useU16 = (cvType == CV_16UC1);

    auto wallStart = std::chrono::steady_clock::now();
    auto lastPrint = wallStart;

    // Contiguous block assignment: each part gets a contiguous range of tile rows
    // for better spatial locality (avoids all VMs reading the same volume region)
    uint32_t rowsPerPart = (numTileRows + uint32_t(numParts) - 1) / uint32_t(numParts);
    // Align to 32 tile-rows for multi-VM pyramid (2^5 = 32 covers L1..L5)
    if (numParts > 1 && !pyramidDs.empty()) {
        constexpr uint32_t kPyrAlign = 32;
        rowsPerPart = ((rowsPerPart + kPyrAlign - 1) / kPyrAlign) * kPyrAlign;
    }
    uint32_t tyStart = uint32_t(partId) * rowsPerPart;
    uint32_t tyEnd = std::min(tyStart + rowsPerPart, numTileRows);

    // ---- Pyramid accumulation buffers (L1..L5) ----
    // At level L, one 128×128 pyramid chunk covers 2^L × 2^L L0 tiles.
    // pyrAccum[li] has one buffer per pyramid chunk column for the current row-group.
    struct PyrAccumLevel {
        size_t chZ, chY, chX;          // chunk dims at this level
        size_t pyrChunksX;             // number of pyramid chunk columns
        std::vector<std::vector<T>> bufs;  // one per pyramid chunk column
    };
    std::vector<PyrAccumLevel> pyrAccum;
    if (!pyramidDs.empty()) {
        pyrAccum.resize(pyramidDs.size());
        for (size_t li = 0; li < pyramidDs.size(); li++) {
            const auto& pc = pyramidDs[li]->meta().chunks;
            pyrAccum[li].chZ = pc[0];
            pyrAccum[li].chY = pc[1];
            pyrAccum[li].chX = pc[2];
            // Number of pyramid chunk columns at this level
            const auto& ps = pyramidDs[li]->meta().shape;
            pyrAccum[li].pyrChunksX = (ps[2] + pc[2] - 1) / pc[2];
            pyrAccum[li].bufs.resize(pyrAccum[li].pyrChunksX);
            for (auto& b : pyrAccum[li].bufs)
                b.assign(pc[0] * pc[1] * pc[2], T(0));
        }
    }

    for (uint32_t ty = tyStart; ty < tyEnd; ty++) {
        uint32_t y0 = ty * uint32_t(CH);
        uint32_t dy = std::min(uint32_t(CH), uint32_t(tgtSize.height) - y0);

        // TIF row buffer: one tile-column's worth of slices per tx
        // tifRowBuf[tx] = vector of cv::Mat (one per output slice), each 128xdx
        std::vector<std::vector<cv::Mat>> tifRowBuf;
        if (wantTif) tifRowBuf.resize(numTileCols);

        // A throw inside the OpenMP tile loop (a remote chunk fetch that failed
        // after its retries, for instance) would end in std::terminate. Keep the
        // first failure and rethrow it once every tile of this row has stopped.
        std::atomic<bool> rowFailed{false};
        std::exception_ptr rowError;
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t tx = 0; tx < numTileCols; tx++) {
            if (rowFailed.load(std::memory_order_relaxed)) continue;
            try {
                // Resume: skip tile if L0 chunk already exists on disk
                if (resume && dsOut) {
                    const bool needsRotFlip = (rotQuad >= 0 || flipAxis >= 0);
                    int dTx = int(tx), dTy = int(ty), dTX, dTY;
                    if (needsRotFlip)
                        mapTileIndex(int(tx), int(ty), int(tilesXSrc), int(tilesYSrc),
                                     std::max(rotQuad, 0), flipAxis, dTx, dTy, dTX, dTY);
                    if (dsOut->chunkExists({0, dTy, dTx})) {
                        // Still need to scatter into pyramid accum buffers
                        // No rotation when inline pyramid is active
                        if (!pyrAccum.empty()) {
                            size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];
                            std::vector<T> existingBuf(chunkZ * chunkY * chunkX, T(0));
                            const auto existing = dsOut->reader().chunk({0, dTy, dTx});
                            if (existing) {
                                if (existing->size() != existingBuf.size() * sizeof(T))
                                    throw std::runtime_error("resume chunk has unexpected dtype or size");
                                std::memcpy(existingBuf.data(), existing->data(), existing->size());
                            }
                            uint32_t dxTile = std::min(uint32_t(CW), uint32_t(tgtSize.width) - tx * uint32_t(CW));
                            size_t dy_actual = std::min(chunkY, size_t(dy));
                            size_t dx_actual = std::min(chunkX, size_t(dxTile));
                            size_t numZ = isComposite ? 1 : size_t(std::max(1, numSlices));
                            size_t l1cx = size_t(tx) >> 1;
                            size_t halfCH = CH / 2, halfCW = CW / 2;
                            size_t offY = (size_t(ty) & 1) * halfCH;
                            size_t offX = (size_t(tx) & 1) * halfCW;
                            auto& pa = pyrAccum[0];
                            if (l1cx < pa.bufs.size()) {
                                downsampleTileIntoPreserveZ(
                                    existingBuf.data(), chunkZ, chunkY, chunkX,
                                    pa.bufs[l1cx].data(), pa.chZ, pa.chY, pa.chX,
                                    numZ, dy_actual, dx_actual,
                                    offY, offX);
                            }
                        }
                        continue;
                    }
                }

                uint32_t x0 = tx * uint32_t(CW);
                uint32_t dx = std::min(uint32_t(CW), uint32_t(tgtSize.width) - x0);

                // 1. Generate surface for this tile
                float u0, v0; computeCanvasOrigin(fullSize, u0, v0);
                u0 += float(crop.x) + float(x0);
                v0 += float(crop.y) + float(y0);
                cv::Mat_<cv::Vec3f> tilePts, tileNrm;
                genTile(surf, cv::Size(int(dx), int(dy)), renderScale, u0, v0, tilePts, tileNrm);

                // 2. Prepare base coords and step directions
                cv::Mat_<cv::Vec3f> base, dirs;
                prepareBaseAndDirs(tilePts, tileNrm, scaleSeg, levelMap, hasAffine, aff, base, dirs);

                // 3. Sample all slices for this tile
                std::vector<cv::Mat_<T>> raw;
                if (isComposite) {
                    // Composite mode: always u8 (the caller rejects uint16 sources).
                    if constexpr (std::is_same_v<T, uint8_t>)
                        raw = {sampleComposite(volume, base, dirs, compositeOffsets, compositeMethod)};
                } else {
                    raw = sampleLayers<T>(volume, type, base, dirs, allOffsets);
                }

                // Accumulate (no rotation: applied per zarr chunk and per TIFF row separately).
                // A composite is one image already.
                std::vector<cv::Mat> slices = isComposite
                    ? std::vector<cv::Mat>(raw.begin(), raw.end())
                    : processRawSlices<T>(raw, numSlices, accumOffsets, accumType, cvType);

                // 4. Pack into zarr chunk (with rotation applied to pixel data)
                if (dsOut) {
                    size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];
                    size_t numZ = slices.size();
                    const bool needsRotFlip = (rotQuad >= 0 || flipAxis >= 0);

                    // Only clone+rotate when rotation/flip is active
                    const std::vector<cv::Mat>* zarrSlices = &slices;
                    std::vector<cv::Mat> rotSlices;
                    if (needsRotFlip) {
                        rotSlices.resize(numZ);
                        for (size_t zi = 0; zi < numZ; zi++) {
                            rotSlices[zi] = slices[zi].clone();
                            rotateFlipIfNeeded(rotSlices[zi], rotQuad, flipAxis);
                        }
                        zarrSlices = &rotSlices;
                    }

                    int dstTx = int(tx), dstTy = int(ty), dTX, dTY;
                    if (needsRotFlip)
                        mapTileIndex(int(tx), int(ty), int(tilesXSrc), int(tilesYSrc),
                                     std::max(rotQuad, 0), flipAxis, dstTx, dstTy, dTX, dTY);

                    std::vector<T> chunkBuf(chunkZ * chunkY * chunkX, T(0));
                    size_t dy_actual = std::min(chunkY, size_t((*zarrSlices)[0].rows));
                    size_t dx_actual = std::min(chunkX, size_t((*zarrSlices)[0].cols));
                    for (size_t zi = 0; zi < numZ; zi++) {
                        size_t sliceOff = zi * chunkY * chunkX;
                        for (size_t yy = 0; yy < dy_actual; yy++) {
                            const T* row = (*zarrSlices)[zi].ptr<T>(int(yy));
                            std::memcpy(&chunkBuf[sliceOff + yy * chunkX], row, dx_actual * sizeof(T));
                        }
                    }
                    dsOut->writeChunkSkipEmpty({0, dstTy, dstTx},
                                               reinterpret_cast<const uint8_t*>(chunkBuf.data()), chunkBuf.size() * sizeof(T));

                    // Scatter L0 tile into L1 pyramid accumulation buffer
                    // No rotation when inline pyramid is active, so tx/ty == dstTx/dstTy.
                    // L0 tile (ty, tx) maps to L1 pyramid chunk (ty/2, tx/2)
                    // at sub-offset ((ty%2)*halfCH, (tx%2)*halfCW) within the L1 chunk
                    if (!pyrAccum.empty()) {
                        size_t l1cx = size_t(tx) >> 1;
                        size_t halfCH = CH / 2, halfCW = CW / 2;
                        size_t offY = (size_t(ty) & 1) * halfCH;
                        size_t offX = (size_t(tx) & 1) * halfCW;
                        auto& pa = pyrAccum[0];
                        if (l1cx < pa.bufs.size()) {
                            downsampleTileIntoPreserveZ(
                                chunkBuf.data(), chunkZ, chunkY, chunkX,
                                pa.bufs[l1cx].data(), pa.chZ, pa.chY, pa.chX,
                                numZ, dy_actual, dx_actual,
                                offY, offX);
                        }
                    }
                }

                // 5. Store unrotated slices for TIF assembly
                if (wantTif) {
                    tifRowBuf[tx] = std::move(slices);
                }
            } catch (...) {
                #pragma omp critical(render_tile_error)
                {
                    if (!rowError) rowError = std::current_exception();
                }
                rowFailed.store(true, std::memory_order_relaxed);
            }
        }
        if (rowError) std::rethrow_exception(rowError);

        // After all tx done for this ty: assemble TIF if needed
        if (wantTif) {
            int outSlices = isComposite ? 1 : numSlices;
            // Assemble full-width unrotated rows from tile columns
            std::vector<cv::Mat> fullWidthSlices(outSlices);
            for (int zi = 0; zi < outSlices; zi++) {
                fullWidthSlices[zi] = cv::Mat(int(dy), tgtSize.width, cvType, cv::Scalar(0));
                for (uint32_t tx = 0; tx < numTileCols; tx++) {
                    uint32_t x0t = tx * uint32_t(CW);
                    const cv::Mat& tileMat = tifRowBuf[tx][zi];
                    tileMat.copyTo(fullWidthSlices[zi](cv::Rect(int(x0t), 0, tileMat.cols, tileMat.rows)));
                }
                // Apply rotation/flip to assembled full-width row
                rotateFlipIfNeeded(fullWidthSlices[zi], rotQuad, flipAxis);
            }

            if (quickTif && !useU16) {
                for (auto& s : fullWidthSlices)
                    for (int r = 0; r < s.rows; r++) {
                        auto* row = s.ptr<uint8_t>(r);
                        for (int c = 0; c < s.cols; c++)
                            row[c] &= 0xF0;
                    }
            }
            writeTifRows(*tifWriters, fullWidthSlices, y0, tiffTileH, uint32_t(tgtSize.height), rotQuad, flipAxis);
        }

        // ---- Pyramid row-group flush ----
        // At each level L (1..5), a pyramid chunk covers 2^L tile rows.
        // When the current tile-row completes a group, flush & cascade.
        if (!pyrAccum.empty()) {
            // No rotation/flip when inline pyramid is active (ensured by caller),
            // so tile-row ty maps directly to dest tile-row ty.
            uint32_t relRow = ty - tyStart + 1;
            bool isLastRow = (ty == tyEnd - 1);

            for (size_t li = 0; li < pyrAccum.size(); li++) {
                uint32_t groupSize = 1u << (li + 1);  // L1=2, L2=4, L3=8, L4=16, L5=32
                bool groupComplete = (relRow % groupSize == 0) || isLastRow;
                if (!groupComplete) continue;

                auto& pa = pyrAccum[li];
                // tile-row ty divided by 2^(li+1) gives the pyramid chunk row
                size_t pyrChunkRow = size_t(ty) >> (li + 1);

                // Write each column's accumulation buffer as a zarr chunk
                for (size_t cx = 0; cx < pa.bufs.size(); cx++) {
                    pyramidDs[li]->writeChunkSkipEmpty({0, int64_t(pyrChunkRow), int64_t(cx)},
                        reinterpret_cast<const uint8_t*>(pa.bufs[cx].data()), pa.bufs[cx].size() * sizeof(T));
                }

                // Cascade: scatter this level's buffers into the next level's accum
                if (li + 1 < pyrAccum.size()) {
                    auto& nextPa = pyrAccum[li + 1];
                    size_t halfY = pa.chY / 2, halfX = pa.chX / 2;
                    for (size_t cx = 0; cx < pa.bufs.size(); cx++) {
                        size_t nextCx = cx >> 1;
                        size_t offY = (pyrChunkRow & 1) * halfY;
                        size_t offX = (cx & 1) * halfX;
                        if (nextCx < nextPa.bufs.size()) {
                            downsampleTileIntoPreserveZ(
                                pa.bufs[cx].data(), pa.chZ, pa.chY, pa.chX,
                                nextPa.bufs[nextCx].data(), nextPa.chZ, nextPa.chY, nextPa.chX,
                                pa.chZ, pa.chY, pa.chX,
                                offY, offX);
                        }
                    }
                }

                // Clear this level's buffers for next group
                for (auto& b : pa.bufs)
                    std::fill(b.begin(), b.end(), T(0));
            }
        }

        // Progress (throttled to ~1/sec)
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - lastPrint).count();
        uint32_t tileRowsThis = tyEnd - tyStart;
        uint32_t done = ty - tyStart + 1;
        if (since >= 1.0 || done == tileRowsThis) {
            lastPrint = now;
            double elapsed = std::chrono::duration<double>(now - wallStart).count();
            double eta = done > 0 ? elapsed * (double(tileRowsThis) / done - 1.0) : 0.0;
            // Chunks written this part: done tile-rows × numTileCols chunks per row
            double chunksWritten = double(done) * double(numTileCols);
            double chunksPerSec = elapsed > 0 ? chunksWritten / elapsed : 0.0;
            const char* prefix = g_logFile ? "  " : "\r  ";
            const char* suffix = g_logFile ? "\n" : "";
            logPrintf(stderr, "%stile-row %u/%u (%d%%)  %.1f chunks/s  %dm%02ds  eta %dm%02ds%s",
                prefix, done, tileRowsThis, int(100.0 * done / tileRowsThis),
                chunksPerSec,
                int(elapsed)/60, int(elapsed)%60, int(eta)/60, int(eta)%60, suffix);
        }
    }
    if (!g_logFile) std::fprintf(stderr, "\n");
}


// ============================================================
// Voxel size resolution
// ============================================================

// Where a usable physical voxel size came from. "Unspecified" means the value is
// a placeholder: callers must not present it as a physical measurement.
enum class VoxelSizeSource {
    Cli,
    LocalStoreMetadata,
    RemoteVolume,
    Unspecified,
};

struct ResolvedVoxelSize {
    double micrometerPerVoxel = 1.0;   // always micrometres per voxel
    VoxelSizeSource source = VoxelSizeSource::Unspecified;

    [[nodiscard]] bool isUsable() const
    {
        return source != VoxelSizeSource::Unspecified;
    }
};

static const char* voxelSizeSourceName(VoxelSizeSource source)
{
    switch (source) {
    case VoxelSizeSource::Cli:                return "command line";
    case VoxelSizeSource::LocalStoreMetadata: return "local store metadata";
    case VoxelSizeSource::RemoteVolume:       return "remote volume metadata";
    case VoxelSizeSource::Unspecified:        return "unspecified";
    }
    return "unspecified";
}

// The OME-Zarr axis unit and the numeric value that must accompany it.
//
// These travel as a PAIR: `writeZarrAttrs` writes `scale = <value> / pixelsPerVoxel`
// and `axis.unit = <unit>` independently, with nothing checking that they agree. A
// micrometre number labelled "nanometer" is therefore a silent 1000x error -- the
// same class of defect this patch exists to remove.
//
// The value resolved internally is always micrometres. How it is *declared* depends
// on where it came from:
//
//   * explicit CLI value -- declared in the caller's own unit, exactly as the
//     pre-patch renderer did. `--voxel-unit` has always meant "the unit of the
//     number I am giving you" (the pre-patch code multiplied by 0.001 for
//     `nanometer` to get micrometres for the TIFF tag), and the flag's help text
//     describes the number the caller supplies. Preserving that is what keeps
//     `--voxel-size 8640 --voxel-unit nanometer` meaning 8640 nm, as it did before.
//   * store metadata / the open volume -- both are micrometres by definition, so
//     both are declared as micrometres.
//
// In every case `.zattrs` and the TIFF resolution describe the same physical size.
static const char* zarrVoxelUnit(const ResolvedVoxelSize& resolved,
                                 const std::string& explicitUnit)
{
    switch (resolved.source) {
    case VoxelSizeSource::Cli:                return explicitUnit.c_str();
    case VoxelSizeSource::LocalStoreMetadata:
    case VoxelSizeSource::RemoteVolume:
        // Both resolve to micrometres by definition: the store document's
        // `voxelsize` is µm and Volume::voxelSize() is µm.
        return "micrometer";
    case VoxelSizeSource::Unspecified:        return nullptr;
    }
    return nullptr;
}

// The number that `unit` describes, for the same resolved size.
//
// For a CLI value this is what the caller typed. Otherwise the resolved value is in
// micrometres and so is the unit, so the number passes through unchanged.
static double zarrVoxelValue(const ResolvedVoxelSize& resolved,
                             double explicitValue)
{
    return (resolved.source == VoxelSizeSource::Cli) ? explicitValue
                                                     : resolved.micrometerPerVoxel;
}

// Convert a --voxel-size / --voxel-unit pair to micrometres per voxel.
//
// nullopt when the value is not positive and finite, or when the unit is not one
// we know how to convert: a unit we cannot read is a user error, not a missing
// measurement, and guessing it is a silent 1000x mistake.
static std::optional<double> explicitMicrometerPerVoxel(double value,
                                                        const std::string& unit)
{
    if (!std::isfinite(value) || value <= 0.0)
        return std::nullopt;

    if (unit == "nanometer" || unit == "nanometre" || unit == "nm")
        return value * 0.001;
    if (unit == "micrometer" || unit == "micrometre" || unit == "um" ||
        unit == "\xC2\xB5m")
        return value;
    if (unit == "millimeter" || unit == "millimetre" || unit == "mm")
        return value * 1000.0;
    if (unit == "meter" || unit == "metre" || unit == "m")
        return value * 1000000.0;
    return std::nullopt;
}

// Resolve the physical voxel size this render must use, in micrometres.
//
// The store's own document is the authority, and it is read the way every other
// VC3D tool reads it -- vc::metadata::resolveLocalStoreVoxelSize(), which
// Volume::NewFromUrl() and vc_grow_seg_from_segments already share. The reader
// that used to live here looked only for a top-level "voxelsize", so a store
// publishing its resolution as an acquisition record resolved to nothing and the
// render continued at a scale of 1.0 with no physical meaning.
//
// Order, and why it is this order:
//
//   1. the opened volume, when there is one. Volume construction has already
//      fetched and normalised the store's own document, so this is the value the
//      rest of the process agrees on: no extra request, and it cannot disagree
//      with the volume actually being streamed.
//   2. the local store document, for a run with no Volume open.
//
// The opened volume comes first because --volume is often a chunk cache for a
// remote source rather than the store itself; where such a cache mirrors the
// store's metadata it may be stale, and it is the streamed volume, not the cache
// directory, that is being rendered. Consulting the cache first would let a stale
// document win over the freshly fetched one.
static ResolvedVoxelSize resolveRenderVoxelSize(
    const std::filesystem::path& volPath,
    const Volume* remoteVolume,
    bool hasExplicitVoxelSize,
    double explicitValue,
    const std::string& explicitUnit)
{
    if (hasExplicitVoxelSize) {
        // Validated by the caller, which errors out on a bad pair; this is not a
        // fallback for an unusable --voxel-size, and must not become one.
        if (const auto cli = explicitMicrometerPerVoxel(explicitValue, explicitUnit))
            return {*cli, VoxelSizeSource::Cli};
    }

    if (remoteVolume != nullptr) {
        const double fromVolume = remoteVolume->voxelSize();
        if (std::isfinite(fromVolume) && fromVolume > 0.0)
            return {fromVolume, VoxelSizeSource::RemoteVolume};
    }

    if (const auto local = vc::metadata::resolveLocalStoreVoxelSize(volPath))
        return {*local, VoxelSizeSource::LocalStoreMetadata};

    return {};
}

// ============================================================
// main
// ============================================================

int main(int argc, char *argv[])
{
    // clang-format off
    po::options_description required("Required arguments");
    required.add_options()
        ("volume,v", po::value<std::string>(), "Local OME-Zarr volume directory, or a remote OME-Zarr URL (http(s)://, s3://) to stream; remote chunks persist in the shared remote cache")
        ("scale", po::value<float>()->required(), "Pixels per level-g voxel (Pg)")
        ("group-idx,g", po::value<int>()->required(), "OME-Zarr group index");

    po::options_description optional("Optional arguments");
    optional.add_options()
        ("help,h", "Show this help message")
        ("segmentation,s", po::value<std::string>(), "Path to a single tifxyz segmentation folder")
        ("cache-gb", po::value<size_t>()->default_value(16), "Zarr chunk cache size in GB")
        ("remote-url", po::value<std::string>(), "[DEPRECATED] Same as passing the remote OME-Zarr URL as --volume")
        ("log-path", po::value<std::string>(), "Log all output to file instead of stdout/stderr")
        ("timeout", po::value<int>()->default_value(0), "Kill process if not finished within N minutes")
        ("num-slices,n", po::value<int>()->default_value(1), "Number of slices to render")
        ("slice-step", po::value<float>()->default_value(1.0f), "Spacing between slices along normal")
        ("accum", po::value<float>()->default_value(0.0f), "Accumulation sub-step (0 = disabled)")
        ("accum-type", po::value<std::string>()->default_value("max"),
            "Reducer: max, mean, median, alpha, beerlambert. max/mean/median only take effect with "
            "--accum or --composite-collapse; alpha and beerlambert always collapse the sampled layers into one "
            "composite image (see --composite-collapse).")
        ("crop-x", po::value<int>()->default_value(0), "Crop X") ("crop-y", po::value<int>()->default_value(0), "Crop Y")
        ("crop-width", po::value<int>()->default_value(0), "Crop width") ("crop-height", po::value<int>()->default_value(0), "Crop height")
        ("auto-crop", po::bool_switch()->default_value(false), "Auto-crop to valid surface bbox")
        ("affine", po::value<std::vector<std::string>>()->multitoken()->composing(), "Affine JSON files (append :inv to invert)")
        ("affine-invert", po::value<std::vector<int>>()->multitoken()->composing(), "0-based indices to invert")
        ("scale-segmentation", po::value<float>()->default_value(1.0), "Scale segmentation")
        ("rotate", po::value<double>()->default_value(0.0), "Rotate output (0/90/180/270)")
        ("flip", po::value<int>()->default_value(-1), "Flip: 0=V, 1=H, 2=Both")
        // The geometric normal is N = dP/dU x dP/dV (see grid_normal in Geometry.cpp),
        // so (U, V, N) is right-handed. With the standard segment orientation (V down-screen
        // aligned with +z, text readable, U winding outside->inside) that normal points toward
        // the OUTSIDE of the scroll, i.e. behind the surface as the reader views it. Our
        // convention is the opposite: w / layer number should increase going IN FRONT of the
        // surface (toward the viewer / toward the scroll center), which makes (U, V, w) a
        // left-handed frame. So --flip-normals is usually what we want: it negates N so the
        // slice stack grows in front of the sheet rather than behind it.
        ("flip-normals", po::bool_switch()->default_value(false), "Negate surface normals (reverses slice ordering along the normal)")
        // How the surface itself is resampled, as opposed to how the volume is
        // sampled along the normal (which is always trilinear here).
        ("surface-interpolation", po::value<std::string>()->default_value("linear"),
            "Surface resampling: linear (bilinear positions, nearest-neighbour normals) or "
            "smooth/bicubic (Catmull-Rom positions with normals differentiated from the same "
            "basis). The source grid is far coarser than the render, so linear leaves the normal "
            "constant across each grid cell and every offset layer steps at the cell edges; "
            "smooth removes those steps and the creases between them, at ~3-5x the surface-warp "
            "cost (a few percent end to end).")
        ("zarr-output", po::value<std::string>(), "Output path for .zarr (optional)")
        ("zarr-compressor", po::value<std::string>()->default_value("blosc"), "Zarr compressor: blosc, zstd, gzip, lz4, none")
        ("zarr-compression-level", po::value<int>()->default_value(-1), "Zarr compression level (<=0 = compressor default)")
        ("zarr-separator", po::value<std::string>()->default_value("/"), "Zarr chunk dimension separator: / or .")
        ("tif-output", po::value<std::string>(), "Output path for per-slice TIFFs (optional)")
        ("quick-tif", po::bool_switch()->default_value(false), "Fast TIF: PACKBITS + zero low nibble")
        ("alpha-min", po::value<float>()->default_value(0.0f), "Alpha min (0-255)")
        ("alpha-max", po::value<float>()->default_value(255.0f), "Alpha max (0-255)")
        ("alpha-opacity", po::value<float>()->default_value(230.0f), "Alpha opacity (0-255)")
        ("alpha-cutoff", po::value<float>()->default_value(9950.0f), "Alpha cutoff (0-10000)")
        ("bl-extinction", po::value<float>()->default_value(1.5f), "Beer-Lambert extinction")
        ("bl-emission", po::value<float>()->default_value(1.5f), "Beer-Lambert emission")
        ("bl-ambient", po::value<float>()->default_value(0.1f), "Beer-Lambert ambient")
        ("iso-cutoff", po::value<int>()->default_value(0), "Highpass (0-255)")
        ("composite-start", po::value<int>(), "Composite start offset")
        ("composite-end", po::value<int>(), "Composite end offset")
        // Named '--composite-collapse' rather than '--composite' so it is not an abbreviation
        // prefix of --composite-start/--composite-end under boost's allow_guessing.
        ("composite-collapse", po::bool_switch()->default_value(false),
            "Collapse the sampled layers into a single image using --accum-type as the reducer "
            "(max/mean/median, in addition to the implicit alpha/beerlambert). The layers span "
            "[--composite-start, --composite-end] layers along the normal at --slice-step spacing; "
            "use a symmetric range (e.g. --composite-start=-54 --composite-end=54) for a centered "
            "projection.")
        ("num-parts", po::value<int>()->default_value(1), "Parts for multi-VM")
        ("part-id", po::value<int>()->default_value(0), "Part ID (0-indexed)")
        ("merge-tiff-parts", po::bool_switch()->default_value(false), "Merge partial TIFFs from multi-VM render")
        ("pyramid", po::value<bool>()->default_value(true), "Build pyramid levels L1-L5 (default: true)")
        ("resume", po::bool_switch()->default_value(false), "Skip chunks that already exist on disk")
        ("pre", po::bool_switch()->default_value(false), "Create zarr + all level datasets")
        ("voxel-size", po::value<double>(), "Physical voxel size for OME-Zarr scale metadata (reads from volume metadata if omitted)")
        ("voxel-unit", po::value<std::string>()->default_value("nanometer"), "Unit of the number given to --voxel-size (nanometer, micrometer, millimeter, meter). Sizes read from volume metadata are always declared in micrometer.");
    StoreToolOptions::add(optional, "Object store holding the --surface to render; with --volume-uuid it is read in "
                                    "that volume's voxels through the store's view of it (its binding or a "
                                    "registration), replacing --affine and --scale-segmentation",
                          {.surfaces = true, .creates = false});
    // clang-format on

    po::options_description all("Usage");
    all.add(required).add(optional);
    po::variables_map parsed;
    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);
        if (parsed.count("help") || argc < 2) {
            std::cout << "vc_render_tifxyz: Render volume data using segmentation surfaces\n\n" << all << '\n';
            return EXIT_SUCCESS;
        }
        po::notify(parsed);
    } catch (po::error& e) {
        std::cerr << "Error: " << e.what() << "\nUse --help for usage information\n";
        return EXIT_FAILURE;
    }

    // --- Log path setup ---
    // Join the flush thread and preserve buffered diagnostics on every return.
    struct LogFlusherGuard {
        ~LogFlusherGuard() { stopLogFlusher(); }
    } logFlusherGuard;
    if (parsed.count("log-path")) {
        const auto& logPath = parsed["log-path"].as<std::string>();
        g_logFile = std::fopen(logPath.c_str(), "a");
        if (!g_logFile) {
            std::cerr << "Error: cannot open log file: " << logPath << "\n";
            return EXIT_FAILURE;
        }
        startLogFlusher();
    }

    // --- Parse common options ---
    const int numParts = parsed["num-parts"].as<int>();
    const int partId = parsed["part-id"].as<int>();
    if (numParts < 1 || partId < 0 || partId >= numParts) {
        logPrintf(stderr, "Error: need 0 <= part-id < num-parts\n"); return EXIT_FAILURE;
    }
    if (g_logFile && numParts > 1) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "[part %d/%d] ", partId, numParts);
        g_logPrefix = buf;
    }
    if (numParts > 1) logPrintf(stdout, "Multi-part mode: part %d of %d\n", partId, numParts);

    const bool pre_flag = parsed["pre"].as<bool>();
    const bool wantPyramid = parsed["pyramid"].as<bool>();
    const bool resumeFlag = parsed["resume"].as<bool>();

    const bool wantZarr = parsed.count("zarr-output") > 0;
    const bool wantTif = parsed.count("tif-output") > 0;
    if (!wantZarr && !wantTif) { logPrintf(stderr, "Error: at least one of --zarr-output or --tif-output required\n"); return EXIT_FAILURE; }
    const std::string zarrOutputArg = wantZarr ? parsed["zarr-output"].as<std::string>() : "";
    const std::string tifOutputArg = wantTif ? parsed["tif-output"].as<std::string>() : "";

    // Zarr output codec / layout
    std::string zarrCompressor = parsed["zarr-compressor"].as<std::string>();
    const int zarrCompressionLevel = parsed["zarr-compression-level"].as<int>();
    const std::string zarrSeparator = parsed["zarr-separator"].as<std::string>();
    {
        static const std::set<std::string> kComp{"blosc", "zstd", "gzip", "lz4", "none"};
        if (kComp.find(zarrCompressor) == kComp.end()) {
            logPrintf(stderr, "Error: --zarr-compressor must be one of blosc, zstd, gzip, lz4, none\n");
            return EXIT_FAILURE;
        }
        if (zarrSeparator != "/" && zarrSeparator != ".") {
            logPrintf(stderr, "Error: --zarr-separator must be \"/\" or \".\"\n");
            return EXIT_FAILURE;
        }
    }

    const bool mergeTiffFlag = parsed["merge-tiff-parts"].as<bool>();
    if (mergeTiffFlag) {
        if (numParts < 2) { logPrintf(stderr, "Error: --merge-tiff-parts needs --num-parts >= 2\n"); return EXIT_FAILURE; }
        if (!wantTif) { logPrintf(stderr, "Error: --merge-tiff-parts requires --tif-output\n"); return EXIT_FAILURE; }
        return mergeTiffParts(tifOutputArg, numParts) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::string volumeSource = parsed.count("volume") ? parsed["volume"].as<std::string>() : "";
    if (parsed.count("remote-url")) {
        const std::string remoteUrlArg = parsed["remote-url"].as<std::string>();
        if (volumeSource.empty()) {
            volumeSource = remoteUrlArg;
        } else if (volumeSource != remoteUrlArg) {
            logPrintf(stderr, "Error: --volume (%s) and --remote-url (%s) name different sources; --remote-url is deprecated, pass the remote URL as --volume\n",
                      volumeSource.c_str(), remoteUrlArg.c_str());
            return EXIT_FAILURE;
        }
        logPrintf(stderr, "Warning: --remote-url is deprecated, pass the remote URL as --volume\n");
    }
    if (volumeSource.empty()) { logPrintf(stderr, "Error: --volume required\n"); return EXIT_FAILURE; }
    std::filesystem::path vol_path = volumeSource;
    const std::string remoteUrl = volumeSource.find("://") != std::string::npos ? volumeSource : std::string{};
    const bool useRemoteCache = !remoteUrl.empty();

    std::optional<StoreToolOptions> store_options;
    try {
        store_options = StoreToolOptions::from(parsed);
    } catch (const std::exception& e) {
        logPrintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const bool store_mode = store_options.has_value();
    if (store_mode ? store_options->surfaces.size() != 1 || parsed.count("segmentation") : !parsed.count("segmentation")) {
        logPrintf(stderr, "Error: give --segmentation, or --store with one --surface\n");
        return EXIT_FAILURE;
    }
    const std::optional<vc::objects::VolumeVoxels> view_volume = store_mode ? store_options->volume : std::nullopt;
    if (view_volume && (parsed.count("affine") || parsed.count("affine-invert") || !parsed["scale-segmentation"].defaulted())) {
        logPrintf(stderr, "Error: --volume-uuid replaces --affine and --scale-segmentation\n");
        return EXIT_FAILURE;
    }
    std::filesystem::path seg_path = store_mode ? store_options->surfaces.front() : parsed["segmentation"].as<std::string>();

    float tgt_scale = parsed["scale"].as<float>();
    int group_idx = parsed["group-idx"].as<int>();
    int num_slices = parsed["num-slices"].as<int>();
    double slice_step = parsed["slice-step"].as<float>();
    if (!std::isfinite(slice_step) || slice_step <= 0) { logPrintf(stderr, "Error: --slice-step must be positive\n"); return EXIT_FAILURE; }

    double accum_step = parsed["accum"].as<float>();
    if (!std::isfinite(accum_step) || accum_step < 0) { logPrintf(stderr, "Error: --accum must be non-negative\n"); return EXIT_FAILURE; }

    std::string accum_type_str = parsed["accum-type"].as<std::string>();
    std::transform(accum_type_str.begin(), accum_type_str.end(), accum_type_str.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });

    AccumType accumType;
    if      (accum_type_str == "max")    accumType = AccumType::Max;
    else if (accum_type_str == "mean")   accumType = AccumType::Mean;
    else if (accum_type_str == "median") accumType = AccumType::Median;
    else if (accum_type_str == "alpha")  accumType = AccumType::Alpha;
    else if (accum_type_str == "beerlam" || accum_type_str == "beerlambert") accumType = AccumType::BeerLambert;
    else { logPrintf(stderr, "Error: invalid --accum-type\n"); return EXIT_FAILURE; }

    std::string surf_interp_str = parsed["surface-interpolation"].as<std::string>();
    std::transform(surf_interp_str.begin(), surf_interp_str.end(), surf_interp_str.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
    GenInterpolation genInterp;
    if      (surf_interp_str == "linear") genInterp = GenInterpolation::Linear;
    else if (surf_interp_str == "smooth" || surf_interp_str == "bicubic")
        genInterp = GenInterpolation::Smooth;
    else { logPrintf(stderr, "Error: invalid --surface-interpolation\n"); return EXIT_FAILURE; }

    // alpha/beerlambert reducers only make sense over collapsed layers, so they always imply
    // composite mode. --composite-collapse extends the same collapsing path to max/mean/median,
    // which the native compositor (vcr::composite::reduce) already implements.
    const bool requestComposite = parsed["composite-collapse"].as<bool>();
    const bool isCompositeMode = requestComposite
        || accumType == AccumType::Alpha || accumType == AccumType::BeerLambert;

    // Without layers to reduce, processRawSlices passes each raw slice through and never reads
    // accumType, so an explicit max/mean/median here changes nothing. Say so rather than stay quiet.
    if (accum_step <= 0 && !isCompositeMode
        && parsed.count("accum-type") > 0 && !parsed["accum-type"].defaulted())
        logPrintf(stderr, "Warning: --accum-type %s has no effect without --accum or --composite-collapse\n",
                  accum_type_str.c_str());
    int compositeStart = 0, compositeEnd = num_slices - 1;

    vcr::composite::Options compositeMethod;
    if (isCompositeMode) {
        using Method = vcr::composite::Method;
        switch (accumType) {
            case AccumType::Alpha:       compositeMethod.method = Method::Alpha; break;
            case AccumType::BeerLambert: compositeMethod.method = Method::BeerLambert; break;
            case AccumType::Max:         compositeMethod.method = Method::Max; break;
            case AccumType::Mean:        compositeMethod.method = Method::Mean; break;
            case AccumType::Median:      compositeMethod.method = Method::Median; break;
        }
        compositeMethod.alphaMin = parsed["alpha-min"].as<float>() / 255.0f;
        compositeMethod.alphaMax = parsed["alpha-max"].as<float>() / 255.0f;
        compositeMethod.alphaOpacity = parsed["alpha-opacity"].as<float>() / 255.0f;
        compositeMethod.alphaCutoff = parsed["alpha-cutoff"].as<float>() / 10000.0f;
        compositeMethod.extinction = parsed["bl-extinction"].as<float>();
        compositeMethod.emission = parsed["bl-emission"].as<float>();
        compositeMethod.ambient = parsed["bl-ambient"].as<float>();
        compositeMethod.isoCutoff = float(std::clamp(parsed["iso-cutoff"].as<int>(), 0, 255));
        if (parsed.count("composite-start")) compositeStart = parsed["composite-start"].as<int>();
        if (parsed.count("composite-end"))   compositeEnd = parsed["composite-end"].as<int>();
        if (compositeEnd < compositeStart) { logPrintf(stderr, "Error: --composite-end < --composite-start\n"); return EXIT_FAILURE; }
        int layers = compositeEnd - compositeStart + 1;
        if (compositeMethod.method == Method::Median && std::size_t(layers) > vcr::composite::kMaxMedianLayers) {
            logPrintf(stderr, "Error: a median composite takes at most %zu layers\n", vcr::composite::kMaxMedianLayers);
            return EXIT_FAILURE;
        }
        logPrintf(stdout, "Composite: %s (%d layers [%d..%d])\n", accum_type_str.c_str(), layers, compositeStart, compositeEnd);
        if (compositeMethod.method == Method::Alpha)
            logPrintf(stdout, "  alpha: min=%.0f max=%.0f opacity=%.0f cutoff=%.0f\n",
                      compositeMethod.alphaMin*255, compositeMethod.alphaMax*255,
                      compositeMethod.alphaOpacity*255, compositeMethod.alphaCutoff*10000);
        else if (compositeMethod.method == Method::BeerLambert)
            logPrintf(stdout, "  BL: ext=%.1f em=%.1f amb=%.1f\n",
                      compositeMethod.extinction, compositeMethod.emission, compositeMethod.ambient);
        if (compositeMethod.isoCutoff > 0) logPrintf(stdout, "  iso cutoff: %d\n", int(compositeMethod.isoCutoff));
    }

    std::vector<float> accumOffsets;
    if (accum_step > 0) {
        if (accum_step > slice_step) { logPrintf(stderr, "Error: --accum > --slice-step\n"); return EXIT_FAILURE; }
        double ratio = slice_step / accum_step, rounded = std::round(ratio);
        if (std::abs(ratio - rounded) > 1e-4) { logPrintf(stderr, "Error: --accum must evenly divide --slice-step\n"); return EXIT_FAILURE; }
        size_t samples = std::max<size_t>(1, size_t(rounded));
        double spacing = slice_step / samples;
        for (size_t i = 0; i < samples; i++) accumOffsets.push_back(float(spacing * i));
        accum_step = spacing;
        logPrintf(stdout, "Accumulation: %zu samples/slice at step %.4f (%s)\n", samples, spacing, accum_type_str.c_str());
    }

    const float ds_scale = std::ldexp(1.0f, -group_idx);
    float scale_seg = parsed["scale-segmentation"].as<float>();
    double rotate_angle = parsed["rotate"].as<double>();
    int flip_axis = parsed["flip"].as<int>();
    g_flipNormals = parsed["flip-normals"].as<bool>();
    const bool quickTif = parsed["quick-tif"].as<bool>();

    // --- Load affines ---
    Affine affineTransform;
    bool hasAffine = false;
    std::vector<std::pair<std::string,bool>> affineSpecs;
    if (parsed.count("affine"))
        for (const auto& s : parsed["affine"].as<std::vector<std::string>>())
            affineSpecs.push_back(parseAffineSpec(s));
    if (parsed.count("affine-invert") && !affineSpecs.empty()) {
        std::set<int> inv;
        for (int idx : parsed["affine-invert"].as<std::vector<int>>()) {
            if (idx < 0 || idx >= int(affineSpecs.size())) { logPrintf(stderr, "Error: --affine-invert index out of range\n"); return EXIT_FAILURE; }
            inv.insert(idx);
        }
        for (int k = 0; k < int(affineSpecs.size()); k++) if (inv.count(k)) affineSpecs[k].second = true;
    }
    if (!affineSpecs.empty()) {
        Affine composed;
        for (int k = 0; k < int(affineSpecs.size()); k++) {
            auto& [path, inv] = affineSpecs[k];
            try {
                auto T = Affine::loadTransformJson(path);
                logPrintf(stdout, "Loaded affine[%d]: %s%s\n", k, path.c_str(), inv ? " (invert)" : "");
                composed = (inv ? T.inverse() : T) * composed;
            } catch (const std::exception& e) { logPrintf(stderr, "Error loading affine[%d]: %s\n", k, e.what()); return EXIT_FAILURE; }
        }
        hasAffine = true;
        affineTransform = composed;
        printMat4x4(affineTransform, "Final composed affine:");
    }

    // Resolved below from the same CLI/metadata value used for OME-Zarr.
    float tifDpi = 0.f;

    // --- Open source volume ---
    const size_t cache_bytes = parsed["cache-gb"].as<size_t>() * 1024ull * 1024ull * 1024ull;
    if (const size_t mem = usableMemoryBytes(); mem > 0 && cache_bytes >= mem)
        logPrintf(stderr, "Warning: --cache-gb %llu is not below the memory available to this process (%.1f GB): the render can stall without any output. Lower --cache-gb.\n",
                  (unsigned long long)parsed["cache-gb"].as<size_t>(), double(mem) / (1024.0 * 1024.0 * 1024.0));

    // A remote volume opens through Volume so the renderer shares VC3D's remote
    // cache: the globally configured cache root and the URL-derived source
    // identity, with fetched chunks persisted. Group g is the volume's working
    // level g, i.e. physical level g + baseScaleLevel().
    std::shared_ptr<Volume> remoteVolume;
    std::string sourceLocation = vol_path.string();
    vcr::zarr::OpenOptions sourceOptions;
    int workingLevel = 0;
    if (useRemoteCache) {
        try {
            remoteVolume = Volume::NewFromUrl(remoteUrl, vc::HttpAuth::from_env());
        } catch (const std::exception& e) {
            logPrintf(stderr, "Error opening remote zarr: %s\n", e.what());
            return EXIT_FAILURE;
        }
        sourceLocation = remoteVolume->sourceLocation();
        sourceOptions = remoteVolume->sourceOpenOptions();
        workingLevel = remoteVolume->baseScaleLevel();
        logPrintf(stdout, "Remote zarr streaming: %s\n", remoteUrl.c_str());
    }
    // The array keeps decoded chunks in its own memory cache of --cache-gb.
    sourceOptions.memoryCache = std::make_shared<vcr::MemoryCache>(cache_bytes);
    std::shared_ptr<vcr::zarr::OpenedArray> source;
    LevelMap levelMap;
    try {
        source = std::make_shared<vcr::zarr::OpenedArray>(
            vcr::zarr::openPyramidLevel(sourceLocation, workingLevel + group_idx, sourceOptions));
        const auto working = group_idx == 0
            ? source->placement
            : vcr::zarr::openPyramidLevel(sourceLocation, workingLevel, sourceOptions).placement;
        levelMap = mapToLevel(working, source->placement);
    } catch (const std::exception& e) {
        logPrintf(stderr, "Error opening %s zarr: group index %d: %s\n", useRemoteCache ? "remote" : "local",
                  group_idx, e.what());
        return EXIT_FAILURE;
    }
    const vcr::ElementType sourceType = source->meta.type;
    if (source->meta.shape.size() != 3 || sourceType.kind != 'u' ||
        (sourceType.itemsize != 1 && sourceType.itemsize != 2)) {
        logPrintf(stderr, "Error: the renderer reads 3-D uint8 or uint16 volumes, not %s\n",
                  source->meta.dtype.c_str());
        return EXIT_FAILURE;
    }
    vcr::Array& volume = *source->array;

    {
        std::ostringstream oss;
        for (auto v : source->meta.shape) oss << v << " ";
        logPrintf(stdout, "zarr dataset size for group %d [%s]\n", group_idx, oss.str().c_str());
    }

    const bool output_is_u16 = sourceType.itemsize == 2;
    logPrintf(stdout, "Source dtype: %s\n", output_is_u16 ? "uint16" : "uint8");

    // The composite path is u8-only (renderTiles runs it under if constexpr T==uint8_t),
    // so on a uint16 source it would silently emit empty output. Fail fast instead.
    if (isCompositeMode && output_is_u16) {
        logPrintf(stderr, "Error: composite/alpha rendering requires a uint8 volume (source is uint16)\n");
        return EXIT_FAILURE;
    }
    {
        std::ostringstream oss;
        for (auto v : source->meta.chunks) oss << v << " ";
        logPrintf(stdout, "chunk shape [%s]\n", oss.str().c_str());
    }

    // --- Resolve voxel size for OME-Zarr metadata ---
    //
    // The value itself is resolved below, once the source volume is open: a
    // streamed volume's own resolved metadata is the authoritative size, and
    // reusing it is both cheaper and impossible to disagree with. What is
    // settled here is only the command line.
    const std::string voxel_unit = parsed["voxel-unit"].as<std::string>();
    bool hasExplicitVoxelSize = false;
    double explicitVoxelSize = 0.0;
    if (parsed.count("voxel-size")) {
        explicitVoxelSize = parsed["voxel-size"].as<double>();
        if (!std::isfinite(explicitVoxelSize) || explicitVoxelSize <= 0.0) {
            logPrintf(stderr, "Error: --voxel-size must be a positive finite number\n");
            return EXIT_FAILURE;
        }
        if (!explicitMicrometerPerVoxel(explicitVoxelSize, voxel_unit)) {
            logPrintf(stderr, "Error: unsupported --voxel-unit: %s\n", voxel_unit.c_str());
            return EXIT_FAILURE;
        }
        hasExplicitVoxelSize = true;
    }

    // Filled in once the source volume is open.
    double base_voxel_size = 1.0;
    double render_level_voxel_size = 1.0;
    // The number and the unit that describe one level-g voxel *in .zattrs*. They
    // are written as a pair by writeZarrAttrs and nothing checks that they agree, so
    // they are computed together below. `base_voxel_size` is always micrometres;
    // these are in the caller's own unit for an explicit --voxel-size.
    //
    // The value starts at 0 = "unknown", which is what writeZarrAttrs needs in
    // order to omit the physical scale rather than declare a placeholder one.
    double zarr_voxel_value = 0.0;
    std::string zarr_voxel_unit = voxel_unit;

    // --- Resolve the physical voxel size -------------------------------------
    // The source volume is open by now, so a streamed volume's own metadata is
    // available to reuse. Computing this earlier is what made the renderer miss
    // remote resolutions entirely.
    {
        const ResolvedVoxelSize resolved = resolveRenderVoxelSize(
            vol_path, remoteVolume.get(), hasExplicitVoxelSize, explicitVoxelSize,
            voxel_unit);

        base_voxel_size = resolved.micrometerPerVoxel;

        // Emit the number and the unit as a matched pair.
        //
        // For store metadata and the opened volume the resolved value is
        // micrometres, so both are micrometres: passing --voxel-unit through
        // unchanged is what let `.zattrs` declare nanometre voxels for micrometre
        // data.
        //
        // For an explicit --voxel-size the caller's own number and unit are kept,
        // as the pre-patch renderer did. Emitting the *converted* micrometre number
        // under the caller's unit label is a silent 1000x error, and is exactly
        // what the first version of this patch did for
        // `--voxel-size 8640 --voxel-unit nanometer`.
        if (const char* unit = zarrVoxelUnit(resolved, voxel_unit)) {
            zarr_voxel_unit = unit;
            zarr_voxel_value = zarrVoxelValue(resolved, explicitVoxelSize);
        }

        if (resolved.isUsable()) {
            // Log exactly what .zattrs will declare, so the log line can be
            // checked against the output instead of needing its own conversion.
            logPrintf(stdout, "Voxel size (%s): %g %s\n",
                      voxelSizeSourceName(resolved.source), zarr_voxel_value,
                      zarr_voxel_unit.c_str());

            const double umPerOutputPixel = (ds_scale > 0 && tgt_scale > 0)
                ? base_voxel_size / double(ds_scale) / double(tgt_scale)
                : base_voxel_size;
            tifDpi = voxelSizeToDpi(umPerOutputPixel);
        } else {
            // There is no measurement to declare. Passing 0 tells writeZarrAttrs
            // that the physical size is unknown: it then writes no axis unit and
            // encodes only the relative pyramid scaling, keeping the multiscales
            // discovery metadata intact. A zero tifDpi leaves the TIFF resolution
            // tags unset. Both outputs therefore say "unknown" instead of
            // inventing a size, and neither loses the image structure.
            logPrintf(stderr,
                      "Warning: this volume publishes no usable voxel size, so its "
                      "physical scale is unknown and none will be declared. Pass "
                      "--voxel-size <value> --voxel-unit micrometer to supply one.\n");
            zarr_voxel_value = 0.0;
            zarr_voxel_unit.clear();
        }

        // Physical size of one *level-g* voxel, stated in `zarr_voxel_unit`. The
        // output raster is not necessarily isotropic in it: in-plane, one output
        // pixel spans 1/tgt_scale level-g voxels (--scale), and through-plane,
        // adjacent layers sit --slice-step level-g voxels apart (buildOffsetList).
        // writeZarrAttrs derives the per-axis .zattrs scale from this base value,
        // so this argument MUST be in the same unit as zarr_voxel_unit -- and 0
        // when there is no size to declare, which is the flag that switches
        // writeZarrAttrs from a physical scale to a relative one.
        render_level_voxel_size = ds_scale > 0
            ? zarr_voxel_value / double(ds_scale)
            : zarr_voxel_value;
    }

    int rotQuadGlobal = -1;
    if (std::abs(rotate_angle) > 1e-6) {
        rotQuadGlobal = normalizeQuadrantRotation(rotate_angle);
        if (rotQuadGlobal < 0) { logPrintf(stderr, "Error: only 0/90/180/270 rotations supported\n"); return EXIT_FAILURE; }
        rotate_angle = rotQuadGlobal * 90.0;
        logPrintf(stdout, "Rotation: %.0f degrees\n", rotate_angle);
    }
    if (flip_axis >= 0) logPrintf(stdout, "Flip: %s\n", flip_axis == 0 ? "V" : flip_axis == 1 ? "H" : "Both");
    if (g_flipNormals) logPrintf(stdout, "Flip normals: on\n");
    if (genInterp == GenInterpolation::Smooth)
        logPrintf(stdout, "Surface interpolation: smooth (bicubic)\n");

    if (wantZarr) {
        if (auto p = std::filesystem::path(zarrOutputArg).parent_path(); !p.empty())
            std::filesystem::create_directories(p);
    }
    if (wantTif) std::filesystem::create_directories(tifOutputArg);

    if (int t = parsed["timeout"].as<int>(); t > 0) {
        logPrintf(stdout, "Timeout: %d minutes\n", t);
        std::thread([t]{ std::this_thread::sleep_for(std::chrono::minutes(t)); logPrintf(stderr, "\n[timeout]\n"); if (g_logFile) std::fflush(g_logFile); _exit(2); }).detach();
    }

    // ============================================================
    // process_one: render a single segmentation to output
    // ============================================================
    auto process_one = [&](const std::filesystem::path& seg_folder) -> bool {
        {
            std::ostringstream oss;
            oss << "Rendering: " << seg_folder.string();
            if (wantZarr) oss << " -> " << zarrOutputArg << " (zarr)";
            if (wantTif)  oss << " -> " << tifOutputArg << " (tif)";
            logPrintf(stdout, "%s\n", oss.str().c_str());
        }

        std::unique_ptr<QuadSurface> surf;
        try {
            surf = store_mode ? StoreSurfaces(store_options->store, "vc_render_tifxyz").load(seg_folder.string(), true, view_volume)
                              : load_quad_from_tifxyz(seg_folder);
        }
        catch (...) { logPrintf(stderr, "Error loading: %s\n", seg_folder.string().c_str()); return false; }

        surf->setGenInterpolation(genInterp);

        // Replace sentinel -1 with NaN
        auto* raw_points = surf->rawPointsPtr();
        for (int j = 0; j < raw_points->rows; j++)
            for (int i = 0; i < raw_points->cols; i++)
                if ((*raw_points)(j,i)[0] == -1) (*raw_points)(j,i) = {NAN,NAN,NAN};

        // Bounding box of valid points
        int col_min = raw_points->cols, col_max = -1, row_min = raw_points->rows, row_max = -1;
        for (int j = 0; j < raw_points->rows; j++)
            for (int i = 0; i < raw_points->cols; i++)
                if (std::isfinite((*raw_points)(j,i)[0])) {
                    col_min = std::min(col_min, i); col_max = std::max(col_max, i);
                    row_min = std::min(row_min, j); row_max = std::max(row_max, j);
                }

        cv::Size full_size = raw_points->size();

        // Compute render scale
        const double sA = hasAffine ? affineTransform.lengthScale() : 1.0;
        double render_scale = double(tgt_scale) * (double(scale_seg) * sA * double(ds_scale));

        {
            double sx = render_scale / surf->_scale[0], sy = render_scale / surf->_scale[1];
            full_size.width  = std::max(1, int(std::lround(full_size.width  * sx)));
            full_size.height = std::max(1, int(std::lround(full_size.height * sy)));
        }

        cv::Size tgt_size = full_size;
        cv::Rect crop = {0, 0, full_size.width, full_size.height};
        const cv::Rect canvasROI = crop;

        logPrintf(stdout, "ds_level=%d ds_scale=%g sA=%g Pg=%g render_scale=%g\n",
                  group_idx, ds_scale, sA, double(tgt_scale), render_scale);

        // Handle crop
        int cx = parsed["crop-x"].as<int>(), cy = parsed["crop-y"].as<int>();
        int cw = parsed["crop-width"].as<int>(), ch = parsed["crop-height"].as<int>();
        bool manual = cw > 0 && ch > 0, autoCrop = parsed["auto-crop"].as<bool>();
        if (autoCrop && manual) { logPrintf(stderr, "Error: --auto-crop and --crop-* are mutually exclusive\n"); return false; }

        if (autoCrop && col_max >= col_min) {
            double sx = render_scale / surf->_scale[0], sy = render_scale / surf->_scale[1];
            crop = cv::Rect(int(std::floor(col_min*sx)), int(std::floor(row_min*sy)),
                            int(std::ceil((col_max+1)*sx)) - int(std::floor(col_min*sx)),
                            int(std::ceil((row_max+1)*sy)) - int(std::floor(row_min*sy))) & canvasROI;
            tgt_size = crop.size();
            logPrintf(stdout, "auto-crop: [%d×%d from (%d,%d)]\n", crop.width, crop.height, crop.x, crop.y);
        } else if (manual) {
            crop = cv::Rect(cx, cy, cw, ch) & canvasROI;
            if (crop.width <= 0 || crop.height <= 0) { logPrintf(stderr, "Error: crop outside canvas\n"); return false; }
            tgt_size = crop.size();
        }

        logPrintf(stdout, "rendering %dx%d at scale %g crop [%d×%d from (%d,%d)]\n",
                  tgt_size.width, tgt_size.height, double(tgt_scale),
                  crop.width, crop.height, crop.x, crop.y);

        const int rotQuad = rotQuadGlobal;

        // Determine output dtype
        const bool useU16 = output_is_u16 && !isCompositeMode;
        const int cvType = useU16 ? CV_16UC1 : CV_8UC1;

        // ---- Output tiles: the zarr chunks, also when only TIFFs are written ----
        const size_t CH = 128, CW = 128;
        size_t baseZ = isCompositeMode ? 1 : size_t(std::max(1, num_slices));
        std::vector<size_t> chunks0 = {baseZ, std::min(CH, size_t(tgt_size.height)), std::min(CW, size_t(tgt_size.width))};
        const size_t tilesYSrc = (tgt_size.height + CH - 1) / CH;
        const size_t tilesXSrc = (tgt_size.width  + CW - 1) / CW;
        std::unique_ptr<vcr::zarr::Writer> dsOut;
        std::filesystem::path outFilePath(wantZarr ? zarrOutputArg : "/dev/null");

        if (wantZarr) {
            cv::Size zarrXY = tgt_size;
            if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(zarrXY.width, zarrXY.height);
            size_t baseY = zarrXY.height, baseX = zarrXY.width;

            outFilePath = zarrOutputArg;
            std::vector<size_t> shape0 = {baseZ, baseY, baseX};
            chunks0 = {shape0[0], std::min(CH, shape0[1]), std::min(CW, shape0[2])};
            vcr::zarr::CreateOptions outputOptions;
            outputOptions.shape.assign(shape0.begin(), shape0.end());
            outputOptions.chunks.assign(chunks0.begin(), chunks0.end());
            outputOptions.dtype = useU16 ? "<u2" : "|u1";
            outputOptions.compressor = zarrCompressor;
            outputOptions.separator = zarrSeparator;
            outputOptions.level = zarrCompressionLevel;

            if (pre_flag) {
                logPrintf(stdout, "[pre] creating zarr + all levels...\n");
                std::filesystem::create_directories(outFilePath);
                vcr::zarr::Writer::create(outFilePath / "0", outputOptions);
                logPrintf(stdout, "[pre] L0 shape: [%zu,%zu,%zu]\n", shape0[0], shape0[1], shape0[2]);
                if (wantPyramid)
                    createPyramidDatasets(outFilePath, shape0, CH, CW, useU16,
                                          zarrCompressor, zarrCompressionLevel, zarrSeparator);

                cv::Size attrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(attrXY.width, attrXY.height);
                writeZarrAttrs(outFilePath, vol_path, group_idx, baseZ, slice_step, accum_step,
                               accum_type_str, accumOffsets.size(), attrXY, baseZ, CH, CW,
                               render_level_voxel_size, zarr_voxel_unit, tgt_scale);
                return true;
            } else if (numParts > 1) {
                if (!std::filesystem::exists(std::filesystem::path(zarrOutputArg) / "0" / ".zarray")) {
                    logPrintf(stderr, "Error: run --pre first in multi-part mode\n"); return false;
                }
                dsOut = std::make_unique<vcr::zarr::Writer>(vcr::zarr::Writer::open(outFilePath / "0"));
            } else if (resumeFlag && std::filesystem::exists(std::filesystem::path(zarrOutputArg) / "0" / ".zarray")) {
                dsOut = std::make_unique<vcr::zarr::Writer>(vcr::zarr::Writer::open(outFilePath / "0"));
                logPrintf(stdout, "[resume] opening existing zarr\n");
            } else {
                std::filesystem::create_directories(outFilePath);
                dsOut = std::make_unique<vcr::zarr::Writer>(
                    vcr::zarr::Writer::create(outFilePath / "0", outputOptions));
            }
        }

        // ---- TIF setup (if requested) ----
        std::vector<TiffWriter> tifWriters;
        uint32_t tiffTileH = 16;

        if (wantTif) {
            int outW = (rotQuad >= 0 && rotQuad % 2 == 1) ? tgt_size.height : tgt_size.width;
            int outH = (rotQuad >= 0 && rotQuad % 2 == 1) ? tgt_size.width  : tgt_size.height;
            int tifSlices = isCompositeMode ? 1 : num_slices;

            auto makePath = [&](int zi) -> std::filesystem::path {
                int pad = 2, v = std::max(0, num_slices-1);
                while (v >= 100) { pad++; v /= 10; }
                std::ostringstream fn; fn << std::setw(pad) << std::setfill('0') << zi;
                return std::filesystem::path(tifOutputArg) / (fn.str() + ".tif");
            };
            auto makePartPath = [&](int zi) -> std::filesystem::path {
                auto p = makePath(zi);
                if (numParts > 1) return p.parent_path() / (p.stem().string() + ".part" + std::to_string(partId) + p.extension().string());
                return p;
            };

            // Skip existing TIFFs only when explicitly resuming. A render
            // killed mid-way (OOM, --timeout) leaves every slice with a
            // header but no directory -- TiffWriter writes the IFD in
            // close() -- and a bare exists() check would treat those torn
            // files as done and skip them on every rerun (#1404).
            // isReadableTiff also decodes every tile, so a slice whose
            // directory survived but whose data did not is re-rendered too.
            bool tifSkip = false;
            if (resumeFlag && numParts <= 1) {
                int missing = 0, torn = 0;
                for (int z = 0; z < tifSlices; z++) {
                    const auto p = makePartPath(z);
                    if (!std::filesystem::exists(p)) missing++;
                    else if (!isReadableTiff(p)) torn++;
                }
                if (missing == 0 && torn == 0) {
                    if (!wantZarr) { logPrintf(stdout, "[tif] all slices exist, skipping.\n"); return true; }
                    logPrintf(stdout, "[tif] all slices exist, skipping tif output.\n");
                    tifSkip = true;
                } else if (torn > 0) {
                    logPrintf(stdout, "[tif] %d of %d existing slices are not complete, readable TIFFs (interrupted render or damaged file?), re-rendering all slices.\n",
                              torn, tifSlices - missing);
                }
            }
            if (!tifSkip) {
                uint32_t tiffTileW = (uint32_t(outW) + 15u) & ~15u;
                uint16_t tifComp = quickTif ? COMPRESSION_PACKBITS : COMPRESSION_LZW;
                for (int z = 0; z < tifSlices; z++)
                    tifWriters.emplace_back(makePartPath(z), uint32_t(outW), uint32_t(outH), cvType, tiffTileW, tiffTileH, 0.0f, tifComp, tifDpi);
            }
        }

        // ---- Create pyramid datasets before render ----
        const bool hasRotFlip = (rotQuad >= 0 || flip_axis >= 0);
        // Inline pyramid only works without rotation/flip (accumulation assumes
        // source tile-rows map 1:1 to destination tile-rows for row-group flushing)
        const bool inlinePyramid = wantZarr && wantPyramid && !pre_flag && !hasRotFlip;
        std::vector<vcr::zarr::Writer*> pyramidDs;
        std::vector<std::unique_ptr<vcr::zarr::Writer>> pyramidOwned;
        if (wantZarr && wantPyramid && !pre_flag) {
            // Single-part: create datasets now; multi-part/resume: already created them
            if (numParts <= 1 && !resumeFlag) {
                cv::Size zarrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(zarrXY.width, zarrXY.height);
                std::vector<size_t> shape0 = {baseZ, size_t(zarrXY.height), size_t(zarrXY.width)};
                createPyramidDatasets(outFilePath, shape0, CH, CW, useU16,
                                      zarrCompressor, zarrCompressionLevel, zarrSeparator);
            }
            if (inlinePyramid) {
                for (int level = 1; level <= 5; level++) {
                    pyramidOwned.push_back(std::make_unique<vcr::zarr::Writer>(
                        vcr::zarr::Writer::open(outFilePath / std::to_string(level))));
                    pyramidDs.push_back(pyramidOwned.back().get());
                }
            }
        }

        // ---- Render pass ----
        {
            auto* tifs = tifWriters.empty() ? nullptr : &tifWriters;
            if (useU16)
                renderTiles<uint16_t>(surf.get(), volume, sourceType, levelMap,
                    full_size, crop, tgt_size, float(render_scale), scale_seg,
                    hasAffine, affineTransform, num_slices, slice_step,
                    accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                    compositeMethod, rotQuad, flip_axis, numParts, partId, cvType,
                    dsOut.get(), chunks0, tilesXSrc, tilesYSrc,
                    pyramidDs, tifs, tiffTileH, quickTif, resumeFlag);
            else
                renderTiles<uint8_t>(surf.get(), volume, sourceType, levelMap,
                    full_size, crop, tgt_size, float(render_scale), scale_seg,
                    hasAffine, affineTransform, num_slices, slice_step,
                    accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                    compositeMethod, rotQuad, flip_axis, numParts, partId, cvType,
                    dsOut.get(), chunks0, tilesXSrc, tilesYSrc,
                    pyramidDs, tifs, tiffTileH, quickTif, resumeFlag);
            tifWriters.clear();
        }

        // ---- Zarr pyramid + attrs ----
        if (wantZarr && !pre_flag) {
            // Rotation/flip: pyramid couldn't be built inline, build from L0 on disk
            if (wantPyramid && hasRotFlip) {
                logPrintf(stdout, "[pyramid] building from L0...\n");
                for (int level = 1; level <= 5; level++) {
                    if (useU16) buildPyramidLevel<uint16_t>(outFilePath, level, CH, CW, numParts, partId);
                    else        buildPyramidLevel<uint8_t>(outFilePath, level, CH, CW, numParts, partId);
                }
            }

            // --pre already writes attrs for multi-part; single-part writes here
            if (numParts <= 1) {
                cv::Size attrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(attrXY.width, attrXY.height);
                writeZarrAttrs(outFilePath, vol_path, group_idx, baseZ, slice_step, accum_step,
                               accum_type_str, accumOffsets.size(), attrXY, baseZ, CH, CW,
                               render_level_voxel_size, zarr_voxel_unit, tgt_scale);
            }
        }
        return true;
    };

    bool ok = false;
    try {
        ok = process_one(seg_path);
    } catch (const std::exception& e) {
        // Typically a remote chunk fetch that failed after its retries:
        // report which chunk failed and why.
        // An HTTP status error carries the server's response body in its
        // message; for a 503 that is a whole HTML page. Keep the first line.
        std::string what = e.what();
        if (auto nl = what.find('\n'); nl != std::string::npos) {
            what.erase(nl);
            while (!what.empty() && (what.back() == '\r' || what.back() == ' '))
                what.pop_back();
            what += " [...]";
        }
        logPrintf(stderr, "\nError: %s\n", what.c_str());
    }
    if (!ok)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
