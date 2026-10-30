#include <iostream>
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StoreToolOptions.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/StreamOperators.hpp"

#include "vc/core/types/Volume.hpp"
#include <vc_reader/composite.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_reader/zarr_metadata.hpp>
#include <vc_reader/array.hpp>
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

// Surface coordinates (working-level voxels, XYZ) to indices of the rendered
// pyramid level: index = scale * c + offset per axis, from the two levels' placements.
struct LevelMap {
    cv::Vec3f scale{1, 1, 1}, offset{0, 0, 0};
};

static LevelMap mapToLevel(const vcr::zarr::LevelPlacement& working, const vcr::zarr::LevelPlacement& rendered)
{
    const auto m = rendered.indexMapFrom(working);
    LevelMap map;
    for (int a = 0; a < 3; ++a) {
        map.scale[a] = float(m.scale[2 - a]);
        map.offset[a] = float(m.offset[2 - a]);
    }
    return map;
}

// The surface with its vertices in rendered-level index space (XYZ): the segmentation
// scale, the affine and the level placement applied to each vertex once. Bilinear and
// Hermite patches commute with affine maps, so positions are those of the mapped
// vertex grid; normals are its patch normals and offsets along them are in
// rendered-level voxels.
struct RenderSurface {
    cv::Mat_<cv::Vec3f> vertices;
    std::unique_ptr<vcr::grid::Surface> grid;
};

static RenderSurface renderSurface(const QuadSurface& surf, vcr::grid::Interpolation interpolation, float scaleSeg,
                                   bool hasAffine, const Affine& aff, const LevelMap& level)
{
    RenderSurface r;
    r.vertices = surf.rawPointsPtr()->clone();
    for (auto& v : r.vertices) {
        if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
            v = cv::Vec3f(NAN, NAN, NAN);
            continue;
        }
        v *= scaleSeg;
        if (hasAffine) {
            const auto q = aff.apply({v[0], v[1], v[2]});
            v = cv::Vec3f(float(q[0]), float(q[1]), float(q[2]));
        }
        for (int a = 0; a < 3; ++a)
            v[a] = v[a] * level.scale[a] + level.offset[a];
    }
    auto options = surf.gridOptions();
    options.interpolation = interpolation;
    r.grid = std::make_unique<vcr::grid::Surface>(
        vcr::grid::Vertices{reinterpret_cast<const float*>(r.vertices.ptr()), r.vertices.rows, r.vertices.cols,
                            r.vertices.step1()},
        std::move(options));
    return r;
}

// A tile's layers at `offsets` along the (--flip-normals: negated) normal, linear or
// nearest, optionally reduced to one image; NaN where the surface has no point.
static vcr::grid::Result sampleTile(const RenderSurface& surface, const vcr::grid::Raster& raster,
                                    vcr::Array& volume, const vcr::ElementType& type, const std::vector<float>& offsets,
                                    int order, const std::optional<vcr::composite::Options>& composite)
{
    vcr::grid::Request request;
    request.levels = {vcr::grid::Level{&volume, type}};
    request.order = order;
    request.offsets = offsets;
    if (g_flipNormals)
        for (auto& o : request.offsets)
            o = -o;
    request.composite = composite;
    return vcr::grid::sample(*surface.grid, raster, request);
}

// One image of a result: pixels without a value read 0; uint8 truncates and uint16
// rounds, as the renders always have.
template <typename T>
static cv::Mat toImage(const vcr::grid::Result& result, int layer)
{
    cv::Mat_<T> out(result.height, result.width);
    const std::size_t n = std::size_t(result.width) * std::size_t(result.height);
    const float* v = result.values.data() + std::size_t(layer) * n;
    constexpr float maxValue = float(std::numeric_limits<T>::max());
    constexpr float rounding = std::is_same_v<T, uint16_t> ? 0.5f : 0.0f;
    T* dst = out[0];
    for (std::size_t p = 0; p < n; ++p)
        dst[p] = std::isnan(v[p]) ? T(0) : T(std::clamp(v[p], 0.0f, maxValue) + rounding);
    return out;
}

// ============================================================
// Reducers: --accum sub-steps per slice (max / mean / median) and composites
// ============================================================

enum class AccumType { Max, Mean, Median, Alpha, BeerLambert };

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

// ============================================================
// Rendering: OMP-parallel over the output tiles (the zarr chunks and TIFF tiles)
// ============================================================

// The source pixels (the unrotated render) that land in output pixels [x0, x1] x
// [y0, y1] after rotation (quarter turns counter-clockwise) and flip.
static cv::Rect sourceRect(int x0, int y0, int x1, int y1, const cv::Size& source, int rotQuad, int flipAxis)
{
    const bool swap = rotQuad == 1 || rotQuad == 3;
    const int W = swap ? source.height : source.width, H = swap ? source.width : source.height;
    auto back = [&](int x, int y) {
        if (flipAxis == 0 || flipAxis == 2) y = H - 1 - y;
        if (flipAxis == 1 || flipAxis == 2) x = W - 1 - x;
        switch (rotQuad) {
            case 1: return cv::Point(source.width - 1 - y, x);
            case 2: return cv::Point(source.width - 1 - x, source.height - 1 - y);
            case 3: return cv::Point(y, source.height - 1 - x);
            default: return cv::Point(x, y);
        }
    };
    const cv::Point a = back(x0, y0), b = back(x1, y1);
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::abs(a.x - b.x) + 1, std::abs(a.y - b.y) + 1};
}

// Renders output tiles [tileRows) of `tiles` (CH x CW pixels each, in output
// space after rotation/flip): samples each tile's source pixels, then writes the
// tile to the zarr level 0 and/or each slice's TIFF.
template <typename T>
static void renderTiles(
    const QuadSurface& surf, const RenderSurface& surface, vcr::Array& volume, const vcr::ElementType& type,
    const cv::Size& fullSize, const cv::Rect& crop, const cv::Size& tgtSize,
    float renderScale,
    int numSlices, double sliceStep,
    const std::vector<float>& accumOffsets, AccumType accumType,
    bool isComposite, int compositeStart, int compositeEnd,
    const vcr::composite::Options& compositeMethod,
    int rotQuad, int flipAxis, int cvType,
    size_t CH, size_t CW, size_t tilesX, uint32_t rowStart, uint32_t rowEnd,
    vcr::zarr::Writer* dsOut,                 // null: no zarr output
    std::vector<TiffWriter>* tifWriters,      // null: no TIFF output
    bool quickTif, bool resume)
{
    const auto allOffsets = buildOffsetList(numSlices, sliceStep, accumOffsets);
    const std::size_t sps = std::max<std::size_t>(1, accumOffsets.size());
    // --accum: each slice reduces its sub-steps.
    std::optional<vcr::composite::Options> accumMethod;
    if (!accumOffsets.empty()) {
        accumMethod.emplace();
        accumMethod->method = accumType == AccumType::Max    ? vcr::composite::Method::Max
                            : accumType == AccumType::Median ? vcr::composite::Method::Median
                                                             : vcr::composite::Method::Mean;
    }
    const auto compositeOffsets = isComposite
        ? buildCompositeOffsetList(compositeStart, compositeEnd, sliceStep)
        : std::vector<float>{};
    const bool swap = rotQuad == 1 || rotQuad == 3;
    const cv::Size outSize = swap ? cv::Size(tgtSize.height, tgtSize.width) : tgtSize;
    const size_t numZ = isComposite ? 1 : size_t(std::max(1, numSlices));
    std::vector<std::mutex> tifLocks(tifWriters ? tifWriters->size() : 0);

    const long long count = (long long)(rowEnd - rowStart) * (long long)tilesX;
    std::atomic<long long> done{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    const auto wallStart = std::chrono::steady_clock::now();
    auto lastPrint = wallStart;

    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < count; i++) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const int ty = int(rowStart + uint32_t(i / (long long)tilesX)), tx = int(i % (long long)tilesX);
            // Resume: a tile whose zarr chunk exists is done, unless TIFFs need it too.
            if (resume && dsOut && !tifWriters && dsOut->chunkExists({0, ty, tx})) {
                ++done;
                continue;
            }
            const int x0 = tx * int(CW), y0 = ty * int(CH);
            const int w = std::min(int(CW), outSize.width - x0), h = std::min(int(CH), outSize.height - y0);
            const cv::Rect src = sourceRect(x0, y0, x0 + w - 1, y0 + h - 1, tgtSize, rotQuad, flipAxis);

            // The tile's framebuffer pixels, the top-left one at canvas offset (u0, v0).
            float u0, v0; computeCanvasOrigin(fullSize, u0, v0);
            u0 += float(crop.x + src.x);
            v0 += float(crop.y + src.y);
            const auto raster = surf.gridRaster(double(u0 / renderScale), double(v0 / renderScale), renderScale,
                                                src.size());

            std::vector<cv::Mat> slices;
            if (isComposite) {
                // Composite mode: nearest voxels, always u8 (the caller rejects uint16 sources).
                slices = {toImage<uint8_t>(sampleTile(surface, raster, volume, type, compositeOffsets, 0, compositeMethod), 0)};
            } else if (accumMethod) {
                for (int zi = 0; zi < numSlices; ++zi) {
                    const std::vector<float> steps(allOffsets.begin() + std::ptrdiff_t(zi * sps),
                                                   allOffsets.begin() + std::ptrdiff_t((zi + 1) * sps));
                    slices.push_back(toImage<T>(sampleTile(surface, raster, volume, type, steps, 1, accumMethod), 0));
                }
            } else {
                const auto layers = sampleTile(surface, raster, volume, type, allOffsets, 1, std::nullopt);
                for (int zi = 0; zi < numSlices; ++zi)
                    slices.push_back(toImage<T>(layers, zi));
            }
            for (auto& slice : slices)
                rotateFlipIfNeeded(slice, rotQuad, flipAxis);

            if (dsOut) {
                std::vector<T> chunkBuf(numZ * CH * CW, T(0));
                for (size_t zi = 0; zi < numZ; zi++)
                    for (int yy = 0; yy < h; yy++)
                        std::memcpy(&chunkBuf[(zi * CH + size_t(yy)) * CW], slices[zi].ptr<T>(yy), size_t(w) * sizeof(T));
                dsOut->writeChunkSkipEmpty({0, ty, tx}, reinterpret_cast<const uint8_t*>(chunkBuf.data()),
                                           chunkBuf.size() * sizeof(T));
            }
            if (tifWriters) {
                for (size_t zi = 0; zi < tifWriters->size(); zi++) {
                    if (quickTif && cvType == CV_8UC1)
                        slices[zi] &= cv::Scalar(0xF0);
                    std::lock_guard lock(tifLocks[zi]);
                    (*tifWriters)[zi].writeTile(uint32_t(x0), uint32_t(y0), slices[zi]);
                }
            }

            const long long d = ++done;
            #pragma omp critical(render_progress)
            {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - lastPrint).count() >= 1.0 || d == count) {
                    lastPrint = now;
                    const double elapsed = std::chrono::duration<double>(now - wallStart).count();
                    const double eta = elapsed * (double(count) / double(d) - 1.0);
                    logPrintf(stderr, "%stile %lld/%lld (%d%%)  %.1f tiles/s  %dm%02ds  eta %dm%02ds%s",
                              g_logFile ? "  " : "\r  ", d, count, int(100.0 * double(d) / double(count)),
                              elapsed > 0 ? double(d) / elapsed : 0.0, int(elapsed) / 60, int(elapsed) % 60,
                              int(eta) / 60, int(eta) % 60, g_logFile ? "\n" : "");
                }
            }
        } catch (...) {
            // A throw out of the OpenMP loop (a remote chunk fetch that failed after
            // its retries, for instance) would end in std::terminate: keep the first.
            #pragma omp critical(render_tile_error)
            {
                if (!error) error = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (error) std::rethrow_exception(error);
    if (!g_logFile && count > 0) std::fprintf(stderr, "\n");
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
// Levels above level 0 of a rendered Zarr: Z keeps its layers, Y/X halve per level.
const vcr::zarr::PyramidOptions kRenderPyramid{5, {1, 2, 2}};

// The render's .zattrs: what it rendered, and its OME multiscales. With a known
// physical size (baseVoxelSize > 0, one source voxel at the rendered level) the
// scales are physical: Z = baseVoxelSize * sliceStep, Y/X = baseVoxelSize /
// pixelsPerVoxel, and the axes carry `voxelUnit`. Without one they state only each
// level's factor over level 0 and no unit, as OME-NGFF 0.4 asks, and say why.
void writeZarrAttrs(const std::filesystem::path& outDir,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW,
                    double baseVoxelSize, const std::string& voxelUnit,
                    double pixelsPerVoxel)
{
    vcr::zarr::MultiscalesOptions ome;
    ome.name = "render";
    ome.levels = kRenderPyramid.levels;
    ome.factors = kRenderPyramid.factors;
    const bool physicalSizeKnown = std::isfinite(baseVoxelSize) && baseVoxelSize > 0.0;
    if (physicalSizeKnown) {
        const double px = std::isfinite(pixelsPerVoxel) && pixelsPerVoxel > 0.0 ? pixelsPerVoxel : 1.0;
        const double step = std::isfinite(sliceStep) && sliceStep > 0.0 ? sliceStep : 1.0;
        ome.scale = {baseVoxelSize * step, baseVoxelSize / px, baseVoxelSize / px};
        ome.unit = voxelUnit;
    }
    auto multiscales = vcr::zarr::omeMultiscales(ome);
    if (!physicalSizeKnown)
        multiscales["metadata"]["physical_size"] = "unknown";

    vcr::zarr::Json attrs{{"source_zarr", volPath.string()},
                          {"source_group", groupIdx},
                          {"num_slices", baseZ},
                          {"slice_step", sliceStep},
                          {"canvas_size", {canvasSize.width, canvasSize.height}},
                          {"chunk_size", {int(CZ), int(CH), int(CW)}},
                          {"note_axes_order", "ZYX (slice, row, col)"},
                          {"multiscales", vcr::zarr::Json::array({std::move(multiscales)})}};
    if (accumSamples > 0) {
        attrs["accum_step"] = accumStep;
        attrs["accum_type"] = accumTypeStr;
        attrs["accum_samples"] = int(accumSamples);
    }
    vcr::zarr::writeAttributes(outDir, attrs);
}

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
// VC3D tool reads it -- vcr::zarr::storeVoxelSizeUm(), which
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

    if (const auto local = vcr::zarr::storeVoxelSizeUm(volPath.string()))
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
    vcr::grid::Interpolation interpolation;
    if      (surf_interp_str == "linear") interpolation = vcr::grid::Interpolation::Linear;
    else if (surf_interp_str == "smooth" || surf_interp_str == "bicubic")
        interpolation = vcr::grid::Interpolation::Smooth;
    else { logPrintf(stderr, "Error: invalid --surface-interpolation\n"); return EXIT_FAILURE; }

    // alpha/beerlambert reducers only make sense over collapsed layers, so they always imply
    // composite mode. --composite-collapse extends the same collapsing path to max/mean/median,
    // which the native compositor (vcr::composite::reduce) already implements.
    const bool requestComposite = parsed["composite-collapse"].as<bool>();
    const bool isCompositeMode = requestComposite
        || accumType == AccumType::Alpha || accumType == AccumType::BeerLambert;

    // Without layers to reduce, the slices are the sampled layers and accumType is never
    // read, so an explicit max/mean/median here changes nothing. Say so rather than stay quiet.
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
            remoteVolume = Volume::NewFromUrl(remoteUrl, vc::loadAwsCredentials());
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
    if (interpolation == vcr::grid::Interpolation::Smooth)
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

        // ---- Output tiles: the zarr chunks (and TIFF tiles) of the rotated output ----
        const size_t CH = 128, CW = 128;
        size_t baseZ = isCompositeMode ? 1 : size_t(std::max(1, num_slices));
        cv::Size outXY = tgt_size;
        if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(outXY.width, outXY.height);
        const std::vector<size_t> shape0 = {baseZ, size_t(outXY.height), size_t(outXY.width)};
        const std::vector<size_t> chunks0 = {baseZ, std::min(CH, shape0[1]), std::min(CW, shape0[2])};
        const size_t tilesY = (shape0[1] + chunks0[1] - 1) / chunks0[1];
        const size_t tilesX = (shape0[2] + chunks0[2] - 1) / chunks0[2];
        std::unique_ptr<vcr::zarr::Writer> dsOut;
        std::filesystem::path outFilePath(wantZarr ? zarrOutputArg : "/dev/null");

        vcr::zarr::CreateOptions outputOptions;
        outputOptions.shape.assign(shape0.begin(), shape0.end());
        outputOptions.chunks.assign(chunks0.begin(), chunks0.end());
        outputOptions.dtype = useU16 ? "<u2" : "|u1";
        outputOptions.compressor = zarrCompressor;
        outputOptions.separator = zarrSeparator;
        outputOptions.level = zarrCompressionLevel;

        if (wantZarr) {
            outFilePath = zarrOutputArg;

            if (pre_flag) {
                logPrintf(stdout, "[pre] creating zarr + all levels...\n");
                std::filesystem::create_directories(outFilePath);
                vcr::zarr::Writer::create(outFilePath / "0", outputOptions);
                logPrintf(stdout, "[pre] L0 shape: [%zu,%zu,%zu]\n", shape0[0], shape0[1], shape0[2]);
                if (wantPyramid)
                    vcr::zarr::createPyramidLevels(outFilePath, outputOptions, kRenderPyramid);

                writeZarrAttrs(outFilePath, vol_path, group_idx, baseZ, slice_step, accum_step,
                               accum_type_str, accumOffsets.size(), outXY, baseZ, CH, CW,
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

        if (wantTif) {
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
                // TIFF tiles are the render tiles (TIFF needs multiples of 16: 128).
                uint16_t tifComp = quickTif ? COMPRESSION_PACKBITS : COMPRESSION_LZW;
                for (int z = 0; z < tifSlices; z++)
                    tifWriters.emplace_back(makePartPath(z), uint32_t(outXY.width), uint32_t(outXY.height), cvType,
                                            uint32_t(CW), uint32_t(CH), 0.0f, tifComp, tifDpi);
            }
        }

        // Single-part: create the pyramid datasets now; multi-part/resume: --pre created them.
        if (wantZarr && wantPyramid && !pre_flag && numParts <= 1 && !resumeFlag)
            vcr::zarr::createPyramidLevels(outFilePath, outputOptions, kRenderPyramid);

        // Each part renders a contiguous range of tile rows. With a pyramid the
        // ranges are multiples of 32 rows, so a part builds levels 1-5 of its rows alone.
        uint32_t rowsPerPart = uint32_t((tilesY + size_t(numParts) - 1) / size_t(numParts));
        if (numParts > 1 && wantZarr && wantPyramid)
            rowsPerPart = (rowsPerPart + 31u) / 32u * 32u;
        const uint32_t rowStart = std::min(uint32_t(partId) * rowsPerPart, uint32_t(tilesY));
        const uint32_t rowEnd = std::min(rowStart + rowsPerPart, uint32_t(tilesY));

        // ---- Render pass ----
        {
            const auto surface = renderSurface(*surf, interpolation, scale_seg, hasAffine, affineTransform, levelMap);
            auto* tifs = tifWriters.empty() ? nullptr : &tifWriters;
            if (useU16)
                renderTiles<uint16_t>(*surf, surface, volume, sourceType,
                    full_size, crop, tgt_size, float(render_scale), num_slices, slice_step,
                    accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                    compositeMethod, rotQuad, flip_axis, cvType,
                    chunks0[1], chunks0[2], tilesX, rowStart, rowEnd,
                    dsOut.get(), tifs, quickTif, resumeFlag);
            else
                renderTiles<uint8_t>(*surf, surface, volume, sourceType,
                    full_size, crop, tgt_size, float(render_scale), num_slices, slice_step,
                    accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                    compositeMethod, rotQuad, flip_axis, cvType,
                    chunks0[1], chunks0[2], tilesX, rowStart, rowEnd,
                    dsOut.get(), tifs, quickTif, resumeFlag);
            tifWriters.clear();
        }

        // ---- Zarr pyramid + attrs ----
        if (wantZarr && !pre_flag) {
            // Levels 1-5 of this part's rows, each from the level below on disk.
            if (wantPyramid) {
                logPrintf(stdout, "[pyramid] building from L0...\n");
                for (int level = 1; level <= 5; level++) {
                    const size_t first = rowStart >> level, last = (size_t(rowEnd) + (1u << level) - 1) >> level;
                    vcr::zarr::writePyramidLevel(
                        outFilePath, level, kRenderPyramid,
                        vcr::zarr::ChunkRange{{0, std::int64_t(first), 0},
                                              {std::numeric_limits<std::int64_t>::max(), std::int64_t(last),
                                               std::numeric_limits<std::int64_t>::max()}},
                        [level](std::size_t done, std::size_t total) {
                            if (done == total)
                                logPrintf(stdout, "[pyramid L%d] %zu chunks\n", level, total);
                        });
                }
            }

            // --pre already writes attrs for multi-part; single-part writes here
            if (numParts <= 1) {
                writeZarrAttrs(outFilePath, vol_path, group_idx, baseZ, slice_step, accum_step,
                               accum_type_str, accumOffsets.size(), outXY, baseZ, CH, CW,
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
