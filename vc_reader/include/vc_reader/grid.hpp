#pragma once
// Surfaces stored as grids of vertices (VC's tifxyz), and sampling volumes along
// them and along planes: positions and normals interpolated from the vertices at
// every pixel of a raster (grid::point, src/cpp/grid/points.hpp), offset along the normal into layers, sampled from a
// pyramid with fallback between levels, optionally from the caches alone, and
// optionally reduced to one value per pixel. Rasters are sampled in tiles, each on
// one thread, which reads the chunks its samples need and writes final pixels.

#include "vc_reader/composite.hpp"
#include "vc_reader/native/element_type.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vcr {
class Array;
}
namespace vcr::reader_cuda {
class Reader;
}

namespace vcr::grid {

// rows x cols vertices of three floats (XYZ), row r starting at xyz + r * stride floats.
// `valid` (rows x cols, contiguous; nonzero where the vertex is valid) says which
// vertices count; without it every vertex with finite coordinates does. A file format's
// own marker for an invalid vertex (tifxyz's -1) is read into `valid` by its reader.
// With `normals` (the layout of `xyz`: a unit normal per vertex, in physical space),
// normals are those interpolated over each patch instead of the patch's own.
struct Vertices {
    const float* xyz = nullptr;
    int rows = 0, cols = 0;
    std::size_t stride = 0;
    const float* normals = nullptr;
    const std::uint8_t* valid = nullptr;
};

enum class Interpolation {
    Linear,  // bilinear patches
    Bicubic,  // bicubic Hermite patches (C1 across cells), from per-vertex derivatives
};
// The interpolation named "linear" or "bicubic"; any other name raises.
[[nodiscard]] Interpolation interpolationNamed(const std::string& name);

// A cell of the grid is a surface patch when its four vertices are valid and in one
// piece; elsewhere there is no point. Normals are the patch's (its derivatives' cross
// product, column direction x row direction).
struct SurfaceOptions {
    Interpolation interpolation = Interpolation::Bicubic;
    // Column ranges [first, last) of separate pieces, never interpolated across;
    // empty: the grid is one piece.
    std::vector<std::pair<int, int>> components;
    // The physical size of one unit of vertex coordinates, per axis (XYZ): normals are
    // taken in physical space and layer offsets are physical lengths along them.
    std::array<float, 3> spacing{1, 1, 1};
    // Negate the patch normals (a grid whose rows and columns are mirrored against the
    // convention). Vertex normals given in `Vertices` keep their sign.
    bool flipNormals = false;
};

// A raster laid over the grid: pixel (i, j) sits at grid column x0 + i * sx + j * kx,
// row y0 + i * ky + j * sy (kx, ky shear it; zero for a raster along the grid's axes).
struct Raster {
    double x0 = 0, y0 = 0, sx = 1, sy = 1;
    int width = 0, height = 0;
    double kx = 0, ky = 0;
};

// The vertices of a surface (a view, not a copy: `xyz` and `normals` must outlive the
// Surface and not change while it is used; build a new Surface after an edit; `valid`
// is copied) and what interpolating them needs (vertex validity, pieces, Hermite
// derivatives).
class Surface {
public:
    explicit Surface(Vertices vertices, SurfaceOptions options = {});
    // An immutable owned copy, suitable for workers that outlive an editing transaction.
    static std::shared_ptr<const Surface> snapshot(Vertices vertices, SurfaceOptions options = {});
    ~Surface();
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    const Vertices& vertices() const;
    const SurfaceOptions& options() const;

    // Positions (vertex coordinates) and normals at every pixel of `raster`, row-major,
    // three floats each; NaN where the surface has no point. A normal is the move in
    // vertex coordinates per unit of physical offset along the unit physical normal
    // (the unit normal itself when spacing is 1): a layer at physical offset d lies at
    // xyz + d * normal. Either output may be null.
    void resample(const Raster& raster, float* xyz, float* normals) const;
    // The same at `n` real-valued grid positions (`xy`: column, row pairs).
    // Optional tangents are derivatives of XYZ with respect to grid column/row,
    // before spacing or normal normalization, from the same position patch.
    // Position-only queries also accept patches with undefined (zero-area) normals.
    void pointsAt(const float* xy, std::size_t n, float* xyz, float* normals,
                  float* columnTangents = nullptr, float* rowTangents = nullptr) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    friend struct SurfaceAccess;
};

// A plane raster: pixel (i, j) sits at origin + i * dx + j * dy, with normal `normal`.
struct Plane {
    std::array<float, 3> origin{}, dx{}, dy{}, normal{};
    int width = 0, height = 0;
};

// Prepared per-pixel geometry, without further interpolation. Contiguous H,W,3
// XYZ positions and optional offset directions (coordinate units per offset unit).
// Nonfinite positions have no sample. Request::pixels supplies optional validity.
struct Points {
    const float* xyz = nullptr;
    const float* normals = nullptr;
    int width = 0, height = 0;
};

// One pyramid level of a volume: its array, how its stored elements read, and where
// its indices sit: index[a] = scale[a] * p[2 - a] + offset[a] for an XYZ point p
// (indices are ZYX). A level placed by a general affine (e.g. a rotated registration)
// sets `linear` instead of scale: index[a] = sum_b linear[3a + b] * p[b] + offset[a].
struct Level {
    Array* array = nullptr;
    ElementType type;
    std::array<double, 3> scale{1, 1, 1}, offset{0, 0, 0};
    std::optional<std::array<double, 9>> linear;
    // The level's CUDA reader (uint8 3-D CT), when a request may sample it on the GPU;
    // null samples on the CPU. Builds without CUDA have none.
    reader_cuda::Reader* cuda = nullptr;
    // Select a scalar channel of a CZYX array. Unset requires a ZYX array.
    // Packed chunks remain shared in the array cache; no extracted volume is made.
    // Channel views currently sample on the CPU (CUDA readers are 3-D only).
    std::optional<std::int64_t> channel;
};

// A level placed by a row-major 3x4 map, index (ZYX) = linear * p (XYZ) + offset: the
// scale form when the map only scales each axis, otherwise `linear`.
Level levelFromMap(Array& array, ElementType type, const std::array<double, 12>& map);

struct Request {
    // In order of preference (e.g. the view's level, finer ones, then coarser): each
    // sample comes from the first level that holds it.
    std::vector<Level> levels;
    int order = 1;  // 0 nearest, 1 linear
    // One layer per offset along the unit normal. A nonzero offset needs a normal:
    // points without one have no sample in that layer.
    std::vector<float> offsets{0.0f};
    // Read the caches alone (Array::cached_chunk): a sample whose chunks only the
    // source can supply is missing at that level, and the next level is tried.
    // Otherwise chunks are read blocking (Array::chunk), fetched as needed.
    bool cachedOnly = false;
    // When set, only pixels p with pixels[p] != 0 are sampled.
    const std::uint8_t* pixels = nullptr;
    // Blocking reads: once set, no further chunk read starts and sample throws
    // vcr::Cancelled. Cache-only requests never wait on a read and ignore it.
    const std::atomic<bool>* cancel = nullptr;
    // Reduce each pixel's layers to one value (layers without a sample are dropped).
    std::optional<composite::Options> composite;
};

struct Result {
    static constexpr std::int8_t kNone = -2;     // no layer of the pixel had a point
    static constexpr std::int8_t kMissing = -1;  // a layer's chunks were in no level's caches (cached-only reads)

    int width = 0, height = 0, layers = 0;
    // One value per pixel and layer, layer-major (layer * width * height + pixel), or
    // one per pixel with a composite; NaN where there is no value.
    std::vector<float> values;
    // Per pixel: the highest index into Request::levels that supplied one of its
    // layers; kMissing when a layer with a point was in no level (whatever the others
    // got); kNone when no layer had a point.
    std::vector<std::int8_t> level;
    // What a GPU pass failed with, when the request was sampled on the CPU instead
    // (see sample); empty otherwise.
    std::string cudaFailure;
    // Cached-only requests sampled on the CPU: per request level, the chunks (keys
    // C-order flattened indices over the level's full chunk grid, ascending) a sample
    // needed and no cache held. Empty otherwise.
    std::vector<std::vector<std::int64_t>> missingChunks;
};

// Samples a request, on the GPU when every level has a CUDA reader (one device
// between them), otherwise on the CPU. A GPU pass that fails (device memory, a request
// over the readers' brick budget, a device error) gives the readers' bricks back
// (Reader::clear), samples on the CPU and names the failure in Result::cudaFailure.
// Invalid requests throw either way.
Result sample(const Surface& surface, const Raster& raster, const Request& request);
Result sample(const Plane& plane, const Request& request);
Result sample(const Points& points, const Request& request);

// ---- views: choosing levels and drawing frames progressively ----------------------
//
// A view (any UI) draws each frame from the caches alone (sampleCached), shows it, and
// reads what the frame lacked in the background (load); the next frame is complete.
// Levels are in the view's working coordinates (as Request::levels): a Level with no
// array is absent and skipped.

// Working voxels one index of `level` spans along each index axis (ZYX).
std::array<double, 3> voxelExtent(const Level& level);
// The level a view at `pixelsPerVoxel` (view pixels per working voxel) samples: the
// coarsest present level whose largest voxel extent is at most
// 1 / (pixelsPerVoxel * qualityBias), else the finest present one. 0 when the zoom or
// bias is not positive and finite.
int levelForView(const std::vector<Level>& levels, float pixelsPerVoxel, float qualityBias = 1);
// How many levels after `start` may fill a width x height view's holes: up to the first
// level whose mean chunk edge spans the view's larger edge, at most `maximum`.
int fallbackLevels(const std::vector<Level>& levels, int start, int width, int height, float pixelsPerVoxel,
                   int maximum = 5);
// The 3x3 (column, row) grid positions at which a raster's magnification is measured.
std::array<float, 18> magnificationPoints(const Raster& raster);
// View pixels per working voxel over a raster, from a surface's column and row tangents
// at magnificationPoints(raster): the most magnified direction (smallest singular value
// of the pixel-step Jacobian). 0 when no tangent pair is usable (holes, boundaries).
float pixelsPerVoxel(const float* columnTangents, const float* rowTangents, std::size_t n, double sx, double sy);
float pixelsPerVoxel(const Surface& surface, const Raster& raster);

// What a frame samples: a plane, or a raster over a surface (kept alive by the frame).
struct Frame {
    std::optional<Plane> plane;
    std::shared_ptr<const Surface> surface;
    Raster raster;
    int width() const { return plane ? plane->width : raster.width; }
    int height() const { return plane ? plane->height : raster.height; }
};
Result sample(const Frame& frame, const Request& request);

// The blocking reads frames could not sample from the caches, each at one level for
// the pixels that lacked it: `coarse` (a quick preview from the coarsest level) is
// read before `target` (the view's level). Requests hold Array pointers: keep their
// arrays open until the reads ran (`owners` may hold whatever keeps them open).
struct Pending {
    struct Read {
        Frame frame;
        Request request;  // one level; sampled for the pixels set in `pixels`
        std::vector<std::uint8_t> pixels;
        // When set, the level's chunks to read (Result::missingChunks keys): load reads
        // them and samples nothing.
        std::vector<std::int64_t> chunks;
    };
    std::vector<Read> coarse, target;
    std::vector<std::shared_ptr<const void>> owners;
    bool empty() const { return coarse.empty() && target.empty(); }
};

// Samples `request` over `frame` from the caches alone (cachedOnly is forced).
// request.levels[0] is the view's level; later levels (coarser) fill its holes. When
// `pending` is given, adds the reads the frame lacked: pixels not from levels[0] read
// it, and pixels no level held also read the last level first.
Result sampleCached(const Frame& frame, Request request, Pending* pending);
// Runs the reads, coarse then target, bringing their chunks into the caches.
// `afterPart` runs after each part that had reads (one cut short too); returning false
// stops. Once `cancel` is set, no further chunk read starts (reads under way finish) and
// load returns after that part's afterPart. Returns the first GPU failure a read fell back from
// (Result::cudaFailure), else empty.
std::string load(const Pending& pending, const std::function<bool()>& afterPart = {},
                 const std::atomic<bool>* cancel = nullptr);

}  // namespace vcr::grid

#ifdef VC_READER_CUDA
namespace vcr::reader_cuda {
class Reader;
}
namespace vcr::grid::cuda {
// The same requests sampled on a GPU alone (grid::sample dispatches to these and falls
// back): readers[i] is the CUDA reader (uint8 3-D CT) of request.levels[i].array
// (Level::cuda is ignored), and every sample reads its level's resident bricks.
// Cache-only requests read those alone; blocking ones first make the bricks the
// first level lacks resident (from the array), then sample again. A surface's
// vertex data is uploaded once and positions and normals are computed on the device.
Result sample(const Surface& surface, const Raster& raster, const Request& request,
              const std::vector<reader_cuda::Reader*>& readers);
Result sample(const Plane& plane, const Request& request, const std::vector<reader_cuda::Reader*>& readers);
Result sample(const Points& points, const Request& request, const std::vector<reader_cuda::Reader*>& readers);
// Frees the surface vertex data kept on devices between samples (grid::sample does when
// a GPU pass fails; the last few surfaces sampled are otherwise kept).
void forgetKeptData();
}  // namespace vcr::grid::cuda
#endif
