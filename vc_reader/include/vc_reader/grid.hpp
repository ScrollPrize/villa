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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace vcr {
class Array;
}

namespace vcr::grid {

// rows x cols vertices of three floats (XYZ), row r starting at xyz + r * stride floats.
// A vertex is missing when a coordinate is NaN or all three are -1. With `normals`
// (the same layout: a unit normal per vertex, in physical space), normals are those
// interpolated over each patch instead of the patch's own.
struct Vertices {
    const float* xyz = nullptr;
    int rows = 0, cols = 0;
    std::size_t stride = 0;
    const float* normals = nullptr;
};

// A missing vertex never enters a position: in a cell with missing corners, the
// present corners are blended with renormalised bilinear weights.
enum class Interpolation {
    Linear,  // bilinear patches
    Smooth,  // bicubic Hermite patches (C1 across cells), from per-vertex derivatives
};

// A cell of the grid is a surface patch when its four vertices are present and in one
// piece; elsewhere there is no point. Normals are the patch's (its derivatives' cross
// product, column direction x row direction).
struct SurfaceOptions {
    Interpolation interpolation = Interpolation::Linear;
    // Column ranges [first, last) of separate pieces, never interpolated across;
    // empty: the grid is one piece.
    std::vector<std::pair<int, int>> components;
    // The physical size of one unit of vertex coordinates, per axis (XYZ): normals are
    // taken in physical space and layer offsets are physical lengths along them.
    std::array<float, 3> spacing{1, 1, 1};
};

// A raster laid over the grid: pixel (i, j) sits at grid column x0 + i * sx + j * kx,
// row y0 + i * ky + j * sy (kx, ky shear it; zero for a raster along the grid's axes).
struct Raster {
    double x0 = 0, y0 = 0, sx = 1, sy = 1;
    int width = 0, height = 0;
    double kx = 0, ky = 0;
};

// The vertices of a surface (a view, not a copy: they must outlive the Surface and
// not change while it is used; build a new Surface after an edit) and what
// interpolating them needs (vertex presence, pieces, Hermite derivatives).
class Surface {
public:
    explicit Surface(Vertices vertices, SurfaceOptions options = {});
    ~Surface();
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    const Vertices& vertices() const;
    const SurfaceOptions& options() const;

    // Positions and normals at every pixel of `raster`, row-major, three floats each;
    // NaN where the surface has no point. A normal is the move in vertex coordinates
    // per unit of physical offset along the unit physical normal (the unit normal when
    // spacing is 1). Either output may be null.
    void resample(const Raster& raster, float* xyz, float* normals) const;

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

// One pyramid level of a volume: its array, how its stored elements read, and where
// its indices sit: index[a] = scale[a] * p[2 - a] + offset[a] for an XYZ point p
// (indices are ZYX). A level placed by a general affine (e.g. a rotated registration)
// sets `linear` instead of scale: index[a] = sum_b linear[3a + b] * p[b] + offset[a].
struct Level {
    Array* array = nullptr;
    ElementType type;
    std::array<double, 3> scale{1, 1, 1}, offset{0, 0, 0};
    std::optional<std::array<double, 9>> linear;
};

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
};

Result sample(const Surface& surface, const Raster& raster, const Request& request);
Result sample(const Plane& plane, const Request& request);

}  // namespace vcr::grid

#ifdef VC_READER_CUDA
namespace vcr::reader_cuda {
class Reader;
}
namespace vcr::grid::cuda {
// The same requests sampled on a GPU: readers[i] is the CUDA reader (uint8 3-D CT)
// of request.levels[i].array, and every sample reads its level's resident bricks.
// Cache-only requests read those alone; blocking ones first make the bricks the
// first level lacks resident (from the array), then sample again. A surface's
// positions and normals are resampled on the CPU and uploaded.
Result sample(const Surface& surface, const Raster& raster, const Request& request,
              const std::vector<reader_cuda::Reader*>& readers);
Result sample(const Plane& plane, const Request& request, const std::vector<reader_cuda::Reader*>& readers);
}  // namespace vcr::grid::cuda
#endif
