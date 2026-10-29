#pragma once
// Surfaces stored as grids of vertices (VC's tifxyz), and sampling volumes along
// them and along planes: positions and normals interpolated from the vertices at
// every pixel of a raster, offset along the normal into layers, sampled from a
// pyramid with fine-to-coarse fallback, optionally from the caches alone, and
// optionally reduced to one value per pixel.

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
// A vertex is missing when a coordinate is NaN or all three are -1.
struct Vertices {
    const float* xyz = nullptr;
    int rows = 0, cols = 0;
    std::size_t stride = 0;
};

// A missing vertex never enters a position: in a cell with missing corners, the
// present corners are blended with renormalised bilinear weights.
enum class Interpolation {
    Linear,  // bilinear positions; each pixel takes its nearest vertex's normal
    Smooth,  // bicubic Hermite patches over complete cells (others: the blend above, no normal)
};

struct SurfaceOptions {
    Interpolation interpolation = Interpolation::Linear;
    // A pixel lies on the surface when all four vertices of its grid cell exist
    // (strict), or when its nearest vertex exists.
    bool strictQuads = false;
    // Column ranges [first, last) of separate pieces, never interpolated across;
    // empty: the grid is one piece.
    std::vector<std::pair<int, int>> components;
};

// A raster laid over the grid: pixel (i, j) sits at grid column x0 + i * sx, row y0 + j * sy.
struct Raster {
    double x0 = 0, y0 = 0, sx = 1, sy = 1;
    int width = 0, height = 0;
};

// The vertices of a surface (a view, not a copy: they must outlive the Surface and
// not change while it is used; build a new Surface after an edit) and what
// interpolating them needs (validity, vertex normals, Hermite derivatives), each
// derived once on first use.
class Surface {
public:
    explicit Surface(Vertices vertices, SurfaceOptions options = {});
    ~Surface();
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    const Vertices& vertices() const;
    const SurfaceOptions& options() const;

    // Positions and unit normals at every pixel of `raster`, row-major, three floats
    // each; NaN where the surface has no point (normals: or no normal). Either
    // output may be null.
    void resample(const Raster& raster, float* xyz, float* normals) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend struct Result sample(const Surface&, const struct Raster&, const struct Request&);
};

// A plane raster: pixel (i, j) sits at origin + i * dx + j * dy, with normal `normal`.
struct Plane {
    std::array<float, 3> origin{}, dx{}, dy{}, normal{};
    int width = 0, height = 0;
};

// One pyramid level of a volume: its array, how its stored elements read, and where
// its indices sit: index[a] = scale[a] * p[2 - a] + offset[a] for an XYZ point p
// (indices are ZYX).
struct Level {
    Array* array = nullptr;
    ElementType type;
    std::array<double, 3> scale{1, 1, 1}, offset{0, 0, 0};
};

struct Request {
    // Finest first: each sample comes from the first level that holds it.
    std::vector<Level> levels;
    int order = 1;  // 0 nearest, 1 linear
    // One layer per offset along the unit normal. A nonzero offset needs a normal:
    // points without one have no sample in that layer.
    std::vector<float> offsets{0.0f};
    // Read the caches alone (Array::sample_*_cached): a sample whose chunks only the
    // source can supply is missing at that level, and the next level is tried.
    bool cachedOnly = false;
    // When set, only pixels p with pixels[p] != 0 are sampled.
    const std::uint8_t* pixels = nullptr;
    // Reduce each pixel's layers to one value (layers without a sample are dropped).
    std::optional<composite::Options> composite;
};

struct Result {
    static constexpr std::int8_t kNone = -2;     // no point here, or outside every level
    static constexpr std::int8_t kMissing = -1;  // no level held it (cached-only reads)

    int width = 0, height = 0, layers = 0;
    // One value per pixel and layer (layer-major: layer * width * height + pixel), or
    // one per pixel with a composite; NaN where there is no value.
    std::vector<float> values;
    // Per pixel and layer, always layer-major: the index into Request::levels that
    // supplied the sample, kMissing or kNone.
    std::vector<std::int8_t> level;
};

Result sample(const Surface& surface, const Raster& raster, const Request& request);
Result sample(const Plane& plane, const Request& request);

}  // namespace vcr::grid
