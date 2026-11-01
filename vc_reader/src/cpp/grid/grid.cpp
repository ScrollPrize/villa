// Surfaces stored as grids of vertices, and sampling volumes along them and along planes.
#include "vc_reader/grid.hpp"

#include "../array.hpp"
#include "points.hpp"
#include "../sample_point.hpp"
#include "level_map.hpp"
#include "../composite/reducer.hpp"
#include "../thread_pool.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <functional>
#include <optional>
#include <thread>

namespace vcr::grid {

Level levelFromMap(Array& array, ElementType type, const std::array<double, 12>& map)
{
    Level L{&array, type};
    bool aligned = true;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 4; ++b) {
            if (!std::isfinite(map[std::size_t(4 * a + b)]))
                throw std::invalid_argument("nonfinite level map");
            aligned &= b == 3 || b == 2 - a || map[std::size_t(4 * a + b)] == 0;
        }
    for (int a = 0; a < 3; ++a)
        L.offset[std::size_t(a)] = map[std::size_t(4 * a + 3)];
    if (aligned) {
        for (int a = 0; a < 3; ++a)
            L.scale[std::size_t(a)] = map[std::size_t(4 * a + 2 - a)];
    } else {
        std::array<double, 9> linear;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                linear[std::size_t(3 * a + b)] = map[std::size_t(4 * a + b)];
        L.linear = linear;
    }
    return L;
}

Interpolation interpolationNamed(const std::string& name)
{
    if (name == "linear")
        return Interpolation::Linear;
    if (name == "smooth")
        return Interpolation::Smooth;
    throw std::invalid_argument("interpolation must be 'linear' or 'smooth'");
}

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

ThreadPool& resamplePool()
{
    static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

}  // namespace

// A surface's vertices with what grid::point needs: which vertices are valid, the
// piece of each column, and (smooth mode) the Hermite derivatives of every vertex.
struct Surface::Impl {
    Vertices v;
    SurfaceOptions options;
    std::vector<std::uint8_t> valid;    // per vertex (bool-sized)
    std::vector<int> piece;             // per column; empty: one piece
    std::vector<float> du, dv, duv;     // per vertex, smooth mode
    std::vector<float> packed, packedNormals;  // without row padding, when stride != cols * 3

    const float* xyz() const { return packed.empty() ? v.xyz : packed.data(); }
    const float* normals() const { return packedNormals.empty() ? v.normals : packedNormals.data(); }
    std::size_t index(int r, int c) const { return std::size_t(r) * std::size_t(v.cols) + std::size_t(c); }
    geometry::V at(int r, int c) const { return geometry::vector(xyz() + 3 * index(r, c)); }

    GridView view() const
    {
        GridView g;
        g.xyz = xyz();
        g.valid = reinterpret_cast<const bool*>(valid.data());
        g.piece = piece.empty() ? nullptr : piece.data();
        g.normals = normals();
        if (options.interpolation == Interpolation::Smooth) {
            g.du = du.data();
            g.dv = dv.data();
            g.duv = duv.data();
        }
        return g;
    }
    // Vertex coordinates to physical space (spacing) and back.
    PointMap map() const
    {
        PointMap g;
        g.height = v.rows;
        g.width = v.cols;
        for (int a = 0; a < 3; ++a) {
            g.physical[a * 5] = options.spacing[std::size_t(a)];
            g.destination[a * 5] = 1.0f / options.spacing[std::size_t(a)];
        }
        g.orientation = options.flipNormals ? -1.0f : 1.0f;
        return g;
    }

    // Per-vertex Hermite derivatives: centred differences inside, one-sided at borders
    // and holes, never across a piece's columns.
    void buildDerivatives()
    {
        const int rows = v.rows, cols = v.cols;
        const std::size_t n = std::size_t(rows) * std::size_t(cols) * 3;
        du.assign(n, 0.0f);
        dv.assign(n, 0.0f);
        duv.assign(n, 0.0f);
        const auto ok = [&](int r, int c) { return r >= 0 && r < rows && c >= 0 && c < cols && valid[index(r, c)]; };
        const auto same = [&](int c, int d) { return piece.empty() || piece[std::size_t(c)] == piece[std::size_t(d)]; };
        const auto put = [&](std::vector<float>& d, int r, int c, geometry::V value) { geometry::store(&d[3 * index(r, c)], value); };
        const auto get = [&](const std::vector<float>& d, int r, int c) { return geometry::vector(&d[3 * index(r, c)]); };
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                if (!ok(r, c))
                    continue;
                const auto p = at(r, c);
                const bool left = ok(r, c - 1) && same(c - 1, c), right = ok(r, c + 1) && same(c, c + 1);
                if (left && right)
                    put(du, r, c, (at(r, c + 1) - at(r, c - 1)) * 0.5f);
                else if (right)
                    put(du, r, c, at(r, c + 1) - p);
                else if (left)
                    put(du, r, c, p - at(r, c - 1));
                const bool up = ok(r - 1, c), down = ok(r + 1, c);
                if (up && down)
                    put(dv, r, c, (at(r + 1, c) - at(r - 1, c)) * 0.5f);
                else if (down)
                    put(dv, r, c, at(r + 1, c) - p);
                else if (up)
                    put(dv, r, c, p - at(r - 1, c));
            }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                if (!ok(r, c))
                    continue;
                const bool left = ok(r, c - 1) && same(c - 1, c), right = ok(r, c + 1) && same(c, c + 1);
                const bool up = ok(r - 1, c), down = ok(r + 1, c);
                if (left && right && up && down && ok(r - 1, c - 1) && ok(r - 1, c + 1) && ok(r + 1, c - 1) &&
                    ok(r + 1, c + 1)) {
                    put(duv, r, c, (at(r + 1, c + 1) - at(r + 1, c - 1) - at(r - 1, c + 1) + at(r - 1, c - 1)) * 0.25f);
                    continue;
                }
                geometry::V sum{0, 0, 0};
                int count = 0;
                if (up && down) {
                    sum = sum + (get(du, r + 1, c) - get(du, r - 1, c)) * 0.5f;
                    ++count;
                } else if (down) {
                    sum = sum + (get(du, r + 1, c) - get(du, r, c));
                    ++count;
                } else if (up) {
                    sum = sum + (get(du, r, c) - get(du, r - 1, c));
                    ++count;
                }
                if (left && right) {
                    sum = sum + (get(dv, r, c + 1) - get(dv, r, c - 1)) * 0.5f;
                    ++count;
                } else if (right) {
                    sum = sum + (get(dv, r, c + 1) - get(dv, r, c));
                    ++count;
                } else if (left) {
                    sum = sum + (get(dv, r, c) - get(dv, r, c - 1));
                    ++count;
                }
                if (count)
                    put(duv, r, c, sum * (1.0f / float(count)));
            }
    }

    // Positions and normals of the raster pixels i0 <= i < i0 + w, j0 <= j < j0 + h, into
    // w x h outputs. Pixel (i, j) sits at grid position (x0 + i sx + j kx, y0 + i ky + j sy),
    // computed in double as the CUDA sampler does.
    void resampleRect(const Raster& raster, int i0, int j0, int w, int h, float* xyz, float* nrm) const
    {
        if (raster.ky != 0)
            resampleRows<true>(raster, i0, j0, w, h, xyz, nrm);
        else  // the rows of an unsheared raster each lie on one grid row
            resampleRows<false>(raster, i0, j0, w, h, xyz, nrm);
    }
    template <bool Sheared>
    void resampleRows(const Raster& raster, int i0, int j0, int w, int h, float* xyz, float* nrm) const
    {
        const PointMap g = map();
        const GridView grid = view();
        for (int j = 0; j < h; ++j) {
            const double x = raster.x0 + double(j0 + j) * raster.kx, y = raster.y0 + double(j0 + j) * raster.sy;
            const float row = float(y);
            for (int i = 0; i < w; ++i) {
                const std::size_t o = (std::size_t(j) * std::size_t(w) + std::size_t(i)) * 3;
                float step[3];
                if (!point(g, grid, float(x + double(i0 + i) * raster.sx), Sheared ? float(y + double(i0 + i) * raster.ky) : row,
                           xyz + o, step)) {
                    xyz[o] = xyz[o + 1] = xyz[o + 2] = kNaN;
                    step[0] = step[1] = step[2] = kNaN;
                }
                if (nrm)
                    std::copy_n(step, 3, nrm + o);
            }
        }
    }
};

// The samplers' access to a surface's derived data.
struct SurfaceAccess {
    static const Surface::Impl& of(const Surface& surface) { return *surface.impl_; }
};

Surface::Surface(Vertices vertices, SurfaceOptions options) : impl_(std::make_unique<Impl>())
{
    if (vertices.rows < 0 || vertices.cols < 0 || (vertices.rows && vertices.cols && !vertices.xyz) ||
        vertices.stride < std::size_t(vertices.cols) * 3)
        throw std::invalid_argument("invalid surface vertices");
    auto& s = *impl_;
    s.v = vertices;
    s.options = std::move(options);
    const std::size_t rows = std::size_t(vertices.rows), cols = std::size_t(vertices.cols);
    if (vertices.stride != cols * 3) {
        s.packed.resize(rows * cols * 3);
        for (std::size_t r = 0; r < rows; ++r)
            std::copy_n(vertices.xyz + r * vertices.stride, cols * 3, s.packed.data() + r * cols * 3);
        if (vertices.normals) {
            s.packedNormals.resize(rows * cols * 3);
            for (std::size_t r = 0; r < rows; ++r)
                std::copy_n(vertices.normals + r * vertices.stride, cols * 3, s.packedNormals.data() + r * cols * 3);
        }
    }
    s.valid.resize(rows * cols);
    if (vertices.valid) {
        for (std::size_t k = 0; k < rows * cols; ++k)
            s.valid[k] = vertices.valid[k] != 0;
    } else {
        for (std::size_t k = 0; k < rows * cols; ++k) {
            const float* p = s.xyz() + 3 * k;
            s.valid[k] = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        }
    }
    if (!s.options.components.empty()) {  // unlisted columns belong to no piece
        s.piece.assign(cols, -1);
        int id = 0;
        for (const auto& [first, last] : s.options.components) {
            for (int c = std::max(first, 0); c < std::min(last, vertices.cols); ++c)
                s.piece[std::size_t(c)] = id;
            ++id;
        }
        for (std::size_t c = 0; c < cols; ++c)
            if (s.piece[c] < 0)
                s.piece[c] = -1 - int(c);  // a piece of its own: no patch spans it
    }
    if (s.options.interpolation == Interpolation::Smooth)
        s.buildDerivatives();
}

Surface::~Surface() = default;
const Vertices& Surface::vertices() const { return impl_->v; }
const SurfaceOptions& Surface::options() const { return impl_->options; }

void Surface::resample(const Raster& raster, float* xyz, float* normals) const
{
    if (raster.width < 0 || raster.height < 0)
        throw std::invalid_argument("invalid raster");
    std::vector<float> scratch;
    if (!xyz) {
        scratch.resize(std::size_t(raster.width) * std::size_t(raster.height) * 3);
        xyz = scratch.data();
    }
    const std::size_t rows = std::size_t(raster.height);
    const std::size_t parts = std::clamp<std::size_t>(rows * std::size_t(raster.width) / 65536, 1, 64);
    resamplePool().parallel_for(parts, [&](std::size_t k) {
        const int j0 = int(rows * k / parts), j1 = int(rows * (k + 1) / parts);
        const std::size_t o = std::size_t(j0) * std::size_t(raster.width) * 3;
        impl_->resampleRect(raster, 0, j0, raster.width, j1 - j0, xyz + o, normals ? normals + o : nullptr);
    });
}

void Surface::pointsAt(const float* xy, std::size_t n, float* xyz, float* normals) const
{
    std::vector<float> scratch;
    if (!xyz) {
        scratch.resize(n * 3);
        xyz = scratch.data();
    }
    const std::size_t parts = std::clamp<std::size_t>(n / 65536, 1, 64);
    const PointMap g = impl_->map();
    const GridView grid = impl_->view();
    resamplePool().parallel_for(parts, [&](std::size_t k) {
        for (std::size_t i = n * k / parts; i < n * (k + 1) / parts; ++i) {
            float step[3];
            if (!point(g, grid, xy[i * 2], xy[i * 2 + 1], xyz + i * 3, step)) {
                xyz[i * 3] = xyz[i * 3 + 1] = xyz[i * 3 + 2] = kNaN;
                step[0] = step[1] = step[2] = kNaN;
            }
            if (normals) std::copy_n(step, 3, normals + i * 3);
        }
    });
}

namespace {

// A level's voxels through a few recently used chunks, for sampling::point: a tile's
// samples mostly share a chunk with the previous one. Cache-only lookups use
// Array::cached_chunk (memory and disk caches); others Array::chunk.
template <class Load>
struct ChunkLookup {
    Array* array;
    Load loader;
    bool cachedOnly;
    float fill;
    std::int64_t shape[3], block[3], stride[3], grid[3];
    int shift[3];

    struct Slot {
        std::int64_t key = -1;
        bool available = false;
        ChunkPtr chunk;  // null: not stored (fill)
    };
    std::array<Slot, 8> slots{};
    std::size_t last = 0, next = 0;

    ChunkLookup(const Level& L, Load l, bool cached) : array(L.array), loader(l), cachedOnly(cached)
    {
        const auto& spec = array->spec();
        fill = float(L.type.load(spec.fill.data()));
        for (std::size_t a = 0; a < 3; ++a) {
            shape[a] = spec.shape[a];
            block[a] = spec.chunks[a];
            grid[a] = (shape[a] + block[a] - 1) / block[a];
            shift[a] = std::has_single_bit(std::uint64_t(block[a])) ? std::countr_zero(std::uint64_t(block[a])) : -1;
        }
        stride[2] = std::int64_t(spec.itemsize);
        stride[1] = block[2] * stride[2];
        stride[0] = block[1] * stride[1];
    }
    std::int64_t chunkOf(std::int64_t v, int a) const { return shift[a] >= 0 ? v >> shift[a] : v / block[a]; }
    float load(const unsigned char* p) const { return float(loader(p)); }

    // A sample's chunk is almost always the last one used: that check stays inline, the
    // rest (other slots, reading a chunk) is out of line.
    const unsigned char* voxel(const std::int64_t* v, bool& available)
    {
        const std::int64_t c[3]{chunkOf(v[0], 0), chunkOf(v[1], 1), chunkOf(v[2], 2)};
        const std::int64_t key = (c[0] * grid[1] + c[1]) * grid[2] + c[2];
        const Slot& slot = slots[last].key == key ? slots[last] : find(key, c);
        available = slot.available;
        if (!available || !slot.chunk)
            return nullptr;
        return slot.chunk->data() + (v[0] - c[0] * block[0]) * stride[0] + (v[1] - c[1] * block[1]) * stride[1] +
               (v[2] - c[2] * block[2]) * stride[2];
    }

    [[gnu::noinline]] const Slot& find(std::int64_t key, const std::int64_t* c)
    {
        for (std::size_t k = 0; k < slots.size(); ++k)
            if (slots[k].key == key)
                return slots[last = k];
        Slot& slot = slots[last = next];
        next = (next + 1) % slots.size();
        slot.key = key;
        const Index idx{c[0], c[1], c[2]};
        if (cachedOnly) {
            auto [found, data] = array->cached_chunk(idx);
            slot.available = found;
            slot.chunk = std::move(data);
        } else {
            slot.available = true;
            slot.chunk = array->chunk(idx);
        }
        return slot;
    }
};

// The points (and normals) of a tile's pixels: rows [j0, j0 + h), columns [i0, i0 + w).
using TilePoints = std::function<void(int i0, int j0, int w, int h, float* xyz, float* nrm)>;

constexpr int kTile = 64;

// Samples a raster tile by tile, each tile on one thread of the first level's pool:
// the tile's points, then per pixel its layers, each from the first level holding
// it, reduced and written once. Level indices are taken about `centre`; General: some
// level has a general (non-axis-aligned) map.
template <bool General, class Load>
Result sampleTiles(int width, int height, const Request& request, const TilePoints& points, const float* centre, Load load)
{
    const std::size_t pixels = std::size_t(width) * std::size_t(height);
    const std::size_t layers = request.offsets.size();
    const bool needNormals = std::any_of(request.offsets.begin(), request.offsets.end(), [](float w) { return w != 0; });
    std::vector<LevelMap> maps;
    for (const auto& L : request.levels)
        maps.push_back(levelMap(L.scale, L.offset, L.linear ? &*L.linear : nullptr, centre));
    Result result;
    result.width = width;
    result.height = height;
    result.layers = int(layers);
    result.values.assign(request.composite ? pixels : pixels * layers, kNaN);
    result.level.assign(pixels, Result::kNone);
    const int tilesX = (width + kTile - 1) / kTile, tilesY = (height + kTile - 1) / kTile;
    request.levels.front().array->parallel_for(std::size_t(tilesX) * std::size_t(tilesY), [&](std::size_t t) {
        const int i0 = int(t % std::size_t(tilesX)) * kTile, j0 = int(t / std::size_t(tilesX)) * kTile;
        const int w = std::min(kTile, width - i0), h = std::min(kTile, height - j0);
        if (request.pixels) {  // skip tiles without a requested pixel
            bool any = false;
            for (int j = j0; j < j0 + h && !any; ++j)
                any = std::any_of(request.pixels + std::size_t(j) * width + i0, request.pixels + std::size_t(j) * width + i0 + w,
                                  [](std::uint8_t m) { return m != 0; });
            if (!any)
                return;
        }
        std::vector<ChunkLookup<Load>> lookups;
        lookups.reserve(request.levels.size());
        for (const auto& L : request.levels)
            lookups.emplace_back(L, load, request.cachedOnly);
        std::vector<float> xyz(std::size_t(w) * h * 3), nrm(needNormals ? xyz.size() : 0);
        points(i0, j0, w, h, xyz.data(), needNormals ? nrm.data() : nullptr);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                const std::size_t q = std::size_t(j0 + j) * std::size_t(width) + std::size_t(i0 + i);
                if (request.pixels && !request.pixels[q])
                    continue;
                const float* p = &xyz[(std::size_t(j) * w + i) * 3];
                const float* n = needNormals ? &nrm[(std::size_t(j) * w + i) * 3] : nullptr;
                std::optional<composite::detail::Reducer> reduce;
                if (request.composite)
                    reduce.emplace(*request.composite);
                std::int8_t level = Result::kNone;
                bool missing = false;
                for (std::size_t k = 0; k < layers && !(reduce && reduce->done); ++k) {
                    const float off = request.offsets[k];
                    const float at[3]{p[0] + (off != 0 ? n[0] * off : 0.0f), p[1] + (off != 0 ? n[1] * off : 0.0f),
                                      p[2] + (off != 0 ? n[2] * off : 0.0f)};
                    if (!(std::isfinite(at[0]) && std::isfinite(at[1]) && std::isfinite(at[2])))
                        continue;  // no point in this layer
                    float value = kNaN;
                    std::size_t li = 0;
                    for (; li < lookups.size(); ++li) {
                        float c[3];
                        levelIndex<General>(maps[li], at[0], at[1], at[2], c);
                        if (sampling::point(lookups[li], maps[li].base, c, request.order, value))
                            break;
                    }
                    if (li == lookups.size()) {
                        missing = true;
                        continue;
                    }
                    level = std::max(level, std::int8_t(li));
                    if (reduce)
                        reduce->add(value);
                    else
                        result.values[k * pixels + q] = value;
                }
                if (reduce)
                    result.values[q] = reduce->result();
                result.level[q] = missing ? Result::kMissing : level;
            }
    });
    return result;
}

void checkRequest(const Request& request, int width, int height)
{
    if (request.levels.empty())
        throw std::invalid_argument("a sample request needs at least one level");
    if (request.levels.size() > 127)
        throw std::invalid_argument("a sample request takes at most 127 levels");
    for (const auto& L : request.levels)
        if (!L.array || L.array->ndim() != 3)
            throw std::invalid_argument("sample levels must be 3-D arrays");
    if (request.offsets.empty())
        throw std::invalid_argument("a sample request needs at least one layer");
    if (request.order != 0 && request.order != 1)
        throw std::invalid_argument("order must be 0 or 1");
    if (width < 0 || height < 0)
        throw std::invalid_argument("invalid raster");
    for (const auto& L : request.levels)
        if (L.type.kind != request.levels.front().type.kind || L.type.itemsize != request.levels.front().type.itemsize ||
            L.type.swap != request.levels.front().type.swap)
            throw std::invalid_argument("sample levels must share one element type");
    if (request.composite)
        composite::detail::checkLayers(request.offsets.size(), *request.composite);
}

// The tile loop with the levels' element load chosen once.
Result sampleRaster(int width, int height, const Request& request, const TilePoints& points, const float* centre)
{
    checkRequest(request, width, height);
    Result result;
    const bool general = std::any_of(request.levels.begin(), request.levels.end(), [](const Level& L) { return L.linear.has_value(); });
    request.levels.front().type.visit_load([&](auto load) {
        result = general ? sampleTiles<true>(width, height, request, points, centre, load)
                         : sampleTiles<false>(width, height, request, points, centre, load);
    });
    return result;
}

// The point at a surface raster's middle pixel (NaN where the surface has none there).
std::array<float, 3> centreOf(const Surface& surface, const Raster& raster)
{
    std::array<float, 3> centre{kNaN, kNaN, kNaN};
    if (raster.width > 0 && raster.height > 0)
        SurfaceAccess::of(surface).resampleRect(raster, raster.width / 2, raster.height / 2, 1, 1, centre.data(), nullptr);
    return centre;
}

std::array<float, 3> centreOf(const Plane& plane)
{
    std::array<float, 3> centre;
    for (std::size_t a = 0; a < 3; ++a)
        centre[a] = plane.origin[a] + plane.dx[a] * float(plane.width / 2) + plane.dy[a] * float(plane.height / 2);
    return centre;
}

}  // namespace

namespace {

// The CPU tile loop.
Result sampleHost(const Surface& surface, const Raster& raster, const Request& request)
{
    checkRequest(request, raster.width, raster.height);
    const auto& impl = SurfaceAccess::of(surface);
    const auto centre = centreOf(surface, raster);
    return sampleRaster(raster.width, raster.height, request, [&](int i0, int j0, int w, int h, float* xyz, float* nrm) {
        impl.resampleRect(raster, i0, j0, w, h, xyz, nrm);
    }, centre.data());
}

Result sampleHost(const Plane& plane, const Request& request)
{
    const auto centre = centreOf(plane);
    return sampleRaster(plane.width, plane.height, request, [&](int i0, int j0, int w, int h, float* xyz, float* nrm) {
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                float* p = xyz + (std::size_t(j) * w + i) * 3;
                for (std::size_t a = 0; a < 3; ++a) {
                    p[a] = plane.origin[a] + plane.dy[a] * float(j0 + j) + plane.dx[a] * float(i0 + i);
                    if (nrm)
                        nrm[(std::size_t(j) * w + i) * 3 + a] = plane.normal[a];
                }
            }
    }, centre.data());
}

}  // namespace

}  // namespace vcr::grid

#ifdef VC_READER_CUDA
#include "../reader_cuda/reader.hpp"
#include "cuda.hpp"

namespace vcr::grid::cuda {

namespace {

// f() with every reader's resident bricks pinned (locks taken in address order).
template <class F>
void withResident(const std::vector<reader_cuda::Reader*>& readers, std::size_t i, std::vector<reader_cuda::Source>& sources,
                  const std::vector<reader_cuda::Reader*>& order, F&& f)
{
    if (i == order.size()) {
        f();
        return;
    }
    order[i]->with_resident([&](const reader_cuda::Source& source) {
        for (std::size_t k = 0; k < readers.size(); ++k)
            if (readers[k] == order[i])
                sources[k] = source;
        withResident(readers, i + 1, sources, order, f);
    });
}

device::Storage upload(const void* data, std::size_t bytes, int deviceId)
{
    auto storage = device::allocate(std::max<std::size_t>(bytes, 16), deviceId);
    if (bytes)
        device::check(cudaMemcpyAsync(storage->ptr, data, bytes, cudaMemcpyHostToDevice, device::stream()));
    return storage;
}

// Samples a raster on the readers' device. `base` carries the points: a plane, or a
// surface grid (device copies of its vertex data, kept alive by `keep`) with raster.
Result sampleDevice(int width, int height, const Request& request, const std::vector<reader_cuda::Reader*>& readers,
                    cuda_detail::Params base, const float* centre)
{
    checkRequest(request, width, height);
    if (readers.size() != request.levels.size())
        throw std::invalid_argument("one CUDA reader per request level");
    for (auto* r : readers)
        if (!r || r->device_id() != readers.front()->device_id())
            throw std::invalid_argument("CUDA readers must share one device");
    const int deviceId = readers.front()->device_id();
    device::Guard guard(deviceId);
    const std::size_t pixels = std::size_t(width) * std::size_t(height);
    const std::size_t layers = request.offsets.size();
    Result result;
    result.width = width;
    result.height = height;
    result.layers = int(layers);
    result.values.resize(request.composite ? pixels : pixels * layers);
    result.level.resize(pixels);
    if (!pixels)
        return result;

    const auto offsets = upload(request.offsets.data(), layers * sizeof(float), deviceId);
    const auto mask = request.pixels ? upload(request.pixels, pixels, deviceId) : device::Storage{};
    const auto values = device::allocate(result.values.size() * sizeof(float), deviceId);
    const auto levels = device::allocate(pixels, deviceId);
    std::vector<cuda_detail::DeviceLevel> host(readers.size());
    for (std::size_t li = 0; li < readers.size(); ++li)
        host[li].map = levelMap(request.levels[li].scale, request.levels[li].offset,
                                request.levels[li].linear ? &*request.levels[li].linear : nullptr, centre);
    // Blocking requests note the first level's lacking bricks, one bit each.
    std::size_t words = 0;
    device::Storage lacking;
    std::vector<reader_cuda::Reader*> order = readers;
    std::sort(order.begin(), order.end());
    order.erase(std::unique(order.begin(), order.end()), order.end());
    std::vector<reader_cuda::Source> sources(readers.size());
    const auto pass = [&] {
        withResident(readers, 0, sources, order, [&] {
            for (std::size_t li = 0; li < readers.size(); ++li)
                host[li].source = sources[li];
            if (!request.cachedOnly) {
                if (!lacking) {
                    const auto& g = sources.front().grid;
                    words = std::size_t(g[0] * g[1] * g[2] + 31) / 32;
                    lacking = device::allocate(words * 4 + 4, deviceId);
                }
                device::check(cudaMemsetAsync(lacking->ptr, 0, words * 4 + 4, device::stream()));
            }
            const auto table = upload(host.data(), host.size() * sizeof(cuda_detail::DeviceLevel), deviceId);
            cuda_detail::Params P = base;
            P.width = width;
            P.height = height;
            P.layers = unsigned(layers);
            P.offsets = static_cast<const float*>(offsets->ptr);
            P.order = request.order;
            P.pixels = mask ? static_cast<const std::uint8_t*>(mask->ptr) : nullptr;
            P.reduce = request.composite.has_value();
            if (request.composite)
                P.composite = *request.composite;
            P.levels = static_cast<const cuda_detail::DeviceLevel*>(table->ptr);
            P.levelCount = int(readers.size());
            P.values = static_cast<float*>(values->ptr);
            P.level = static_cast<std::int8_t*>(levels->ptr);
            P.lacking = lacking ? static_cast<unsigned*>(lacking->ptr) : nullptr;
            cuda_detail::launch(P);
            device::complete();
        });
    };
    pass();
    if (lacking) {  // load what the first level lacked, then sample again
        std::vector<unsigned> bits(words);
        device::check(cudaMemcpy(bits.data(), lacking->ptr, words * 4, cudaMemcpyDeviceToHost));
        std::vector<std::int64_t> keys;
        for (std::size_t w = 0; w < bits.size(); ++w)
            for (unsigned b = bits[w]; b; b &= b - 1)
                keys.push_back(std::int64_t(w * 32 + std::size_t(std::countr_zero(b))));
        if (!keys.empty()) {
            readers.front()->load_bricks(keys);
            pass();
        }
    }
    device::check(cudaMemcpy(result.values.data(), values->ptr, result.values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    device::check(cudaMemcpy(result.level.data(), levels->ptr, pixels, cudaMemcpyDeviceToHost));
    return result;
}

}  // namespace

Result sample(const Surface& surface, const Raster& raster, const Request& request,
              const std::vector<reader_cuda::Reader*>& readers)
{
    checkRequest(request, raster.width, raster.height);
    if (readers.empty())
        throw std::invalid_argument("one CUDA reader per request level");
    device::Guard guard(readers.front()->device_id());
    const int id = readers.front()->device_id();
    const auto& impl = SurfaceAccess::of(surface);
    const std::size_t vertices = std::size_t(impl.v.rows) * std::size_t(impl.v.cols);
    const GridView host = impl.view();
    const auto xyz = upload(host.xyz, vertices * 12, id);
    const auto valid = upload(host.valid, vertices, id);
    const auto piece = upload(host.piece, host.piece ? std::size_t(impl.v.cols) * sizeof(int) : 0, id);
    const auto normals = upload(host.normals, host.normals ? vertices * 12 : 0, id);
    const bool smooth = host.du != nullptr;
    const auto du = upload(host.du, smooth ? vertices * 12 : 0, id), dv = upload(host.dv, smooth ? vertices * 12 : 0, id),
               duv = upload(host.duv, smooth ? vertices * 12 : 0, id);
    cuda_detail::Params base;
    base.x0 = raster.x0;
    base.y0 = raster.y0;
    base.sx = raster.sx;
    base.sy = raster.sy;
    base.kx = raster.kx;
    base.ky = raster.ky;
    base.map = impl.map();
    base.grid.xyz = static_cast<const float*>(xyz->ptr);
    base.grid.valid = static_cast<const bool*>(valid->ptr);
    base.grid.piece = host.piece ? static_cast<const int*>(piece->ptr) : nullptr;
    base.grid.normals = host.normals ? static_cast<const float*>(normals->ptr) : nullptr;
    if (smooth) {
        base.grid.du = static_cast<const float*>(du->ptr);
        base.grid.dv = static_cast<const float*>(dv->ptr);
        base.grid.duv = static_cast<const float*>(duv->ptr);
    }
    const auto centre = centreOf(surface, raster);
    return sampleDevice(raster.width, raster.height, request, readers, base, centre.data());
}

Result sample(const Plane& plane, const Request& request, const std::vector<reader_cuda::Reader*>& readers)
{
    cuda_detail::Params base;
    base.plane = true;
    std::copy(plane.origin.begin(), plane.origin.end(), base.origin);
    std::copy(plane.dx.begin(), plane.dx.end(), base.dx);
    std::copy(plane.dy.begin(), plane.dy.end(), base.dy);
    std::copy(plane.normal.begin(), plane.normal.end(), base.normal);
    const auto centre = centreOf(plane);
    return sampleDevice(plane.width, plane.height, request, readers, base, centre.data());
}

}  // namespace vcr::grid::cuda
#endif

namespace vcr::grid {

namespace {

// Samples on the GPU when every level has a CUDA reader, falling back to the CPU
// when a GPU pass fails; otherwise on the CPU.
template <class Host, class Device>
Result dispatch(const Request& request, Host&& host, Device&& device)
{
#ifdef VC_READER_CUDA
    std::vector<reader_cuda::Reader*> readers;
    for (const auto& L : request.levels)
        if (L.cuda)
            readers.push_back(L.cuda);
    if (!readers.empty() && readers.size() == request.levels.size()) {
        std::string failure;
        try {
            return device(readers);
        } catch (const std::exception& e) {
            failure = e.what();
        }
        std::sort(readers.begin(), readers.end());
        readers.erase(std::unique(readers.begin(), readers.end()), readers.end());
        for (auto* reader : readers)
            reader->clear();
        Result result = host();
        result.cudaFailure = failure.empty() ? "GPU pass failed" : failure;
        return result;
    }
#else
    (void)device;
#endif
    return host();
}

}  // namespace

Result sample(const Surface& surface, const Raster& raster, const Request& request)
{
    checkRequest(request, raster.width, raster.height);
    return dispatch(request, [&] { return sampleHost(surface, raster, request); },
                    [&](const std::vector<reader_cuda::Reader*>& readers) { return cuda::sample(surface, raster, request, readers); });
}

Result sample(const Plane& plane, const Request& request)
{
    checkRequest(request, plane.width, plane.height);
    return dispatch(request, [&] { return sampleHost(plane, request); },
                    [&](const std::vector<reader_cuda::Reader*>& readers) { return cuda::sample(plane, request, readers); });
}

}  // namespace vcr::grid
