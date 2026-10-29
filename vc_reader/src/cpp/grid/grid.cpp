// Surfaces stored as grids of vertices, and sampling volumes along them and along planes.
#include "vc_reader/grid.hpp"

#include "../array.hpp"
#include "../rounding.hpp"
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

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

struct Vec {
    float x = 0, y = 0, z = 0;
};
inline Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec operator*(Vec a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline bool finite(Vec a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }
inline Vec nanVec() { return {kNaN, kNaN, kNaN}; }

// A unit normal of the cross product, or NaN when it has no direction.
inline Vec unitCross(Vec u, Vec v)
{
    const Vec n{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
    const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
    if (len2 == 0.0f || len2 != len2)
        return nanVec();
    const float inv = 1.0f / std::sqrt(len2);
    return {n.x * inv, n.y * inv, n.z * inv};
}

inline Vec bilerp(Vec p00, Vec p01, Vec p10, Vec p11, float wx, float wy)
{
    const float iwx = 1.0f - wx, iwy = 1.0f - wy;
    return (p00 * iwx + p01 * wx) * iwy + (p10 * iwx + p11 * wx) * wy;
}

// Cubic Hermite basis for a unit cell: p weights endpoint values, m endpoint
// derivatives; the d* fields are their derivatives with respect to t.
struct Hermite {
    float p0, m0, p1, m1, dp0, dm0, dp1, dm1;
};
inline Hermite hermite(float t)
{
    const float t2 = t * t, t3 = t2 * t;
    return {2.f * t3 - 3.f * t2 + 1.f, t3 - 2.f * t2 + t, -2.f * t3 + 3.f * t2, t3 - t2,
            6.f * t2 - 6.f * t,       3.f * t2 - 4.f * t + 1.f, -6.f * t2 + 6.f * t, 3.f * t2 - 2.f * t};
}

ThreadPool& resamplePool()
{
    static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

}  // namespace

struct Surface::Impl {
    Vertices v;
    SurfaceOptions options;
    std::vector<std::pair<int, int>> pieces;  // column ranges, clamped; one for the whole grid without components
    bool replicate = true;                    // a one-piece grid replicates its border vertices

    std::once_flag validOnce, normalsOnce, derivativesOnce;
    std::vector<std::uint8_t> valid;          // per vertex
    std::vector<Vec> normals;                 // per vertex (Linear)
    std::vector<Vec> du, dv, duv;             // per vertex (Smooth)

    Vec at(int r, int c) const
    {
        const float* p = v.xyz + std::size_t(r) * v.stride + std::size_t(c) * 3;
        return {p[0], p[1], p[2]};
    }
    std::size_t index(int r, int c) const { return std::size_t(r) * std::size_t(v.cols) + std::size_t(c); }

    void buildValid()
    {
        valid.assign(std::size_t(v.rows) * std::size_t(v.cols), 0);
        for (int r = 0; r < v.rows; ++r)
            for (int c = 0; c < v.cols; ++c) {
                const Vec p = at(r, c);
                const bool nan = p.x != p.x || p.y != p.y || p.z != p.z;
                const bool sentinel = p.x == -1.f && p.y == -1.f && p.z == -1.f;
                valid[index(r, c)] = nan || sentinel ? 0 : 255;
            }
    }

    // A stored point a normal may difference: finite, no -1 coordinate, z above 0.
    static bool stored(Vec p)
    {
        return finite(p) && p.x != -1.f && p.y != -1.f && p.z != -1.f && p.z > 0;
    }

    // Central-difference vertex normals; NaN on the border and where a neighbour is missing.
    void buildNormals()
    {
        normals.assign(std::size_t(v.rows) * std::size_t(v.cols), nanVec());
        for (int r = 1; r < v.rows - 1; ++r)
            for (int c = 1; c < v.cols - 1; ++c) {
                if (at(r, c).x == -1.f)
                    continue;
                const Vec xl = at(r, c - 1), xr = at(r, c + 1), yu = at(r - 1, c), yd = at(r + 1, c);
                if (!stored(xl) || !stored(xr) || !stored(yu) || !stored(yd))
                    continue;
                normals[index(r, c)] = unitCross(xr - xl, yd - yu);
            }
    }

    // Per-vertex derivatives of the Hermite patches: centred differences inside,
    // one-sided at borders and holes, never across a piece's columns.
    void buildDerivatives()
    {
        std::call_once(validOnce, [this] { buildValid(); });
        const int rows = v.rows, cols = v.cols;
        const std::size_t n = std::size_t(rows) * std::size_t(cols);
        du.assign(n, Vec{});
        dv.assign(n, Vec{});
        duv.assign(n, Vec{});
        std::vector<int> begin(std::size_t(cols), 0), end(std::size_t(cols), cols);
        if (!options.components.empty()) {
            std::fill(begin.begin(), begin.end(), -1);
            std::fill(end.begin(), end.end(), -1);
            for (const auto& [first, last] : pieces)
                for (int c = first; c < last; ++c) {
                    begin[std::size_t(c)] = first;
                    end[std::size_t(c)] = last;
                }
        }
        const auto ok = [&](int r, int c) {
            return r >= 0 && r < rows && c >= 0 && c < cols && begin[std::size_t(c)] >= 0 && valid[index(r, c)];
        };
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                if (!ok(r, c))
                    continue;
                const Vec p = at(r, c);
                const bool left = c > begin[std::size_t(c)] && ok(r, c - 1);
                const bool right = c + 1 < end[std::size_t(c)] && ok(r, c + 1);
                Vec& u = du[index(r, c)];
                if (left && right)
                    u = (at(r, c + 1) - at(r, c - 1)) * 0.5f;
                else if (right)
                    u = at(r, c + 1) - p;
                else if (left)
                    u = p - at(r, c - 1);
                const bool up = ok(r - 1, c), down = ok(r + 1, c);
                Vec& w = dv[index(r, c)];
                if (up && down)
                    w = (at(r + 1, c) - at(r - 1, c)) * 0.5f;
                else if (down)
                    w = at(r + 1, c) - p;
                else if (up)
                    w = p - at(r - 1, c);
            }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                if (!ok(r, c))
                    continue;
                const bool left = c > begin[std::size_t(c)] && ok(r, c - 1);
                const bool right = c + 1 < end[std::size_t(c)] && ok(r, c + 1);
                const bool up = ok(r - 1, c), down = ok(r + 1, c);
                if (left && right && up && down && ok(r - 1, c - 1) && ok(r - 1, c + 1) && ok(r + 1, c - 1) &&
                    ok(r + 1, c + 1)) {
                    duv[index(r, c)] =
                        (at(r + 1, c + 1) - at(r + 1, c - 1) - at(r - 1, c + 1) + at(r - 1, c - 1)) * 0.25f;
                    continue;
                }
                Vec sum{};
                int count = 0;
                if (up && down) {
                    sum = sum + (du[index(r + 1, c)] - du[index(r - 1, c)]) * 0.5f;
                    ++count;
                } else if (down) {
                    sum = sum + (du[index(r + 1, c)] - du[index(r, c)]);
                    ++count;
                } else if (up) {
                    sum = sum + (du[index(r, c)] - du[index(r - 1, c)]);
                    ++count;
                }
                if (left && right) {
                    sum = sum + (dv[index(r, c + 1)] - dv[index(r, c - 1)]) * 0.5f;
                    ++count;
                } else if (right) {
                    sum = sum + (dv[index(r, c + 1)] - dv[index(r, c)]);
                    ++count;
                } else if (left) {
                    sum = sum + (dv[index(r, c)] - dv[index(r, c - 1)]);
                    ++count;
                }
                if (count)
                    duv[index(r, c)] = sum * (1.0f / float(count));
            }
    }

    void prepare(bool wantNormals)
    {
        std::call_once(validOnce, [this] { buildValid(); });
        if (options.interpolation == Interpolation::Smooth)
            std::call_once(derivativesOnce, [this] { buildDerivatives(); });
        else if (wantNormals)
            std::call_once(normalsOnce, [this] { buildNormals(); });
    }

    // Whether the pixel at grid position (gx, gy) lies on piece [c0, c1): the
    // validity rule of the piece's vertices, outside the piece never.
    bool onPiece(int c0, int c1, double gx, double gy) const
    {
        const int sc = c1 - c0, sr = v.rows;
        const double fx = gx - c0;
        if (options.strictQuads) {
            if (sc < 2 || sr < 2 || fx < 0.0 || fx > double(sc - 1) || gy < 0.0 || gy > double(sr - 1))
                return false;
            const int col = std::min(int(std::floor(fx)), sc - 2) + c0;
            const int row = std::min(int(std::floor(gy)), sr - 2);
            return valid[index(row, col)] && valid[index(row, col + 1)] && valid[index(row + 1, col)] &&
                   valid[index(row + 1, col + 1)];
        }
        const long ix = std::lround(fx), iy = std::lround(gy);
        return ix >= 0 && ix < sc && iy >= 0 && iy < sr && valid[index(int(iy), int(ix) + c0)];
    }

    // Position (and normal) of piece [c0, c1) at grid position (gx, gy); NaN outside a
    // piece with constant borders.
    void evaluate(int c0, int c1, double gx, double gy, bool wantNormal, Vec& p, Vec& n) const
    {
        const int sc = c1 - c0, sr = v.rows;
        double fx = gx - c0, fy = gy;
        const double xmax = double(sc - 1), ymax = double(sr - 1);
        if (replicate) {
            fx = std::clamp(fx, 0.0, xmax);
            fy = std::clamp(fy, 0.0, ymax);
        } else if (fx < 0.0 || fx > xmax || fy < 0.0 || fy > ymax) {
            p = n = nanVec();
            return;
        }
        n = nanVec();
        if (options.interpolation == Interpolation::Smooth) {
            int x0 = int(fx), y0 = int(fy);
            if (sc >= 2 && x0 >= sc - 1)
                x0 = sc - 2;
            if (sr >= 2 && y0 >= sr - 1)
                y0 = sr - 2;
            const int x1 = std::min(x0 + 1, sc - 1), y1 = std::min(y0 + 1, sr - 1);
            const float wx = float(fx - x0), wy = float(fy - y0);
            const int a0 = x0 + c0, a1 = x1 + c0;
            const bool complete = sc >= 2 && sr >= 2 && valid[index(y0, a0)] && valid[index(y0, a1)] &&
                                  valid[index(y1, a0)] && valid[index(y1, a1)];
            if (!complete) {
                p = blendPresent(y0, a0, y1, a1, wx, wy);
                return;
            }
            const Hermite bx = hermite(wx), by = hermite(wy);
            const auto D = [&](const std::vector<Vec>& d, int r, int c) { return d[index(r, c)]; };
            const Vec p00 = at(y0, a0), p01 = at(y0, a1), p10 = at(y1, a0), p11 = at(y1, a1);
            const Vec q0 = p00 * bx.p0 + D(du, y0, a0) * bx.m0 + p01 * bx.p1 + D(du, y0, a1) * bx.m1;
            const Vec q1 = p10 * bx.p0 + D(du, y1, a0) * bx.m0 + p11 * bx.p1 + D(du, y1, a1) * bx.m1;
            const Vec r0 = D(dv, y0, a0) * bx.p0 + D(duv, y0, a0) * bx.m0 + D(dv, y0, a1) * bx.p1 + D(duv, y0, a1) * bx.m1;
            const Vec r1 = D(dv, y1, a0) * bx.p0 + D(duv, y1, a0) * bx.m0 + D(dv, y1, a1) * bx.p1 + D(duv, y1, a1) * bx.m1;
            p = q0 * by.p0 + r0 * by.m0 + q1 * by.p1 + r1 * by.m1;
            if (wantNormal) {
                const Vec q0u = p00 * bx.dp0 + D(du, y0, a0) * bx.dm0 + p01 * bx.dp1 + D(du, y0, a1) * bx.dm1;
                const Vec q1u = p10 * bx.dp0 + D(du, y1, a0) * bx.dm0 + p11 * bx.dp1 + D(du, y1, a1) * bx.dm1;
                const Vec r0u = D(dv, y0, a0) * bx.dp0 + D(duv, y0, a0) * bx.dm0 + D(dv, y0, a1) * bx.dp1 +
                                D(duv, y0, a1) * bx.dm1;
                const Vec r1u = D(dv, y1, a0) * bx.dp0 + D(duv, y1, a0) * bx.dm0 + D(dv, y1, a1) * bx.dp1 +
                                D(duv, y1, a1) * bx.dm1;
                const Vec pu = q0u * by.p0 + r0u * by.m0 + q1u * by.p1 + r1u * by.m1;
                const Vec pv = q0 * by.dp0 + r0 * by.dm0 + q1 * by.dp1 + r1 * by.dm1;
                n = unitCross(pu, pv);
            }
            return;
        }
        const int x0 = int(fx), y0 = int(fy);
        const int x1 = std::min(x0 + 1, sc - 1), y1 = std::min(y0 + 1, sr - 1);
        p = blendPresent(y0, x0 + c0, y1, x1 + c0, float(fx - x0), float(fy - y0));
    }

    // The bilinear blend of a cell's corners; with a missing corner, of the present
    // ones only, their weights renormalised (a missing vertex's coordinates never
    // enter a position). NaN when no present corner has weight.
    Vec blendPresent(int y0, int x0, int y1, int x1, float wx, float wy) const
    {
        const bool v00 = valid[index(y0, x0)], v01 = valid[index(y0, x1)];
        const bool v10 = valid[index(y1, x0)], v11 = valid[index(y1, x1)];
        if (v00 && v01 && v10 && v11)
            return bilerp(at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1), wx, wy);
        const float w[4]{(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy};
        const bool ok[4]{v00, v01, v10, v11};
        const Vec c[4]{at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1)};
        Vec sum{};
        float total = 0;
        for (int k = 0; k < 4; ++k)
            if (ok[k] && w[k] > 0) {
                sum = sum + c[k] * w[k];
                total += w[k];
            }
        return total > 0 ? sum * (1.0f / total) : nanVec();
    }

    // Positions and normals of the raster rows [j0, j1), into out rows starting at j0.
    void resampleRows(const Raster& raster, int j0, int j1, float* xyz, float* nrm) const
    {
        const bool wantNormals = nrm != nullptr;
        const bool smooth = options.interpolation == Interpolation::Smooth;
        for (int j = j0; j < j1; ++j) {
            const double gy = raster.y0 + double(j) * raster.sy;
            for (int i = 0; i < raster.width; ++i) {
                const double gx = raster.x0 + double(i) * raster.sx;
                Vec p = nanVec(), n = nanVec();
                bool on = false;
                for (const auto& [c0, c1] : pieces) {  // later pieces win where they overlap
                    if (!onPiece(c0, c1, gx, gy))
                        continue;
                    Vec pp, pn;
                    evaluate(c0, c1, gx, gy, wantNormals && smooth, pp, pn);
                    p = pp;
                    n = pn;
                    on = true;
                }
                if (on && finite(p) && wantNormals && !smooth) {  // the nearest vertex's normal
                    const long ix = std::lround(gx), iy = std::lround(gy);
                    n = ix >= 0 && ix < v.cols && iy >= 0 && iy < v.rows ? normals[index(int(iy), int(ix))] : nanVec();
                }
                if (!on || !finite(p))
                    p = n = nanVec();
                const std::size_t o = (std::size_t(j - j0) * std::size_t(raster.width) + std::size_t(i)) * 3;
                xyz[o] = p.x;
                xyz[o + 1] = p.y;
                xyz[o + 2] = p.z;
                if (wantNormals) {
                    nrm[o] = n.x;
                    nrm[o + 1] = n.y;
                    nrm[o + 2] = n.z;
                }
            }
        }
    }
};

Surface::Surface(Vertices vertices, SurfaceOptions options) : impl_(std::make_unique<Impl>())
{
    if (vertices.rows < 0 || vertices.cols < 0 || (vertices.rows && vertices.cols && !vertices.xyz) ||
        vertices.stride < std::size_t(vertices.cols) * 3)
        throw std::invalid_argument("invalid surface vertices");
    impl_->v = vertices;
    impl_->options = std::move(options);
    if (impl_->options.components.empty()) {
        impl_->pieces = {{0, vertices.cols}};
    } else {
        impl_->replicate = false;
        for (const auto& [first, last] : impl_->options.components) {
            const int c0 = std::clamp(first, 0, vertices.cols), c1 = std::clamp(last, 0, vertices.cols);
            if (c1 > c0)
                impl_->pieces.emplace_back(c0, c1);
        }
    }
}

Surface::~Surface() = default;
const Vertices& Surface::vertices() const { return impl_->v; }
const SurfaceOptions& Surface::options() const { return impl_->options; }

void Surface::resample(const Raster& raster, float* xyz, float* normals) const
{
    if (raster.width < 0 || raster.height < 0)
        throw std::invalid_argument("invalid raster");
    impl_->prepare(normals != nullptr);
    std::vector<float> scratch;
    if (!xyz) {
        scratch.resize(std::size_t(raster.width) * std::size_t(raster.height) * 3);
        xyz = scratch.data();
    }
    if (impl_->v.rows == 0 || impl_->v.cols == 0 || impl_->pieces.empty()) {
        const std::size_t n = std::size_t(raster.width) * std::size_t(raster.height) * 3;
        std::fill_n(xyz, n, kNaN);
        if (normals)
            std::fill_n(normals, n, kNaN);
        return;
    }
    // Even parts of the rows: plain work division.
    const std::size_t rows = std::size_t(raster.height);
    const std::size_t parts = std::clamp<std::size_t>(rows * std::size_t(raster.width) / 65536, 1, 64);
    resamplePool().parallel_for(parts, [&](std::size_t k) {
        const int j0 = int(rows * k / parts), j1 = int(rows * (k + 1) / parts);
        const std::size_t o = std::size_t(j0) * std::size_t(raster.width) * 3;
        impl_->resampleRows(raster, j0, j1, xyz + o, normals ? normals + o : nullptr);
    });
}

namespace {

// One level as the tile loop reads it: its array, its chunk grid, and a few
// recently used chunks (a pixel's samples and its neighbours' mostly share one).
template <class Load>
struct LevelReader {
    Array* array;
    Load load;
    float fill;
    std::array<double, 3> scale, offset;
    std::array<std::int64_t, 3> shape, chunks, grid, stride;  // stride: bytes per step in a chunk
    std::array<int, 3> shift;  // log2 of a power-of-two chunk edge, or -1
    bool cachedOnly;

    struct Slot {
        std::array<std::int64_t, 3> index{-1, -1, -1};
        bool available = false;
        ChunkPtr chunk;  // null: not stored (reads as fill)
    };
    std::array<Slot, 8> slots{};
    std::size_t last = 0, next = 0;

    LevelReader(const Level& L, Load l, bool cached) : array(L.array), load(l), scale(L.scale), offset(L.offset), cachedOnly(cached)
    {
        const auto& spec = array->spec();
        fill = float(L.type.load(spec.fill.data()));
        for (std::size_t a = 0; a < 3; ++a) {
            shape[a] = spec.shape[a];
            chunks[a] = spec.chunks[a];
            grid[a] = (shape[a] + chunks[a] - 1) / chunks[a];
            shift[a] = std::has_single_bit(std::uint64_t(chunks[a])) ? std::countr_zero(std::uint64_t(chunks[a])) : -1;
        }
        stride = {chunks[1] * chunks[2] * std::int64_t(spec.itemsize), chunks[2] * std::int64_t(spec.itemsize),
                  std::int64_t(spec.itemsize)};
    }

    std::int64_t chunkIndex(std::int64_t v, std::size_t a) const { return shift[a] >= 0 ? v >> shift[a] : v / chunks[a]; }

    // The chunk with chunk-grid index c, or null when it is not available (cache-only
    // reads of a chunk only the source has).
    const Slot* chunk(const std::array<std::int64_t, 3>& c)
    {
        if (slots[last].index == c)
            return slots[last].available ? &slots[last] : nullptr;
        for (std::size_t k = 0; k < slots.size(); ++k)
            if (slots[k].index == c) {
                last = k;
                return slots[k].available ? &slots[k] : nullptr;
            }
        Slot& slot = slots[next];
        last = next;
        next = (next + 1) % slots.size();
        slot.index = c;
        const Index idx{c[0], c[1], c[2]};
        if (cachedOnly) {
            auto [found, data] = array->cached_chunk(idx);
            slot.available = found;
            slot.chunk = std::move(data);
        } else {
            slot.available = true;
            slot.chunk = array->chunk(idx);
        }
        return slot.available ? &slot : nullptr;
    }

    // Byte offset of voxel v in its chunk, whose first voxel is `base`.
    std::int64_t at(const std::int64_t* v, const std::array<std::int64_t, 3>& base) const
    {
        return (v[0] - base[0]) * stride[0] + (v[1] - base[1]) * stride[1] + (v[2] - base[2]) * stride[2];
    }
    float voxel(const Slot& slot, std::int64_t byte) const
    {
        return slot.chunk ? float(load(slot.chunk->data() + byte)) : fill;
    }
    bool inside(const std::int64_t* v) const
    {
        return v[0] >= 0 && v[1] >= 0 && v[2] >= 0 && v[0] < shape[0] && v[1] < shape[1] && v[2] < shape[2];
    }
    // The value of voxel v (inside the array) through the chunk memo; false when unavailable.
    bool read(const std::int64_t* v, float& value)
    {
        const std::array<std::int64_t, 3> c{chunkIndex(v[0], 0), chunkIndex(v[1], 1), chunkIndex(v[2], 2)};
        const Slot* slot = chunk(c);
        if (!slot)
            return false;
        value = voxel(*slot, at(v, {c[0] * chunks[0], c[1] * chunks[1], c[2] * chunks[2]}));
        return true;
    }

    // The value at the XYZ point p: nearest (order 0) or trilinear; voxels outside the
    // array read as fill. False when a chunk it needs is not available.
    bool sample(const float* p, int order, float& value)
    {
        double c[3];
        for (std::size_t a = 0; a < 3; ++a)
            c[a] = scale[a] * double(p[2 - a]) + offset[a];
        if (order == 0) {
            std::int64_t v[3];
            for (std::size_t a = 0; a < 3; ++a)
                v[a] = nearest_index(c[a]);
            if (!inside(v)) {
                value = fill;
                return true;
            }
            return read(v, value);
        }
        std::int64_t lo[3];
        float f[3];
        for (std::size_t a = 0; a < 3; ++a) {
            const double low = std::floor(c[a]);
            lo[a] = std::int64_t(low);
            f[a] = float(c[a] - low);
        }
        float corner[8];
        // Usually the whole cell lies in one chunk inside the array: one lookup, strides.
        const std::array<std::int64_t, 3> c0{chunkIndex(lo[0], 0), chunkIndex(lo[1], 1), chunkIndex(lo[2], 2)};
        const std::int64_t hi[3]{lo[0] + 1, lo[1] + 1, lo[2] + 1};
        if (inside(lo) && inside(hi) && chunkIndex(hi[0], 0) == c0[0] && chunkIndex(hi[1], 1) == c0[1] &&
            chunkIndex(hi[2], 2) == c0[2]) {
            const Slot* slot = chunk(c0);
            if (!slot)
                return false;
            if (!slot->chunk) {
                value = fill;
                return true;
            }
            const std::uint8_t* d = slot->chunk->data() + at(lo, {c0[0] * chunks[0], c0[1] * chunks[1], c0[2] * chunks[2]});
            for (int k = 0; k < 8; ++k)
                corner[k] = float(load(d + (k >> 2 & 1) * stride[0] + (k >> 1 & 1) * stride[1] + (k & 1) * stride[2]));
        } else {
            for (int k = 0; k < 8; ++k) {
                const std::int64_t v[3]{lo[0] + (k >> 2 & 1), lo[1] + (k >> 1 & 1), lo[2] + (k & 1)};
                if (!inside(v))
                    corner[k] = fill;
                else if (!read(v, corner[k]))
                    return false;
            }
        }
        const float x00 = corner[0] + f[2] * (corner[1] - corner[0]), x01 = corner[2] + f[2] * (corner[3] - corner[2]);
        const float x10 = corner[4] + f[2] * (corner[5] - corner[4]), x11 = corner[6] + f[2] * (corner[7] - corner[6]);
        const float y0 = x00 + f[1] * (x01 - x00), y1 = x10 + f[1] * (x11 - x10);
        value = y0 + f[0] * (y1 - y0);
        return true;
    }
};

// The points (and normals) of a tile's pixels: rows [j0, j0 + h), columns [i0, i0 + w).
using TilePoints = std::function<void(int i0, int j0, int w, int h, float* xyz, float* nrm)>;

constexpr int kTile = 64;

// Samples a raster tile by tile, each tile on one thread of the first level's pool:
// the tile's points, then per pixel its layers, each from the first level holding
// it, reduced and written once.
template <class Load>
Result sampleTiles(int width, int height, const Request& request, const TilePoints& points, Load load)
{
    const std::size_t pixels = std::size_t(width) * std::size_t(height);
    const std::size_t layers = request.offsets.size();
    const bool needNormals = std::any_of(request.offsets.begin(), request.offsets.end(), [](float w) { return w != 0; });
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
        std::vector<LevelReader<Load>> readers;
        readers.reserve(request.levels.size());
        for (const auto& L : request.levels)
            readers.emplace_back(L, load, request.cachedOnly);
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
                    while (li < readers.size() && !readers[li].sample(at, request.order, value))
                        ++li;
                    if (li == readers.size()) {
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
Result sampleRaster(int width, int height, const Request& request, const TilePoints& points)
{
    checkRequest(request, width, height);
    Result result;
    request.levels.front().type.visit_load([&](auto load) {
        result = sampleTiles(width, height, request, points, load);
    });
    return result;
}

}  // namespace

Result sample(const Surface& surface, const Raster& raster, const Request& request)
{
    checkRequest(request, raster.width, raster.height);
    const bool needNormals = std::any_of(request.offsets.begin(), request.offsets.end(), [](float w) { return w != 0; });
    auto& impl = *surface.impl_;
    impl.prepare(needNormals);
    const bool empty = impl.v.rows == 0 || impl.v.cols == 0 || impl.pieces.empty();
    return sampleRaster(raster.width, raster.height, request, [&](int i0, int j0, int w, int h, float* xyz, float* nrm) {
        if (empty) {
            std::fill_n(xyz, std::size_t(w) * h * 3, kNaN);
            if (nrm)
                std::fill_n(nrm, std::size_t(w) * h * 3, kNaN);
            return;
        }
        const Raster tile{raster.x0 + double(i0) * raster.sx, raster.y0 + double(j0) * raster.sy, raster.sx, raster.sy, w, h};
        impl.resampleRows(tile, 0, h, xyz, nrm);
    });
}

Result sample(const Plane& plane, const Request& request)
{
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
    });
}

}  // namespace vcr::grid
