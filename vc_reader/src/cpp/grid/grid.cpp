// Surfaces stored as grids of vertices, and sampling volumes along them and along planes.
#include "vc_reader/grid.hpp"

#include "../array.hpp"
#include "../composite/reducer.hpp"
#include "../thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
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

// Where each pixel's point of one layer comes from: a plane (computed from the
// pixel), or resampled surface positions and normals.
struct Source {
    const Plane* plane = nullptr;
    const float* xyz = nullptr;
    const float* nrm = nullptr;
    int width = 0;

    // The point of pixel q at offset w along the normal (non-finite: no point).
    void point(std::size_t q, float w, float* p) const
    {
        if (plane) {
            const float i = float(q % std::size_t(width)), j = float(q / std::size_t(width));
            for (int a = 0; a < 3; ++a)
                p[a] = plane->origin[a] + plane->dy[a] * j + plane->dx[a] * i + plane->normal[a] * w;
            return;
        }
        for (int a = 0; a < 3; ++a)
            p[a] = w != 0 ? xyz[q * 3 + a] + nrm[q * 3 + a] * w : xyz[q * 3 + a];
    }
};

// The points of the pixels `which` lists (all pixels when null) of one layer, as
// a level's ZYX indices: index[a] = scale[a] * p[2 - a] + offset[a]. Computed as
// the sampler asks: no point or index buffer.
struct LayerPoints {
    using value_type = double;
    const Source* source;
    float w;
    const std::uint32_t* which;
    std::array<double, 3> scale, offset;

    std::int64_t base(std::size_t) const { return 0; }
    std::size_t pixel(std::size_t p) const { return which ? which[p] : p; }
    const double* operator()(std::size_t p, double* c) const
    {
        float q[3];
        source->point(pixel(p), w, q);
        c[0] = scale[0] * q[2] + offset[0];
        c[1] = scale[1] * q[1] + offset[1];
        c[2] = scale[2] * q[0] + offset[2];
        return c;
    }
    void prefetch(std::size_t) const {}
};

// Runs fn(part, begin, end) over [0, n) in even parts on the pool (plain work division).
template <class Fn>
std::size_t forParts(Array& pool, std::size_t n, Fn&& fn)
{
    const std::size_t parts = std::clamp<std::size_t>(n / 32768, 1, 64);
    pool.parallel_for(parts, [&](std::size_t k) { fn(k, n * k / parts, n * (k + 1) / parts); });
    return parts;
}

// The entries i < n with keep(i), in order.
template <class Keep>
std::vector<std::uint32_t> select(Array& pool, std::size_t n, Keep&& keep)
{
    std::vector<std::vector<std::uint32_t>> kept(std::clamp<std::size_t>(n / 32768, 1, 64));
    forParts(pool, n, [&](std::size_t k, std::size_t a, std::size_t b) {
        for (std::size_t i = a; i < b; ++i)
            if (keep(i))
                kept[k].push_back(std::uint32_t(i));
    });
    std::vector<std::uint32_t> all;
    for (auto& part : kept)
        all.insert(all.end(), part.begin(), part.end());
    return all;
}

// Per-thread scratch reused across calls (no page faults on every frame).
std::vector<float>& scratch(int which)
{
    thread_local std::vector<float> buffers[3];
    return buffers[which];
}

// Samples every layer of a raster whose points `source` gives: per layer, one
// Array call for all of its pixels at the first level, then calls at the next
// levels for the pixels still unresolved.
Result sampleRaster(int width, int height, const Request& request, const Source& source)
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
    if (std::size_t(width) * std::size_t(height) > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("raster too large");
    if (request.composite)
        composite::detail::checkLayers(request.offsets.size(), *request.composite);
    Array& pool = *request.levels.front().array;
    const std::size_t pixels = std::size_t(width) * std::size_t(height);
    const std::size_t layers = request.offsets.size();
    Result result;
    result.width = width;
    result.height = height;
    result.layers = int(layers);
    result.level.resize(pixels * layers);
    result.values.resize(request.composite ? pixels : pixels * layers);
    auto& stack = scratch(2);  // the layers of a composite
    if (request.composite)
        stack.resize(pixels * layers);

    std::vector<float> linear;
    std::vector<std::uint8_t> raw, missing;
    for (std::size_t k = 0; k < layers; ++k) {
        const float w = request.offsets[k];
        float* values = request.composite ? stack.data() + k * pixels : result.values.data() + k * pixels;
        std::int8_t* level = result.level.data() + k * pixels;
        // Masked reads sample only their pixels; frames sample every pixel.
        std::vector<std::uint32_t> pending;
        if (request.pixels)
            pending = select(pool, pixels, [&](std::size_t q) { return request.pixels[q] != 0; });
        const std::uint32_t* which = request.pixels ? pending.data() : nullptr;
        std::size_t n = request.pixels ? pending.size() : pixels;
        if (request.pixels) {
            std::fill_n(values, pixels, kNaN);
            std::fill_n(level, pixels, Result::kNone);
        }
        for (std::size_t li = 0; li < request.levels.size() && n; ++li) {
            const Level& L = request.levels[li];
            const auto& shape = L.array->spec().shape;
            const LayerPoints points{&source, w, which, L.scale, L.offset};
            std::uint8_t* miss = nullptr;
            if (request.cachedOnly) {
                missing.resize(n);
                miss = missing.data();
            }
            const std::size_t isz = L.array->itemsize();
            if (request.order == 0) {
                raw.resize(n * isz);
                L.array->sample_nearest_coordinates(points, n, raw.data(), miss);
            } else {
                linear.resize(n);
                L.array->sample_linear_coordinates(points, n, L.type, false, linear.data(), miss);
            }
            // Write back what this level supplied; count what it did not.
            std::vector<std::size_t> unresolved(64, 0);
            forParts(pool, n, [&](std::size_t part, std::size_t a, std::size_t b) {
                for (std::size_t p = a; p < b; ++p) {
                    const std::size_t q = points.pixel(p);
                    double c[3];
                    points(p, c);
                    const bool point = std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2]);
                    bool inside = point;
                    for (int axis = 0; axis < 3 && inside; ++axis)
                        inside = c[axis] >= -0.5 && c[axis] <= double(shape[std::size_t(axis)]) - 0.5;
                    if (li == 0) {
                        values[q] = kNaN;
                        level[q] = Result::kNone;
                    }
                    if (!inside || (miss && miss[p])) {
                        if (point) {
                            ++unresolved[part];
                            if (inside)
                                level[q] = Result::kMissing;
                        }
                        continue;
                    }
                    values[q] = request.order == 0 ? float(L.type.load(raw.data() + p * isz)) : linear[p];
                    level[q] = std::int8_t(li);
                }
            });
            std::size_t left = 0;
            for (auto u : unresolved)
                left += u;
            if (!left || li + 1 == request.levels.size())
                break;
            // The next level samples only the pixels with a point and no value yet.
            auto next = select(pool, n, [&](std::size_t p) {
                if (level[points.pixel(p)] >= 0)
                    return false;
                double c[3];
                points(p, c);
                return std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2]);
            });
            for (auto& p : next)  // `points` reads `pending`: remap before replacing it
                p = std::uint32_t(points.pixel(p));
            pending = std::move(next);
            which = pending.data();
            n = pending.size();
        }
    }
    if (request.composite) {
        forParts(pool, pixels, [&](std::size_t, std::size_t a, std::size_t b) {
            for (std::size_t q = a; q < b; ++q) {
                composite::detail::Reducer r(*request.composite);
                for (std::size_t k = 0; k < layers && !r.done; ++k)
                    r.add(stack[k * pixels + q]);
                result.values[q] = r.result();
            }
        });
    }
    return result;
}

}  // namespace

Result sample(const Surface& surface, const Raster& raster, const Request& request)
{
    if (request.levels.empty())
        throw std::invalid_argument("a sample request needs at least one level");
    const bool needNormals = std::any_of(request.offsets.begin(), request.offsets.end(), [](float w) { return w != 0; });
    auto& impl = *surface.impl_;
    impl.prepare(needNormals);
    const std::size_t pixels = std::size_t(std::max(raster.width, 0)) * std::size_t(std::max(raster.height, 0));
    auto& xyz = scratch(0);
    auto& nrm = scratch(1);
    xyz.resize(pixels * 3);
    nrm.resize(needNormals ? pixels * 3 : 0);
    if (impl.v.rows == 0 || impl.v.cols == 0 || impl.pieces.empty()) {
        std::fill(xyz.begin(), xyz.end(), kNaN);
        std::fill(nrm.begin(), nrm.end(), kNaN);
    } else {
        const std::size_t rows = std::size_t(std::max(raster.height, 0));
        const std::size_t parts = std::clamp<std::size_t>(pixels / 32768, 1, 64);
        request.levels.front().array->parallel_for(parts, [&](std::size_t k) {
            const std::size_t r0 = rows * k / parts, r1 = rows * (k + 1) / parts;
            const std::size_t o = r0 * std::size_t(raster.width) * 3;
            impl.resampleRows(raster, int(r0), int(r1), xyz.data() + o, needNormals ? nrm.data() + o : nullptr);
        });
    }
    return sampleRaster(raster.width, raster.height, request,
                        Source{nullptr, xyz.data(), needNormals ? nrm.data() : nullptr, raster.width});
}

Result sample(const Plane& plane, const Request& request)
{
    return sampleRaster(plane.width, plane.height, request, Source{&plane, nullptr, nullptr, plane.width});
}

}  // namespace vcr::grid
