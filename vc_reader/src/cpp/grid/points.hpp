// Points on a surface stored as a grid of vertices, at real-valued grid positions:
// the position, the unit normal and an offset along it, shared by the CPU and CUDA
// samplers and the Python surface reader.
#pragma once
#include "../geometry/math.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define VCR_POINTS_HD __host__ __device__
#else
#define VCR_POINTS_HD
#endif

namespace vcr::grid {

// How grid positions become points: `physical` maps vertex coordinates to physical
// space (where normals are taken), `destination` maps physical points to the output
// space; both row-major 3x4. `offset` moves the point along the unit normal (in
// physical units); `orientation` (+-1) flips the normal.
struct PointMap {
    int height = 0, width = 0;
    float physical[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    float destination[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    float orientation = 1, offset = 0;
};

// A grid's vertices (height x width x 3 floats, rows contiguous) and what
// interpolating them needs. A cell is a surface patch only when its four vertices are
// present (`present`) and in one piece (`piece`, per column; null: one piece). With
// `normals`, normals are those vertex normals interpolated; otherwise they are the
// patch's. With the Hermite derivatives (du, dv, duv per vertex), patches are bicubic
// Hermite (smooth); otherwise bilinear.
struct GridView {
    const float* xyz = nullptr;
    const bool* present = nullptr;
    const int* piece = nullptr;
    const float* normals = nullptr;
    const float* du = nullptr;
    const float* dv = nullptr;
    const float* duv = nullptr;
};

namespace detail {
struct Hermite {
    float p0, m0, p1, m1, dp0, dm0, dp1, dm1;
};
VCR_POINTS_HD inline Hermite hermite(float t)
{
    const float t2 = t * t, t3 = t2 * t;
    return {2.f * t3 - 3.f * t2 + 1.f, t3 - 2.f * t2 + t, -2.f * t3 + 3.f * t2, t3 - t2,
            6.f * t2 - 6.f * t,       3.f * t2 - 4.f * t + 1.f, -6.f * t2 + 6.f * t, 3.f * t2 - 2.f * t};
}
}  // namespace detail

// The point at grid position (x = column, y = row): false when no patch covers it.
// The patch's tangents are taken in physical space, so its normal accounts for
// anisotropic spacing. `point` gets the mapped point (offset included), `normal`
// (optional) the unit normal in physical space, and `step` (optional) the move in
// output coordinates per unit of physical offset along it (destination's linear
// part applied to the normal).
VCR_POINTS_HD inline bool point(const PointMap& g, const GridView& v, float x, float y, float* point, float* normal,
                                float* step = nullptr)
{
    using geometry::V;
    using geometry::vector;
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 || x > float(g.width - 1) || y > float(g.height - 1) ||
        g.width < 2 || g.height < 2)
        return false;
    const int col = int(fminf(floorf(x), float(g.width - 2))), row = int(fminf(floorf(y), float(g.height - 2)));
    const float u = x - float(col), w = y - float(row);
    const int at[4]{row * g.width + col, row * g.width + col + 1, (row + 1) * g.width + col, (row + 1) * g.width + col + 1};
    for (int k = 0; k < 4; ++k)
        if (!v.present[at[k]])
            return false;
    if (v.piece && v.piece[col] != v.piece[col + 1])
        return false;
    V p[4];
    for (int k = 0; k < 4; ++k)
        p[k] = vector(v.xyz + 3 * at[k]);
    V position, du, dv;
    if (v.du) {
        const detail::Hermite bx = detail::hermite(u), by = detail::hermite(w);
        const auto D = [&](const float* d, int k) { return vector(d + 3 * at[k]); };
        const V q0 = p[0] * bx.p0 + D(v.du, 0) * bx.m0 + p[1] * bx.p1 + D(v.du, 1) * bx.m1;
        const V q1 = p[2] * bx.p0 + D(v.du, 2) * bx.m0 + p[3] * bx.p1 + D(v.du, 3) * bx.m1;
        const V r0 = D(v.dv, 0) * bx.p0 + D(v.duv, 0) * bx.m0 + D(v.dv, 1) * bx.p1 + D(v.duv, 1) * bx.m1;
        const V r1 = D(v.dv, 2) * bx.p0 + D(v.duv, 2) * bx.m0 + D(v.dv, 3) * bx.p1 + D(v.duv, 3) * bx.m1;
        const V q0u = p[0] * bx.dp0 + D(v.du, 0) * bx.dm0 + p[1] * bx.dp1 + D(v.du, 1) * bx.dm1;
        const V q1u = p[2] * bx.dp0 + D(v.du, 2) * bx.dm0 + p[3] * bx.dp1 + D(v.du, 3) * bx.dm1;
        const V r0u = D(v.dv, 0) * bx.dp0 + D(v.duv, 0) * bx.dm0 + D(v.dv, 1) * bx.dp1 + D(v.duv, 1) * bx.dm1;
        const V r1u = D(v.dv, 2) * bx.dp0 + D(v.duv, 2) * bx.dm0 + D(v.dv, 3) * bx.dp1 + D(v.duv, 3) * bx.dm1;
        position = q0 * by.p0 + r0 * by.m0 + q1 * by.p1 + r1 * by.m1;
        du = q0u * by.p0 + r0u * by.m0 + q1u * by.p1 + r1u * by.m1;
        dv = q0 * by.dp0 + r0 * by.dm0 + q1 * by.dp1 + r1 * by.dm1;
    } else {
        position = (p[0] * (1 - u) + p[1] * u) * (1 - w) + (p[2] * (1 - u) + p[3] * u) * w;
        du = (p[1] - p[0]) * (1 - w) + (p[3] - p[2]) * w;
        dv = (p[2] - p[0]) * (1 - u) + (p[3] - p[1]) * u;
    }
    V n = geometry::cross(geometry::linear(g.physical, du), geometry::linear(g.physical, dv)) * g.orientation;
    if (v.normals) {
        V m[4];
        for (int k = 0; k < 4; ++k)
            m[k] = vector(v.normals + 3 * at[k]);
        n = (m[0] * (1 - u) + m[1] * u) * (1 - w) + (m[2] * (1 - u) + m[3] * u) * w;
    }
    const float scale = fmaxf(fabsf(n.x), fmaxf(fabsf(n.y), fabsf(n.z)));
    if (!(scale > 0) || !geometry::finite(n))
        return false;
    n = n * (1 / scale);
    n = n * (1 / sqrtf(geometry::dot(n, n)));
    const V out = geometry::transform(g.destination, geometry::transform(g.physical, position) + n * g.offset);
    if (!geometry::finite(out))
        return false;
    geometry::store(point, out);
    if (normal)
        geometry::store(normal, n);
    if (step)
        geometry::store(step, geometry::linear(g.destination, n));
    return true;
}

}  // namespace vcr::grid

#undef VCR_POINTS_HD
