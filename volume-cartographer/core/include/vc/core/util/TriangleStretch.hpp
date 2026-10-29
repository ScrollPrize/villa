#pragma once

// Per-triangle stretch metrics shared by flatboi's SLIM flattening
// (libs/flatboi/flatboi.cpp) and the tifxyz grid isometry block (GridIsometry.hpp).
// ScrollPrize/villa issue #2279 item 4 asks for exactly this: flatboi's
// `triangle_stretch` moved into a small Eigen-free header and called from both
// places, so a grid's L2 and Linf come from the same code as flatboi's numbers
// and the two cannot drift. Eigen-free on purpose -- flatboi passes its Eigen
// rows in as plain arrays, the grid passes grid coordinates, and this header
// depends only on the C++ standard library.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace vc::surface
{
// One triangle's stretch, from the affine map between its 2D parameter
// coordinates and its 3D vertices. `l2_squared` is the mean of the squared
// singular values of that map's 2x3 Jacobian (flatboi's "L2" before its square
// root, kept squared so an area-weighted caller can use the exact value);
// `linf` is the largest singular value; `area3d` is the triangle's 3D area
// (Heron) and `area2d` the absolute area of its parameter triangle.
struct TriangleStretch
{
    double l2_squared = 0.0;
    double linf = 0.0;
    double area3d = 0.0;
    double area2d = 0.0;
};

// p[k] is the k-th vertex in 3D, uv[k] its 2D parameter coordinate. A degenerate
// parameter triangle (2D area below 1e-30) has no Jacobian and yields all zeros,
// which is what flatboi's original returned.
inline TriangleStretch triangleStretch(const double p[3][3], const double uv[3][2])
{
    TriangleStretch out;
    const double s1 = uv[0][0], t1 = uv[0][1];
    const double s2 = uv[1][0], t2 = uv[1][1];
    const double s3 = uv[2][0], t3 = uv[2][1];

    const double area2d = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2.0;
    if (std::abs(area2d) < 1e-30) {
        return out;
    }

    // Columns of the Jacobian: the 3D derivatives along the two parameter axes.
    double ss[3];
    double st[3];
    for (int k = 0; k < 3; ++k) {
        ss[k] = (p[0][k] * (t2 - t3) + p[1][k] * (t3 - t1) + p[2][k] * (t1 - t2)) / (2.0 * area2d);
        st[k] = (p[0][k] * (s3 - s2) + p[1][k] * (s1 - s3) + p[2][k] * (s2 - s1)) / (2.0 * area2d);
    }
    const double a = ss[0] * ss[0] + ss[1] * ss[1] + ss[2] * ss[2];
    const double b = ss[0] * st[0] + ss[1] * st[1] + ss[2] * st[2];
    const double c = st[0] * st[0] + st[1] * st[1] + st[2] * st[2];
    out.linf = std::sqrt(((a + c) + std::sqrt((a - c) * (a - c) + 4.0 * b * b)) / 2.0);
    out.l2_squared = (a + c) / 2.0;

    const double ab = std::sqrt((p[0][0] - p[1][0]) * (p[0][0] - p[1][0]) +
                                (p[0][1] - p[1][1]) * (p[0][1] - p[1][1]) +
                                (p[0][2] - p[1][2]) * (p[0][2] - p[1][2]));
    const double bc = std::sqrt((p[1][0] - p[2][0]) * (p[1][0] - p[2][0]) +
                                (p[1][1] - p[2][1]) * (p[1][1] - p[2][1]) +
                                (p[1][2] - p[2][2]) * (p[1][2] - p[2][2]));
    const double ca = std::sqrt((p[2][0] - p[0][0]) * (p[2][0] - p[0][0]) +
                                (p[2][1] - p[0][1]) * (p[2][1] - p[0][1]) +
                                (p[2][2] - p[0][2]) * (p[2][2] - p[0][2]));
    const double s = (ab + bc + ca) / 2.0;
    out.area3d = std::sqrt(std::max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
    out.area2d = std::abs(area2d);
    return out;
}

// The area-weighted median flatboi reports: a stable sort of the values, the
// cumulative weight, and the first value whose cumulative weight reaches half
// the total.
inline double weightedMedian(const std::vector<double>& data, const std::vector<double>& weights)
{
    if (data.empty() || data.size() != weights.size()) {
        return 0.0;
    }
    double total = 0.0;
    for (double w : weights) {
        total += w;
    }
    if (!(total > 0.0)) {
        return 0.0;
    }
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
        return data[i] < data[j];
    });
    const double half = 0.5 * total;
    double acc = 0.0;
    std::size_t k = 0;
    for (; k < order.size(); ++k) {
        acc += weights[order[k]];
        if (acc >= half) {
            break;
        }
    }
    if (k >= order.size()) {
        k = order.size() - 1;
    }
    return data[order[k]];
}

// flatboi's `stretch_metrics` over the shared per-triangle values: the
// area-weighted L2 mean and median, the largest Linf, and the mean per-triangle
// area error (share of 3D area against share of 2D area, so a uniform map
// scores 0).
struct StretchMetrics
{
    double l2_mean = 0.0;
    double l2_median = 0.0;
    double linf = 0.0;
    double area_error = 0.0;
};

inline StretchMetrics aggregateStretch(const std::vector<TriangleStretch>& tris)
{
    StretchMetrics out;
    if (tris.empty()) {
        return out;
    }
    std::vector<double> l2sq(tris.size());
    std::vector<double> area3d(tris.size());
    std::vector<double> area2d(tris.size());
    double sumA3 = 0.0;
    double sumA2 = 0.0;
    for (std::size_t i = 0; i < tris.size(); ++i) {
        l2sq[i] = tris[i].l2_squared;
        area3d[i] = tris[i].area3d;
        area2d[i] = tris[i].area2d;
        sumA3 += area3d[i];
        sumA2 += area2d[i];
        out.linf = std::max(out.linf, tris[i].linf);
    }

    double weightedL2 = 0.0;
    for (std::size_t i = 0; i < tris.size(); ++i) {
        weightedL2 += l2sq[i] * area3d[i];
    }
    if (sumA3 > 0.0) {
        out.l2_mean = std::sqrt(weightedL2 / sumA3);
    }
    out.l2_median = weightedMedian(l2sq, area3d);

    double areaError = 0.0;
    for (std::size_t i = 0; i < tris.size(); ++i) {
        const double alpha = sumA3 > 0.0 ? area3d[i] / sumA3 : 0.0;
        const double beta = sumA2 > 0.0 ? area2d[i] / sumA2 : 0.0;
        areaError += alpha > beta ? 1.0 - beta / (alpha + 1e-30)
                                  : 1.0 - alpha / (beta + 1e-30);
    }
    out.area_error = areaError / static_cast<double>(tris.size());
    return out;
}
} // namespace vc::surface
