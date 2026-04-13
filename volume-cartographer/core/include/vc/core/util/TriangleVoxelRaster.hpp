#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <opencv2/core.hpp>

namespace vc::core::util {

enum class RasterSliceAxis {
    Z,
    Y,
    X
};

struct RasterVoxel {
    int z;
    int y;
    int x;

    bool operator==(const RasterVoxel& other) const noexcept
    {
        return z == other.z && y == other.y && x == other.x;
    }
};

namespace detail {

constexpr float kAxisPlaneEps = 1e-5f;
constexpr float kSegmentPointEps2 = 1e-12f;

struct SegmentPoint2 {
    float u;
    float v;
};

inline float axisCoord(const cv::Vec3f& p, RasterSliceAxis axis) noexcept
{
    switch (axis) {
    case RasterSliceAxis::Z: return p[2];
    case RasterSliceAxis::Y: return p[1];
    case RasterSliceAxis::X: return p[0];
    }
    return p[2];
}

inline SegmentPoint2 projectPoint(const cv::Vec3f& p, RasterSliceAxis axis) noexcept
{
    switch (axis) {
    case RasterSliceAxis::Z: return {p[0], p[1]};
    case RasterSliceAxis::Y: return {p[0], p[2]};
    case RasterSliceAxis::X: return {p[1], p[2]};
    }
    return {p[0], p[1]};
}

inline bool addSegmentPoint(std::array<SegmentPoint2, 4>& pts,
                            int& n,
                            float u,
                            float v) noexcept
{
    for (int i = 0; i < n; ++i) {
        const float du = pts[i].u - u;
        const float dv = pts[i].v - v;
        if (du * du + dv * dv < kSegmentPointEps2) {
            return false;
        }
    }
    if (n < static_cast<int>(pts.size())) {
        pts[n++] = {u, v};
        return true;
    }
    return false;
}

template <typename EmitVoxel>
inline bool rasterizeProjectedLine(RasterSliceAxis axis,
                                   int fixedIndex,
                                   float u0,
                                   float v0,
                                   float u1,
                                   float v1,
                                   const std::array<size_t, 3>& shape,
                                   EmitVoxel&& emit)
{
    const float du = u1 - u0;
    const float dv = v1 - v0;
    const float maxDelta = std::max(std::fabs(du), std::fabs(dv));
    const int steps = static_cast<int>(std::floor(maxDelta + 0.5f));

    const float su = (steps > 0) ? (du / std::max(1.0f, static_cast<float>(steps))) : 0.0f;
    const float sv = (steps > 0) ? (dv / std::max(1.0f, static_cast<float>(steps))) : 0.0f;

    bool any = false;
    auto emitProjected = [&](int iu, int iv) {
        switch (axis) {
        case RasterSliceAxis::Z:
            if (fixedIndex >= 0 && fixedIndex < static_cast<int>(shape[0]) &&
                iv >= 0 && iv < static_cast<int>(shape[1]) &&
                iu >= 0 && iu < static_cast<int>(shape[2])) {
                emit(RasterVoxel{fixedIndex, iv, iu});
                any = true;
            }
            break;
        case RasterSliceAxis::Y:
            if (iv >= 0 && iv < static_cast<int>(shape[0]) &&
                fixedIndex >= 0 && fixedIndex < static_cast<int>(shape[1]) &&
                iu >= 0 && iu < static_cast<int>(shape[2])) {
                emit(RasterVoxel{iv, fixedIndex, iu});
                any = true;
            }
            break;
        case RasterSliceAxis::X:
            if (iv >= 0 && iv < static_cast<int>(shape[0]) &&
                iu >= 0 && iu < static_cast<int>(shape[1]) &&
                fixedIndex >= 0 && fixedIndex < static_cast<int>(shape[2])) {
                emit(RasterVoxel{iv, iu, fixedIndex});
                any = true;
            }
            break;
        }
    };

    if (steps <= 0) {
        emitProjected(static_cast<int>(std::llround(u0)),
                      static_cast<int>(std::llround(v0)));
        return any;
    }

    float fu = u0;
    float fv = v0;
    for (int i = 0; i <= steps; ++i) {
        emitProjected(static_cast<int>(std::llround(fu)),
                      static_cast<int>(std::llround(fv)));
        fu += su;
        fv += sv;
    }
    return any;
}

template <typename EmitVoxel>
inline bool rasterizeTriangleAtPlane(const cv::Vec3f& a,
                                     const cv::Vec3f& b,
                                     const cv::Vec3f& c,
                                     RasterSliceAxis axis,
                                     float planeCoord,
                                     int fixedIndex,
                                     const std::array<size_t, 3>& shape,
                                     EmitVoxel&& emit)
{
    std::array<SegmentPoint2, 4> pts{};
    int n = 0;

    auto visitEdge = [&](const cv::Vec3f& p0, const cv::Vec3f& p1) {
        const float c0 = axisCoord(p0, axis);
        const float c1 = axisCoord(p1, axis);
        const float dc = c1 - c0;
        if (std::fabs(dc) <= kAxisPlaneEps) {
            if (std::fabs(c0 - planeCoord) <= kAxisPlaneEps) {
                const auto q0 = projectPoint(p0, axis);
                const auto q1 = projectPoint(p1, axis);
                addSegmentPoint(pts, n, q0.u, q0.v);
                addSegmentPoint(pts, n, q1.u, q1.v);
            }
            return;
        }

        float t = (planeCoord - c0) / dc;
        if (t < -kAxisPlaneEps || t > 1.0f + kAxisPlaneEps) {
            return;
        }
        t = std::clamp(t, 0.0f, 1.0f);
        const cv::Vec3f p = p0 + t * (p1 - p0);
        const auto q = projectPoint(p, axis);
        addSegmentPoint(pts, n, q.u, q.v);
    };

    visitEdge(a, b);
    visitEdge(b, c);
    visitEdge(c, a);

    if (n < 2) {
        return false;
    }

    bool any = false;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            any |= rasterizeProjectedLine(axis,
                                          fixedIndex,
                                          pts[i].u,
                                          pts[i].v,
                                          pts[j].u,
                                          pts[j].v,
                                          shape,
                                          emit);
        }
    }
    return any;
}

}  // namespace detail

template <typename EmitVoxel>
inline bool rasterizeTriangleOnAxisSlices(const cv::Vec3f& a,
                                          const cv::Vec3f& b,
                                          const cv::Vec3f& c,
                                          RasterSliceAxis axis,
                                          float planeOffset,
                                          const std::array<size_t, 3>& shape,
                                          EmitVoxel&& emit)
{
    const float axisMin = std::min({detail::axisCoord(a, axis),
                                    detail::axisCoord(b, axis),
                                    detail::axisCoord(c, axis)});
    const float axisMax = std::max({detail::axisCoord(a, axis),
                                    detail::axisCoord(b, axis),
                                    detail::axisCoord(c, axis)});

    int startIndex = static_cast<int>(std::floor(axisMin - planeOffset + 1e-6f));
    int endIndex = static_cast<int>(std::floor(axisMax - planeOffset + 1e-6f));

    const int axisExtent = [&]() -> int {
        switch (axis) {
        case RasterSliceAxis::Z: return static_cast<int>(shape[0]);
        case RasterSliceAxis::Y: return static_cast<int>(shape[1]);
        case RasterSliceAxis::X: return static_cast<int>(shape[2]);
        }
        return static_cast<int>(shape[0]);
    }();

    if (endIndex < 0 || startIndex >= axisExtent) {
        return false;
    }

    startIndex = std::max(0, startIndex);
    endIndex = std::min(endIndex, axisExtent - 1);

    bool any = false;
    for (int idx = startIndex; idx <= endIndex; ++idx) {
        any |= detail::rasterizeTriangleAtPlane(a,
                                                b,
                                                c,
                                                axis,
                                                static_cast<float>(idx) + planeOffset,
                                                idx,
                                                shape,
                                                emit);
    }
    return any;
}

}  // namespace vc::core::util
