#pragma once

// Measurements drawn over a volume viewer (the "HUD"): a histogram of what is on screen,
// an intensity profile along a line, a volume-coordinate grid and a magnifying loupe.
//
// Everything here works on the viewer's rendered framebuffer and on an affine
// viewport-to-volume map, so it needs nothing from the surface, fiber or segment code
// and is header-only so the view's tests can use it directly.

#include <QImage>
#include <QLineF>
#include <QPointF>
#include <QRectF>
#include <QString>

#include "FiberMapRulerMath.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace vc3d::hud {

struct Options {
    bool histogram = false;
    bool lineProfile = false;
    bool grid = false;
    bool loupe = false;
    double gridStep = 0.0;  // volume voxels between grid lines; 0 picks one from the zoom

    bool any() const { return histogram || lineProfile || grid || loupe; }
    bool operator==(const Options&) const = default;
};

inline int grayAt(const QImage& img, int x, int y)
{
    if (img.format() == QImage::Format_Grayscale8) {
        return img.constScanLine(y)[x];
    }
    return qGray(img.pixel(x, y));
}

struct Histogram {
    std::array<std::uint32_t, 256> bins{};
    std::uint64_t count = 0;
    double mean = 0.0;
    int low = 0;    // 1st percentile
    int high = 0;   // 99th percentile
    double black = 0.0;  // fraction at 0
    double white = 0.0;  // fraction at 255
};

// Histogram of the displayed gray values, reading every `stride`-th pixel in x and y,
// and only pixels for which `include(x, y)` holds when it is given.
inline Histogram computeHistogram(const QImage& img, int stride = 2,
                                  const std::function<bool(int, int)>& include = {})
{
    Histogram h;
    if (img.isNull()) {
        return h;
    }
    stride = std::max(1, stride);
    double sum = 0.0;
    for (int y = 0; y < img.height(); y += stride) {
        for (int x = 0; x < img.width(); x += stride) {
            if (include && !include(x, y)) {
                continue;
            }
            const int v = grayAt(img, x, y);
            ++h.bins[static_cast<std::size_t>(v)];
            sum += v;
            ++h.count;
        }
    }
    if (h.count == 0) {
        return h;
    }
    h.mean = sum / static_cast<double>(h.count);
    h.black = static_cast<double>(h.bins[0]) / static_cast<double>(h.count);
    h.white = static_cast<double>(h.bins[255]) / static_cast<double>(h.count);
    const std::uint64_t lowTarget = h.count / 100;
    const std::uint64_t highTarget = h.count - h.count / 100;
    std::uint64_t seen = 0;
    bool lowSet = false;
    for (int v = 0; v < 256; ++v) {
        seen += h.bins[static_cast<std::size_t>(v)];
        if (!lowSet && seen > lowTarget) {
            h.low = v;
            lowSet = true;
        }
        if (seen >= highTarget) {
            h.high = v;
            break;
        }
    }
    return h;
}

// Gray values along a line in image pixel coordinates, bilinear, one sample per pixel of
// length (at least two). Samples outside the image are NaN.
inline std::vector<float> sampleLine(const QImage& img, QPointF a, QPointF b)
{
    const double length = QLineF(a, b).length();
    const int n = std::max(2, static_cast<int>(std::ceil(length)) + 1);
    std::vector<float> out(static_cast<std::size_t>(n), std::numeric_limits<float>::quiet_NaN());
    if (img.isNull()) {
        return out;
    }
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / (n - 1);
        const double x = a.x() + (b.x() - a.x()) * t;
        const double y = a.y() + (b.y() - a.y()) * t;
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        if (x0 < 0 || y0 < 0 || x0 >= img.width() || y0 >= img.height()) {
            continue;
        }
        const int x1 = std::min(x0 + 1, img.width() - 1);
        const int y1 = std::min(y0 + 1, img.height() - 1);
        const double fx = x - x0;
        const double fy = y - y0;
        const double top = grayAt(img, x0, y0) * (1 - fx) + grayAt(img, x1, y0) * fx;
        const double bottom = grayAt(img, x0, y1) * (1 - fx) + grayAt(img, x1, y1) * fx;
        out[static_cast<std::size_t>(i)] = static_cast<float>(top * (1 - fy) + bottom * fy);
    }
    return out;
}

// Indices of peaks that rise at least `minProminence` above the higher of the two lowest
// points separating them from a taller neighbour (topographic prominence). NaNs never peak.
inline std::vector<int> findPeaks(const std::vector<float>& v, float minProminence)
{
    std::vector<int> peaks;
    const int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i) {
        const float x = v[static_cast<std::size_t>(i)];
        if (std::isnan(x)) {
            continue;
        }
        // Plateaus count once, at their left edge.
        if (i > 0 && !(v[static_cast<std::size_t>(i - 1)] < x)) {
            continue;
        }
        int j = i;
        while (j + 1 < n && v[static_cast<std::size_t>(j + 1)] == x) {
            ++j;
        }
        if (j + 1 < n && !(v[static_cast<std::size_t>(j + 1)] < x)) {
            continue;
        }
        float leftMin = x;
        for (int k = i - 1; k >= 0; --k) {
            const float y = v[static_cast<std::size_t>(k)];
            if (std::isnan(y) || y > x) {
                break;
            }
            leftMin = std::min(leftMin, y);
        }
        float rightMin = x;
        for (int k = j + 1; k < n; ++k) {
            const float y = v[static_cast<std::size_t>(k)];
            if (std::isnan(y) || y > x) {
                break;
            }
            rightMin = std::min(rightMin, y);
        }
        if (x - std::max(leftMin, rightMin) >= minProminence) {
            peaks.push_back(i);
        }
        i = j;
    }
    return peaks;
}

// Volume position as an affine function of viewport position: v = origin + dx * x + dy * y.
struct AffineMap {
    std::array<double, 3> origin{};
    std::array<double, 3> dx{};
    std::array<double, 3> dy{};

    std::array<double, 3> at(QPointF p) const
    {
        return {origin[0] + dx[0] * p.x() + dy[0] * p.y(),
                origin[1] + dx[1] * p.x() + dy[1] * p.y(),
                origin[2] + dx[2] * p.x() + dy[2] * p.y()};
    }
};

struct GridLine {
    QLineF line;
    int axis = 0;     // 0 = x, 1 = y, 2 = z in volume coordinates
    double value = 0.0;
};

// Lines where a volume coordinate is a multiple of `step`, clipped to `rect`. Coordinates
// that barely change across the view (the slice axis of a plane) get no lines.
inline std::vector<GridLine> gridLines(const AffineMap& map, const QRectF& rect, double step,
                                       int maxLinesPerAxis = 60)
{
    std::vector<GridLine> out;
    if (!(step > 0.0) || rect.isEmpty()) {
        return out;
    }
    const std::array<QPointF, 4> corners{rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()};
    for (int axis = 0; axis < 3; ++axis) {
        const double gx = map.dx[static_cast<std::size_t>(axis)];
        const double gy = map.dy[static_cast<std::size_t>(axis)];
        const double grad = std::hypot(gx, gy);
        if (grad * std::max(rect.width(), rect.height()) < step * 0.5) {
            continue;
        }
        double lo = std::numeric_limits<double>::max();
        double hi = std::numeric_limits<double>::lowest();
        for (const auto& c : corners) {
            const double v = map.at(c)[static_cast<std::size_t>(axis)];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        const double first = std::ceil(lo / step) * step;
        if ((hi - first) / step > maxLinesPerAxis) {
            continue;
        }
        for (double value = first; value <= hi; value += step) {
            // Points on the rectangle's edges where the coordinate equals value.
            std::vector<QPointF> hits;
            for (int e = 0; e < 4; ++e) {
                const QPointF a = corners[static_cast<std::size_t>(e)];
                const QPointF b = corners[static_cast<std::size_t>((e + 1) % 4)];
                const double va = map.at(a)[static_cast<std::size_t>(axis)] - value;
                const double vb = map.at(b)[static_cast<std::size_t>(axis)] - value;
                if ((va <= 0 && vb > 0) || (va > 0 && vb <= 0)) {
                    const double t = va / (va - vb);
                    hits.push_back(a + (b - a) * t);
                }
            }
            if (hits.size() >= 2) {
                out.push_back({QLineF(hits[0], hits[1]), axis, value});
            }
        }
    }
    return out;
}

inline QString axisName(int axis)
{
    static const char* names[] = {"x", "y", "z"};
    return QString::fromLatin1(names[std::clamp(axis, 0, 2)]);
}

}  // namespace vc3d::hud
