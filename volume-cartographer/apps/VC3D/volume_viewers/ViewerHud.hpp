#pragma once

// Measurements drawn over a volume viewer (the "HUD"): a histogram of the visible
// samples, an intensity profile along a line, a volume-coordinate grid and a
// magnifying loupe.
//
// Everything here works on one rendered frame: its 8-bit base-volume samples before the
// display window, the mask of pixels that frame measured (sampled inside the volume
// from that frame's own geometry), and a map from frame pixels to volume positions.
// Pixels outside the mask (loading gaps, holes in a flattened surface, the fill the
// reader returns beyond the volume's edges) never enter a measurement. The code needs
// nothing from the surface, fiber or segment code and is header-only so the view's
// tests can use it directly.

#include <QLineF>
#include <QPointF>
#include <QRect>
#include <QRectF>
#include <QSizeF>
#include <QString>

#include "FiberMapRulerMath.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// One frame's 8-bit samples and the mask of the pixels it measured. `coverage` may be
// null, meaning every pixel counts. Strides are in bytes.
struct Raster {
    const std::uint8_t* values = nullptr;
    const std::uint8_t* coverage = nullptr;
    int width = 0;
    int height = 0;
    std::size_t valuesStride = 0;
    std::size_t coverageStride = 0;

    bool empty() const { return !values || width <= 0 || height <= 0; }
    bool contains(int x, int y) const { return x >= 0 && y >= 0 && x < width && y < height; }
    bool covered(int x, int y) const
    {
        return contains(x, y) && (!coverage || coverage[std::size_t(y) * coverageStride + std::size_t(x)] != 0);
    }
    int at(int x, int y) const { return values[std::size_t(y) * valuesStride + std::size_t(x)]; }
};

// std::clamp requires lo <= hi; panes can be smaller than what is placed in them.
inline double clampInto(double v, double lo, double hi)
{
    if (hi < lo) {
        return lo;
    }
    return std::min(std::max(v, lo), hi);
}

// A volume position the overlays may use: finite, and not the (-1, -1, -1) that marks
// a hole in a flattened surface.
inline bool validPosition(const std::array<double, 3>& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
           !(p[0] == -1.0 && p[1] == -1.0 && p[2] == -1.0);
}

// The gray the display window gives each 8-bit sample (0 black, 255 white), as the
// renderer's lookup table maps it. The viewer fills it from that table.
using DisplayMap = std::array<std::uint8_t, 256>;

inline DisplayMap identityDisplay()
{
    DisplayMap out{};
    for (int i = 0; i < 256; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
    }
    return out;
}

// Viewport pixel for a fraction of an extent in pixels, 0 the first pixel and 1 the
// last one (extent - 1), so scripts can reach every pixel of the frame.
inline double fractionToPixel(double fraction, double extent)
{
    const double f = std::isfinite(fraction) ? std::clamp(fraction, 0.0, 1.0) : 0.0;
    return f * std::max(0.0, extent - 1.0);
}

struct Histogram {
    std::array<std::uint32_t, 256> bins{};
    std::uint64_t count = 0;
    double mean = 0.0;
    int low = 0;    // 1st percentile
    int high = 0;   // 99th percentile
    double belowWindow = 0.0;  // fraction the display window shows as black (gray 0)
    double aboveWindow = 0.0;  // fraction it shows as white (gray 255)
};

// Histogram of the measured values inside `region`, reading every `stride`-th pixel in
// x and y. Pixels outside the frame's mask are left out. `displayed` is the gray the
// display window gives each value, so the result can say how much of the view the
// window shows as black or white exactly as the renderer draws it.
inline Histogram computeHistogram(const Raster& r, QRect region, int stride,
                                  const DisplayMap& displayed)
{
    Histogram h;
    if (r.empty()) {
        return h;
    }
    region = region.intersected(QRect(0, 0, r.width, r.height));
    if (region.isEmpty()) {
        return h;
    }
    stride = std::max(1, stride);
    double sum = 0.0;
    std::uint64_t below = 0;
    std::uint64_t above = 0;
    for (int y = region.top(); y <= region.bottom(); y += stride) {
        for (int x = region.left(); x <= region.right(); x += stride) {
            if (!r.covered(x, y)) {
                continue;
            }
            const int v = r.at(x, y);
            ++h.bins[static_cast<std::size_t>(v)];
            sum += v;
            ++h.count;
            const std::uint8_t gray = displayed[static_cast<std::size_t>(v)];
            below += gray == 0 ? 1 : 0;
            above += gray == 255 ? 1 : 0;
        }
    }
    if (h.count == 0) {
        return h;
    }
    h.mean = sum / static_cast<double>(h.count);
    h.belowWindow = static_cast<double>(below) / static_cast<double>(h.count);
    h.aboveWindow = static_cast<double>(above) / static_cast<double>(h.count);
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

inline Histogram computeHistogram(const Raster& r, QRect region, int stride = 2)
{
    return computeHistogram(r, region, stride, identityDisplay());
}

// Position of sample i of n along a line, at least two samples.
inline QPointF samplePoint(QPointF a, QPointF b, int i, int n)
{
    const double t = n > 1 ? static_cast<double>(i) / (n - 1) : 0.0;
    return a + (b - a) * t;
}

inline int sampleCount(QPointF a, QPointF b)
{
    return std::max(2, static_cast<int>(std::ceil(QLineF(a, b).length())) + 1);
}

// Values along a line in frame pixel coordinates, bilinear, one sample per pixel of
// length (at least two). A sample is NaN unless every pixel that contributes to it
// (has a nonzero weight) was sampled by the frame, so loading gaps and surface holes
// read as missing rather than as gray.
inline std::vector<float> sampleLine(const Raster& r, QPointF a, QPointF b)
{
    const int n = sampleCount(a, b);
    std::vector<float> out(static_cast<std::size_t>(n), std::numeric_limits<float>::quiet_NaN());
    if (r.empty()) {
        return out;
    }
    for (int i = 0; i < n; ++i) {
        const QPointF p = samplePoint(a, b, i, n);
        const int x0 = static_cast<int>(std::floor(p.x()));
        const int y0 = static_cast<int>(std::floor(p.y()));
        const double fx = p.x() - x0;
        const double fy = p.y() - y0;
        double value = 0.0;
        bool ok = true;
        for (int dy = 0; dy < 2 && ok; ++dy) {
            const double wy = dy ? fy : 1.0 - fy;
            if (wy <= 0.0) {
                continue;
            }
            for (int dx = 0; dx < 2; ++dx) {
                const double wx = dx ? fx : 1.0 - fx;
                if (wx <= 0.0) {
                    continue;
                }
                if (!r.covered(x0 + dx, y0 + dy)) {
                    ok = false;
                    break;
                }
                value += wx * wy * r.at(x0 + dx, y0 + dy);
            }
        }
        if (ok) {
            out[static_cast<std::size_t>(i)] = static_cast<float>(value);
        }
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

// Distance along a path of volume positions: entry i is the length from the first
// position to position i. Once a position is missing (NaN, or a surface hole) the
// length is unknown, so that entry and every later one are NaN.
inline std::vector<double> cumulativeLength(const std::vector<std::array<double, 3>>& positions)
{
    std::vector<double> out(positions.size(), std::numeric_limits<double>::quiet_NaN());
    double total = 0.0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto& p = positions[i];
        if (!validPosition(p)) {
            break;
        }
        if (i > 0) {
            const auto& q = positions[i - 1];
            total += std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) +
                               (p[2] - q[2]) * (p[2] - q[2]));
        }
        out[i] = total;
    }
    return out;
}

// Whether `inCell` holds at a point inside every unit grid cell the straight segment
// from a to b (grid coordinates) crosses, including cells it only clips at a corner.
template <class InCell>
bool everyCrossedCell(QPointF a, QPointF b, InCell inCell)
{
    std::vector<double> t{0.0, 1.0};
    const auto crossings = [&t](double p0, double p1) {
        for (double k = std::floor(std::min(p0, p1)) + 1.0; k < std::max(p0, p1); k += 1.0) {
            t.push_back((k - p0) / (p1 - p0));
        }
    };
    crossings(a.x(), b.x());
    crossings(a.y(), b.y());
    std::sort(t.begin(), t.end());
    for (std::size_t i = 0; i + 1 < t.size(); ++i) {
        if (t[i] < t[i + 1] && !inCell(a + (b - a) * (0.5 * (t[i] + t[i + 1])))) {
            return false;
        }
    }
    return true;
}

// Volume position as an affine function of frame pixel position: v = origin + dx * x + dy * y.
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
// that barely change across the view (the slice axis of a plane) get no lines, nor do
// axes that would need more than `maxLinesPerAxis`.
inline std::vector<GridLine> gridLines(const AffineMap& map, const QRectF& rect, double step,
                                       int maxLinesPerAxis = 60)
{
    std::vector<GridLine> out;
    if (!(step > 0.0) || !std::isfinite(step) || rect.isEmpty()) {
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
        const double firstIndex = std::ceil(lo / step);
        const double lastIndex = std::floor(hi / step);
        if (!(lastIndex - firstIndex <= maxLinesPerAxis)) {
            continue;
        }
        for (double k = firstIndex; k <= lastIndex; k += 1.0) {
            // Multiply rather than accumulate, so every value is an exact multiple.
            const double value = k * step;
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

// Decimals that show every multiple of `step` exactly enough to tell neighbours apart:
// 0 for whole steps, 1 for 0.5 or 2.5, 2 for 0.25, up to 6.
inline int gridLabelDecimals(double step)
{
    if (!(step > 0.0) || !std::isfinite(step)) {
        return 0;
    }
    double scale = 1.0;
    for (int d = 0; d < 6; ++d) {
        const double scaled = step * scale;
        if (std::abs(scaled - std::round(scaled)) <= 1e-6 * std::max(1.0, scaled)) {
            return d;
        }
        scale *= 10.0;
    }
    return 6;
}

inline QString gridLabel(int axis, double value, double step)
{
    static const char* names[] = {"x", "y", "z"};
    const int decimals = gridLabelDecimals(step);
    // Avoid "-0" for values that round to zero.
    const double shown = std::abs(value) < 0.5 * std::pow(10.0, -decimals) ? 0.0 : value;
    return QString::fromLatin1(names[std::clamp(axis, 0, 2)]) + QStringLiteral("=") +
           QString::number(shown, 'f', decimals);
}

// Where a square loupe of up to `preferred` pixels goes in a `pane`, up and to the right
// of `cursor` and fully inside the pane with `margin` pixels to spare and `below` pixels
// under it for a label. Smaller panes get a smaller loupe; none below `minimum`.
inline std::optional<QRectF> loupeRect(QSizeF pane, QPointF cursor, double preferred,
                                       double minimum = 48.0, double margin = 4.0, double below = 22.0)
{
    const double size = std::min({preferred, pane.width() - 2 * margin, pane.height() - 2 * margin - below});
    if (!(size >= minimum)) {
        return std::nullopt;
    }
    const double half = size / 2.0;
    const QPointF wanted = cursor + QPointF(size * 0.75, -size * 0.75);
    const QPointF center(clampInto(wanted.x(), margin + half, pane.width() - margin - half),
                         clampInto(wanted.y(), margin + half, pane.height() - margin - below - half));
    return QRectF(center - QPointF(half, half), QSizeF(size, size));
}



// Where the profile and histogram panels go in a pane: anchored to the bottom, the
// profile on the left and the histogram on the right, each lifted above anything in
// `reserved` (the scale bar) and kept clear of the other. When they do not fit side by
// side they stack, and both shrink together down to `minScale` of their size. A panel
// that still does not fit is left out, the histogram before the profile.
struct PanelLayout {
    std::optional<QRectF> profile;
    std::optional<QRectF> histogram;
};

inline PanelLayout layoutPanels(QSizeF pane, std::optional<QSizeF> profile, std::optional<QSizeF> histogram,
                                const std::vector<QRectF>& reserved = {}, double margin = 10.0,
                                double gap = 6.0, double minScale = 0.6)
{
    const QRectF area(margin, margin, pane.width() - 2 * margin, pane.height() - 2 * margin);
    // A panel of `size` at the bottom of the left or right edge, moved up past whatever
    // it would cover.
    auto place = [&](QSizeF size, bool right, const std::vector<QRectF>& taken) -> std::optional<QRectF> {
        if (area.width() <= 0 || area.height() <= 0 || size.width() > area.width()) {
            return std::nullopt;
        }
        const double x = right ? area.right() - size.width() : area.left();
        double bottom = area.bottom();
        for (int tries = 0; tries < 32; ++tries) {
            const QRectF r(x, bottom - size.height(), size.width(), size.height());
            if (r.top() < area.top()) {
                return std::nullopt;
            }
            std::optional<double> blockedTop;
            for (const auto* list : {&reserved, &taken}) {
                for (const QRectF& b : *list) {
                    if (!b.isEmpty() && r.intersects(b.adjusted(-gap, -gap, gap, gap))) {
                        blockedTop = blockedTop ? std::min(*blockedTop, b.top()) : b.top();
                    }
                }
            }
            if (!blockedTop) {
                return r;
            }
            bottom = *blockedTop - gap;
        }
        return std::nullopt;
    };
    auto scaled = [](const std::optional<QSizeF>& s, double k) {
        return s ? std::optional<QSizeF>(QSizeF(s->width() * k, s->height() * k)) : std::nullopt;
    };
    for (double k = 1.0; k >= minScale - 1e-9; k -= 0.1) {
        const auto p = scaled(profile, k);
        const auto h = scaled(histogram, k);
        PanelLayout out;
        if (p && h) {
            // Side by side, then stacked on the right, then stacked on the left.
            for (int arrangement = 0; arrangement < 3; ++arrangement) {
                if (arrangement == 1) {
                    out.histogram = place(*h, true, {});
                    out.profile = out.histogram ? place(*p, true, {*out.histogram}) : std::nullopt;
                } else {
                    out.profile = place(*p, false, {});
                    out.histogram = out.profile ? place(*h, arrangement == 0, {*out.profile}) : std::nullopt;
                }
                if (out.profile && out.histogram) {
                    return out;
                }
            }
            continue;
        }
        if (p) {
            out.profile = place(*p, false, {});
        }
        if (h) {
            out.histogram = place(*h, true, {});
        }
        if (out.profile || out.histogram) {
            return out;
        }
    }
    if (profile && histogram) {
        return layoutPanels(pane, profile, std::nullopt, reserved, margin, gap, minScale);
    }
    return {};
}

// ---- frames carried forward ----------------------------------------------------------

// A frame that still lacks chunks shows the previous frame's pixel wherever it has none
// of its own and the pixel is waiting on a read (missing[q] == missingCode), so loading
// regions do not flash blank. Measurement validity travels with the pixel: `prevMeasure`
// is the earlier frame's mask when that frame had the same geometry and one (null
// otherwise: after an edit, or rendered with the overlays off), and a carried pixel
// keeps its mark from it, or 0. So a pixel carried across an edit stays unmeasured
// however many frames carry it on. `measure` is null when this frame has no mask.
// All buffers hold n pixels, row-major.
inline void carryMissingPixels(std::size_t n, std::uint8_t* values, std::uint8_t* coverage, std::uint8_t* measure,
                               const std::uint8_t* prevValues, const std::uint8_t* prevCoverage,
                               const std::uint8_t* prevMeasure, const std::vector<std::int8_t>& missing,
                               std::int8_t missingCode)
{
    const std::size_t count = std::min(n, missing.size());
    for (std::size_t q = 0; q < count; ++q) {
        if (coverage[q] || !prevCoverage[q] || missing[q] != missingCode) {
            continue;
        }
        values[q] = prevValues[q];
        coverage[q] = 1;
        if (measure) {
            measure[q] = prevMeasure ? prevMeasure[q] : 0;
        }
    }
}

}  // namespace vc3d::hud
