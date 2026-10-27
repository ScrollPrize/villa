#pragma once
// How plane viewers draw surface intersections: the viewer's own (active and
// named surfaces) and the store overlay's (the surface list's visible surfaces).
#include <QColor>
#include <QRgb>
#include <array>
#include <cstddef>
#include <functional>
#include <string>

namespace vc3d {

constexpr std::array<QRgb, 12> kIntersectionPalette = {
    qRgb(255, 120, 120), qRgb(120, 200, 255), qRgb(120, 255, 140),
    qRgb(255, 220, 100), qRgb(220, 140, 255), qRgb(255, 160, 200),
    qRgb(140, 255, 220), qRgb(200, 255, 140), qRgb(255, 180, 120),
    qRgb(180, 200, 255), qRgb(255, 140, 180), qRgb(160, 255, 180),
};
constexpr int kIntersectionZ = 100;
constexpr int kHighlightedIntersectionZ = 110;
constexpr int kActiveIntersectionZ = 120;
inline QColor highlightedIntersectionColor() { return QColor(0, 220, 255); }
// A surface's colour, the same in every viewer and overlay.
inline QColor intersectionColor(const std::string& id)
{
    return QColor::fromRgba(kIntersectionPalette[std::hash<std::string>{}(id) % kIntersectionPalette.size()]);
}

} // namespace vc3d
