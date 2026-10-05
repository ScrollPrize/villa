#pragma once

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include <QPainterPath>
#include <QLineF>
#include <algorithm>
#include <cmath>
#include <optional>

namespace vc3d::spiral {

inline std::optional<QPointF> projectPatchPoint(
    const QPointF& point, const std::shared_ptr<QuadSurface>& from,
    const std::shared_ptr<QuadSurface>& to, SurfacePatchIndex* index, float tolerance)
{
    if (!from || !to) return std::nullopt;
    if (from == to) return point;
    const auto sample = from->sampleAtSurface({point.x(), point.y()});
    const auto volume = sample ? std::optional<cv::Vec3f>(sample.volume) : std::nullopt;
    if (!volume || !index || !index->containsSurface(to)) return std::nullopt;
    SurfacePatchIndex::PointQuery query;
    query.worldPoint = *volume;
    query.tolerance = tolerance;
    query.surfaces.only = to;
    const auto hit = index->locate(query);
    if (!hit) return std::nullopt;
    const auto loc = to->loc(hit->ptr);
    const auto originalGrid = from->surfaceToGrid({point.x(), point.y()});
    const auto targetGrid = to->surfaceToGrid({loc[0], loc[1]});
    const float originalWinding = lookupDepthIndex(from.get(),
        static_cast<int>(std::round(originalGrid[1])), static_cast<int>(std::round(originalGrid[0])));
    const float targetWinding = lookupDepthIndex(to.get(),
        static_cast<int>(std::round(targetGrid[1])), static_cast<int>(std::round(targetGrid[0])));
    if (std::isfinite(originalWinding) && std::isfinite(targetWinding)
        && std::round(originalWinding) != std::round(targetWinding)) return std::nullopt;
    return QPointF(loc[0], loc[1]);
}

inline std::optional<QPainterPath> projectPatchShape(
    const QPainterPath& path, const std::shared_ptr<QuadSurface>& from,
    const std::shared_ptr<QuadSurface>& to, SurfacePatchIndex* index, float tolerance)
{
    if (from == to) return path;
    if (!from || !to) return std::nullopt;
    QPainterPath mapped;
    mapped.setFillRule(path.fillRule());
    const double step = 1.0 / std::max(from->scale()[0], from->scale()[1]);
    for (const auto& polygon : path.toSubpathPolygons()) {
        if (polygon.isEmpty()) continue;
        const auto first = projectPatchPoint(polygon.front(), from, to, index, tolerance);
        if (!first) return std::nullopt;
        mapped.moveTo(*first);
        for (int i = 1; i < polygon.size(); ++i) {
            const int count = std::max(1, static_cast<int>(std::ceil(QLineF(polygon[i - 1], polygon[i]).length() / step)));
            for (int j = 1; j <= count; ++j) {
                const auto point = projectPatchPoint(polygon[i - 1]
                    + (polygon[i] - polygon[i - 1]) * (double(j) / count), from, to, index, tolerance);
                if (!point) return std::nullopt;
                mapped.lineTo(*point);
            }
        }
        mapped.closeSubpath();
    }
    return mapped;
}

} // namespace vc3d::spiral
