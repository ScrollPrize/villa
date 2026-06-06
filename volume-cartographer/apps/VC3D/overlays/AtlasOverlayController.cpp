#include "AtlasOverlayController.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QRectF>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr const char* kOverlayGroup = "atlas_objects";

ViewerOverlayControllerBase::OverlayStyle atlasLineStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(220, 60, 50);
    style.brushColor = Qt::transparent;
    style.penWidth = 2.0;
    style.z = 50.0;
    return style;
}

ViewerOverlayControllerBase::OverlayStyle atlasAnchorStyle()
{
    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 230, 70);
    style.brushColor = QColor(255, 230, 70);
    style.penWidth = 0.0;
    style.z = 60.0;
    return style;
}

std::optional<std::pair<int, int>> controlAnchorSourceRange(
    const vc::atlas::FiberMapping& fiber)
{
    if (fiber.lineAnchors.empty()) {
        return std::nullopt;
    }

    int first = std::numeric_limits<int>::max();
    int last = std::numeric_limits<int>::min();
    int count = 0;
    for (const auto& anchor : fiber.controlAnchors) {
        const vc::atlas::AtlasAnchor* bestLineAnchor = nullptr;
        double bestDistance2 = std::numeric_limits<double>::infinity();
        for (const auto& lineAnchor : fiber.lineAnchors) {
            const double du = lineAnchor.atlasU - anchor.atlasU;
            const double dv = lineAnchor.atlasV - anchor.atlasV;
            const double distance2 = du * du + dv * dv;
            if (distance2 < bestDistance2) {
                bestDistance2 = distance2;
                bestLineAnchor = &lineAnchor;
            }
        }
        if (!bestLineAnchor) {
            continue;
        }
        first = std::min(first, bestLineAnchor->sourceIndex);
        last = std::max(last, bestLineAnchor->sourceIndex);
        ++count;
    }
    if (count < 2 || first > last) {
        return std::nullopt;
    }
    return std::make_pair(first, last);
}

bool anchorInControlRange(const vc::atlas::AtlasAnchor& anchor,
                          const std::optional<std::pair<int, int>>& range)
{
    if (!range) {
        return true;
    }
    return anchor.sourceIndex >= range->first && anchor.sourceIndex <= range->second;
}
}

AtlasOverlayController::AtlasOverlayController(QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
{
}

void AtlasOverlayController::setAtlas(vc::atlas::Atlas atlas,
                                      std::shared_ptr<const QuadSurface> displaySurface,
                                      vc::atlas::AtlasDisplayRange displayRange)
{
    _atlas = std::move(atlas);
    _displaySurface = std::move(displaySurface);
    _displayRange = displayRange;
    refreshAll();
}

void AtlasOverlayController::clearAtlas()
{
    _atlas.reset();
    _displaySurface.reset();
    _displayRange = {};
    refreshAll();
}

std::optional<cv::Vec2f> AtlasOverlayController::atlasAnchorToSurface(
    const vc::atlas::AtlasAnchor& anchor,
    const vc::atlas::FiberMapping& fiber) const
{
    if (!_displaySurface || !std::isfinite(anchor.atlasU) || !std::isfinite(anchor.atlasV)) {
        return std::nullopt;
    }
    const double atlasU = vc::atlas::actualAtlasU(anchor, fiber, _displayRange.baseColumns);
    const cv::Vec2f surfaceCoord =
        vc::atlas::atlasGridToSurfaceCoords(atlasU,
                                            anchor.atlasV,
                                            *_displaySurface,
                                            _displayRange.atlasUOffset);
    if (!std::isfinite(surfaceCoord[0]) || !std::isfinite(surfaceCoord[1])) {
        return std::nullopt;
    }
    return surfaceCoord;
}

std::optional<QRectF> AtlasOverlayController::surfaceBounds() const
{
    if (!_atlas) {
        return std::nullopt;
    }

    bool havePoint = false;
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    for (const auto& fiber : _atlas->fibers) {
        const auto sourceRange = controlAnchorSourceRange(fiber);
        for (const auto& anchor : fiber.lineAnchors) {
            if (!anchorInControlRange(anchor, sourceRange)) {
                continue;
            }
            const auto surfaceCoord = atlasAnchorToSurface(anchor, fiber);
            if (!surfaceCoord) {
                continue;
            }
            havePoint = true;
            minX = std::min(minX, (*surfaceCoord)[0]);
            minY = std::min(minY, (*surfaceCoord)[1]);
            maxX = std::max(maxX, (*surfaceCoord)[0]);
            maxY = std::max(maxY, (*surfaceCoord)[1]);
        }
    }
    if (!havePoint) {
        return std::nullopt;
    }
    return QRectF(QPointF(minX, minY), QPointF(maxX, maxY)).normalized();
}

bool AtlasOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && _atlas.has_value() && _displaySurface != nullptr;
}

void AtlasOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                               OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) {
        return;
    }

    const auto lineStyle = atlasLineStyle();
    const auto anchorStyle = atlasAnchorStyle();
    for (const auto& fiber : _atlas->fibers) {
        const auto sourceRange = controlAnchorSourceRange(fiber);
        std::vector<cv::Vec2f> linePoints;
        linePoints.reserve(fiber.lineAnchors.size());
        for (const auto& anchor : fiber.lineAnchors) {
            if (!anchorInControlRange(anchor, sourceRange)) {
                continue;
            }
            const auto surfaceCoord = atlasAnchorToSurface(anchor, fiber);
            if (!surfaceCoord) {
                continue;
            }
            linePoints.push_back(*surfaceCoord);
        }
        if (linePoints.size() >= 2) {
            builder.addSurfaceLineStrip(linePoints, false, lineStyle);
        }
        for (const auto& anchor : fiber.controlAnchors) {
            const auto surfaceCoord = atlasAnchorToSurface(anchor, fiber);
            if (!surfaceCoord) {
                continue;
            }
            builder.addSurfacePoint(*surfaceCoord, 4.0, anchorStyle);
        }
    }
}
