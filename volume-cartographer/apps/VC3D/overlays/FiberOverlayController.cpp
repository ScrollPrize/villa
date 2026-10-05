#include "FiberOverlayController.hpp"

#include "OverlayBatchItem.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QGraphicsObject>
#include <QGraphicsScene>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QStyleOptionGraphicsItem>

#include <algorithm>
#include <cmath>
#include <unordered_map>


struct FiberOverlayController::PersistentItems
{
    struct ViewerItems {
        QPointer<OverlayBatchItem> lines;
        QPointer<OverlayBatchItem> points;
    };

    std::unordered_map<VolumeViewerBase*, ViewerItems> viewers;
};

FiberOverlayController::FiberOverlayController(QObject* parent)
    : ViewerOverlayControllerBase("fiber_overlay", parent)
    , _persistentItems(std::make_unique<PersistentItems>())
{
}

FiberOverlayController::~FiberOverlayController() = default;

void FiberOverlayController::setChains(std::vector<Chain> chains)
{
    clearPointChainProjectionCache();
    _transformedChains.clear();
    _chains = std::move(chains);
    if (_chains.empty()) {
        _visible = false;
    }
    refreshAll();
}

void FiberOverlayController::setViewerBaseToViewerFactor(
    VolumeViewerBase* viewer, double factor)
{
    if (!viewer || !(factor > 0.0) || !std::isfinite(factor)) {
        return;
    }
    const auto found = _viewerFactors.find(viewer);
    if (found != _viewerFactors.end() && found->second == factor) {
        return;
    }
    _viewerFactors[viewer] = factor;
    _transformedChains.erase(viewer);
    clearPointChainProjectionCache();
    refreshViewer(viewer);
}

double FiberOverlayController::viewerBaseToViewerFactor(
    VolumeViewerBase* viewer) const
{
    const auto found = _viewerFactors.find(viewer);
    return found == _viewerFactors.end() ? 1.0 : found->second;
}

void FiberOverlayController::detachViewer(VolumeViewerBase* viewer)
{
    _viewerFactors.erase(viewer);
    _transformedChains.erase(viewer);
    ViewerOverlayControllerBase::detachViewer(viewer);
}

const std::vector<FiberOverlayController::Chain>&
FiberOverlayController::chainsForViewer(VolumeViewerBase* viewer) const
{
    const double factor = viewerBaseToViewerFactor(viewer);
    if (factor == 1.0) {
        return _chains;
    }
    if (auto found = _transformedChains.find(viewer);
        found != _transformedChains.end()) {
        return found->second;
    }
    std::vector<Chain> transformed = _chains;
    const float scale = static_cast<float>(factor);
    for (auto& chain : transformed) {
        for (auto& point : chain.points) {
            point *= scale;
        }
    }
    return _transformedChains.emplace(viewer, std::move(transformed)).first->second;
}

void FiberOverlayController::setViewDistance(double distance)
{
    const float clamped = static_cast<float>(std::clamp(distance, 0.0, 10000.0));
    if (_viewDistance == clamped) {
        return;
    }
    _viewDistance = clamped;
    clearPointChainProjectionCache();
    refreshAll();
}

void FiberOverlayController::setVisible(bool visible)
{
    visible = visible && !_chains.empty();
    if (_visible == visible) {
        return;
    }
    _visible = visible;
    refreshAll();
}

void FiberOverlayController::setShowLinked(bool show)
{
    if (_showLinked == show) {
        return;
    }
    _showLinked = show;
    // Only colors and link markers change; projections stay valid.
    refreshAll();
}

ViewerOverlayControllerBase::PointChainStyle
FiberOverlayController::fiberStyle(const QColor& color, float distanceTolerance)
{
    PointChainStyle style;
    style.color = color;
    style.pointBorderColor = color;
    style.lineOpacity = 0.75f;
    style.distanceTolerance = distanceTolerance;
    return style;
}

QColor FiberOverlayController::fiberColor(uint64_t fiberId)
{
    // Consecutive fibers are separated by the golden angle. Additional
    // saturation/value bands retain distinct colors for unusually large sets.
    const int hue = static_cast<int>((fiberId * 137ULL) % 360ULL);
    const int saturation = 190 + static_cast<int>((fiberId / 360ULL) % 3ULL) * 25;
    const int value = 255 - static_cast<int>((fiberId / 1080ULL) % 3ULL) * 20;
    return QColor::fromHsv(hue, saturation, value);
}

bool FiberOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && _visible && !_chains.empty();
}

std::optional<FiberOverlayController::ControlPointHit>
FiberOverlayController::hitTestControlPoint(VolumeViewerBase* viewer,
                                            const QPointF& scenePoint,
                                            qreal maxDistancePx) const
{
    if (!isOverlayEnabledFor(viewer)) {
        return std::nullopt;
    }

    std::optional<ControlPointHit> best;
    qreal bestDistanceSq = maxDistancePx * maxDistancePx;
    std::vector<float> opacities;
    for (const Chain& chain : chainsForViewer(viewer)) {
        const FilteredPoints filtered =
            projectedPointChain(viewer, chain.points, _viewDistance, &opacities);
        for (std::size_t i = 0; i < filtered.scenePoints.size(); ++i) {
            if (i < opacities.size() && opacities[i] <= 0.0f) {
                // Boundary projection: line vertex only, no dot drawn.
                continue;
            }
            const QPointF delta = filtered.scenePoints[i] - scenePoint;
            const qreal distanceSq = delta.x() * delta.x() + delta.y() * delta.y();
            if (distanceSq < bestDistanceSq && i < filtered.sourceIndices.size()) {
                bestDistanceSq = distanceSq;
                best = ControlPointHit{chain.id,
                                       static_cast<int>(filtered.sourceIndices[i])};
            }
        }
    }
    return best;
}

void FiberOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                               OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) {
        return;
    }

    const auto& viewerChains = chainsForViewer(viewer);
    // When link rings are shown, keep each ringed chain's projection from the
    // first pass instead of projecting it a second time below.
    struct RingSource {
        const Chain* chain{nullptr};
        FilteredPoints filtered;
        std::vector<float> opacities;
    };
    std::vector<RingSource> ringSources;

    for (std::size_t index = 0; index < viewerChains.size(); ++index) {
        const Chain& chain = viewerChains[index];
        const uint64_t colorId =
            (_showLinked && chain.colorId != 0) ? chain.colorId : chain.id;
        const PointChainStyle style = fiberStyle(fiberColor(colorId), _viewDistance);
        const bool needsRings = _showLinked && !chain.pointLinkStates.empty();
        if (!needsRings) {
            renderPointChain(viewer, builder, chain.points, style);
            continue;
        }
        RingSource source;
        source.chain = &chain;
        renderPointChain(viewer, builder, chain.points, style, std::nullopt,
                         &source.filtered, &source.opacities);
        ringSources.push_back(std::move(source));
    }

    if (!_showLinked) {
        return;
    }

    // Linked-control-point rings, appended after every chain's primitives so
    // they paint last (OverlayBatchItem draws commands in insertion order).
    // Colors match the Line Annotation GUI's branch control-point markers.
    for (const RingSource& source : ringSources) {
        const Chain& chain = *source.chain;
        const FilteredPoints& filtered = source.filtered;
        const std::vector<float>& opacities = source.opacities;
        for (std::size_t i = 0; i < filtered.scenePoints.size(); ++i) {
            const float opacity = i < opacities.size() ? opacities[i] : 1.0f;
            if (opacity <= 0.0f || i >= filtered.sourceIndices.size()) {
                continue;
            }
            const std::size_t sourceIndex = filtered.sourceIndices[i];
            if (sourceIndex >= chain.pointLinkStates.size()) {
                continue;
            }
            const uint8_t state = chain.pointLinkStates[sourceIndex];
            if (state == 0) {
                continue;
            }
            const bool pending = state == 1;
            OverlayStyle style;
            style.penColor = pending ? QColor(80, 150, 255, 245)
                                     : QColor(210, 95, 255, 245);
            style.brushColor = pending ? QColor(80, 150, 255, 175)
                                       : QColor(210, 95, 255, 175);
            style.penColor.setAlphaF(style.penColor.alphaF() * opacity);
            style.brushColor.setAlphaF(style.brushColor.alphaF() * opacity);
            style.penWidth = 2.0;
            style.z = 95.0;
            builder.addPoint(filtered.scenePoints[i], 6.25, style);
        }
    }
}

void FiberOverlayController::applyOverlayPrimitives(
    VolumeViewerBase* viewer,
    std::vector<OverlayPrimitive> primitives)
{
    if (!viewer || primitives.empty()) {
        clearOverlay(viewer);
        return;
    }

    std::vector<OverlayLineCommand> lineCommands;
    std::vector<OverlayPointCommand> pointCommands;
    buildOverlayBatchCommands(primitives, lineCommands, pointCommands);

    auto& items = _persistentItems->viewers[viewer];
    if (!items.lines || !items.points) {
        // A viewer may clear all overlay groups independently. QPointer lets
        // us detect that and recreate the retained pair safely on demand.
        if (items.lines || items.points) {
            ViewerOverlayControllerBase::clearOverlay(viewer);
        }

        QGraphicsScene* scene = viewerScene(viewer);
        if (!scene) {
            clearOverlay(viewer);
            return;
        }

        auto* lines = new OverlayBatchItem();
        auto* points = new OverlayBatchItem();
        lines->setZValue(94.0);
        points->setZValue(95.0);
        lines->setLineCommands(std::move(lineCommands));
        points->setPointCommands(std::move(pointCommands));

        scene->addItem(lines);
        scene->addItem(points);
        viewer->setOverlayGroup(overlayGroupKey(), {lines, points});
        items.lines = lines;
        items.points = points;
        return;
    }

    items.lines->setLineCommands(std::move(lineCommands));
    items.points->setPointCommands(std::move(pointCommands));
}

void FiberOverlayController::clearOverlay(VolumeViewerBase* viewer) const
{
    if (_persistentItems && viewer) {
        _persistentItems->viewers.erase(viewer);
    }
    ViewerOverlayControllerBase::clearOverlay(viewer);
}
