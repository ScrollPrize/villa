#include "FiberOverlayController.hpp"

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

namespace
{

QPen penForStyle(const ViewerOverlayControllerBase::OverlayStyle& style)
{
    QPen pen(style.penColor);
    pen.setWidthF(style.penWidth);
    pen.setStyle(style.penStyle);
    pen.setCapStyle(style.penCap);
    pen.setJoinStyle(style.penJoin);
    if (!style.dashPattern.empty()) {
        QVector<qreal> pattern;
        pattern.reserve(static_cast<int>(style.dashPattern.size()));
        for (qreal value : style.dashPattern) {
            pattern.append(value);
        }
        pen.setDashPattern(pattern);
    }
    return pen;
}

class FiberBatchItem final : public QGraphicsObject
{
public:
    enum class Content {
        Lines,
        Points,
    };

    explicit FiberBatchItem(Content content)
        : _content(content)
    {
        setAcceptedMouseButtons(Qt::NoButton);
        setFlag(QGraphicsItem::ItemUsesExtendedStyleOption, true);
    }

    QRectF boundingRect() const override { return _bounds; }

    void setPrimitives(const std::vector<ViewerOverlayControllerBase::OverlayPrimitive>& primitives)
    {
        std::vector<DrawCommand> commands;
        commands.reserve(primitives.size());
        QRectF bounds;
        bool haveBounds = false;

        auto append = [&](QPainterPath path,
                          const ViewerOverlayControllerBase::OverlayStyle& style) {
            if (path.isEmpty()) {
                return;
            }
            DrawCommand command;
            command.path = std::move(path);
            command.pen = penForStyle(style);
            command.brush = QBrush(style.brushColor);
            const qreal padding = std::max<qreal>(0.5, command.pen.widthF() * 0.5) + 1.0;
            command.bounds = command.path.boundingRect().adjusted(
                -padding, -padding, padding, padding);
            if (!haveBounds) {
                bounds = command.bounds;
                haveBounds = true;
            } else {
                bounds = bounds.united(command.bounds);
            }
            commands.push_back(std::move(command));
        };

        for (const auto& primitive : primitives) {
            if (_content == Content::Lines) {
                const auto* line =
                    std::get_if<ViewerOverlayControllerBase::LineStripPrimitive>(&primitive);
                if (!line || line->points.size() < 2) {
                    continue;
                }
                QPainterPath path(line->points.front());
                for (std::size_t i = 1; i < line->points.size(); ++i) {
                    path.lineTo(line->points[i]);
                }
                if (line->closed) {
                    path.closeSubpath();
                }
                append(std::move(path), line->style);
            } else {
                const auto* point =
                    std::get_if<ViewerOverlayControllerBase::PointPrimitive>(&primitive);
                if (!point || point->radius <= 0.0) {
                    continue;
                }
                QPainterPath path;
                path.addEllipse(point->position, point->radius, point->radius);
                append(std::move(path), point->style);
            }
        }

        prepareGeometryChange();
        _commands = std::move(commands);
        _bounds = haveBounds ? bounds : QRectF{};
        update();
    }

    void paint(QPainter* painter,
               const QStyleOptionGraphicsItem* option,
               QWidget* /*widget*/) override
    {
        if (!painter) {
            return;
        }
        painter->save();
        const QRectF exposed = option ? option->exposedRect : _bounds;
        for (const DrawCommand& command : _commands) {
            if (!exposed.isEmpty() && !command.bounds.intersects(exposed)) {
                continue;
            }
            painter->setPen(command.pen);
            painter->setBrush(command.brush);
            painter->drawPath(command.path);
        }
        painter->restore();
    }

private:
    struct DrawCommand {
        QPainterPath path;
        QPen pen;
        QBrush brush;
        QRectF bounds;
    };

    Content _content;
    std::vector<DrawCommand> _commands;
    QRectF _bounds;
};

} // namespace

struct FiberOverlayController::PersistentItems
{
    struct ViewerItems {
        QPointer<FiberBatchItem> lines;
        QPointer<FiberBatchItem> points;
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
    _chains = std::move(chains);
    if (_chains.empty()) {
        _visible = false;
    }
    refreshAll();
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
    for (const Chain& chain : _chains) {
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

    for (std::size_t index = 0; index < _chains.size(); ++index) {
        const Chain& chain = _chains[index];
        const uint64_t colorId =
            (_showLinked && chain.colorId != 0) ? chain.colorId : chain.id;
        const PointChainStyle style = fiberStyle(fiberColor(colorId), _viewDistance);
        renderPointChain(viewer, builder, chain.points, style);
    }

    if (!_showLinked) {
        return;
    }

    // Linked-control-point rings, appended after every chain's primitives so
    // they paint last (FiberBatchItem draws commands in insertion order).
    // Colors match the Line Annotation GUI's branch control-point markers.
    std::vector<float> opacities;
    for (const Chain& chain : _chains) {
        if (chain.pointLinkStates.empty()) {
            continue;
        }
        const FilteredPoints filtered =
            projectedPointChain(viewer, chain.points, _viewDistance, &opacities);
        for (std::size_t i = 0; i < filtered.scenePoints.size(); ++i) {
            const float opacity = i < opacities.size() ? opacities[i] : 1.0f;
            if (opacity <= 0.0f || i >= filtered.sourceIndices.size()) {
                continue;
            }
            const std::size_t source = filtered.sourceIndices[i];
            if (source >= chain.pointLinkStates.size()) {
                continue;
            }
            const uint8_t state = chain.pointLinkStates[source];
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

        auto* lines = new FiberBatchItem(FiberBatchItem::Content::Lines);
        auto* points = new FiberBatchItem(FiberBatchItem::Content::Points);
        lines->setZValue(94.0);
        points->setZValue(95.0);
        lines->setPrimitives(primitives);
        points->setPrimitives(primitives);

        scene->addItem(lines);
        scene->addItem(points);
        viewer->setOverlayGroup(overlayGroupKey(), {lines, points});
        items.lines = lines;
        items.points = points;
        return;
    }

    items.lines->setPrimitives(primitives);
    items.points->setPrimitives(primitives);
}

void FiberOverlayController::clearOverlay(VolumeViewerBase* viewer) const
{
    if (_persistentItems && viewer) {
        _persistentItems->viewers.erase(viewer);
    }
    ViewerOverlayControllerBase::clearOverlay(viewer);
}
