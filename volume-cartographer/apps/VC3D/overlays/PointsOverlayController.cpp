#include "PointsOverlayController.hpp"

#include "OverlayBatchItem.hpp"
#include "ScreenSpacePointIndex.hpp"
#include "../volume_viewers/CVolumeViewerView.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../ViewerManager.hpp"

#include "vc/ui/VCCollection.hpp"

#include <QGraphicsScene>
#include <QPointer>
#include <QtGlobal>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
constexpr const char* kOverlayGroupPoints = "point_collection_overlay";
constexpr qreal kBaseRadius = 5.0;
constexpr qreal kHighlightRadiusMultiplier = 1.4;
constexpr qreal kSelectedRadiusMultiplier = 1.4;
constexpr qreal kBasePenWidth = 1.5;
constexpr qreal kHighlightPenWidth = 2.5;
constexpr qreal kSelectedPenWidth = 2.5;
constexpr qreal kSameWrapPolylineWidth = kBaseRadius;
constexpr qreal kZValue = 95.0;
constexpr qreal kPolylineZValue = kZValue - 1.0;
constexpr qreal kTextZValue = 96.0;

QColor toColor(const cv::Vec3f& c, float opacity)
{
    QColor color;
    color.setRedF(std::clamp(c[0], 0.0f, 1.0f));
    color.setGreenF(std::clamp(c[1], 0.0f, 1.0f));
    color.setBlueF(std::clamp(c[2], 0.0f, 1.0f));
    color.setAlphaF(std::clamp(opacity, 0.0f, 1.0f));
    return color;
}

QString formatWinding(float winding, bool absolute)
{
    if (std::isnan(winding)) {
        return {};
    }

    QString text = QString::number(winding, 'g');
    if (!absolute && winding >= 0.0f) {
        text.prepend('+');
    }
    return text;
}


std::vector<ColPoint> orderedCollectionPoints(const VCCollection::Collection& collection)
{
    std::vector<ColPoint> points;
    points.reserve(collection.points.size());
    for (const auto& [id, point] : collection.points) {
        (void)id;
        points.push_back(point);
    }
    std::sort(points.begin(), points.end(), [](const ColPoint& a, const ColPoint& b) {
        if (a.creation_time != b.creation_time) {
            return a.creation_time < b.creation_time;
        }
        return a.id < b.id;
    });
    return points;
}

} // namespace

struct PointsOverlayController::PersistentItems
{
    struct ViewerItems {
        QPointer<OverlayBatchItem> lines;
        QPointer<OverlayBatchItem> points;
        std::vector<DisplayPointHit> hitRecords;
        ScreenSpacePointIndex hitIndex;
    };

    std::unordered_map<VolumeViewerBase*, ViewerItems> viewers;
};

PointsOverlayController::PointsOverlayController(VCCollection* collection, QObject* parent,
                                                 bool displayOnly)
    : ViewerOverlayControllerBase(displayOnly
                                      ? "display_only_point_collection_overlay"
                                      : kOverlayGroupPoints,
                                  parent)
    , _persistentItems(std::make_unique<PersistentItems>())
    , _collection(collection)
    , _displayOnly(displayOnly)
{
    connectCollectionSignals();
}

void PointsOverlayController::applyOverlayPrimitives(
    VolumeViewerBase* viewer,
    std::vector<OverlayPrimitive> primitives)
{
    if (!viewer || primitives.empty()) {
        clearOverlay(viewer);
        return;
    }

    std::vector<OverlayLineCommand> lineCommands;
    std::vector<OverlayPointCommand> pointCommands;
    if (!buildOverlayBatchCommands(primitives, lineCommands, pointCommands)) {
        // Winding labels (and anything else the batch item does not paint)
        // fall back to the general materialization, which renders them
        // exactly as before. Only the label-free case -- the display-only
        // point clouds that made this overlay slow -- takes the fast path.
        clearOverlay(viewer);
        ViewerOverlayControllerBase::applyOverlayPrimitives(viewer, std::move(primitives));
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

        auto* lines = new OverlayBatchItem();
        auto* points = new OverlayBatchItem();
        lines->setZValue(kPolylineZValue);
        points->setZValue(kZValue);
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

void PointsOverlayController::clearOverlay(VolumeViewerBase* viewer) const
{
    if (_persistentItems && viewer) {
        _persistentItems->viewers.erase(viewer);
    }
    ViewerOverlayControllerBase::clearOverlay(viewer);
}

void PointsOverlayController::setCoordinateScale(double scale)
{
    if (!std::isfinite(scale) || scale <= 0.0) return;
    if (std::abs(_coordinateScale - scale) < 1.0e-12) return;
    _coordinateScale = scale;
    _orderedCollections.clear();
    ++_pointsRevision;
    refreshAll();
}

void PointsOverlayController::setHiddenCollectionIds(const QSet<qulonglong>& ids)
{
    if (_hiddenCollectionIds == ids) return;
    _hiddenCollectionIds = ids;
    refreshAll();
}

const PointsOverlayController::OrderedCollection&
PointsOverlayController::orderedCollection(
    uint64_t collectionId, const PointCollections::Collection& collection) const
{
    auto found = _orderedCollections.find(collectionId);
    if (found != _orderedCollections.end() &&
        found->second.coordinateScale == _coordinateScale) {
        return found->second;
    }

    OrderedCollection ordered;
    ordered.coordinateScale = _coordinateScale;
    ordered.points = orderedCollectionPoints(collection);
    ordered.scaledPositions.reserve(ordered.points.size());
    for (const ColPoint& colPoint : ordered.points) {
        ordered.scaledPositions.push_back(colPoint.p * static_cast<float>(_coordinateScale));
    }
    return _orderedCollections.insert_or_assign(collectionId, std::move(ordered))
        .first->second;
}

void PointsOverlayController::setVisible(bool visible)
{
    if (_visible == visible) return;
    _visible = visible;
    refreshAll();
}

void PointsOverlayController::setShowWindingLabels(bool show)
{
    if (_showWindingLabels == show) return;
    _showWindingLabels = show;
    refreshAll();
}

std::optional<PointsOverlayController::DisplayPointHit>
PointsOverlayController::displayPointHitAt(
    VolumeViewerBase* viewer, const QPointF& devicePosition, qreal radius,
    const QSet<qulonglong>& allowedCollectionIds) const
{
    if (!_displayOnly || !_visible || !viewer || allowedCollectionIds.isEmpty()
        || !_persistentItems) return std::nullopt;
    const auto found = _persistentItems->viewers.find(viewer);
    if (found == _persistentItems->viewers.end()) return std::nullopt;
    const auto& items = found->second;
    if (!items.points) return std::nullopt;
    const auto hit = items.hitIndex.closest(
        devicePosition, radius, [&items, &allowedCollectionIds](std::size_t index) {
            return index < items.hitRecords.size()
                && allowedCollectionIds.contains(
                    items.hitRecords[index].ref.collectionId);
        });
    return hit && *hit < items.hitRecords.size()
        ? std::optional<DisplayPointHit>(items.hitRecords[*hit])
        : std::nullopt;
}

PointsOverlayController::~PointsOverlayController()
{
    disconnectCollectionSignals();
}

void PointsOverlayController::setViewTolerance(double tolerance)
{
    tolerance = std::clamp(tolerance, 0.0, 10000.0);
    if (std::abs(_viewTolerance - tolerance) < 0.001) {
        return;
    }
    _viewTolerance = tolerance;
    refreshAll();
}

void PointsOverlayController::setCollection(VCCollection* collection)
{
    if (_collection == collection) {
        return;
    }
    disconnectCollectionSignals();
    _collection = collection;
    connectCollectionSignals();
    _orderedCollections.clear();
    ++_pointsRevision;
    refreshAll();
}

bool PointsOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return _visible && _collection && viewer;
}

void PointsOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    PersistentItems::ViewerItems* viewerItems = nullptr;
    if (_displayOnly && _persistentItems && viewer) {
        viewerItems = &_persistentItems->viewers[viewer];
        viewerItems->hitRecords.clear();
        viewerItems->hitIndex.clear();
    }
    if (!_collection || !viewer) {
        return;
    }

    if (!_displayOnly && viewer->pointCollection() != _collection) {
        return;
    }

    const auto& collections = _collection->getAllCollections();
    if (collections.empty()) {
        return;
    }

    const std::optional<vc::PointRef> highlighted = _displayOnly
        ? std::nullopt : viewer->highlightedPoint();
    const std::optional<vc::PointRef> selected = _displayOnly
        ? std::nullopt : viewer->selectedPoint();
    const bool drawSameWrapPolylines = !_displayOnly
        && viewer->isSameWrapAnnotationModeEnabled();
    auto* graphicsView = _displayOnly ? viewer->graphicsView() : nullptr;

    for (const auto& [collectionId, collection] : collections) {
        if (_hiddenCollectionIds.contains(collectionId)) continue;
        const cv::Vec3f collectionColor = collection.color;
        const bool absoluteWinding = collection.metadata.absolute_winding_number;
        struct Entry {
            uint64_t pointId;
            float opacity{1.0f};
            bool isHighlighted{false};
            bool isSelected{false};
            bool hasLabel{false};
            QString label;
        };

        const OrderedCollection& ordered = orderedCollection(collectionId, collection);
        const std::vector<cv::Vec3f>& positions = ordered.scaledPositions;
        std::vector<Entry> entries;
        entries.reserve(ordered.points.size());

        for (const ColPoint& colPoint : ordered.points) {
            Entry entry;
            entry.pointId = colPoint.id;
            const vc::PointRef ref{collectionId, colPoint.id};
            entry.isHighlighted = highlighted && *highlighted == ref;
            entry.isSelected = selected && *selected == ref;
            // A display-only overlay (the Spiral same-winding PCLs) draws whole
            // point clouds rather than a handful of annotations. One text item
            // per point flushes the point-batching groups in applyPrimitives,
            // costing two QGraphicsItems per point; the labels are not
            // actionable there, so skip them unless the overlay asked for
            // them (relative-winding PCLs, whose annotations are the display).
            if ((!_displayOnly || _showWindingLabels)
                && !std::isnan(colPoint.winding_annotation)) {
                const QString text = formatWinding(colPoint.winding_annotation, absoluteWinding);
                entry.hasLabel = !text.isEmpty();
                entry.label = text;
            }

            entries.push_back(std::move(entry));
        }

        std::vector<float> opacities;
        auto filtered = filterPointsNearViewerSurfaceCached(
            viewer, collectionId, _pointsRevision, positions,
            static_cast<float>(_viewTolerance), &opacities);
        if (viewerItems) {
            const std::size_t hitCount = viewerItems->hitRecords.size()
                + filtered.scenePoints.size();
            viewerItems->hitRecords.reserve(hitCount);
            viewerItems->hitIndex.reserve(hitCount);
        }
        for (size_t i = 0; i < filtered.sourceIndices.size(); ++i) {
            entries[filtered.sourceIndices[i]].opacity = opacities[i];
        }
        if (drawSameWrapPolylines && filtered.scenePoints.size() >= 2 && collection.name.rfind("same_wrap", 0) == 0) {
            OverlayStyle lineStyle;
            lineStyle.penColor = toColor(collectionColor, static_cast<float>(viewer->sameWrapAnnotationPolylineOpacity()));
            lineStyle.penWidth = kSameWrapPolylineWidth;
            lineStyle.brushColor = Qt::transparent;
            lineStyle.z = kPolylineZValue;
            addBrokenLineStrips(builder, filtered, polylineBreakDistance(positions), lineStyle);
        }

        for (size_t i = 0; i < filtered.volumePoints.size(); ++i) {
            size_t srcIndex = filtered.sourceIndices.empty() ? i : filtered.sourceIndices[i];
            const auto& entry = entries[srcIndex];
            const QPointF& scenePos = filtered.scenePoints[i];

            qreal radius = kBaseRadius;
            qreal penWidth = kBasePenWidth;
            QColor borderColor(255, 255, 255, 200);

            if (entry.isHighlighted) {
                radius *= kHighlightRadiusMultiplier;
                penWidth = kHighlightPenWidth;
                borderColor = QColor(Qt::yellow);
            }
            if (entry.isSelected) {
                radius *= kSelectedRadiusMultiplier;
                penWidth = kSelectedPenWidth;
                borderColor = QColor(255, 0, 255);
            }

            OverlayStyle style;
            style.penColor = borderColor;
            style.brushColor = toColor(collectionColor, entry.opacity);
            style.penWidth = penWidth;
            style.z = kZValue;
            style.penColor.setAlphaF(entry.opacity);

            builder.addPoint(scenePos, radius, style);

            if (viewerItems && graphicsView) {
                const std::size_t hitIndex = viewerItems->hitRecords.size();
                const QPointF devicePosition =
                    graphicsView->viewportTransform().map(scenePos);
                viewerItems->hitRecords.push_back({
                    vc::PointRef{collectionId, entry.pointId}, scenePos,
                    devicePosition, toColor(collectionColor, 1.0f)});
                viewerItems->hitIndex.insert({
                    devicePosition, collectionId, entry.pointId, hitIndex});
            }

            if (entry.hasLabel) {
                OverlayStyle textStyle;
                QColor textColor = Qt::white;
                textColor.setAlphaF(entry.opacity);
                textStyle.penColor = textColor;
                textStyle.z = kTextZValue;
                builder.addText(scenePos + QPointF(radius, -radius), entry.label, QFont(), textStyle);
            }
        }
    }
}

void PointsOverlayController::connectCollectionSignals()
{
    if (!_collection) {
        return;
    }

    disconnectCollectionSignals();

    _collectionConnections[0] = connect(_collection, &VCCollection::collectionsAdded,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[1] = connect(_collection, &VCCollection::collectionRemoved,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[2] = connect(_collection, &VCCollection::collectionChanged,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[3] = connect(_collection, &VCCollection::pointAdded,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[4] = connect(_collection, &VCCollection::pointChanged,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[5] = connect(_collection, &VCCollection::pointRemoved,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[6] = connect(_collection, &VCCollection::pointsAdded,
                                        this, &PointsOverlayController::handleCollectionMutated);
    _collectionConnections[7] = connect(_collection, &VCCollection::pointsRemoved,
                                        this, &PointsOverlayController::handleCollectionMutated);
}

void PointsOverlayController::disconnectCollectionSignals()
{
    for (auto& connection : _collectionConnections) {
        QObject::disconnect(connection);
        connection = QMetaObject::Connection();
    }
}

void PointsOverlayController::handleCollectionMutated()
{
    // Several collection mutations (notably VCCollection::addPoints) emit both
    // per-point and batch signals, and a batch add fires pointAdded N times plus
    // pointsAdded once. Coalesce the resulting refreshes onto a single deferred
    // call so a burst of signals in one event-loop turn triggers one refreshAll().
    // The ordered-point and surface-projection caches are keyed on a revision
    // that has to move before the deferred rebuild reads them, not when it
    // runs -- a mutation and its rebuild are separated by an event-loop turn.
    // Dropping the projection cache outright also reclaims entries for
    // collections that have been removed; the revision bump would have forced
    // every surviving entry to recompute anyway.
    _orderedCollections.clear();
    clearSurfacePointsCache();
    ++_pointsRevision;
    if (_refreshPending) {
        return;
    }
    _refreshPending = true;
    QTimer::singleShot(0, this, [this]() {
        _refreshPending = false;
        refreshAll();
    });
}
