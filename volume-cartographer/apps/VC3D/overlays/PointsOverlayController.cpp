#include "PointsOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../ViewerManager.hpp"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "vc/core/util/QuadSurface.hpp"

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

float sameWrapPolylineBreakDistance(const std::vector<cv::Vec3f>& positions)
{
    if (positions.size() < 4) {
        return std::numeric_limits<float>::infinity();
    }

    std::vector<float> distances;
    distances.reserve(positions.size() - 1);
    for (size_t i = 1; i < positions.size(); ++i) {
        const float distance = cv::norm(positions[i] - positions[i - 1]);
        if (distance > 1e-4f) {
            distances.push_back(distance);
        }
    }
    if (distances.size() < 3) {
        return std::numeric_limits<float>::infinity();
    }

    auto medianIt = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), medianIt, distances.end());
    const float medianDistance = *medianIt;
    if (medianDistance <= 1e-4f) {
        return std::numeric_limits<float>::infinity();
    }
    return medianDistance * 4.0f;
}

} // namespace

PointsOverlayController::PointsOverlayController(VCCollection* collection, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupPoints, parent)
    , _collection(collection)
{
    connectCollectionSignals();
}

PointsOverlayController::~PointsOverlayController()
{
    disconnectCollectionSignals();
}

void PointsOverlayController::setCollection(VCCollection* collection)
{
    if (_collection == collection) {
        return;
    }
    disconnectCollectionSignals();
    _collection = collection;
    connectCollectionSignals();
    refreshAll();
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

bool PointsOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return _collection && viewer;
}

void PointsOverlayController::collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder)
{
    if (!_collection || !viewer) {
        return;
    }

    if (viewer->pointCollection() != _collection) {
        return;
    }

    const auto& collections = _collection->getAllCollections();
    if (collections.empty()) {
        return;
    }

    const uint64_t highlightId = viewer->highlightedPointId();
    const uint64_t selectedId = viewer->selectedPointId();
    const bool drawSameWrapPolylines = viewer->isSameWrapAnnotationModeEnabled();

    for (const auto& [collectionId, collection] : collections) {
        const cv::Vec3f collectionColor = collection.color;
        const bool absoluteWinding = collection.metadata.absolute_winding_number;
        struct Entry {
            cv::Vec3f world;
            uint64_t pointId;
            float opacity{1.0f};
            bool isHighlighted{false};
            bool isSelected{false};
            bool hasLabel{false};
            QString label;
        };

        std::vector<cv::Vec3f> positions;
        std::vector<Entry> entries;
        positions.reserve(collection.points.size());
        entries.reserve(collection.points.size());

        Surface* surface = viewerSurface(viewer);
        auto* planeSurface = dynamic_cast<PlaneSurface*>(surface);
        auto* quadSurface = dynamic_cast<QuadSurface*>(surface);

        for (const ColPoint& colPoint : orderedCollectionPoints(collection)) {
            Entry entry;
            entry.world = colPoint.p;
            entry.pointId = colPoint.id;
            entry.isHighlighted = colPoint.id == highlightId;
            entry.isSelected = colPoint.id == selectedId;
            if (!std::isnan(colPoint.winding_annotation)) {
                const QString text = formatWinding(colPoint.winding_annotation, absoluteWinding);
                entry.hasLabel = !text.isEmpty();
                entry.label = text;
            }

            positions.push_back(entry.world);
            entries.push_back(std::move(entry));
        }

        PointFilterOptions filter;
        filter.clipToSurface = false;
        filter.requireSceneVisibility = true;
        filter.computeScenePoints = true;
        auto* patchIndex = manager() ? manager()->surfacePatchIndex() : nullptr;
        const float viewTolerance = static_cast<float>(_viewTolerance);
        filter.volumePredicate = [planeSurface, quadSurface, patchIndex, &entries, viewTolerance](const cv::Vec3f&, size_t index) {
            auto& entry = entries[index];
            float opacity = 1.0f;
            auto opacityForDistance = [viewTolerance](float dist) {
                if (dist < 0.0f) {
                    return 0.0f;
                }
                if (viewTolerance <= 0.0f) {
                    return dist <= 0.0f ? 1.0f : 0.0f;
                }
                return dist >= viewTolerance ? 0.0f : 1.0f - (dist / viewTolerance);
            };
            if (planeSurface) {
                opacity = opacityForDistance(std::fabs(planeSurface->pointDist(entry.world)));
            } else if (quadSurface) {
                cv::Vec3f ptr(0, 0, 0);
                float dist = quadSurface->pointTo(ptr, entry.world, std::max(viewTolerance, 0.0f), 100, patchIndex);
                opacity = opacityForDistance(dist);
            }
            entry.opacity = opacity;
            return opacity > 0.0f;
        };

        auto filtered = filterPoints(viewer, positions, filter);
        if (drawSameWrapPolylines && filtered.scenePoints.size() >= 2 && collection.name.rfind("same_wrap", 0) == 0) {
            const float maxSegmentDistance = sameWrapPolylineBreakDistance(positions);
            OverlayStyle lineStyle;
            lineStyle.penColor = toColor(collectionColor, static_cast<float>(viewer->sameWrapAnnotationPolylineOpacity()));
            lineStyle.penWidth = kSameWrapPolylineWidth;
            lineStyle.brushColor = Qt::transparent;
            lineStyle.z = kPolylineZValue;

            std::vector<QPointF> lineStrip;
            lineStrip.reserve(filtered.scenePoints.size());
            size_t previousSourceIndex = 0;
            bool hasPreviousSourceIndex = false;
            auto flushLineStrip = [&]() {
                if (lineStrip.size() >= 2) {
                    builder.addLineStrip(lineStrip, false, lineStyle);
                }
                lineStrip.clear();
            };
            for (size_t i = 0; i < filtered.scenePoints.size(); ++i) {
                const size_t sourceIndex = filtered.sourceIndices.empty() ? i : filtered.sourceIndices[i];
                if (hasPreviousSourceIndex && sourceIndex != previousSourceIndex + 1) {
                    flushLineStrip();
                } else if (hasPreviousSourceIndex && i < filtered.volumePoints.size()) {
                    const float segmentDistance =
                        cv::norm(filtered.volumePoints[i] - filtered.volumePoints[i - 1]);
                    if (segmentDistance > maxSegmentDistance) {
                        flushLineStrip();
                    }
                }
                lineStrip.push_back(filtered.scenePoints[i]);
                previousSourceIndex = sourceIndex;
                hasPreviousSourceIndex = true;
            }
            flushLineStrip();
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
    refreshAll();
}
