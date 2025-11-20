#include "PointsOverlayController.hpp"

#include "../CVolumeViewer.hpp"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <QtGlobal>

#include <algorithm>
#include <cmath>

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
constexpr qreal kZValue = 95.0;
constexpr qreal kTextZValue = 96.0;
constexpr float kFadeThreshold = 10.0f;

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

bool PointsOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    return _collection && viewer;
}

void PointsOverlayController::collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder)
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
    const uint64_t selectedCollectionId = viewer->selectedCollectionId();

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

        for (const auto& [pointId, colPoint] : collection.points) {
            Entry entry;
            entry.world = colPoint.p;
            entry.pointId = pointId;
            entry.isHighlighted = pointId == highlightId;
            entry.isSelected = pointId == selectedId;
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
        filter.volumePredicate = [planeSurface, quadSurface, &entries](const cv::Vec3f&, size_t index) {
            auto& entry = entries[index];
            float opacity = 1.0f;
            if (planeSurface) {
                float dist = std::fabs(planeSurface->pointDist(entry.world));
                if (dist >= kFadeThreshold) {
                    opacity = 0.0f;
                } else {
                    opacity = 1.0f - (dist / kFadeThreshold);
                }
            } else if (quadSurface) {
                auto ptr = quadSurface->pointer();
                float dist = quadSurface->pointTo(ptr, entry.world, 10.0, 100);
                if (dist >= kFadeThreshold) {
                    opacity = 0.0f;
                } else if (dist >= 0.0f) {
                    opacity = 1.0f - (dist / kFadeThreshold);
                }
            }
            entry.opacity = opacity;
            return opacity > 0.0f;
        };

        auto filtered = filterPoints(viewer, positions, filter);
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
