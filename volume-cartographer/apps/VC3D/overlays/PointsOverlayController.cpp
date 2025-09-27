#include "PointsOverlayController.hpp"

#include "../CVolumeViewer.hpp"

#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"

#include <QtGlobal>

#include <algorithm>
#include <cmath>

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

    Surface* surface = viewerSurface(viewer);
    auto* planeSurface = dynamic_cast<PlaneSurface*>(surface);
    auto* quadSurface = dynamic_cast<QuadSurface*>(surface);

    for (const auto& [collectionId, collection] : collections) {
        const cv::Vec3f collectionColor = collection.color;
        const bool absoluteWinding = collection.metadata.absolute_winding_number;

        for (const auto& [pointId, colPoint] : collection.points) {
            QPointF scenePos = volumeToScene(viewer, colPoint.p);

            if (!isScenePointVisible(viewer, scenePos)) {
                continue;
            }

            float opacity = 1.0f;
            if (planeSurface) {
                float dist = std::abs(planeSurface->pointDist(colPoint.p));
                if (dist >= 0.0f) {
                    if (dist < kFadeThreshold) {
                        opacity = 1.0f - (dist / kFadeThreshold);
                    } else {
                        opacity = 0.0f;
                    }
                }
            } else if (quadSurface) {
                auto ptr = quadSurface->pointer();
                float dist = quadSurface->pointTo(ptr, colPoint.p, 10.0, 100);
                if (dist >= 0.0f) {
                    if (dist < kFadeThreshold) {
                        opacity = 1.0f - (dist / kFadeThreshold);
                    } else {
                        opacity = 0.0f;
                    }
                }
            }

            if (opacity <= 0.0f) {
                continue;
            }

            qreal radius = kBaseRadius;
            qreal penWidth = kBasePenWidth;
            QColor borderColor(255, 255, 255, 200);

            const bool isHighlighted = pointId == highlightId;
            const bool isSelected = pointId == selectedId;

            if (isHighlighted) {
                radius *= kHighlightRadiusMultiplier;
                penWidth = kHighlightPenWidth;
                borderColor = QColor(Qt::yellow);
            }
            if (isSelected) {
                radius *= kSelectedRadiusMultiplier;
                penWidth = kSelectedPenWidth;
                borderColor = QColor(255, 0, 255); // Magenta
            }

            OverlayStyle style;
            style.penColor = borderColor;
            style.brushColor = toColor(collectionColor, opacity);
            style.penWidth = penWidth;
            style.z = kZValue;
            style.penColor.setAlphaF(opacity);

            builder.addPoint(scenePos, radius, style);

            if (!std::isnan(colPoint.winding_annotation)) {
                const QString text = formatWinding(colPoint.winding_annotation, absoluteWinding);
                if (!text.isEmpty()) {
                    OverlayStyle textStyle;
                    QColor textColor = Qt::white;
                    textColor.setAlphaF(opacity);
                    textStyle.penColor = textColor;
                    textStyle.z = kTextZValue;

                    builder.addText(scenePos + QPointF(radius, -radius), text, QFont(), textStyle);
                }
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

    _collectionConnections[0] = connect(_collection, &VCCollection::collectionAdded,
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
