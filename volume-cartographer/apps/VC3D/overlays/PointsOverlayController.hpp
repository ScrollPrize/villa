#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QMetaObject>
#include <QColor>
#include <QPointF>
#include <QSet>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "vc/core/PointCollections.hpp"

class VCCollection;

class PointsOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct DisplayPointHit {
        vc::PointRef ref;
        QPointF scenePosition;
        QPointF devicePosition;
        QColor color;
    };

    PointsOverlayController(VCCollection* collection, QObject* parent = nullptr,
                            bool displayOnly = false);
    ~PointsOverlayController() override;

    void setCollection(VCCollection* collection);
    void setViewTolerance(double tolerance);
    [[nodiscard]] double viewTolerance() const { return _viewTolerance; }
    void setCoordinateScale(double scale);
    void setHiddenCollectionIds(const QSet<qulonglong>& ids);
    void setVisible(bool visible);
    // Display-only overlays skip per-point winding labels by default (see
    // collectPrimitives); an overlay whose annotations are the point of the
    // display (relative-winding PCLs) opts back in.
    void setShowWindingLabels(bool show);
    std::optional<DisplayPointHit> displayPointHitAt(
        VolumeViewerBase* viewer, const QPointF& devicePosition, qreal radius,
        const QSet<qulonglong>& allowedCollectionIds) const;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    void applyOverlayPrimitives(VolumeViewerBase* viewer,
                                std::vector<OverlayPrimitive> primitives) override;
    void clearOverlay(VolumeViewerBase* viewer) const override;

private:
    // Retained batch items, so a rebuild refills them instead of deleting and
    // re-creating one scene item per point.
    struct PersistentItems;
    std::unique_ptr<PersistentItems> _persistentItems;

    void connectCollectionSignals();
    void disconnectCollectionSignals();
    void handleCollectionMutated();

    // Ordering a collection means copying every ColPoint out of its hash map
    // and sorting; the overlay is rebuilt on every pan/zoom tick, so the
    // result is kept until the collection actually changes.
    struct OrderedCollection {
        std::vector<ColPoint> points;
        std::vector<cv::Vec3f> scaledPositions;
        double coordinateScale{1.0};
    };
    const OrderedCollection& orderedCollection(
        uint64_t collectionId, const PointCollections::Collection& collection) const;

    VCCollection* _collection{nullptr};
    std::array<QMetaObject::Connection, 8> _collectionConnections{};
    double _viewTolerance{10.0};
    double _coordinateScale{1.0};
    bool _displayOnly{false};
    bool _showWindingLabels{false};
    QSet<qulonglong> _hiddenCollectionIds;
    bool _visible{true};
    bool _refreshPending{false};
    mutable std::unordered_map<uint64_t, OrderedCollection> _orderedCollections;
    // Bumped whenever the ordered points or their scaling change, so the
    // surface-projection cache in the base class knows to recompute.
    std::uint64_t _pointsRevision{1};
};
