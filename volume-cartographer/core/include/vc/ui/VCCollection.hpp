#pragma once

#include <QObject>
#include <QMetaType>
#include "vc/core/PointCollections.hpp"

Q_DECLARE_METATYPE(vc::PointRef)

// Thin Qt shim over the Qt-free PointCollections: turns base change hooks
// into signals. Data/IO all live in PointCollections.
class VCCollection : public QObject, public PointCollections
{
    Q_OBJECT

public:
    using PointCollections::Collection;
    using PointCollections::WindingFillMode;

    explicit VCCollection(QObject* parent = nullptr) : QObject(parent) {}

signals:
    void collectionChanged(uint64_t collectionId); // Generic signal for name/metadata changes
    void collectionsAdded(const std::vector<uint64_t>& collectionIds);
    void collectionRemoved(uint64_t collectionId);

    void pointAdded(const ColPoint& point);
    void pointsAdded(const std::vector<ColPoint>& points);
    void pointChanged(const ColPoint& point);
    void pointRemoved(vc::PointRef point);
    void pointsRemoved(const std::vector<vc::PointRef>& points);

protected:
    void onCollectionChanged(uint64_t id) override { emit collectionChanged(id); }
    void onCollectionsAdded(const std::vector<uint64_t>& ids) override { emit collectionsAdded(ids); }
    void onCollectionRemoved(uint64_t id) override { emit collectionRemoved(id); }
    void onPointAdded(const ColPoint& p) override { emit pointAdded(p); }
    void onPointsAdded(const std::vector<ColPoint>& points) override { emit pointsAdded(points); }
    void onPointChanged(const ColPoint& p) override { emit pointChanged(p); }
    void onPointRemoved(vc::PointRef point) override { emit pointRemoved(point); }
    void onPointsRemoved(const std::vector<vc::PointRef>& points) override { emit pointsRemoved(points); }
};
