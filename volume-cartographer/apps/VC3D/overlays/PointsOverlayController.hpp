#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QMetaObject>

#include <array>

class VCCollection;

class PointsOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    PointsOverlayController(VCCollection* collection, QObject* parent = nullptr,
                            bool displayOnly = false);
    ~PointsOverlayController() override;

    void setCollection(VCCollection* collection);
    void setViewTolerance(double tolerance);
    [[nodiscard]] double viewTolerance() const { return _viewTolerance; }
    void setCoordinateScale(double scale);
    void setVisible(bool visible);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    void connectCollectionSignals();
    void disconnectCollectionSignals();
    void handleCollectionMutated();

    VCCollection* _collection{nullptr};
    std::array<QMetaObject::Connection, 8> _collectionConnections{};
    double _viewTolerance{10.0};
    double _coordinateScale{1.0};
    bool _displayOnly{false};
    bool _visible{true};
    bool _refreshPending{false};
};
