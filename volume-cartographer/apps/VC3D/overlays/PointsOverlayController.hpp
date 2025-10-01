#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QMetaObject>

#include <array>

class VCCollection;

class PointsOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    PointsOverlayController(VCCollection* collection, QObject* parent = nullptr);
    ~PointsOverlayController() override;

    void setCollection(VCCollection* collection);

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;

private:
    void connectCollectionSignals();
    void disconnectCollectionSignals();
    void handleCollectionMutated();

    VCCollection* _collection{nullptr};
    std::array<QMetaObject::Connection, 6> _collectionConnections{};
};
