#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QImage>
#include <memory>

class QuadSurface;

class SpiralOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    explicit SpiralOverlayController(QObject* parent = nullptr);
    void publishRunDiff(std::shared_ptr<QuadSurface> surface, QImage image);
    void publishLossMap(std::shared_ptr<QuadSurface> surface, QImage image,
                        qreal opacity);
    void setRunDiffVisible(bool visible);
    void reset();

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    bool hasRunDiffFor(VolumeViewerBase* viewer) const;

    std::shared_ptr<QuadSurface> _runDiffSurface;
    QImage _runDiffImage;
    bool _runDiffVisible = false;
    std::shared_ptr<QuadSurface> _lossMapSurface;
    QImage _lossMapImage;
    qreal _lossMapOpacity = 0.8;
};
