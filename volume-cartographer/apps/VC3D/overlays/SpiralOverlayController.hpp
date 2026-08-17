#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QImage>
#include <memory>
#include <vector>

#include <opencv2/core/matx.hpp>

class QuadSurface;

class SpiralOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    // A winding-to-winding boundary on the displayed preview surface. The
    // segment points are grid coordinates (column, row) of that surface.
    struct WindingTransitionCurve {
        int fromWinding = 0;
        int toWinding = 0;
        std::vector<std::vector<cv::Vec2f>> segments;
    };

    explicit SpiralOverlayController(QObject* parent = nullptr);
    void publishRunDiff(std::shared_ptr<QuadSurface> surface, QImage image);
    void publishLossMap(std::shared_ptr<QuadSurface> surface, QImage image,
                        qreal opacity);
    void publishWindingTransitions(std::shared_ptr<QuadSurface> surface,
                                   std::vector<WindingTransitionCurve> curves);
    void setRunDiffVisible(bool visible);
    void reset();

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    bool hasRunDiffFor(VolumeViewerBase* viewer) const;
    bool hasTransitionsFor(VolumeViewerBase* viewer) const;

    std::shared_ptr<QuadSurface> _runDiffSurface;
    QImage _runDiffImage;
    bool _runDiffVisible = false;
    std::shared_ptr<QuadSurface> _lossMapSurface;
    QImage _lossMapImage;
    qreal _lossMapOpacity = 0.8;
    std::shared_ptr<QuadSurface> _transitionSurface;
    std::vector<WindingTransitionCurve> _transitionCurves;
};
