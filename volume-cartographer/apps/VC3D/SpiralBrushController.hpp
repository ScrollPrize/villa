#pragma once

#include "overlays/ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <QPainterPath>
#include <QPointF>
#include <QSet>
#include <QString>

#include <memory>
#include <optional>
#include <vector>

class QuadSurface;
class VolumeViewerBase;
class SpiralBrushCursorWidget;

// Spiral-only surface paint. This deliberately does not use VC3D's annotation
// or segmentation drawing paths: every mark is an antialiased filled vector
// shape whose primitive is a true swept circle.
class SpiralBrushController final : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    struct PreparedPatch {
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> surface;
    };

    explicit SpiralBrushController(QObject* parent = nullptr);

    void bindFlattenedViewer(VolumeViewerBase* viewer);
    void setPaintSurface(const std::shared_ptr<QuadSurface>& surface) { _paintSurface = surface; }
    void resetSession();
    bool hasUnfinalizedPaint() const;
    int brushDiameter() const { return _diameterPx; }

    std::vector<PreparedPatch> preparePatches(QStringList& warnings);
    void finalizationSucceeded(const QString& id);
    void finalizationFailed(const QString& id);
    void discardUnfinalized();

signals:
    void paintStateChanged();
    void brushDiameterChanged(int diameterPx);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    enum class GestureState { Painted, Finalizing };
    struct Gesture {
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> source;
        QPainterPath shape;
        QPointF gridOrigin;
        QPointF columnStep;
        QPointF rowStep;
        GestureState state = GestureState::Painted;
    };
    enum class DragMode { None, Paint, Erase };

    QColor nextColor();
    QPainterPath deviceDisk(const QPointF& center) const;
    QPainterPath deviceSweep(const QPointF& from, const QPointF& to) const;
    QPainterPath deviceToSurface(const QPainterPath& path) const;
    QPainterPath surfaceToScene(const QPainterPath& path) const;
    void beginPaint(const QPointF& devicePos);
    void beginErase(const QPointF& devicePos);
    void extendDrag(const QPointF& devicePos);
    void finishDrag(const QPointF& devicePos);
    void eraseWith(const QPainterPath& surfaceShape);
    void updateCursor(const QPointF& devicePos);
    void updateCursorWidget();
    void sampleColor(const QPointF& scenePos);
    PreparedPatch makePatch(Gesture& gesture) const;

    VolumeViewerBase* _viewer = nullptr;
    std::shared_ptr<QuadSurface> _paintSurface;
    QObject* _viewport = nullptr;
    QObject* _viewObject = nullptr;
    SpiralBrushCursorWidget* _cursorWidget = nullptr;
    std::vector<Gesture> _gestures;
    QSet<QRgb> _usedColors;
    std::optional<QColor> _sampledColor;
    QPointF _cursorDevicePos;
    bool _cursorInside = false;
    QPointF _lastDevicePos;
    DragMode _dragMode = DragMode::None;
    int _activeGesture = -1;
    int _diameterPx = 32;
    bool _gHeld = false;
    bool _shiftHeld = false;
};
