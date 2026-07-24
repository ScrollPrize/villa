#pragma once

#include "SpiralPointChain.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <QJsonDocument>
#include <QPainterPath>
#include <QPointF>
#include <QSet>
#include <QString>

#include <opencv2/core/types.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

class QuadSurface;
class VolumeViewerBase;
class SpiralBrushCursorWidget;

// Spiral-only drawn inputs. This deliberately does not use VC3D's annotation
// or segmentation drawing paths: brush marks are true swept-circle vector
// shapes, while control-point lines retain their ordered surface/volume samples.
class SpiralBrushController final : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    struct PreparedPatch {
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> surface;
    };
    struct PreparedPointCollections {
        QString id;
        QString role;
        QJsonDocument document;
    };

    explicit SpiralBrushController(QObject* parent = nullptr);

    void bindFlattenedViewer(VolumeViewerBase* viewer);
    void setPaintSurface(const std::shared_ptr<QuadSurface>& surface);
    void setVisiblePointCollectionIds(const QSet<QString>& ids);
    void resetSession();
    bool hasUnfinalizedPaint() const;
    bool hasUnfinalizedPolylines() const;
    int brushDiameter() const { return _diameterPx; }

    std::vector<PreparedPatch> preparePatches(QStringList& warnings);
    std::vector<PreparedPointCollections> preparePointCollections(QStringList& warnings);
    void finalizationSucceeded(const QString& id);
    void finalizationFailed(const QString& id);
    void discardUnfinalized();

signals:
    void paintStateChanged();
    void brushDiameterChanged(int diameterPx);
    void pointPlacementRejected(const QString& message);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    enum class GestureState { Painted, Finalizing, Finalized };
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
    struct PolylineGesture {
        enum class Kind { Freehand, Anchored, PointCollection };
        QString id;
        QColor color;
        std::shared_ptr<QuadSurface> source;
        std::vector<vc3d::spiral::PointChainAnchor> anchors;
        std::vector<QPointF> surfacePoints;
        std::vector<cv::Vec3f> volumePoints;
        qint64 creationTime = 0;
        int sequence = 0;
        Kind kind = Kind::Freehand;
        GestureState state = GestureState::Painted;
    };
    enum class DragMode { None, Paint, Polyline, Erase };

    QColor nextColor();
    QPainterPath deviceDisk(const QPointF& center) const;
    QPainterPath deviceSweep(const QPointF& from, const QPointF& to) const;
    QPainterPath deviceToSurface(const QPainterPath& path) const;
    QPainterPath surfaceToScene(const QPainterPath& path) const;
    void beginPaint(const QPointF& devicePos);
    void beginPolyline(const QPointF& devicePos);
    void appendAnchoredPoint(const QPointF& devicePos);
    void finishAnchoredPolyline();
    void appendPointCollectionPoint(const QPointF& devicePos);
    void finishPointCollection();
    void beginErase(const QPointF& devicePos);
    void extendDrag(const QPointF& devicePos);
    void finishDrag(const QPointF& devicePos);
    void eraseWith(const QPainterPath& deviceShape);
    void updateCursor(const QPointF& devicePos);
    void updateCursorWidget();
    void sampleColor(const QPointF& scenePos);
    bool appendPolylinePoint(const QPointF& devicePos);
    bool rebuildAnchoredPolyline(PolylineGesture& gesture);
    std::optional<std::pair<QPointF, cv::Vec3f>> pointOnSurface(
        const QPointF& devicePos, const std::shared_ptr<QuadSurface>& source) const;
    std::optional<cv::Vec3f> volumePointOnSurface(
        const QPointF& surfacePos, const std::shared_ptr<QuadSurface>& source) const;
    void resamplePolyline(PolylineGesture& gesture);
    PreparedPatch makePatch(Gesture& gesture) const;

    VolumeViewerBase* _viewer = nullptr;
    std::shared_ptr<QuadSurface> _paintSurface;
    QObject* _viewport = nullptr;
    QObject* _viewObject = nullptr;
    SpiralBrushCursorWidget* _cursorWidget = nullptr;
    std::vector<Gesture> _gestures;
    std::vector<PolylineGesture> _polylines;
    QSet<QString> _visiblePointCollectionIds;
    QSet<QRgb> _usedColors;
    std::optional<QColor> _sampledColor;
    QPointF _cursorDevicePos;
    bool _cursorInside = false;
    QPointF _lastDevicePos;
    DragMode _dragMode = DragMode::None;
    int _activeGesture = -1;
    int _activePolyline = -1;
    int _nextPolylineSequence = 1;
    bool _polylineBlocked = false;
    int _diameterPx = 32;
    bool _gHeld = false;
    bool _shiftHeld = false;
    bool _controlHeld = false;
    bool _vHeld = false;
    bool _vClickConsumed = false;
    bool _qHeld = false;
    bool _qClickConsumed = false;
};
