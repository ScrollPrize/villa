#pragma once

#include <QObject>
#include <QMetaObject>

#include <QColor>
#include <QFont>
#include <QPointF>
#include <QRectF>
#include <QString>

#include <opencv2/core.hpp>

#include <string>
#include <variant>
#include <vector>

class CVolumeViewer;
class ViewerManager;
class QGraphicsItem;
class QGraphicsScene;
class Surface;

class ViewerOverlayControllerBase : public QObject
{
    Q_OBJECT

public:
    struct OverlayStyle {
        QColor penColor{Qt::white};
        QColor brushColor{Qt::transparent};
        qreal penWidth{0.0};
        Qt::PenStyle penStyle{Qt::SolidLine};
        qreal z{0.0};
    };

    struct PointPrimitive {
        QPointF position;
        qreal radius{3.0};
        OverlayStyle style{};
    };

    struct LineStripPrimitive {
        std::vector<QPointF> points;
        bool closed{false};
        OverlayStyle style{};
    };

    struct RectPrimitive {
        QRectF rect;
        bool filled{true};
        OverlayStyle style{};
    };

    struct TextPrimitive {
        QPointF position;
        QString text;
        QFont font{};
        OverlayStyle style{};
    };

    using OverlayPrimitive = std::variant<PointPrimitive, LineStripPrimitive, RectPrimitive, TextPrimitive>;

    explicit ViewerOverlayControllerBase(std::string overlayGroupKey, QObject* parent = nullptr);
    ~ViewerOverlayControllerBase() override;

    void attachViewer(CVolumeViewer* viewer);
    void detachViewer(CVolumeViewer* viewer);

    void bindToViewerManager(ViewerManager* manager);

    void refreshAll();
    void refreshViewer(CVolumeViewer* viewer);

protected:
    const std::string& overlayGroupKey() const { return _overlayGroupKey; }

    class OverlayBuilder {
    public:
        explicit OverlayBuilder(CVolumeViewer* viewer);

        void addPoint(const QPointF& position,
                      qreal radius,
                      OverlayStyle style);

        void addLineStrip(const std::vector<QPointF>& points,
                          bool closed,
                          OverlayStyle style);

        void addRect(const QRectF& rect,
                     bool filled,
                     OverlayStyle style);

        void addText(const QPointF& position,
                     const QString& text,
                     const QFont& font,
                     OverlayStyle style);

        bool empty() const { return _primitives.empty(); }
        std::vector<OverlayPrimitive> takePrimitives();
        CVolumeViewer* viewer() const { return _viewer; }

    private:
        CVolumeViewer* _viewer{nullptr};
        std::vector<OverlayPrimitive> _primitives;
    };

    virtual bool isOverlayEnabledFor(CVolumeViewer* viewer) const;
    virtual void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) = 0;

    QPointF volumeToScene(CVolumeViewer* viewer, const cv::Vec3f& volumePoint) const;
    cv::Vec3f sceneToVolume(CVolumeViewer* viewer, const QPointF& scenePoint) const;
    std::vector<QPointF> volumeToScene(CVolumeViewer* viewer,
                                       const std::vector<cv::Vec3f>& volumePoints) const;
    QGraphicsScene* viewerScene(CVolumeViewer* viewer) const;
    QRectF visibleSceneRect(CVolumeViewer* viewer) const;
    bool isScenePointVisible(CVolumeViewer* viewer, const QPointF& scenePoint) const;
    Surface* viewerSurface(CVolumeViewer* viewer) const;

    void clearOverlay(CVolumeViewer* viewer) const;

private:
    struct ViewerEntry {
        CVolumeViewer* viewer{nullptr};
        QMetaObject::Connection overlaysUpdatedConn;
        QMetaObject::Connection destroyedConn;
    };

    void rebuildOverlay(CVolumeViewer* viewer);
    void detachAllViewers();

    std::string _overlayGroupKey;
    std::vector<ViewerEntry> _viewers;

    ViewerManager* _manager{nullptr};
    QMetaObject::Connection _managerCreatedConn;
    QMetaObject::Connection _managerDestroyedConn;
};
