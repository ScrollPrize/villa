#pragma once

#include <QObject>
#include <QMetaObject>
#include <QMetaType>

#include <QColor>
#include <QFont>
#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QString>

#include <opencv2/core.hpp>

#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

class CVolumeViewer;
class ViewerManager;
class QGraphicsItem;
class QGraphicsScene;
class Surface;
class QPainterPath;

class ViewerOverlayControllerBase : public QObject
{
    Q_OBJECT

public:
    struct OverlayStyle {
        QColor penColor{Qt::white};
        QColor brushColor{Qt::transparent};
        qreal penWidth{0.0};
        Qt::PenStyle penStyle{Qt::SolidLine};
        Qt::PenCapStyle penCap{Qt::RoundCap};
        Qt::PenJoinStyle penJoin{Qt::RoundJoin};
        std::vector<qreal> dashPattern{};
        qreal z{0.0};
    };

    struct PointPrimitive {
        QPointF position;
        qreal radius{3.0};
        OverlayStyle style{};
    };

    struct CirclePrimitive {
        QPointF center;
        qreal radius{3.0};
        bool filled{true};
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

    enum class PathRenderMode {
        LineStrip,
        Points
    };

    enum class PathBrushShape {
        Circle,
        Square
    };

    struct PathPrimitive {
        std::vector<cv::Vec3f> points;
        QColor color{Qt::white};
        qreal lineWidth{3.0};
        qreal opacity{1.0};
        bool isEraser{false};
        int pathId{0};
        PathBrushShape brushShape{PathBrushShape::Circle};
        PathRenderMode renderMode{PathRenderMode::LineStrip};
        qreal pointRadius{3.0};
        bool closed{false};
        qreal z{25.0};

        PathPrimitive densify(float samplingInterval = 0.5f) const;

    private:
        float interpolateZ(float percent, float totalLength, const QPainterPath& path) const;
    };

    struct ArrowPrimitive {
        QPointF start;
        QPointF end;
        qreal headLength{10.0};
        qreal headWidth{6.0};
        OverlayStyle style{};
    };

    struct ImagePrimitive {
        QImage image;
        QPointF offset{0.0, 0.0};  // Scene-space offset (like setOffset)
        qreal scale{1.0};           // Uniform scale factor (like setScale)
        qreal opacity{1.0};
        qreal z{0.0};
    };

    using OverlayPrimitive = std::variant<PointPrimitive,
                                          CirclePrimitive,
                                          LineStripPrimitive,
                                          RectPrimitive,
                                          TextPrimitive,
                                          PathPrimitive,
                                          ArrowPrimitive,
                                          ImagePrimitive>;

    struct FilteredPoints {
        std::vector<cv::Vec3f> volumePoints;
        std::vector<QPointF> scenePoints;
        std::vector<size_t> sourceIndices;
    };

    struct PointFilterOptions {
        bool clipToSurface{false};
        float planeDistanceTolerance{std::numeric_limits<float>::infinity()};
        float quadDistanceTolerance{std::numeric_limits<float>::infinity()};
        bool requireSceneVisibility{false};
        std::optional<QRectF> customSceneRect;
        bool computeScenePoints{true};
        std::function<bool(const cv::Vec3f&, size_t)> volumePredicate;
        std::function<bool(const QPointF&, size_t)> scenePredicate;
    };

    explicit ViewerOverlayControllerBase(std::string overlayGroupKey, QObject* parent = nullptr);
    ~ViewerOverlayControllerBase() override;

    void attachViewer(CVolumeViewer* viewer);
    void detachViewer(CVolumeViewer* viewer);

    void bindToViewerManager(ViewerManager* manager);

    void refreshAll();
    void refreshViewer(CVolumeViewer* viewer);
    static void applyPrimitives(CVolumeViewer* viewer,
                                const std::string& overlayKey,
                                std::vector<OverlayPrimitive> primitives);

protected:
    const std::string& overlayGroupKey() const { return _overlayGroupKey; }

    class OverlayBuilder {
    public:
        explicit OverlayBuilder(CVolumeViewer* viewer);

        void addPoint(const QPointF& position,
                      qreal radius,
                      OverlayStyle style);

        void addCircle(const QPointF& center,
                       qreal radius,
                       bool filled,
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

        void addPath(const PathPrimitive& path);

        void addArrow(const QPointF& start,
                      const QPointF& end,
                      qreal headLength,
                      qreal headWidth,
                      OverlayStyle style);

        void addImage(const QImage& image,
                      const QPointF& offset,
                      qreal scale,
                      qreal opacity,
                      qreal z);

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

    FilteredPoints filterPoints(CVolumeViewer* viewer,
                                const std::vector<cv::Vec3f>& points,
                                const PointFilterOptions& options) const;

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

Q_DECLARE_METATYPE(ViewerOverlayControllerBase::PathPrimitive)
