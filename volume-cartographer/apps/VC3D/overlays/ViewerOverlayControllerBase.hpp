#pragma once

#include <QObject>
#include <QMetaObject>
#include <QMetaType>
#include <QTimer>

#include <QColor>
#include <QFont>
#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QTransform>

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

class VolumeViewerBase;
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

    struct VolumePointPrimitive {
        cv::Vec3f position{0.0f, 0.0f, 0.0f};
        qreal radius{3.0};
        OverlayStyle style{};
    };

    struct SurfacePointPrimitive {
        cv::Vec2f position{0.0f, 0.0f};
        qreal radius{3.0};
        OverlayStyle style{};
    };

    struct CirclePrimitive {
        QPointF center;
        qreal radius{3.0};
        bool filled{true};
        OverlayStyle style{};
    };

    struct RotatedEllipsePrimitive {
        QPointF center;
        qreal radiusX{3.0};
        qreal radiusY{2.0};
        qreal rotationRadians{0.0};
        bool filled{true};
        OverlayStyle style{};
    };

    struct LineStripPrimitive {
        std::vector<QPointF> points;
        bool closed{false};
        OverlayStyle style{};
    };

    struct SurfaceLineStripPrimitive {
        std::vector<cv::Vec2f> points;
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
        bool outlined{false};
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
        QTransform transform{};
        qreal opacity{1.0};
        qreal z{0.0};
    };

    using OverlayPrimitive = std::variant<PointPrimitive,
                                          VolumePointPrimitive,
                                          SurfacePointPrimitive,
                                          CirclePrimitive,
                                          RotatedEllipsePrimitive,
                                          LineStripPrimitive,
                                          SurfaceLineStripPrimitive,
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

    // Point-collection-style rendering of an ordered chain of volume points:
    // white-bordered dots connected by polylines that only ever join
    // consecutive chain points, fading out with distance to the viewer's
    // surface.
    struct PointChainStyle {
        QColor color{Qt::white};
        QColor pointBorderColor{255, 255, 255, 200};
        qreal pointRadius{5.0};
        qreal pointPenWidth{1.5};
        qreal lineWidth{5.0};
        float lineOpacity{1.0f};
        qreal pointZ{95.0};
        qreal lineZ{94.0};
        float distanceTolerance{std::numeric_limits<float>::infinity()};
        // Break a polyline segment whose scene-space length exceeds this many
        // multiples of its volume-space length times the viewer scale. Guards
        // against projection discontinuities (invalid surface regions, wrap
        // jumps) where volume-close points land far apart on screen. <= 0
        // disables the guard.
        float sceneJumpRatio{4.0f};
        bool drawPoints{true};
        bool drawLines{true};
    };

    // Inclusive volume-space AABB used to cheaply pre-cull points before any
    // per-point surface-distance search runs.
    struct VolumeBounds {
        cv::Vec3f lo{0, 0, 0};
        cv::Vec3f hi{0, 0, 0};
        bool contains(const cv::Vec3f& point) const
        {
            return point[0] >= lo[0] && point[0] <= hi[0]
                && point[1] >= lo[1] && point[1] <= hi[1]
                && point[2] >= lo[2] && point[2] <= hi[2];
        }
    };

    explicit ViewerOverlayControllerBase(std::string overlayGroupKey, QObject* parent = nullptr);
    ~ViewerOverlayControllerBase() override;

    void attachViewer(VolumeViewerBase* viewer);
    virtual void detachViewer(VolumeViewerBase* viewer);

    void bindToViewerManager(ViewerManager* manager);

    void refreshAll();
    void refreshViewer(VolumeViewerBase* viewer);
    static void applyPrimitives(VolumeViewerBase* viewer,
                                const std::string& overlayKey,
                                std::vector<OverlayPrimitive> primitives);

protected:
    const std::string& overlayGroupKey() const { return _overlayGroupKey; }

    class OverlayBuilder {
    public:
        explicit OverlayBuilder(VolumeViewerBase* viewer);

        void addPoint(const QPointF& position,
                      qreal radius,
                      OverlayStyle style);

        void addSurfacePoint(const cv::Vec2f& position,
                             qreal radius,
                             OverlayStyle style);

        void addCircle(const QPointF& center,
                       qreal radius,
                       bool filled,
                       OverlayStyle style);

        void addRotatedEllipse(const QPointF& center,
                               qreal radiusX,
                               qreal radiusY,
                               qreal rotationRadians,
                               bool filled,
                               OverlayStyle style);

        void addLineStrip(const std::vector<QPointF>& points,
                          bool closed,
                          OverlayStyle style);

        void addSurfaceLineStrip(const std::vector<cv::Vec2f>& points,
                                 bool closed,
                                 OverlayStyle style);

        void addRect(const QRectF& rect,
                     bool filled,
                     OverlayStyle style);

        void addText(const QPointF& position,
                     const QString& text,
                     const QFont& font,
                     OverlayStyle style,
                     bool outlined = false);

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

        void addImage(const QImage& image,
                      const QPointF& offset,
                      qreal scaleX,
                      qreal scaleY,
                      qreal opacity,
                      qreal z);

        void addImage(const QImage& image,
                      const QTransform& transform,
                      qreal opacity,
                      qreal z);

        bool empty() const { return _primitives.empty(); }
        std::vector<OverlayPrimitive> takePrimitives();
        VolumeViewerBase* viewer() const { return _viewer; }

    private:
        VolumeViewerBase* _viewer{nullptr};
        std::vector<OverlayPrimitive> _primitives;
    };

    virtual bool isOverlayEnabledFor(VolumeViewerBase* viewer) const;
    virtual void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) = 0;

    QPointF volumeToScene(VolumeViewerBase* viewer, const cv::Vec3f& volumePoint) const;
    cv::Vec3f sceneToVolume(VolumeViewerBase* viewer, const QPointF& scenePoint) const;
    std::vector<QPointF> volumeToScene(VolumeViewerBase* viewer,
                                       const std::vector<cv::Vec3f>& volumePoints) const;
    QGraphicsScene* viewerScene(VolumeViewerBase* viewer) const;
    QRectF visibleSceneRect(VolumeViewerBase* viewer) const;
    bool isScenePointVisible(VolumeViewerBase* viewer, const QPointF& scenePoint) const;
    Surface* viewerSurface(VolumeViewerBase* viewer) const;

    FilteredPoints filterPoints(VolumeViewerBase* viewer,
                                const std::vector<cv::Vec3f>& points,
                                const PointFilterOptions& options) const;

    // Keeps points within `tolerance` of the viewer's plane or quad surface,
    // fading opacity linearly with distance (binary at the surface when
    // tolerance <= 0). `opacities`, when given, is filled aligned with the
    // returned points. `bounds`, when given, rejects points outside the box
    // before the (potentially expensive) surface-distance search.
    FilteredPoints filterPointsNearViewerSurface(VolumeViewerBase* viewer,
                                                 const std::vector<cv::Vec3f>& points,
                                                 float tolerance,
                                                 std::vector<float>* opacities = nullptr,
                                                 const std::optional<VolumeBounds>& bounds = std::nullopt,
                                                 bool requireSceneVisibility = true) const;

    // Longest volume-space distance two consecutive chain points may span and
    // still be joined by a polyline: 4x the median inter-point distance, or
    // infinity when the chain is too short to yield a robust median.
    static float polylineBreakDistance(const std::vector<cv::Vec3f>& positions);

    // Emits line strips joining consecutive filtered points, breaking the
    // strip whenever a source point was filtered out in between, the
    // volume-space gap exceeds maxSegmentDistance, or the scene-space gap
    // exceeds maxScenePerVolume times the volume-space gap (a projection
    // discontinuity).
    static void addBrokenLineStrips(OverlayBuilder& builder,
                                    const FilteredPoints& filtered,
                                    float maxSegmentDistance,
                                    const OverlayStyle& style,
                                    qreal maxScenePerVolume = std::numeric_limits<qreal>::infinity());

    void renderPointChain(VolumeViewerBase* viewer,
                          OverlayBuilder& builder,
                          const std::vector<cv::Vec3f>& points,
                          const PointChainStyle& style,
                          const std::optional<VolumeBounds>& bounds = std::nullopt) const;

    // Invalidates the surface-coordinate projections retained by
    // renderPointChain(). Controllers must call this whenever their source
    // chain storage is replaced or mutated.
    void clearPointChainProjectionCache();

    // Projects a chain onto the viewer's surface with per-point opacity;
    // cached — see clearPointChainProjectionCache().
    FilteredPoints projectedPointChain(VolumeViewerBase* viewer,
                                       const std::vector<cv::Vec3f>& points,
                                       float tolerance,
                                       std::vector<float>* opacities) const;

    // The default implementation materializes primitives as QGraphicsItems.
    // High-volume overlays may override this to retain their graphics items
    // across refreshes while continuing to use the common primitive builder.
    virtual void applyOverlayPrimitives(VolumeViewerBase* viewer,
                                        std::vector<OverlayPrimitive> primitives);
    virtual void clearOverlay(VolumeViewerBase* viewer) const;

    ViewerManager* manager() const { return _manager; }

private:
    struct PointChainProjectionCacheKey {
        VolumeViewerBase* viewer{nullptr};
        const cv::Vec3f* pointsData{nullptr};
        std::size_t pointCount{0};

        friend bool operator==(const PointChainProjectionCacheKey& lhs,
                               const PointChainProjectionCacheKey& rhs)
        {
            return lhs.viewer == rhs.viewer &&
                   lhs.pointsData == rhs.pointsData &&
                   lhs.pointCount == rhs.pointCount;
        }
    };

    struct PointChainProjectionCacheKeyHash {
        std::size_t operator()(const PointChainProjectionCacheKey& key) const;
    };

    struct PointChainProjectionCacheEntry {
        Surface* surface{nullptr};
        std::uint64_t surfaceGeneration{0};
        float tolerance{0.0f};
        cv::Vec3f planeOrigin{0.0f, 0.0f, 0.0f};
        cv::Vec3f planeBasisX{0.0f, 0.0f, 0.0f};
        cv::Vec3f planeBasisY{0.0f, 0.0f, 0.0f};
        std::vector<cv::Vec3f> volumePoints;
        std::vector<cv::Vec2f> surfacePoints;
        std::vector<std::size_t> sourceIndices;
        std::vector<float> opacities;
    };

    void clearPointChainProjectionCache(VolumeViewerBase* viewer);

    struct ViewerEntry {
        VolumeViewerBase* viewer{nullptr};
        QMetaObject::Connection overlaysUpdatedConn;
        QMetaObject::Connection destroyedConn;
        // Coalesce the rebuildOverlay fan-out onto a 16 ms single-shot
        // timer. overlaysUpdated() gets emitted on every viewport pan/zoom
        // (post-render side-effect), and collectDirectionHints' pointTo
        // search was measured at ~7.5% of total CPU in the live profile —
        // debouncing drops that to one rebuild per tick window regardless
        // of signal frequency.
        QTimer* rebuildTimer{nullptr};
        bool rebuildDirty{false};
    };

    void rebuildOverlay(VolumeViewerBase* viewer);
    void scheduleRebuild(VolumeViewerBase* viewer);
    void detachAllViewers();

    std::string _overlayGroupKey;
    std::vector<ViewerEntry> _viewers;

    ViewerManager* _manager{nullptr};
    mutable std::unordered_map<PointChainProjectionCacheKey,
                               PointChainProjectionCacheEntry,
                               PointChainProjectionCacheKeyHash>
        _pointChainProjectionCache;
    QMetaObject::Connection _managerCreatedConn;
    QMetaObject::Connection _managerClosingConn;
    QMetaObject::Connection _managerDestroyedConn;
};

Q_DECLARE_METATYPE(ViewerOverlayControllerBase::PathPrimitive)
