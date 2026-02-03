#pragma once

#include <opencv2/core.hpp>
#include <QElapsedTimer>
#include <QPointF>
#include <unordered_set>
#include <vector>

#include "vc/core/util/PointIndex.hpp"

class QuadSurface;
class SegmentationModule;
class SegmentationWidget;
class SegmentationEditManager;

/**
 * Brush tool for painting approval/unapproved regions on segmentation surfaces.
 *
 * This tool allows users to interactively mark portions of a surface as approved
 * or unapproved by brushing in any viewer. The approval mask is stored as a channel
 * on the QuadSurface and persists through segment growth operations.
 */
class ApprovalMaskBrushTool
{
public:
    enum class PaintMode {
        Approve,    // Paint approval (value 255)
        Unapprove   // Paint unapproved (value 0)
    };

    ApprovalMaskBrushTool(SegmentationModule& module,
                          SegmentationEditManager* editManager,
                          SegmentationWidget* widget);

    void setDependencies(SegmentationWidget* widget);
    void setSurface(QuadSurface* surface);
    void setPaintMode(PaintMode mode) { _paintMode = mode; }
    [[nodiscard]] PaintMode paintMode() const { return _paintMode; }

    void setActive(bool active);
    [[nodiscard]] bool brushActive() const { return _brushActive; }
    [[nodiscard]] bool strokeActive() const { return _strokeActive; }
    [[nodiscard]] bool hasPendingStrokes() const { return !_pendingStrokes.empty(); }

    // scenePos is the raw scene position, viewerScale is the viewer's current scale (_scale member)
    // Grid position is computed as: (scenePos / viewerScale + surface_center) * surface_scale
    void startStroke(const cv::Vec3f& worldPos, const QPointF& scenePos, float viewerScale);
    void extendStroke(const cv::Vec3f& worldPos, const QPointF& scenePos, float viewerScale, bool forceSample);
    void finishStroke();

    // For plane viewers: paint approval based on 3D world position, plane normal, and radius
    // Finds all grid cells within a cylinder: within `radius` of the plane AND within `radius`
    // (in 2D plane space) of the mouse position
    void startStrokeFromPlane(const cv::Vec3f& worldPos, const cv::Vec3f& planeNormal, float worldRadius);
    void extendStrokeFromPlane(const cv::Vec3f& worldPos, const cv::Vec3f& planeNormal, float worldRadius, bool forceSample);
    void finishStrokeFromPlane();

    // Legacy methods (for compatibility) - delegate to plane methods with zero normal (sphere mode)
    void startStrokeFromWorld(const cv::Vec3f& worldPos, float worldRadius);
    void extendStrokeFromWorld(const cv::Vec3f& worldPos, float worldRadius, bool forceSample);
    void finishStrokeFromWorld();
    bool applyPending(float dragRadiusSteps);
    void clear();

    [[nodiscard]] const std::vector<cv::Vec3f>& overlayPoints() const { return _overlayPoints; }
    [[nodiscard]] const std::vector<std::vector<cv::Vec3f>>& overlayStrokeSegments() const { return _overlayStrokeSegments; }
    [[nodiscard]] const std::vector<cv::Vec3f>& currentStrokePoints() const { return _currentStroke; }
    [[nodiscard]] float effectivePaintRadius() const { return _effectivePaintRadiusNative; }
    [[nodiscard]] std::optional<cv::Vec3f> hoverWorldPos() const { return _hoverWorldPos; }
    [[nodiscard]] float hoverEffectiveRadius() const { return _hoverEffectiveRadius; }
    [[nodiscard]] std::optional<QPointF> hoverScenePos() const { return _hoverScenePos; }
    [[nodiscard]] float hoverViewerScale() const { return _hoverViewerScale; }
    [[nodiscard]] std::optional<cv::Vec3f> hoverPlaneNormal() const { return _hoverPlaneNormal; }

    void setHoverWorldPos(const cv::Vec3f& pos, float brushRadius, const QPointF& scenePos, float viewerScale,
                          const std::optional<cv::Vec3f>& planeNormal = std::nullopt);
    void clearHoverWorldPos() { _hoverWorldPos = std::nullopt; _hoverEffectiveRadius = 0.0f; _hoverScenePos = std::nullopt; _hoverViewerScale = 0.0f; _hoverPlaneNormal = std::nullopt; }

    void cancel() { clear(); }
    [[nodiscard]] bool isActive() const { return brushActive() || strokeActive(); }

private:
    // Convert scene position to integer grid indices using surface coordinate transform
    // Formula: gridPos = (scenePos / viewerScale + center) * surfaceScale
    std::optional<std::pair<int, int>> sceneToGridIndex(const QPointF& scenePos, float viewerScale) const;

    // Find all grid cells whose 3D world positions are within radius of the given world position
    std::vector<std::pair<int, int>> findGridCellsInSphere(const cv::Vec3f& worldPos, float radius) const;

    // Find all grid cells within a cylinder centered at worldPos.
    // The cylinder has its axis along planeNormal, with given radius and depth (half-depth on each side).
    // For plane viewers. If outMinDist is provided, returns the minimum distance to any surface point.
    std::vector<std::pair<int, int>> findGridCellsInCylinder(const cv::Vec3f& worldPos,
                                                              const cv::Vec3f& planeNormal,
                                                              float radius,
                                                              float depth,
                                                              float* outMinDist = nullptr) const;

    // Paint accumulated points into QImage (for real-time painting)
    void paintAccumulatedPointsToImage();

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationWidget* _widget{nullptr};
    QuadSurface* _surface{nullptr};

    PaintMode _paintMode{PaintMode::Approve};
    bool _brushActive{false};
    bool _strokeActive{false};
    std::vector<cv::Vec3f> _currentStroke;
    std::vector<std::vector<cv::Vec3f>> _pendingStrokes;
    std::vector<cv::Vec3f> _overlayPoints;  // Current active stroke overlay
    std::vector<std::vector<cv::Vec3f>> _overlayStrokeSegments;  // Completed stroke segments for overlay
    cv::Vec3f _lastSample{0.0f, 0.0f, 0.0f};
    bool _hasLastSample{false};
    cv::Vec3f _lastOverlaySample{0.0f, 0.0f, 0.0f};
    bool _hasLastOverlaySample{false};

    // Throttling for overlay refresh during painting
    QElapsedTimer _lastRefreshTimer;
    qint64 _lastRefreshTime{0};
    bool _pendingRefresh{false};

    // Accumulated grid positions for real-time painting
    std::vector<std::pair<int, int>> _accumulatedGridPositions;
    std::unordered_set<uint64_t> _accumulatedGridPosSet;  // For deduplication

    // For plane viewer strokes: effective paint radius = brushRadius - distanceFromLine
    float _effectivePaintRadiusNative{0.0f};
    bool _usePlaneEffectiveRadius{false};

    // Current hover position for brush circle visualization
    std::optional<cv::Vec3f> _hoverWorldPos;
    float _hoverEffectiveRadius{0.0f};  // Cached effective radius for hover preview
    std::optional<QPointF> _hoverScenePos;  // Cached scene position (avoids expensive pointTo)
    float _hoverViewerScale{0.0f};          // Viewer scale for the cached scene position
    std::optional<cv::Vec3f> _hoverPlaneNormal;  // Plane normal when hovering in XY/XZ/YZ viewers

    // Cache for grid search optimization - avoids expensive pointTo calls during continuous painting
    mutable cv::Vec3f _lastSearchWorldPos{0.0f, 0.0f, 0.0f};
    mutable int _lastSearchGridRow{-1};
    mutable int _lastSearchGridCol{-1};
    mutable bool _hasLastSearchCache{false};

    // Spatial index for fast 3D point queries - built when surface is set
    PointIndex _pointIndex;
    int _pointIndexCols{0};  // Needed to decode ID back to (row, col)
};
