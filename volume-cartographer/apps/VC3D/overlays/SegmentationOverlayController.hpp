#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <chrono>
#include <deque>
#include <map>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class CSurfaceCollection;
class SegmentationEditManager;
class Surface;
class QuadSurface;
class PlaneSurface;
class ViewerManager;

class SegmentationOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct VertexMarker
    {
        int row{0};
        int col{0};
        cv::Vec3f world{0.0f, 0.0f, 0.0f};
        bool isActive{false};
        bool isGrowth{false};
    };

    struct State
    {
        enum class FalloffMode
        {
            Drag,
            Line,
            PushPull
        };

        std::optional<VertexMarker> activeMarker;
        std::vector<VertexMarker> neighbours;
        std::vector<cv::Vec3f> maskPoints;
        bool maskVisible{false};
        bool brushActive{false};
        bool brushStrokeActive{false};
        bool lineStrokeActive{false};
        bool hasLineStroke{false};
        bool pushPullActive{false};
        FalloffMode falloff{FalloffMode::Drag};
        float gaussianRadiusSteps{0.0f};
        float gaussianSigmaSteps{0.0f};
        float displayRadiusSteps{0.0f};
        float gridStepWorld{1.0f};

        // Approval mask state - cylinder brush model
        // Radius = circle in plane views (XY/XZ/YZ), rectangle width in flattened view
        // Depth = cylinder thickness, rectangle height in flattened view
        bool approvalMaskMode{false};
        bool approvalStrokeActive{false};
        std::vector<std::vector<cv::Vec3f>> approvalStrokeSegments;  // Completed segments
        std::vector<cv::Vec3f> approvalCurrentStroke;  // Current active stroke
        float approvalBrushRadius{50.0f};     // Cylinder radius (native voxels)
        float approvalBrushDepth{15.0f};      // Cylinder depth (native voxels)
        float approvalEffectiveRadius{0.0f};  // For plane viewers: brush radius adjusted for distance
        bool paintingApproval{true};
        QuadSurface* surface{nullptr};
        std::optional<cv::Vec3f> approvalHoverWorld;  // Current hover position for brush circle
        std::optional<QPointF> approvalHoverScenePos; // Scene position (avoids expensive pointTo)
        float approvalHoverViewerScale{1.0f};         // Viewer scale for the hover position
        std::optional<cv::Vec3f> approvalHoverPlaneNormal;  // Plane normal when hovering in XY/XZ/YZ viewers

        bool operator==(const State& rhs) const;
        bool operator!=(const State& rhs) const { return !(*this == rhs); }
    };

    explicit SegmentationOverlayController(CSurfaceCollection* surfaces, QObject* parent = nullptr);

    void setEditingEnabled(bool enabled);
    void setEditManager(SegmentationEditManager* manager);
    void setViewerManager(ViewerManager* manager) { _viewerManager = manager; }
    void applyState(const State& state);

    // Load approval mask from surface into QImage (call once when entering approval mode)
    void loadApprovalMaskImage(QuadSurface* surface);

    // Paint directly into the approval mask QImage (fast, in-place editing)
    // If useRectangle is true, paints a rectangle using widthSteps x heightSteps dimensions
    // If useRectangle is false, paints a circle using radiusSteps
    void paintApprovalMaskDirect(const std::vector<std::pair<int, int>>& gridPositions,
                                  float radiusSteps,
                                  uint8_t paintValue,
                                  bool useRectangle = false,
                                  float widthSteps = 0.0f,
                                  float heightSteps = 0.0f);

    // Save the approval mask QImage back to the surface
    void saveApprovalMaskToSurface(QuadSurface* surface);

    // Undo support for approval mask painting
    // Undo the last paint stroke (repaints with inverse value)
    bool undoLastApprovalMaskPaint();
    // Check if there are any undo operations available
    [[nodiscard]] bool canUndoApprovalMaskPaint() const;
    // Clear all undo history (e.g., when applying changes to disk)
    void clearApprovalMaskUndoHistory();

    // Query approval status for a grid position (integer coords, nearest neighbor)
    // Returns: 0 = not approved, 1 = saved approved, 2 = pending approved, 3 = pending unapproved
    int queryApprovalStatus(int row, int col) const;

    // Query approval value with bilinear interpolation (float coords)
    // Returns approval intensity 0.0-1.0 using bilinear interpolation for smooth edges
    // Also returns status: 0 = not approved, 1 = saved, 2 = pending approve, 3 = pending unapprove
    float queryApprovalBilinear(float row, float col, int* outStatus = nullptr) const;

    // Check if approval mask mode is active and we have mask data
    bool hasApprovalMaskData() const;

    // Trigger re-rendering of intersections on all plane viewers
    void invalidatePlaneIntersections();

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer,
                           ViewerOverlayControllerBase::OverlayBuilder& builder) override;

private slots:
    void onSurfaceChanged(std::string name, Surface* surface);

private:
    void buildRadiusOverlay(const State& state,
                            CVolumeViewer* viewer,
                            ViewerOverlayControllerBase::OverlayBuilder& builder) const;
    void buildVertexMarkers(const State& state,
                            CVolumeViewer* viewer,
                            ViewerOverlayControllerBase::OverlayBuilder& builder) const;
    void buildApprovalMaskOverlay(const State& state,
                                  CVolumeViewer* viewer,
                                  ViewerOverlayControllerBase::OverlayBuilder& builder) const;

    ViewerOverlayControllerBase::PathPrimitive buildMaskPrimitive(const State& state) const;
    bool shouldShowMask(const State& state) const;

    CSurfaceCollection* _surfaces{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    ViewerManager* _viewerManager{nullptr};
    bool _editingEnabled{false};
    std::optional<State> _currentState;
    std::chrono::steady_clock::time_point _lastRefreshTime;

    // Approval mask images - separate saved and pending
    QImage _savedApprovalMaskImage;   // Dark green: what's saved to disk
    QImage _pendingApprovalMaskImage; // Light green: pending strokes

    // Per-viewer cached scene-space rasterized images
    struct ViewerImageCache {
        QImage compositeImage;
        QPointF topLeft;
        qreal scale{1.0};
        QuadSurface* surface{nullptr};
        uint64_t savedImageVersion{0};
        uint64_t pendingImageVersion{0};
    };
    mutable std::map<CVolumeViewer*, ViewerImageCache> _viewerCaches;
    mutable uint64_t _savedImageVersion{0};
    mutable uint64_t _pendingImageVersion{0};

    void rebuildViewerCache(CVolumeViewer* viewer, QuadSurface* surface) const;

    // Bilinear interpolation helper for QImage alpha channel
    // Returns interpolated alpha value (0.0-255.0) at floating point coordinates
    static float sampleImageBilinear(const QImage& image, float row, float col);

    // Undo stack for approval mask painting - stores affected region before painting
    struct ApprovalMaskUndoEntry {
        QImage savedRegion;  // Copy of the affected region before painting
        QPoint topLeft;      // Position of the saved region in the full image
    };
    std::deque<ApprovalMaskUndoEntry> _approvalMaskUndoStack;
    static constexpr size_t kMaxUndoEntries = 100;
};
