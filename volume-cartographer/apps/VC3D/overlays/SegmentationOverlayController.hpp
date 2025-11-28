#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QColor>
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

        // Approval mask state
        bool approvalMaskMode{false};
        bool approvalStrokeActive{false};
        std::vector<std::vector<cv::Vec3f>> approvalStrokeSegments;  // Completed segments
        std::vector<cv::Vec3f> approvalCurrentStroke;  // Current active stroke
        float approvalBrushRadius{5.0f};
        bool paintingApproval{true};
        QuadSurface* surface{nullptr};

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
    void paintApprovalMaskDirect(const std::vector<std::pair<int, int>>& gridPositions,
                                  float radiusSteps,
                                  uint8_t paintValue);

    // Save the approval mask QImage back to the surface
    void saveApprovalMaskToSurface(QuadSurface* surface);

    // Query approval status for a grid position
    // Returns: 0 = not approved, 1 = saved approved, 2 = pending approved, 3 = pending unapproved
    int queryApprovalStatus(int row, int col) const;

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
};
