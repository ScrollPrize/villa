#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <QFutureWatcher>
#include <QImage>
#include <atomic>
#include <chrono>
#include <array>
#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core/mat.hpp>

class CState;
class SegmentationEditManager;
class Surface;
class QuadSurface;
class PlaneSurface;
class ViewerManager;
class QTimer;

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
        struct ManualAddLine
        {
            std::vector<cv::Vec3f> worldPoints;
            std::vector<QPointF> surfacePoints;
            bool committed{false};
        };

        struct ManualAddFillSpan
        {
            int row{0};
            int colFirst{0};
            int colLast{0};
        };

        std::optional<VertexMarker> activeMarker;
        std::vector<VertexMarker> neighbours;
        std::vector<QPointF> surfaceMaskPoints;
        bool pushPullActive{false};
        float gaussianRadiusSteps{0.0f};
        float gaussianSigmaSteps{0.0f};
        float displayRadiusSteps{0.0f};
        float gridStepWorld{1.0f};

        // Correction drag state - for drag-and-drop corrections
        bool correctionDragActive{false};
        cv::Vec3f correctionDragStart{0.0f, 0.0f, 0.0f};
        cv::Vec3f correctionDragCurrent{0.0f, 0.0f, 0.0f};

        bool manualAddActive{false};
        std::vector<ManualAddLine> manualAddHoverLines;
        std::vector<ManualAddLine> manualAddCommittedLines;
        std::optional<QPointF> manualAddHoverVertex;
        std::vector<ManualAddFillSpan> manualAddHoverFillSpans;
        bool manualAddHoverCrossFill{false};
        std::vector<cv::Vec3f> manualAddPreviewVertices;
        std::vector<std::array<cv::Vec3f, 4>> manualAddPreviewQuads;
        std::vector<cv::Vec3f> manualAddPlaneConstraints;
        float manualAddTintOpacity{0.45f};
        uint64_t manualAddRevision{0};

        bool operator==(const State& rhs) const;
        bool operator!=(const State& rhs) const { return !(*this == rhs); }
    };

    explicit SegmentationOverlayController(CState* state, QObject* parent = nullptr);

    void setEditingEnabled(bool enabled);
    void setViewerManager(ViewerManager* manager) { _viewerManager = manager; }
    void applyState(const State& state);

    // Force refresh of all viewer overlays (bypasses state comparison optimization)

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer,
                           ViewerOverlayControllerBase::OverlayBuilder& builder) override;
    void detachViewer(VolumeViewerBase* viewer) override;

private slots:
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surface);

private:
    void buildRadiusOverlay(const State& state,
                            VolumeViewerBase* viewer,
                            ViewerOverlayControllerBase::OverlayBuilder& builder) const;
    void buildVertexMarkers(const State& state,
                            VolumeViewerBase* viewer,
                            ViewerOverlayControllerBase::OverlayBuilder& builder) const;
    void buildSurfaceOverlapOverlay(VolumeViewerBase* viewer,
                                    ViewerOverlayControllerBase::OverlayBuilder& builder);


    CState* _state{nullptr};
    ViewerManager* _viewerManager{nullptr};
    bool _editingEnabled{false};
    std::optional<State> _currentState;
    std::chrono::steady_clock::time_point _lastRefreshTime;

    // Surface overlap overlay cache
    struct OverlapCache {
        QImage image;                                       // ARGB32 overlap image (grid-sized)
        std::uint64_t overlaysRevision{0};                  // Selected overlay-set revision
        std::uint64_t surfacesVersion{0};                   // CState surface map version
        float threshold{0.0f};                              // Threshold when image was built
        QuadSurface* surface{nullptr};                      // Current surface when image was built
    };
    mutable std::map<VolumeViewerBase*, OverlapCache> _overlapCaches;

    // Async overlap computation
    struct PendingOverlapResult {
        VolumeViewerBase* viewer{nullptr};
        OverlapCache cache;
    };
    QFutureWatcher<QImage>* _overlapWatcher{nullptr};
    PendingOverlapResult _pendingOverlap;
    bool _overlapComputeRunning{false};
    // Cooperative cancel for the in-flight compute; a fresh flag is created per
    // launch so a canceled worker can't clear a successor's flag.
    std::shared_ptr<std::atomic_bool> _overlapCancel;
    void handleOverlapComputeFinished();

    // Deferred refresh timer - fires after throttle window to apply pending state
    QTimer* _deferredRefreshTimer{nullptr};
};
