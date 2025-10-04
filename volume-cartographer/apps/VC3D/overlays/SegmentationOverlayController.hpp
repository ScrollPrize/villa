#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QColor>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class CSurfaceCollection;
class SegmentationEditManager;
class Surface;

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

        bool operator==(const State& rhs) const;
        bool operator!=(const State& rhs) const { return !(*this == rhs); }
    };

    explicit SegmentationOverlayController(CSurfaceCollection* surfaces, QObject* parent = nullptr);

    void setEditingEnabled(bool enabled);
    void setEditManager(SegmentationEditManager* manager);
    void applyState(const State& state);

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

    ViewerOverlayControllerBase::PathPrimitive buildMaskPrimitive(const State& state) const;
    bool shouldShowMask(const State& state) const;

    CSurfaceCollection* _surfaces{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    bool _editingEnabled{false};
    std::optional<State> _currentState;
};
