#pragma once

#include "ViewerOverlayControllerBase.hpp"

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

    explicit SegmentationOverlayController(CSurfaceCollection* surfaces, QObject* parent = nullptr);

    void setEditingEnabled(bool enabled);
    void setEditManager(SegmentationEditManager* manager);
    void setGaussianParameters(float radiusSteps, float sigmaSteps, float gridStepWorld);
    void setActiveVertex(std::optional<VertexMarker> marker);
    void setTouchedVertices(const std::vector<VertexMarker>& markers);
    void setMaskOverlay(const std::vector<cv::Vec3f>& points,
                        bool visible,
                        float pointRadius,
                        float opacity);

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer,
                           ViewerOverlayControllerBase::OverlayBuilder& builder) override;

private slots:
    void onSurfaceChanged(std::string name, Surface* surface);

private:
    void buildRadiusOverlay(CVolumeViewer* viewer,
                            ViewerOverlayControllerBase::OverlayBuilder& builder) const;
    void buildVertexMarkers(CVolumeViewer* viewer,
                            ViewerOverlayControllerBase::OverlayBuilder& builder) const;

    CSurfaceCollection* _surfaces{nullptr};
    SegmentationEditManager* _editManager{nullptr};
    bool _editingEnabled{false};
    float _radiusSteps{3.0f};
    float _sigmaSteps{1.5f};
    float _gridStepWorld{1.0f};

    std::optional<VertexMarker> _activeVertex;
    std::vector<VertexMarker> _touchedVertices;

    bool _maskVisible{false};
    std::vector<cv::Vec3f> _maskPoints;
    float _maskPointRadius{3.0f};
    float _maskOpacity{0.35f};
};
