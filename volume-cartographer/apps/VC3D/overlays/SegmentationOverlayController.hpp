#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class CSurfaceCollection;
class CVolumeViewer;
class SegmentationEditManager;
class Surface;

class SegmentationOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit SegmentationOverlayController(CSurfaceCollection* surfCollection, QObject* parent = nullptr);

    void setEditingEnabled(bool enabled);
    void setDownsample(int value);
    void setRadius(float radius);
    void setEditManager(SegmentationEditManager* manager) { _editManager = manager; }
    void setActiveHandle(std::optional<std::pair<int,int>> key, bool refresh = true);
    void setHoverHandle(std::optional<std::pair<int,int>> key, bool refresh = true);
    void setKeyboardHandle(std::optional<std::pair<int,int>> key, bool refresh = true);
    void setHandleVisibility(bool showAll, float distance);
    void setCursorWorld(const cv::Vec3f& world, bool valid);

private slots:
    void onSurfaceChanged(std::string name, Surface* surf);

private:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer,
                           ViewerOverlayControllerBase::OverlayBuilder& builder) override;
    [[nodiscard]] float gridStepWorld() const;
    [[nodiscard]] float planeToleranceWorld() const;

    CSurfaceCollection* _surfCollection;
    bool _editingEnabled{false};
    int _downsample{12};
    float _radius{1.0f};
    SegmentationEditManager* _editManager{nullptr};
    std::optional<std::pair<int,int>> _activeHandle;
    std::optional<std::pair<int,int>> _hoverHandle;
    std::optional<std::pair<int,int>> _keyboardHandle;
    bool _showAllHandles{true};
    float _handleDisplayDistance{25.0f};
    bool _cursorValid{false};
    cv::Vec3f _cursorWorld{0.0f, 0.0f, 0.0f};
};
