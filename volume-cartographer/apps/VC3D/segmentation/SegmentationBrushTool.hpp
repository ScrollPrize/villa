#pragma once

#include "SegmentationTool.hpp"

#include <opencv2/core.hpp>

#include <vector>

class SegmentationEditManager;
class SegmentationWidget;
class CSurfaceCollection;
class SegmentationModule;

class SegmentationBrushTool : public SegmentationTool
{
public:
    SegmentationBrushTool(SegmentationModule& module,
                          SegmentationEditManager* editManager,
                          SegmentationWidget* widget,
                          CSurfaceCollection* surfaces);

    void setDependencies(SegmentationEditManager* editManager,
                         SegmentationWidget* widget,
                         CSurfaceCollection* surfaces);

    void setActive(bool active);
    [[nodiscard]] bool brushActive() const { return _brushActive; }
    [[nodiscard]] bool strokeActive() const { return _strokeActive; }
    [[nodiscard]] bool hasPendingStrokes() const { return !_pendingStrokes.empty(); }

    void startStroke(const cv::Vec3f& worldPos);
    void extendStroke(const cv::Vec3f& worldPos, bool forceSample);
    void finishStroke();
    bool applyPending(float dragRadiusSteps);
    void clear();

    [[nodiscard]] const std::vector<cv::Vec3f>& overlayPoints() const { return _overlayPoints; }
    [[nodiscard]] const std::vector<cv::Vec3f>& currentStrokePoints() const { return _currentStroke; }

    void cancel() override { clear(); }
    [[nodiscard]] bool isActive() const override { return brushActive() || strokeActive(); }

private:
    void ensureFalloff();

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationWidget* _widget{nullptr};
    CSurfaceCollection* _surfaces{nullptr};

    bool _brushActive{false};
    bool _strokeActive{false};
    std::vector<cv::Vec3f> _currentStroke;
    std::vector<std::vector<cv::Vec3f>> _pendingStrokes;
    std::vector<cv::Vec3f> _overlayPoints;
    cv::Vec3f _lastSample{0.0f, 0.0f, 0.0f};
    bool _hasLastSample{false};
};

