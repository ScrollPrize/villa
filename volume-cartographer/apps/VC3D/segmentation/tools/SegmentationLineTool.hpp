#pragma once

#include "SegmentationTool.hpp"

#include <opencv2/core.hpp>

#include <vector>

class SegmentationEditManager;
class CState;
class SegmentationModule;

class SegmentationLineTool : public SegmentationTool
{
public:
    SegmentationLineTool(SegmentationModule& module,
                         SegmentationEditManager* editManager,
                         CState* state,
                         float& smoothStrength,
                         int& smoothIterations);

    void setDependencies(SegmentationEditManager* editManager,
                         CState* state);
    void setSmoothing(float& smoothStrength, int& smoothIterations);

    void startStroke(const cv::Vec3f& worldPos);
    void extendStroke(const cv::Vec3f& worldPos, bool forceSample);
    void finishStroke(bool keepLineFalloff);
    bool applyStroke(const std::vector<cv::Vec3f>& stroke);
    void clear();

    [[nodiscard]] bool strokeActive() const { return _strokeActive; }
    [[nodiscard]] const std::vector<cv::Vec3f>& overlayPoints() const { return _overlayPoints; }

    void cancel() override { clear(); }
    [[nodiscard]] bool isActive() const override { return _strokeActive; }

private:
    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    CState* _state{nullptr};
    float* _smoothStrength{nullptr};
    int* _smoothIterations{nullptr};

    bool _strokeActive{false};
    std::vector<cv::Vec3f> _strokePoints;
    std::vector<cv::Vec3f> _overlayPoints;
    cv::Vec3f _lastSample{0.0f, 0.0f, 0.0f};
    bool _hasLastSample{false};
};

