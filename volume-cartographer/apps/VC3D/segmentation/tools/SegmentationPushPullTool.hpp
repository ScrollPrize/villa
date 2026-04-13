#pragma once

#include "SegmentationTool.hpp"
#include "../SegmentationPushPullConfig.hpp"

#include <memory>
#include <optional>

#include <opencv2/core.hpp>

class SegmentationEditManager;
class SegmentationWidget;
class SegmentationOverlayController;
class CState;
class SegmentationModule;
class CTiledVolumeViewer;
class QuadSurface;
class QTimer;

class SegmentationPushPullTool : public SegmentationTool
{
public:
    SegmentationPushPullTool(SegmentationModule& module,
                             SegmentationEditManager* editManager,
                             SegmentationWidget* widget,
                             SegmentationOverlayController* overlay,
                             CState* state);

    void setDependencies(SegmentationEditManager* editManager,
                         SegmentationWidget* widget,
                         SegmentationOverlayController* overlay,
                         CState* state);

    void setStepMultiplier(float multiplier);
    [[nodiscard]] float stepMultiplier() const { return _stepMultiplier; }

    void setAlphaConfig(const AlphaPushPullConfig& config);
    [[nodiscard]] const AlphaPushPullConfig& alphaConfig() const { return _alphaConfig; }

    static AlphaPushPullConfig sanitizeConfig(const AlphaPushPullConfig& config);
    static bool configsEqual(const AlphaPushPullConfig& lhs, const AlphaPushPullConfig& rhs);

    bool start(int direction, std::optional<bool> alphaOverride = std::nullopt);
    void stop(int direction);
    void stopAll();
    bool applyStep();

    void cancel() override { stopAll(); }
    [[nodiscard]] bool isActive() const override { return _ppState.active; }

private:
    bool applyStepInternal();
    void ensureTimer();
    std::optional<cv::Vec3f> computeAlphaTarget(const cv::Vec3f& centerWorld,
                                                const cv::Vec3f& normal,
                                                int direction,
                                                QuadSurface* surface,
                                                CTiledVolumeViewer* viewer,
                                                bool* outUnavailable) const;

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationWidget* _widget{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    CState* _state{nullptr};

    struct State
    {
        bool active{false};
        int direction{0};
    };

    State _ppState;
    QTimer* _timer{nullptr};
    float _stepMultiplier{4.0f};
    bool _activeAlphaEnabled{false};
    bool _alphaOverrideActive{false};
    AlphaPushPullConfig _alphaConfig{};
    bool _undoCaptured{false};

    // Cached state to avoid rebuilding samples every tick
    int _cachedRow{-1};
    int _cachedCol{-1};
    bool _samplesValid{false};
};
