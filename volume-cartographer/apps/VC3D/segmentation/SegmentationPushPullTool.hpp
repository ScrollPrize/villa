#pragma once

#include "SegmentationTool.hpp"
#include "SegmentationPushPullConfig.hpp"

#include <memory>

class SegmentationEditManager;
class SegmentationWidget;
class SegmentationOverlayController;
class CSurfaceCollection;
class SegmentationModule;
class QTimer;

class SegmentationPushPullTool : public SegmentationTool
{
public:
    SegmentationPushPullTool(SegmentationModule& module,
                             SegmentationEditManager* editManager,
                             SegmentationWidget* widget,
                             SegmentationOverlayController* overlay,
                             CSurfaceCollection* surfaces);

    void setDependencies(SegmentationEditManager* editManager,
                         SegmentationWidget* widget,
                         SegmentationOverlayController* overlay,
                         CSurfaceCollection* surfaces);

    void setStepMultiplier(float multiplier);
    [[nodiscard]] float stepMultiplier() const { return _stepMultiplier; }

    void setAlphaEnabled(bool enabled);
    [[nodiscard]] bool alphaEnabled() const { return _alphaEnabled; }

    void setAlphaConfig(const AlphaPushPullConfig& config);
    [[nodiscard]] const AlphaPushPullConfig& alphaConfig() const { return _alphaConfig; }

    bool start(int direction);
    void stop(int direction);
    void stopAll();
    bool applyStep();

    void cancel() override { stopAll(); }
    [[nodiscard]] bool isActive() const override { return _state.active; }

private:
    bool applyStepInternal();
    void ensureTimer();

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationWidget* _widget{nullptr};
    SegmentationOverlayController* _overlay{nullptr};
    CSurfaceCollection* _surfaces{nullptr};

    struct State
    {
        bool active{false};
        int direction{0};
    };

    State _state;
    QTimer* _timer{nullptr};
    float _stepMultiplier{4.0f};
    bool _alphaEnabled{false};
    AlphaPushPullConfig _alphaConfig{};
    bool _undoCaptured{false};
};

