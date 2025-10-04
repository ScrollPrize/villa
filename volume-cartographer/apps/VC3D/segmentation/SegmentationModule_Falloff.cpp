#include "SegmentationModule.hpp"

#include "SegmentationEditManager.hpp"
#include "SegmentationLineTool.hpp"
#include "SegmentationPushPullTool.hpp"
#include "SegmentationWidget.hpp"

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kFloatEpsilon = 1e-4f;

bool nearlyEqual(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < kFloatEpsilon;
}
}

void SegmentationModule::setDragRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (nearlyEqual(sanitized, _dragRadiusSteps)) {
        return;
    }
    _dragRadiusSteps = sanitized;
    if (_activeFalloff == FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }
    if (_widget) {
        _widget->setDragRadius(_dragRadiusSteps);
    }
}

void SegmentationModule::setDragSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (nearlyEqual(sanitized, _dragSigmaSteps)) {
        return;
    }
    _dragSigmaSteps = sanitized;
    if (_activeFalloff == FalloffTool::Drag) {
        useFalloff(FalloffTool::Drag);
    }
    if (_widget) {
        _widget->setDragSigma(_dragSigmaSteps);
    }
}

void SegmentationModule::setLineRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (nearlyEqual(sanitized, _lineRadiusSteps)) {
        return;
    }
    _lineRadiusSteps = sanitized;
    if (_activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Line);
    }
    if (_widget) {
        _widget->setLineRadius(_lineRadiusSteps);
    }
}

void SegmentationModule::setLineSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (nearlyEqual(sanitized, _lineSigmaSteps)) {
        return;
    }
    _lineSigmaSteps = sanitized;
    if (_activeFalloff == FalloffTool::Line) {
        useFalloff(FalloffTool::Line);
    }
    if (_widget) {
        _widget->setLineSigma(_lineSigmaSteps);
    }
}

void SegmentationModule::setPushPullRadius(float radiusSteps)
{
    const float sanitized = std::clamp(radiusSteps, 0.25f, 128.0f);
    if (nearlyEqual(sanitized, _pushPullRadiusSteps)) {
        return;
    }
    _pushPullRadiusSteps = sanitized;
    if (_activeFalloff == FalloffTool::PushPull) {
        useFalloff(FalloffTool::PushPull);
    }
    if (_widget) {
        _widget->setPushPullRadius(_pushPullRadiusSteps);
    }
}

void SegmentationModule::setPushPullSigma(float sigmaSteps)
{
    const float sanitized = std::clamp(sigmaSteps, 0.05f, 64.0f);
    if (nearlyEqual(sanitized, _pushPullSigmaSteps)) {
        return;
    }
    _pushPullSigmaSteps = sanitized;
    if (_activeFalloff == FalloffTool::PushPull) {
        useFalloff(FalloffTool::PushPull);
    }
    if (_widget) {
        _widget->setPushPullSigma(_pushPullSigmaSteps);
    }
}

float SegmentationModule::falloffRadius(FalloffTool tool) const
{
    switch (tool) {
    case FalloffTool::Drag:
        return _dragRadiusSteps;
    case FalloffTool::Line:
        return _lineRadiusSteps;
    case FalloffTool::PushPull:
        return _pushPullRadiusSteps;
    }
    return _dragRadiusSteps;
}

float SegmentationModule::falloffSigma(FalloffTool tool) const
{
    switch (tool) {
    case FalloffTool::Drag:
        return _dragSigmaSteps;
    case FalloffTool::Line:
        return _lineSigmaSteps;
    case FalloffTool::PushPull:
        return _pushPullSigmaSteps;
    }
    return _dragSigmaSteps;
}

void SegmentationModule::updateOverlayFalloff(FalloffTool)
{
    refreshOverlay();
}

void SegmentationModule::useFalloff(FalloffTool tool)
{
    _activeFalloff = tool;
    const float radius = falloffRadius(tool);
    const float sigma = falloffSigma(tool);
    if (_editManager) {
        _editManager->setRadius(radius);
        _editManager->setSigma(sigma);
    }
    updateOverlayFalloff(tool);
}

void SegmentationModule::setPushPullStepMultiplier(float multiplier)
{
    const float sanitized = std::clamp(multiplier, 0.05f, 10.0f);
    if (_pushPullTool && std::fabs(sanitized - _pushPullTool->stepMultiplier()) < kFloatEpsilon) {
        if (_widget && std::fabs(_widget->pushPullStep() - sanitized) >= kFloatEpsilon) {
            _widget->setPushPullStep(sanitized);
        }
        return;
    }
    if (_pushPullTool) {
        _pushPullTool->setStepMultiplier(sanitized);
    }
    if (_widget && std::fabs(_widget->pushPullStep() - sanitized) >= kFloatEpsilon) {
        _widget->setPushPullStep(sanitized);
    }
}

void SegmentationModule::setSmoothingStrength(float strength)
{
    const float clamped = std::clamp(strength, 0.0f, 1.0f);
    if (std::fabs(clamped - _smoothStrength) < kFloatEpsilon) {
        return;
    }
    _smoothStrength = clamped;
    if (_widget) {
        _widget->setSmoothingStrength(_smoothStrength);
    }
    if (_lineTool) {
        _lineTool->setSmoothing(_smoothStrength, _smoothIterations);
    }
}

void SegmentationModule::setSmoothingIterations(int iterations)
{
    const int clamped = std::clamp(iterations, 1, 25);
    if (_smoothIterations == clamped) {
        return;
    }
    _smoothIterations = clamped;
    if (_widget) {
        _widget->setSmoothingIterations(_smoothIterations);
    }
    if (_lineTool) {
        _lineTool->setSmoothing(_smoothStrength, _smoothIterations);
    }
}

void SegmentationModule::setAlphaPushPullConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = SegmentationPushPullTool::sanitizeConfig(config);
    if (_pushPullTool && SegmentationPushPullTool::configsEqual(_pushPullTool->alphaConfig(), sanitized)) {
        if (_widget) {
            _widget->setAlphaPushPullConfig(sanitized);
        }
        return;
    }
    if (_pushPullTool) {
        _pushPullTool->setAlphaConfig(sanitized);
    }
    if (_widget) {
        _widget->setAlphaPushPullConfig(sanitized);
    }
}
