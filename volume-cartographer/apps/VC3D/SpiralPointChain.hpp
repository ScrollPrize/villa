#pragma once

#include <QPointF>

#include <opencv2/core/types.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace vc3d::spiral {

struct PointChainAnchor {
    QPointF surface;
    cv::Vec3f volume;
};

struct PointChainSample {
    QPointF surface;
    cv::Vec3f volume;
    bool isAnchor = false;
    std::size_t anchorIndex = 0;
};

enum class PointChainBuildError {
    None,
    DegenerateSpan,
    InvalidSurface,
    SelfIntersection,
};

struct PointChainBuildResult {
    std::vector<PointChainSample> samples;
    PointChainBuildError error = PointChainBuildError::None;
};

using SurfacePointSampler =
    std::function<std::optional<cv::Vec3f>(const QPointF&)>;

// Interpolate the anchors in their existing order. Each anchor is retained
// exactly, and derived samples are placed at spacing, 2*spacing, ... strictly
// inside each anchor-to-anchor span.
PointChainBuildResult buildPointChain(
    const std::vector<PointChainAnchor>& anchors,
    const SurfacePointSampler& sampleSurface,
    float spacing = 30.0f);

enum class AnchorEraseAction {
    None,
    Trim,
    DeleteChain,
};

struct AnchorEraseDecision {
    AnchorEraseAction action = AnchorEraseAction::None;
    std::size_t removePrefix = 0;
    std::size_t removeSuffix = 0;
};

// Touched anchors may be removed only when they form a prefix and/or suffix.
// Any touched anchor with an untouched anchor on both sides deletes the chain.
AnchorEraseDecision classifyAnchorErase(
    const std::vector<bool>& touchedAnchors);

} // namespace vc3d::spiral
