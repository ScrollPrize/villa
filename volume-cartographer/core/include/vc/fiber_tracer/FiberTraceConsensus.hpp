#pragma once

#include "vc/fiber_tracer/FiberTraceConstraints.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberTraceConsensusLabel : unsigned char {
    Unassigned,
    H,
    V,
    Broken,
};

struct FiberTraceConsensusConfig {
    double brokenCostPerConstraint = 0.5;
    cv::Vec3d cropMinimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d cropMaximumBaseXYZ{0.0, 0.0, 0.0};
};

struct FiberTraceConsensusStep {
    std::size_t addedCount = 0;
    std::size_t traceIndex = 0;
    std::size_t componentIndex = 0;
    FiberTraceConsensusLabel label = FiberTraceConsensusLabel::Unassigned;
    bool componentSeed = false;
    double seedStraightness = 0.0;
    double seedCenterDistanceBaseVoxels = 0.0;
    double seedArcLengthBaseVoxels = 0.0;
    std::size_t evidenceCount = 0;
    double meanDistanceBaseVoxels = 0.0;
    double connectivityScore = 0.0;
    double hCost = 0.0;
    double vCost = 0.0;
    double brokenCost = 0.0;
    double selectedCost = 0.0;
};

struct FiberTraceConsensusReport {
    std::vector<FiberTraceConsensusLabel> labels;
    std::vector<FiberTraceConsensusStep> steps;
    std::vector<std::size_t> snapshotAddedCounts;
    std::array<std::size_t, 4> labelCounts{};
    std::size_t components = 0;
    std::size_t degenerateTraces = 0;
    std::size_t retainedCrossTraceConstraints = 0;
    double orientationCost = 0.0;
    double brokenCost = 0.0;
    double objective = 0.0;
};

struct FiberTraceConsensusObjPaths {
    std::filesystem::path h;
    std::filesystem::path v;
    std::filesystem::path broken;
};

struct FiberTraceConsensusSnapshotObjReport {
    std::size_t addedCount = 0;
    FiberTraceConsensusObjPaths paths;
    std::size_t hCount = 0;
    std::size_t vCount = 0;
    std::size_t brokenCount = 0;
};

struct FiberTraceConsensusObjReport {
    FiberTraceConsensusObjPaths finalPaths;
    std::size_t hCount = 0;
    std::size_t vCount = 0;
    std::size_t brokenCount = 0;
    std::vector<FiberTraceConsensusSnapshotObjReport> snapshots;
};

[[nodiscard]] FiberTraceConsensusReport growFiberTraceConsensus(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceConsensusConfig& config);

[[nodiscard]] FiberTraceConsensusObjReport writeFiberTraceConsensusObjs(
    const std::vector<FiberletCropTraceLine>& traces, const FiberTraceConsensusReport& consensus, const std::filesystem::path& outputBase);

}  // namespace vc::fiber_tracer
