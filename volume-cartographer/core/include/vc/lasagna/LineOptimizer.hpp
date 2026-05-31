#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vc::lasagna {

struct LineOptimizationLossReport {
    std::string name;
    double weight = 0.0;
    int residuals = 0;
    double rawCost = 0.0;
    double weightedCost = 0.0;
};

struct LineOptimizationIterationReport {
    int iteration = 0;
    double cost = 0.0;
    double costChange = 0.0;
    double gradientMaxNorm = 0.0;
    double stepNorm = 0.0;
    double trustRegionRadius = 0.0;
    int linearSolverIterations = 0;
    bool stepSuccessful = false;
};

struct LineOptimizationReport {
    double initialCost = 0.0;
    double finalCost = 0.0;
    int iterations = 0;
    int validNormalSamples = 0;
    int invalidNormalSamples = 0;
    bool converged = false;
    int normalPrefetchCalls = 0;
    double ceresSolveMs = 0.0;
    double normalChunkPrefetchMs = 0.0;
    double normalMaterializeMs = 0.0;
    uint64_t normalPrefetchRequestedChunks = 0;
    uint64_t normalPrefetchChunksRead = 0;
    std::string message;
    std::vector<LineOptimizationLossReport> finalLosses;
    std::vector<LineOptimizationIterationReport> iterationProgress;
};

struct LineOptimizationResult {
    LineModel line;
    LineOptimizationReport report;
};

struct LineControlPoint {
    double linePosition = 0.0;
    cv::Vec3d volumePoint{0.0, 0.0, 0.0};
    bool isSeed = false;
    int optimizedIndex = -1;
};

struct LineControlPointUpdateResult {
    std::vector<cv::Vec3d> linePoints;
    std::vector<LineControlPoint> controlPoints;
    int changedControlIndex = -1;
    int activeStart = -1;
    int activeEnd = -1;
};

[[nodiscard]] LineControlPointUpdateResult updateExistingLineControlPoint(
    std::vector<cv::Vec3d> linePoints,
    std::vector<LineControlPoint> controlPoints,
    size_t changedControlIndex,
    double segmentLength);

[[nodiscard]] LineControlPointUpdateResult updateExistingLineControlPoint(
    std::vector<cv::Vec3d> linePoints,
    std::vector<LineControlPoint> controlPoints,
    size_t changedControlIndex,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config);

class LineOptimizer {
public:
    explicit LineOptimizer(const NormalSampler& normalSampler);

    [[nodiscard]] LineOptimizationResult optimizeFromSeed(
        const cv::Vec3d& seedPoint,
        const LineOptimizationConfig& config = {}) const;

    [[nodiscard]] LineOptimizationResult optimizeFromSeeds(
        const std::vector<cv::Vec3d>& seedPoints,
        const LineOptimizationConfig& config = {}) const;

    [[nodiscard]] LineOptimizationResult optimizeFromControlPoints(
        std::vector<LineControlPoint> controlPoints,
        const LineOptimizationConfig& config = {}) const;

    [[nodiscard]] LineOptimizationResult optimizeExistingLine(
        std::vector<cv::Vec3d> linePoints,
        std::vector<int> fixedPointIndices,
        int displayFrameAnchorIndex,
        const LineOptimizationConfig& config = {},
        int activeStart = -1,
        int activeEnd = -1,
        std::string candidateName = "existing-line+global") const;

private:
    const NormalSampler& normalSampler_;
};

} // namespace vc::lasagna
