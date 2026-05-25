#pragma once

#include "vc/lasagna/LineModel.hpp"

#include <string>
#include <vector>

namespace vc::lasagna {

struct LineOptimizationReport {
    double initialCost = 0.0;
    double finalCost = 0.0;
    int iterations = 0;
    int validNormalSamples = 0;
    int invalidNormalSamples = 0;
    bool converged = false;
    std::string message;
};

struct LineOptimizationResult {
    LineModel line;
    LineOptimizationReport report;
};

class LineOptimizer {
public:
    explicit LineOptimizer(const NormalSampler& normalSampler);

    [[nodiscard]] LineOptimizationResult optimizeFromSeed(
        const cv::Vec3d& seedPoint,
        const LineOptimizationConfig& config = {}) const;

    [[nodiscard]] LineOptimizationResult optimizeFromSeeds(
        const std::vector<cv::Vec3d>& seedPoints,
        const LineOptimizationConfig& config = {}) const;

private:
    const NormalSampler& normalSampler_;
};

} // namespace vc::lasagna
