#pragma once

#include <cstdint>

#include <opencv2/core.hpp>
#include <vector>

struct ThinningStats {
    double distanceTransformSeconds = 0.0;
    double seedDetectionSeconds = 0.0;
    double tracePathsSeconds = 0.0;
    uint64_t seedCount = 0;
    uint64_t traceCount = 0;
    uint64_t traceSteps = 0;
    uint64_t candidateEvaluations = 0;

    void accumulate(const ThinningStats& other) {
        distanceTransformSeconds += other.distanceTransformSeconds;
        seedDetectionSeconds += other.seedDetectionSeconds;
        tracePathsSeconds += other.tracePathsSeconds;
        seedCount += other.seedCount;
        traceCount += other.traceCount;
        traceSteps += other.traceSteps;
        candidateEvaluations += other.candidateEvaluations;
    }
};

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<std::vector<cv::Point>>* traces = nullptr);
void customThinning(const cv::Mat& inputImage,
                    cv::Mat& outputImage,
                    std::vector<std::vector<cv::Point>>* traces,
                    ThinningStats* stats);
void customThinningTraceOnly(const cv::Mat& inputImage,
                             std::vector<std::vector<cv::Point>>& traces,
                             ThinningStats* stats = nullptr);
